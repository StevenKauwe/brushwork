//! The painting worker: receives `[shapes: Float32Array, seed, width, height, backend]`, paints with brushwork, and
//! posts back progress (a number from 0 to 1) and then `[pixels: Uint8ClampedArray (RGBA), milliseconds, backend]`.
//!
//! Where the strokes are laid: on the GPU when the browser offers WebGPU with a real adapter, otherwise on the CPU.
//! The worker finds out once, as it starts, and tells the page with a message `["backend", kind, detail]` (`kind`
//! is `gpu` or `cpu`). The GPU needs no canvas here: it is used for compute only, and its results are read back with
//! `await`, so the worker never blocks and the page's thread never touches the GPU. A request with `backend` set to
//! `cpu` is painted on the CPU whatever the worker found. If the GPU fails in the middle of a painting (the device is
//! lost, say), that painting is done again on the CPU, and the next one tries to open the GPU afresh, once.

// (the GPU half only exists in the browser build; natively this module is built for its tests and `paint`)
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

use crate::scene;

thread_local! {
    /// The last progress posted, so the page gets a message per 2 % rather than per shape.
    static LAST: Cell<f32> = const { Cell::new(-1.0) };
}

fn scope() -> Option<DedicatedWorkerGlobalScope> {
    js_sys::global().dyn_into::<DedicatedWorkerGlobalScope>().ok()
}

fn post(v: &JsValue) {
    if let Some(s) = scope() {
        let _ = s.post_message(v);
    }
}

fn progress(f: f32) {
    if f - LAST.with(Cell::get) >= 0.02 {
        LAST.with(|l| l.set(f));
        post(&JsValue::from_f64(f as f64));
    }
}

/// The painted region (`w` x `h` at the top left of brushwork's canvas) of a frame, as RGBA.
fn crop(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (w, h) = (w.min(brushwork::WI), h.min(brushwork::HI));
    let mut out = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for px in rgb[y * brushwork::WI * 3..(y * brushwork::WI + w) * 3].as_chunks::<3>().0 {
            out.extend_from_slice(px);
            out.push(255);
        }
    }
    out
}

/// Paint `shapes` (canvas pixels in a `w` x `h` region at the top left of brushwork's canvas) on the CPU: RGBA for
/// the region.
pub fn paint(shapes: Vec<scene::Shape>, seed: u32, w: usize, h: usize) -> Vec<u8> {
    brushwork::set_style(scene::STYLE);
    crop(&brushwork::paint_frame(&scene::film(shapes), 0.0, seed), w, h)
}

/// One paint request from the page.
struct Request {
    shapes: Vec<scene::Shape>,
    seed: u32,
    w: usize,
    h: usize,
    /// the page asked for the CPU
    cpu: bool,
}

impl Request {
    fn from(data: &JsValue) -> Request {
        let msg = js_sys::Array::from(data);
        Request {
            shapes: scene::decode(&js_sys::Float32Array::new(&msg.get(0)).to_vec()),
            seed: msg.get(1).as_f64().unwrap_or(1.0) as u32,
            w: msg.get(2).as_f64().unwrap_or(1024.0) as usize,
            h: msg.get(3).as_f64().unwrap_or(576.0) as usize,
            cpu: msg.get(4).as_string().is_some_and(|b| b == "cpu"),
        }
    }
}

/// What the worker knows about the GPU.
#[derive(Clone, Copy, PartialEq)]
enum Gpu {
    /// not looked for yet
    Unknown,
    Ready,
    /// none usable
    Missing,
    /// it failed during a painting: look for it again before the next one (only after its first failure; after a
    /// second the worker stays on the CPU)
    Failed,
}

thread_local! {
    static QUEUE: RefCell<VecDeque<Request>> = const { RefCell::new(VecDeque::new()) };
    /// a task is working through the queue
    static RUNNING: Cell<bool> = const { Cell::new(false) };
    static GPU: Cell<Gpu> = const { Cell::new(Gpu::Unknown) };
    /// paintings the GPU has failed
    static FAILS: Cell<u32> = const { Cell::new(0) };
}

/// Tell the page which backend paints, and why.
fn announce(kind: &str, detail: &str) {
    let msg = js_sys::Array::new();
    msg.push(&JsValue::from_str("backend"));
    msg.push(&JsValue::from_str(kind));
    msg.push(&JsValue::from_str(detail));
    post(&msg);
}

/// Look for the GPU (again): the browser's WebGPU, with an adapter that is a real graphics card.
#[cfg(target_arch = "wasm32")]
async fn look_for_gpu() {
    match brushwork::open_gpu().await {
        Ok(name) => {
            brushwork::set_backend(brushwork::Backend::Gpu);
            GPU.with(|g| g.set(Gpu::Ready));
            announce("gpu", if name.is_empty() { "WebGPU" } else { &name });
        }
        Err(why) => {
            brushwork::set_backend(brushwork::Backend::Cpu);
            GPU.with(|g| g.set(Gpu::Missing));
            announce("cpu", &why);
        }
    }
}

/// Paint one request, on the GPU if there is one and it was not declined: the pixels and the backend that laid them.
#[cfg(target_arch = "wasm32")]
async fn paint_request(r: Request) -> (Vec<u8>, &'static str) {
    if GPU.with(Cell::get) == Gpu::Ready && !r.cpu {
        brushwork::set_style(scene::STYLE);
        brushwork::set_backend(brushwork::Backend::Gpu);
        match brushwork::paint_frame_async(&scene::film(r.shapes.clone()), 0.0, r.seed).await {
            Ok(rgb) => return (crop(&rgb, r.w, r.h), "gpu"),
            Err(why) => {
                let fails = FAILS.with(|f| f.replace(f.get() + 1)) + 1;
                GPU.with(|g| g.set(if fails < 2 { Gpu::Failed } else { Gpu::Missing }));
                announce("cpu", &format!("the GPU failed ({why}); painted on the CPU"));
                LAST.with(|l| l.set(-1.0));
            }
        }
    }
    brushwork::set_backend(brushwork::Backend::Cpu);
    (paint(r.shapes, r.seed, r.w, r.h), "cpu")
}

/// Work through the queued requests, one at a time (looking for the GPU first, if that is still to be done).
#[cfg(target_arch = "wasm32")]
async fn run_queue() {
    while let Some(r) = QUEUE.with(|q| q.borrow_mut().pop_front()) {
        if matches!(GPU.with(Cell::get), Gpu::Unknown | Gpu::Failed) {
            look_for_gpu().await;
        }
        LAST.with(|l| l.set(-1.0));
        let t0 = js_sys::Date::now();
        let (rgba, kind) = paint_request(r).await;
        let ms = js_sys::Date::now() - t0;
        let reply = js_sys::Array::new();
        reply.push(&js_sys::Uint8ClampedArray::from(rgba.as_slice()));
        reply.push(&JsValue::from_f64(ms));
        reply.push(&JsValue::from_str(kind));
        post(&reply);
    }
    RUNNING.with(|r| r.set(false));
}

/// Start answering paint requests (called by worker.js once the module is loaded).
#[wasm_bindgen]
pub fn worker_start() {
    let Some(s) = scope() else { return };
    brushwork::set_progress(Some(progress));
    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        QUEUE.with(|q| q.borrow_mut().push_back(Request::from(&e.data())));
        if !RUNNING.with(|r| r.replace(true)) {
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(run_queue());
        }
    });
    s.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();
    // look for the GPU straight away, so the page can say which backend paints before anything is asked of it
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async {
        if !RUNNING.with(|r| r.replace(true)) {
            look_for_gpu().await;
            run_queue().await;
        }
    });
}
