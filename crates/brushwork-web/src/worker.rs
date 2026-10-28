//! The painting worker: receives `[shapes: Float32Array, seed, width, height]`, paints with brushwork on the CPU,
//! and posts back progress (a number from 0 to 1) and then `[pixels: Uint8ClampedArray (RGBA), milliseconds]`.

use std::cell::Cell;

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

fn progress(f: f32) {
    if f - LAST.with(Cell::get) >= 0.02 {
        LAST.with(|l| l.set(f));
        if let Some(s) = scope() {
            let _ = s.post_message(&JsValue::from_f64(f as f64));
        }
    }
}

/// Paint `shapes` (canvas pixels in a `w` x `h` region at the top left of brushwork's canvas): RGBA for the region.
pub fn paint(shapes: Vec<scene::Shape>, seed: u32, w: usize, h: usize) -> Vec<u8> {
    brushwork::set_style(scene::STYLE);
    let rgb = brushwork::paint_frame(&scene::film(shapes), 0.0, seed);
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

/// Start answering paint requests (called by worker.js once the module is loaded).
#[wasm_bindgen]
pub fn worker_start() {
    let Some(s) = scope() else { return };
    brushwork::set_progress(Some(progress));
    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        let msg = js_sys::Array::from(&e.data());
        let shapes = scene::decode(&js_sys::Float32Array::new(&msg.get(0)).to_vec());
        let seed = msg.get(1).as_f64().unwrap_or(1.0) as u32;
        let w = msg.get(2).as_f64().unwrap_or(1024.0) as usize;
        let h = msg.get(3).as_f64().unwrap_or(576.0) as usize;
        LAST.with(|l| l.set(-1.0));
        let t0 = js_sys::Date::now();
        let rgba = paint(shapes, seed, w, h);
        let ms = js_sys::Date::now() - t0;
        let reply = js_sys::Array::new();
        reply.push(&js_sys::Uint8ClampedArray::from(rgba.as_slice()));
        reply.push(&JsValue::from_f64(ms));
        if let Some(s) = scope() {
            let _ = s.post_message(&reply);
        }
    });
    s.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();
}
