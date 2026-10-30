//! The page: lay down flat shapes with a finger or the mouse, then Paint hands them to the worker, which paints
//! them in oil, on the GPU where the browser offers WebGPU, else on the CPU. The canvas says where it is in
//! `data-state` (`flat`, `painting`, `painted`), how long the last painting took in `data-paint-ms` and what laid
//! its strokes in `data-paint-backend` (`gpu` or `cpu`); the small `#backend` label over the canvas says the same to
//! the visitor (`data-kind`). `?backend=cpu` in the address keeps the painting on the CPU. A drawn outline is calmed
//! by [`stabilize`] (jitter out, corners and size kept) before it becomes a shape.

use std::cell::RefCell;
use std::rc::Rc;

use brushwork_kit::{Col, rgb};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, Document, HtmlCanvasElement, HtmlElement, ImageData, MessageEvent, PointerEvent, Worker, WorkerOptions, WorkerType};

use crate::scene::{self, FROM, Kind, LIGHTS, PALETTE, Shape};
use crate::stabilize;

struct App {
    doc: Document,
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    worker: Worker,
    w: usize,
    h: usize,
    shapes: Vec<Shape>,
    /// the shape being drawn: the pointer's samples, `[x, y, time in ms]` in canvas pixels, as they came
    drawing: Option<Vec<[f64; 3]>>,
    col: Col,
    subject: bool,
    outline: bool,
    /// the next shape is a cast shadow or a glowing light
    kind: Kind,
    light: scene::Light,
    seed: u32,
    painted: Option<ImageData>,
    /// the painting is of the shapes as they are now
    fresh: bool,
    show_painted: bool,
    busy: bool,
    /// the address asked for the CPU (`?backend=cpu`)
    force_cpu: bool,
}

type Shared = Rc<RefCell<App>>;

/// A kit colour (channels 0-255) as CSS.
fn hex(c: Col) -> String {
    let b = |v: f64| v.clamp(0.0, 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

fn el(doc: &Document, id: &str) -> Option<HtmlElement> {
    doc.get_element_by_id(id).and_then(|e| e.dyn_into::<HtmlElement>().ok())
}

impl App {
    fn draw_flat(&self) {
        self.ctx.set_fill_style_str("#d8cfc0");
        self.ctx.fill_rect(0.0, 0.0, self.w as f64, self.h as f64);
        // the shape being drawn, calmed as it will be kept (its end stays on the pen)
        let live = self.drawing.as_ref().map(|d| stabilize::path(d, self.unit(), false));
        let mut all: Vec<(&[[f64; 2]], Col, bool, Kind)> = self.shapes.iter().map(|s| (s.pts.as_slice(), s.col, s.outline, s.kind)).collect();
        if let Some(d) = &live {
            all.push((d.as_slice(), self.col, self.outline, self.kind));
        }
        for (pts, col, outline, kind) in all {
            let Some(first) = pts.first() else { continue };
            self.ctx.begin_path();
            self.ctx.move_to(first[0], first[1]);
            for p in &pts[1..] {
                self.ctx.line_to(p[0], p[1]);
            }
            self.ctx.close_path();
            // (a shadow shows where it falls, as a dusky veil; the painter derives its colour)
            self.ctx.set_fill_style_str(&if kind == Kind::Shadow { "rgba(29, 26, 48, 0.45)".to_string() } else { hex(col) });
            self.ctx.fill();
            if kind == Kind::Glow {
                self.ctx.set_stroke_style_str("#fff6d8");
                self.ctx.set_line_width(3.0);
                self.ctx.stroke();
            }
            if outline && kind == Kind::Plain {
                self.ctx.set_stroke_style_str("#2a2420");
                self.ctx.set_line_width(1.5);
                self.ctx.stroke();
            }
        }
    }

    fn show(&self) {
        match (&self.painted, self.show_painted && self.drawing.is_none()) {
            (Some(img), true) => {
                let _ = self.ctx.put_image_data(img, 0.0, 0.0);
            }
            _ => self.draw_flat(),
        }
        let state = if self.busy {
            "painting"
        } else if self.painted.is_some() && self.show_painted && self.fresh {
            "painted"
        } else {
            "flat"
        };
        let _ = self.canvas.set_attribute("data-state", state);
        if let Some(b) = el(&self.doc, "compare") {
            b.set_text_content(Some(if self.show_painted { "Show shapes" } else { "Show painting" }));
            let _ = b.toggle_attribute_with_force("disabled", self.painted.is_none());
        }
        for id in ["paint", "reseed"] {
            if let Some(b) = el(&self.doc, id) {
                let _ = b.toggle_attribute_with_force("disabled", self.busy || self.shapes.is_empty());
            }
        }
    }

    fn status(&self, text: &str, progress: Option<f64>) {
        if let Some(s) = el(&self.doc, "status") {
            s.set_text_content(Some(text));
        }
        if let Some(bar) = el(&self.doc, "bar") {
            let _ = bar.style().set_property("width", &format!("{:.0}%", progress.unwrap_or(0.0) * 100.0));
            let _ = bar.style().set_property("opacity", if progress.is_some() { "1" } else { "0" });
        }
    }

    fn paint(&mut self) {
        if self.busy || self.shapes.is_empty() {
            return;
        }
        self.busy = true;
        let msg = js_sys::Array::new();
        msg.push(&js_sys::Float32Array::from(scene::encode(&self.shapes).as_slice()));
        msg.push(&JsValue::from_f64(self.seed as f64));
        msg.push(&JsValue::from_f64(self.w as f64));
        msg.push(&JsValue::from_f64(self.h as f64));
        msg.push(&JsValue::from_str(if self.force_cpu { "cpu" } else { "auto" }));
        msg.push(&JsValue::from_f64(self.light.preset as f64));
        msg.push(&JsValue::from_f64(self.light.from as f64));
        let _ = self.worker.post_message(&msg);
        self.status("Painting…", Some(0.0));
        self.show();
    }

    /// Say which backend paints (`gpu` or `cpu`) in the small label over the canvas; `detail` goes in its tooltip.
    fn backend_label(&self, kind: &str, detail: &str) {
        let Some(l) = el(&self.doc, "backend") else { return };
        let (text, why) = match kind {
            "gpu" => ("GPU", format!("Painting on the graphics card through WebGPU ({detail}).")),
            _ if self.force_cpu => ("CPU", "Painting on the processor, as the address asks (?backend=cpu).".to_string()),
            _ => ("CPU", format!("Painting on the processor: {detail}.")),
        };
        l.set_text_content(Some(text));
        let _ = l.set_attribute("data-kind", kind);
        let _ = l.set_attribute("title", &why);
        let _ = l.set_attribute("aria-label", &format!("Backend: {text}. {why}"));
        l.set_hidden(false);
    }

    /// canvas pixels under a pointer event, and when it happened (ms)
    fn at(&self, e: &PointerEvent) -> [f64; 3] {
        let r = self.canvas.get_bounding_client_rect();
        [
            (e.client_x() as f64 - r.left()) * self.w as f64 / r.width().max(1.0),
            (e.client_y() as f64 - r.top()) * self.h as f64 / r.height().max(1.0),
            e.time_stamp(),
        ]
    }

    /// canvas pixels per screen (CSS) pixel
    fn unit(&self) -> f64 {
        self.w as f64 / self.canvas.get_bounding_client_rect().width().max(1.0)
    }
}

fn on<E: JsCast + 'static>(target: &web_sys::EventTarget, kind: &str, app: &Shared, f: fn(&mut App, E)) {
    let app = app.clone();
    let cb = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
        if let Ok(e) = e.dyn_into::<E>() {
            f(&mut app.borrow_mut(), e);
        }
    });
    let _ = target.add_event_listener_with_callback(kind, cb.as_ref().unchecked_ref());
    cb.forget();
}

fn click(doc: &Document, id: &str, app: &Shared, f: fn(&mut App)) {
    let Some(b) = el(doc, id) else { return };
    let app = app.clone();
    let cb = Closure::<dyn FnMut()>::new(move || f(&mut app.borrow_mut()));
    b.set_onclick(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
}

fn set_pressed(doc: &Document, id: &str, on: bool) {
    if let Some(b) = el(doc, id) {
        let _ = b.set_attribute("aria-pressed", if on { "true" } else { "false" });
    }
}

/// Start the page (called by index.html once the module is loaded).
#[wasm_bindgen]
pub fn app_start() -> Result<(), JsValue> {
    let win = web_sys::window().ok_or("no window")?;
    let doc = win.document().ok_or("no document")?;
    let canvas: HtmlCanvasElement = doc.get_element_by_id("canvas").ok_or("no canvas")?.dyn_into()?;
    // the painted region: 1024 x 576 on a desktop, smaller on a phone (paint time grows with the area)
    let narrow = win.inner_width()?.as_f64().unwrap_or(1024.0) < 700.0;
    let (w, h) = if narrow { (768, 432) } else { (1024, 576) };
    canvas.set_width(w as u32);
    canvas.set_height(h as u32);
    let ctx: CanvasRenderingContext2d = canvas.get_context("2d")?.ok_or("no 2d context")?.dyn_into()?;
    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let worker = Worker::new_with_options("./worker.js", &opts)?;
    let force_cpu = win.location().search().unwrap_or_default().contains("backend=cpu");
    let app: Shared = Rc::new(RefCell::new(App {
        doc: doc.clone(),
        canvas: canvas.clone(),
        ctx,
        worker: worker.clone(),
        w,
        h,
        shapes: scene::sample(w as f64, h as f64),
        drawing: None,
        col: rgb(PALETTE[3]),
        subject: true,
        outline: true,
        kind: Kind::Plain,
        light: scene::Light::default(),
        seed: 1,
        painted: None,
        fresh: false,
        show_painted: false,
        busy: false,
        force_cpu,
    }));

    // the worker: progress, then the painting
    {
        let app1 = app.clone();
        let cb = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            let mut a = app1.borrow_mut();
            let data = e.data();
            if let Some(f) = data.as_f64() {
                a.status(&format!("Painting… {:.0}%", f * 100.0), Some(f));
                return;
            }
            let msg = js_sys::Array::from(&data);
            // the worker found (or lost) the GPU: ["backend", kind, detail]
            if msg.get(0).as_string().as_deref() == Some("backend") {
                let kind = msg.get(1).as_string().unwrap_or_default();
                let shown = if a.force_cpu { "cpu" } else { kind.as_str() };
                a.backend_label(shown, &msg.get(2).as_string().unwrap_or_default());
                return;
            }
            let px = js_sys::Uint8ClampedArray::new(&msg.get(0)).to_vec();
            let ms = msg.get(1).as_f64().unwrap_or(0.0);
            let kind = msg.get(2).as_string().unwrap_or_else(|| "cpu".into());
            a.busy = false;
            match ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&px), a.w as u32, a.h as u32) {
                Ok(img) => {
                    a.painted = Some(img);
                    a.fresh = true;
                    a.show_painted = true;
                    let _ = a.canvas.set_attribute("data-paint-ms", &format!("{ms:.0}"));
                    let _ = a.canvas.set_attribute("data-paint-backend", &kind);
                    let on = if kind == "gpu" { "on the GPU, through WebGPU" } else { "on the CPU" };
                    a.status(&format!("Painted in {:.1} s ({on}, in a Web Worker). Seed {}.", ms / 1000.0, a.seed), None);
                }
                Err(_) => a.status("The painting came back the wrong size.", None),
            }
            a.show();
        });
        worker.set_onmessage(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
        let app2 = app.clone();
        let err = Closure::<dyn FnMut()>::new(move || {
            let mut a = app2.borrow_mut();
            a.busy = false;
            a.status("Painting failed in the worker.", None);
            a.show();
        });
        worker.set_onerror(Some(err.as_ref().unchecked_ref()));
        err.forget();
    }

    // palette
    if let Some(pal) = el(&doc, "palette") {
        for (i, c) in PALETTE.iter().enumerate() {
            let b: HtmlElement = doc.create_element("button")?.dyn_into()?;
            b.set_class_name("swatch");
            let _ = b.set_attribute("aria-label", &format!("colour {}", i + 1));
            let _ = b.style().set_property("background", &hex(rgb(*c)));
            if i == 3 {
                let _ = b.set_attribute("aria-pressed", "true");
            }
            let app = app.clone();
            let col = rgb(*c);
            let cb = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
                let a = &mut app.borrow_mut();
                a.col = col;
                if let Ok(list) = a.doc.query_selector_all(".swatch") {
                    for k in 0..list.length() {
                        if let Some(n) = list.item(k).and_then(|n| n.dyn_into::<HtmlElement>().ok()) {
                            let _ = n.set_attribute("aria-pressed", "false");
                        }
                    }
                }
                if let Some(t) = e.current_target().and_then(|t| t.dyn_into::<HtmlElement>().ok()) {
                    let _ = t.set_attribute("aria-pressed", "true");
                }
            });
            b.set_onclick(Some(cb.as_ref().unchecked_ref()));
            cb.forget();
            pal.append_child(&b)?;
        }
    }
    set_pressed(&doc, "subject", true);
    set_pressed(&doc, "outline", true);
    click(&doc, "subject", &app, |a| {
        a.subject = !a.subject;
        set_pressed(&a.doc, "subject", a.subject);
    });
    click(&doc, "outline", &app, |a| {
        a.outline = !a.outline;
        set_pressed(&a.doc, "outline", a.outline);
    });
    click(&doc, "shadow", &app, |a| {
        a.kind = if a.kind == Kind::Shadow { Kind::Plain } else { Kind::Shadow };
        set_pressed(&a.doc, "shadow", a.kind == Kind::Shadow);
        set_pressed(&a.doc, "glow", a.kind == Kind::Glow);
        a.show();
    });
    click(&doc, "glow", &app, |a| {
        a.kind = if a.kind == Kind::Glow { Kind::Plain } else { Kind::Glow };
        set_pressed(&a.doc, "shadow", a.kind == Kind::Shadow);
        set_pressed(&a.doc, "glow", a.kind == Kind::Glow);
        a.show();
    });
    let light_text = |a: &App| {
        if let Some(b) = el(&a.doc, "light") {
            b.set_text_content(Some(&format!("Light: {}", LIGHTS[a.light.preset].0)));
        }
        if let Some(b) = el(&a.doc, "from") {
            b.set_text_content(Some(&format!("From the {}", FROM[a.light.from].0)));
        }
    };
    light_text(&app.borrow());
    click(&doc, "light", &app, |a| {
        a.light.preset = (a.light.preset + 1) % LIGHTS.len();
        a.fresh = false;
        if let Some(b) = el(&a.doc, "light") {
            b.set_text_content(Some(&format!("Light: {}", LIGHTS[a.light.preset].0)));
        }
    });
    click(&doc, "from", &app, |a| {
        a.light.from = (a.light.from + 1) % FROM.len();
        a.fresh = false;
        if let Some(b) = el(&a.doc, "from") {
            b.set_text_content(Some(&format!("From the {}", FROM[a.light.from].0)));
        }
    });
    click(&doc, "undo", &app, |a| {
        a.shapes.pop();
        a.fresh = false;
        a.show_painted = false;
        a.show();
    });
    click(&doc, "clear", &app, |a| {
        a.shapes.clear();
        a.painted = None;
        a.fresh = false;
        a.show_painted = false;
        a.status("Draw shapes, then Paint.", None);
        a.show();
    });
    click(&doc, "sample", &app, |a| {
        a.shapes = scene::sample(a.w as f64, a.h as f64);
        a.fresh = false;
        a.show_painted = false;
        a.show();
        a.paint();
    });
    click(&doc, "paint", &app, App::paint);
    click(&doc, "reseed", &app, |a| {
        a.seed = a.seed.wrapping_add(1);
        a.paint();
    });
    click(&doc, "compare", &app, |a| {
        a.show_painted = !a.show_painted;
        a.show();
    });

    // drawing: press, drag round a shape, let go
    let target: &web_sys::EventTarget = canvas.as_ref();
    on::<PointerEvent>(target, "pointerdown", &app, |a, e| {
        e.prevent_default();
        let _ = a.canvas.set_pointer_capture(e.pointer_id());
        let p = a.at(&e);
        a.drawing = Some(vec![p]);
        a.show();
    });
    on::<PointerEvent>(target, "pointermove", &app, |a, e| {
        if a.drawing.is_none() {
            return;
        }
        // every sample the device took since the last event (a touch screen often takes two per frame), where the
        // browser offers them: the stabilizer reads the pen's speed from their times
        let mut samples = vec![];
        if js_sys::Reflect::has(&e, &JsValue::from_str("getCoalescedEvents")).unwrap_or(false) {
            for c in e.get_coalesced_events().iter() {
                if let Ok(c) = c.dyn_into::<PointerEvent>() {
                    samples.push(a.at(&c));
                }
            }
        }
        if samples.is_empty() {
            samples.push(a.at(&e));
        }
        let Some(d) = a.drawing.as_mut() else { return };
        let mut moved = false;
        for p in samples {
            if d.last().is_some_and(|q| (q[0] - p[0]).hypot(q[1] - p[1]) >= 0.5) {
                d.push(p);
                moved = true;
            }
        }
        if moved {
            a.show();
        }
    });
    for kind in ["pointerup", "pointercancel"] {
        on::<PointerEvent>(target, kind, &app, |a, _e| {
            let Some(raw) = a.drawing.take() else { return };
            let d = stabilize::outline(&raw, a.unit());
            let n = d.len();
            let area = (0..n).map(|i| d[i][0] * d[(i + 1) % n][1] - d[(i + 1) % n][0] * d[i][1]).sum::<f64>().abs() / 2.0;
            if n >= 3 && area > 30.0 {
                a.shapes.push(Shape { col: a.col, subject: a.subject && a.kind == Kind::Plain, outline: a.outline, kind: a.kind, pts: d });
                a.fresh = false;
                a.show_painted = false;
                a.status("Draw more shapes, or Paint.", None);
            }
            a.show();
        });
    }

    let mut a = app.borrow_mut();
    a.show();
    a.paint();
    Ok(())
}
