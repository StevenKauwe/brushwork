//! Opt-in fast photo renderer. The approximations are isolated from quality/semantic painting.
#![forbid(unsafe_code)]
use super::{BODY, Dynamics, Painting, PhotoOptions, SMALLEST, Tool, gpu, h32, hf, noise, shifted, sweep};
#[cfg(any(test, not(target_arch = "wasm32")))]
const DEFAULT_CANVAS: [usize; 2] = [960, 540];
#[cfg(target_arch = "wasm32")]
trait WorkerChunks<T> {
    fn par_chunks_mut(&mut self, n: usize) -> std::slice::ChunksMut<'_, T>;
}
#[cfg(target_arch = "wasm32")]
impl<T> WorkerChunks<T> for Vec<T> {
    fn par_chunks_mut(&mut self, n: usize) -> std::slice::ChunksMut<'_, T> {
        self.chunks_mut(n)
    }
}
const CANVASES: [[usize; 2]; 4] = [[480, 270], [960, 540], [1280, 720], [1920, 1080]];
fn canvas_slot(canvas: [usize; 2]) -> Option<usize> {
    CANVASES.iter().position(|s| *s == canvas)
}
// Serialize native submissions because wgpu device error scopes are shared.
#[cfg(not(target_arch = "wasm32"))]
static JOB: std::sync::Mutex<()> = std::sync::Mutex::new(());
// Nonblocking across await: overlapping async callers receive a clear busy error.
static BUSY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
struct Job;
impl Job {
    fn acquire() -> Result<Self, String> {
        BUSY.compare_exchange(false, true, std::sync::atomic::Ordering::Acquire, std::sync::atomic::Ordering::Relaxed)
            .map(|_| Self)
            .map_err(|_| "A fast photo job is already running; queue requests in the worker".into())
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        BUSY.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// Bounded GPU working canvases. Choose a smaller canvas for quicker previews.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FastPhotoResolution {
    P270,
    #[default]
    P540,
    P720,
    P1080,
}
impl FastPhotoResolution {
    pub fn canvas(self) -> [usize; 2] {
        match self {
            Self::P270 => [480, 270],
            Self::P540 => [960, 540],
            Self::P720 => [1280, 720],
            Self::P1080 => [1920, 1080],
        }
    }
}
/// Fast painting is approximate and opt-in; it never silently replaces the detailed painter.
#[derive(Clone, Copy, Debug)]
pub struct FastPhotoOptions {
    pub resolution: FastPhotoResolution,
    /// Fixed, seeded stroke layouts instead of image-adaptive paths.
    pub steady: bool,
    /// Enlarge a small input to fit the canvas, preserving aspect ratio.
    pub upscale: bool,
}
impl Default for FastPhotoOptions {
    fn default() -> Self {
        Self { resolution: FastPhotoResolution::default(), steady: false, upscale: true }
    }
}
async fn finish_batch(p: &mut Painting) -> Result<(), String> {
    if let Some(recorder) = p.rec.take() {
        let (col, thick) = gpu::render_async(&recorder).await?;
        p.col = col;
        p.thick = thick;
    }
    Ok(())
}
fn new_canvas(canvas: [usize; 2]) -> Painting {
    let [wi, hi] = canvas;
    static GROUNDS: [std::sync::OnceLock<Vec<[f32; 3]>>; 4] = [const { std::sync::OnceLock::new() }; 4];
    let paper = false;
    let col = GROUNDS[canvas_slot(canvas).expect("validated canvas")]
        .get_or_init(|| {
            // Cached immutable procedural ground; not cached input or output frames.
            let mut col = vec![[0.0; 3]; wi * hi];
            for y in 0..hi {
                for x in 0..wi {
                    let n = noise(x as f32 / 140.0, y as f32 / 140.0, 7) * 0.06 - 0.03;
                    col[y * wi + x] = if paper { [0.955 + n * 0.3, 0.93 + n * 0.3, 0.875 + n * 0.3] } else { [0.60 + n, 0.50 + n, 0.37 + n] };
                }
            }
            col
        })
        .clone();
    Painting {
        canvas: [wi, hi],
        fast_photo: true,
        mix_cache: None,
        dynamics: Dynamics::default(),
        flat: col.clone(),
        behind: col.clone(),
        col,
        thick: vec![0.0; wi * hi],
        wet: vec![0.0; wi * hi],
        strokes: 0,
        rec: None,
        planning_clip: None,
        view: 1.0,
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn paint_photo_image_fast(
    img: image::RgbImage,
    opts: &PhotoOptions,
    focus_input: [f32; 3],
    dynamics: Dynamics,
    seed: u32,
    progress: &mut dyn FnMut(&str, usize, usize) -> Result<(), String>,
) -> Result<image::RgbImage, String> {
    paint_photo_image_fast_at_size(img, opts, focus_input, dynamics, seed, DEFAULT_CANVAS, false, true, progress)
}

/// Same full-photo wet painter with a seeded, image-independent stroke layout.
/// Repainting a different frame changes pigment, not stroke selection or paths.
#[cfg(not(target_arch = "wasm32"))]
pub fn paint_photo_image_steady(
    img: image::RgbImage,
    opts: &PhotoOptions,
    focus_input: [f32; 3],
    dynamics: Dynamics,
    seed: u32,
    progress: &mut dyn FnMut(&str, usize, usize) -> Result<(), String>,
) -> Result<image::RgbImage, String> {
    paint_photo_image_fast_at_size(img, opts, focus_input, dynamics, seed, DEFAULT_CANVAS, true, true, progress)
}

/// Paint at a real GPU canvas size; legacy entry points retain their 540p output.
/// Explicit resolution callers can disable input upscaling.
#[cfg(not(target_arch = "wasm32"))]
pub fn paint_photo_image_fast_at_size(
    img: image::RgbImage,
    opts: &PhotoOptions,
    focus_input: [f32; 3],
    dynamics: Dynamics,
    seed: u32,
    canvas: [usize; 2],
    steady: bool,
    upscale: bool,
    progress: &mut dyn FnMut(&str, usize, usize) -> Result<(), String>,
) -> Result<image::RgbImage, String> {
    let resolution = match canvas {
        [480, 270] => FastPhotoResolution::P270,
        [960, 540] => FastPhotoResolution::P540,
        [1280, 720] => FastPhotoResolution::P720,
        [1920, 1080] => FastPhotoResolution::P1080,
        _ => return Err("Unsupported fast photo canvas".into()),
    };
    let _job = JOB.lock().map_err(|_| "Fast painter unavailable")?;
    pollster::block_on(paint_photo_image_fast_async(img, opts, focus_input, dynamics, seed, FastPhotoOptions { resolution, steady, upscale }, progress))
}

/// The same fast painter on native GPU and browser WebGPU. In a browser call
/// [`crate::open_gpu`] first and await this function from a Web Worker. GPU failure
/// is returned to the caller; it can explicitly fall back to the detailed CPU painter.
/// Only one job per browser worker may run at a time; dropping the future cancels it.
pub async fn paint_photo_image_fast_async(
    img: image::RgbImage,
    opts: &PhotoOptions,
    focus_input: [f32; 3],
    dynamics: Dynamics,
    seed: u32,
    fast: FastPhotoOptions,
    progress: &mut dyn FnMut(&str, usize, usize) -> Result<(), String>,
) -> Result<image::RgbImage, String> {
    let canvas = fast.resolution.canvas();
    let (steady, upscale) = (fast.steady, fast.upscale);
    if canvas_slot(canvas).is_none() {
        return Err("Unsupported fast photo canvas".into());
    }
    let [wi, hi] = canvas;
    if img.width() == 0 || img.height() == 0 {
        return Err("Empty photo".into());
    }
    if opts.attention {
        return Err("Fast photo requires geometry-preserving attention off".into());
    }
    // A renderer can be used after a semantic frame without inheriting its state.
    let _job = Job::acquire()?;
    progress("Preparing photo", 0, 0)?;
    // the canvas is landscape: a portrait picture is painted on its side and turned back afterwards
    let turned = img.height() > img.width();
    let (iw, ih) = if turned { (img.height(), img.width()) } else { img.dimensions() };
    let scale = (hi as f32 / ih as f32).min(wi as f32 / iw as f32);
    let scale = if upscale { scale } else { scale.min(1.) };
    let (rw, rh) = (((iw as f32 * scale) as usize).max(1), ((ih as f32 * scale) as usize).max(1));
    let small = if turned { image::imageops::rotate270(&photo_resize(&img, rh as u32, rw as u32)) } else { photo_resize(&img, rw as u32, rh as u32) };
    let (ox, oy) = ((wi - rw) / 2, (hi - rh) / 2);
    let mut p = new_canvas(canvas);
    p.dynamics = dynamics;
    let mut subject = p.col.clone();
    for y in 0..rh {
        for x in 0..rw {
            let c = small.get_pixel(x as u32, y as u32).0;
            subject[(y + oy) * wi + x + ox] = [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0];
        }
    }
    // DEEP-SEA TEST (crabscape): `OIL_NOCTURNE=g` paints a night picture as a painter does. Their eye adapts to the
    // dark and sees its structure (so here the subject is seen lifted by 1/g, and every judgement of shape, edge and
    // colour is made on that), then they transpose the whole value scale down when mixing (the canvas is mapped
    // back by g just before it is lit), so streaks and warm shifts stay in proportion to the darks they sit in.
    let nocturne = opts.nocturne;
    if let Some(g) = nocturne {
        for c in subject.iter_mut() {
            for v in c.iter_mut() {
                *v = v.max(0.0).powf(1.0 / g);
            }
        }
    }
    // ---- see it as a person does, not as a lens does. A camera gives every part of the scene the same attention
    // through a wide lens. A person looks AT something: that thing is large and clear, and the rest of the room is
    // a few quiet planes. `focus` is where the painter looks (x, y and reach, as fractions of the upright picture).
    let [fu, fv, fr] = focus_input;
    let focus =
        if turned { [ox as f32 + fv * rw as f32, oy as f32 + (1.0 - fu) * rh as f32] } else { [ox as f32 + fu * rw as f32, oy as f32 + fv * rh as f32] };
    let reach_px = fr * rw.max(rh) as f32;
    // DEEP-SEA TEST (crabscape): `OIL_ATTENTION=off` paints every part of the picture as if it were looked at: no
    // magnification around the focus, no quieting away from it, small brushes allowed everywhere. For a subject
    // that must keep its exact drawn shape (a game sprite painted on its own).
    let attend = opts.attention;
    let attention = |q: [f32; 2]| {
        if attend { (-(((q[0] - focus[0]).hypot(q[1] - focus[1])) / (1.2 * reach_px)).powi(2)).exp() } else { 1.0 }
    };
    // DEEP-SEA TEST (crabscape): `OIL_KEY` is the value the unattended room relaxes towards (default 0.56, a daylight
    // middle grey). A nocturne's middle value is near black: there the quiet parts must sink, not lift to grey.
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let inside = |q: [f32; 2]| q[0] >= ox as f32 && q[1] >= oy as f32 && q[0] < (ox + rw) as f32 && q[1] < (oy + rh) as f32;
    let px = |q: [f32; 2]| (q[1].clamp(0.0, hi as f32 - 1.0) as usize) * wi + q[0].clamp(0.0, wi as f32 - 1.0) as usize;
    // DEEP-SEA TEST (crabscape): `OIL_GROUND=subject` tones the ground with a broad wash of the subject's own colours
    // (an imprimatura laid in to suit the picture) instead of the default warm mid ochre, so a dark picture's
    // skipped bristles show dark ground, not bright brown.
    progress("Ground wash", 0, 0)?;
    if opts.ground_subject {
        let wash = box_blur(&subject, 24, canvas);
        p.col.copy_from_slice(&wash);
        p.flat.copy_from_slice(&wash);
        p.wet.fill(1.0);
    }
    let start_batch = |p: &mut Painting| {
        let mut recorder = gpu::Recorder::new(std::mem::take(&mut p.col));
        recorder.canvas = [wi, hi];
        recorder.initial_thick = Some(std::mem::take(&mut p.thick));
        p.rec = Some(Box::new(recorder));
    };
    let mut proxy = PhotoProxy::new(&p.col, canvas);
    start_batch(&mut p);
    // ---- block the whole canvas in with the big flat, squinting
    {
        let seen = box_blur(&subject, 13, canvas);
        let w = 35.0f32;
        let mut spots: Vec<([f32; 2], u32)> = vec![];
        for gy in 0..(rh as f32 / (w * 0.5)) as u32 + 1 {
            for gx in 0..(rw as f32 / (w * 0.5)) as u32 + 1 {
                let id = (0x7700_0000 | gy << 12 | gx) ^ seed;
                let q = [ox as f32 + (gx as f32 + hf(id, 1)) * w * 0.5, oy as f32 + (gy as f32 + hf(id, 2)) * w * 0.5];
                if inside(q) {
                    spots.push((q, id));
                }
            }
        }
        spots.sort_by_key(|(_, id)| h32(*id));
        let total = spots.len();
        for (index, (q, id)) in spots.into_iter().enumerate() {
            progress("Photo block-in", index, total)?;
            let a = -0.6 + (noise(q[0] / 210.0, q[1] / 210.0, 5) - 0.5) * 0.7;
            let d = [a.cos(), a.sin()];
            let path: Vec<[f32; 2]> =
                [0, 6].into_iter().map(|i| [q[0] + d[0] * w * 0.45 * (i as f32 - 3.0), q[1] + d[1] * w * 0.45 * (i as f32 - 3.0)]).collect();
            let c = seen[px(q)];
            proxy.sweep(&path, w, c);
            sweep(&mut p, Tool::Flat, &path, w * (0.85 + 0.3 * hf(id, 4)), c, shifted(c, 1.06, 0.4), 0.97, 0.2, 0.5, id, &|_, _, _| 1.0);
        }
    }

    // ---- then look again, and touch in with the small brushes only where it is still wrong
    for (layer, (w, tool, wrong, load)) in
        [(26.0f32, Tool::Filbert, 0.08f32, 0.9f32), (13.0f32, Tool::Filbert, 0.11f32, 0.9f32), (SMALLEST, Tool::Round, 0.10, 1.0)].into_iter().enumerate()
    {
        // Photo's finishing passes must obey the same stroke-size control as
        // its shape pass. Previously these stayed coarse at every setting.
        let w = w * 0.5 * dynamics.stroke_scale.clamp(0.25, 2.0);
        let layer = layer + 2;
        if layer == 4 {
            progress("GPU coarse painting", 0, 0)?;
            finish_batch(&mut p).await?;
            proxy = PhotoProxy::new(&p.col, canvas);
            start_batch(&mut p);
            progress("Detail planning", 0, 0)?;
        }
        // squint: see the subject no more finely than this brush can paint it
        let seen = box_blur(&subject, (w * 0.38) as usize, canvas);
        let grad = |q: [f32; 2]| -> [f32; 2] {
            let d = (w * 0.3).max(1.5);
            [lum(seen[px([q[0] + d, q[1]])]) - lum(seen[px([q[0] - d, q[1]])]), lum(seen[px([q[0], q[1] + d])]) - lum(seen[px([q[0], q[1] - d])])]
        };
        let step = w * if steady { 1.15 } else { 0.55 };
        let mut spots: Vec<([f32; 2], u32)> = vec![];
        for gy in 0..(rh as f32 / step) as u32 + 1 {
            for gx in 0..(rw as f32 / step) as u32 + 1 {
                let id = ((layer as u32) << 24 | gy << 12 | gx) ^ seed;
                let q = [ox as f32 + (gx as f32 + hf(id, 1)) * step, oy as f32 + (gy as f32 + hf(id, 2)) * step];
                if !inside(q) {
                    continue;
                }
                if steady {
                    spots.push((q, id));
                    continue;
                }
                // is the canvas still wrong here, as far as this brush can tell?
                let mut err = 0.0;
                for (dx, dy) in [(0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)] {
                    let s = [q[0] + dx * step, q[1] + dy * step];
                    err += dist(proxy.at(s), seen[px(s)]) / 5.0;
                }
                if attention(q) > [0.0, 0.0, 0.42, 0.6][layer.min(3)] && err > wrong {
                    spots.push((q, id));
                }
            }
        }
        spots.sort_by_key(|(_, id)| h32(*id));
        let total = spots.len();
        for (index, (q, id)) in spots.into_iter().enumerate() {
            progress("Photo finishing strokes", index, total)?;
            if !steady && dist(proxy.at(q), seen[px(q)]) <= wrong {
                continue;
            }
            let c = seen[px(q)];
            // along the edge, not across it; where there is no edge, the painter's own diagonal
            let mut dir = {
                let g = grad(q);
                if !steady && g[0].hypot(g[1]) > 0.006 {
                    let l = g[0].hypot(g[1]);
                    [-g[1] / l, g[0] / l]
                } else {
                    let a = -0.6 + (noise(q[0] / 190.0, q[1] / 190.0, 5) - 0.5) * 0.9;
                    [a.cos(), a.sin()]
                }
            };
            // pull, don't push
            if dir[0] * 0.75 + dir[1] < 0.0 {
                dir = [-dir[0], -dir[1]];
            }
            let mut path = vec![[q[0] - dir[0] * w * 0.6, q[1] - dir[1] * w * 0.6], q];
            let (seg, most) = (w * 0.45, (if steady { 4.0 + 3.0 * hf(id, 3) } else { 7.0 + 9.0 * hf(id, 3) }) as usize);
            let mut at = q;
            for i in 0..most {
                let next = [at[0] + dir[0] * seg, at[1] + dir[1] * seg];
                // the stroke ends where its colour stops being right for what is under it
                if !inside(next) || (!steady && i >= 2 && dist(seen[px(next)], c) > dist(seen[px(next)], proxy.at(next)).max(0.06)) {
                    break;
                }
                path.push(next);
                at = next;
                let g = grad(at);
                let l = g[0].hypot(g[1]);
                if !steady && l > 0.006 {
                    let mut d = [-g[1] / l, g[0] / l];
                    if d[0] * dir[0] + d[1] * dir[1] < 0.0 {
                        d = [-d[0], -d[1]];
                    }
                    // a wrist turns smoothly
                    let nd = [dir[0] * 0.55 + d[0] * 0.45, dir[1] * 0.55 + d[1] * 0.45];
                    let nl = nd[0].hypot(nd[1]).max(1e-3);
                    dir = [nd[0] / nl, nd[1] / nl];
                }
            }
            let path = simplify_photo_path(&path, (w * 0.08).max(0.25));
            let width = w * (0.8 + 0.35 * hf(id, 4));
            proxy.sweep(&path, width, c);
            sweep(&mut p, tool, &path, width, c, shifted(c, 1.1, 0.5), 0.96, 0.12 + 0.14 * hf(id, 5), load, id, &|_, _, _| 1.0);
        }
    }
    progress("GPU painting", 0, 0)?;
    finish_batch(&mut p).await?;
    // DEEP-SEA TEST (crabscape): the nocturne's values transposed down (see `OIL_NOCTURNE` above)
    if let Some(g) = nocturne {
        for c in p.col.iter_mut() {
            for v in c.iter_mut() {
                *v = v.max(0.0).powf(g);
            }
        }
    }
    progress("Lighting and output", 0, 0)?;
    let lit = light(&p);
    let mut crop = vec![0u8; rw * rh * 3];
    for y in 0..rh {
        crop[y * rw * 3..(y + 1) * rw * 3].copy_from_slice(&lit[((y + oy) * wi + ox) * 3..((y + oy) * wi + ox + rw) * 3]);
    }
    let done = image::RgbImage::from_raw(rw as u32, rh as u32, crop).ok_or("image")?;
    let done = if turned { image::imageops::rotate90(&done) } else { done };
    progress("Photo complete", 1, 1)?;
    Ok(done)
}

fn light(p: &Painting) -> Vec<u8> {
    let [wi, hi] = p.canvas;
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;
    let mut out = vec![0u8; wi * hi * 3];
    let l = {
        let v = [-0.52f32, -0.62, 0.59];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let hv = {
        let v = [l[0], l[1], l[2] + 1.0];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    };
    // wet paint settles: soften the relief a little before lighting it
    let mut soft = vec![0.0f32; wi * hi];
    soft.par_chunks_mut(wi).enumerate().for_each(|(y, row)| {
        for x in 0..wi {
            let mut acc = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    acc += p.thick[(y as i32 + dy).clamp(0, hi as i32 - 1) as usize * wi + (x as i32 + dx).clamp(0, wi as i32 - 1) as usize];
                }
            }
            row[x] = acc / 9.0;
        }
    });
    static WEAVE: [std::sync::OnceLock<Vec<f32>>; 4] = [const { std::sync::OnceLock::new() }; 4];
    let weave = WEAVE[canvas_slot(p.canvas).expect("validated canvas")].get_or_init(|| {
        (0..(wi + 1) * (hi + 1))
            .map(|i| {
                let (x, y) = (i % (wi + 1), i / (wi + 1));
                0.02 * ((x as f32 * 1.55).sin() * (y as f32 * 1.55).sin())
            })
            .collect()
    });
    let hgt = |x: usize, y: usize| -> f32 {
        let weave = weave[y * (wi + 1) + x];
        let t = soft[y.min(hi - 1) * wi + x.min(wi - 1)];
        t + weave * (1.0 - t).clamp(0.15, 1.0)
    };
    out.par_chunks_mut(wi * 3).enumerate().for_each(|(y, row)| {
        for x in 0..wi {
            let gx = (hgt(x + 1, y) - hgt(x.saturating_sub(1), y)) * 1.9 * BODY * p.view;
            let gy = (hgt(x, y + 1) - hgt(x, y.saturating_sub(1))) * 1.9 * BODY * p.view;
            let inv = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
            let n = [-gx * inv, -gy * inv, inv];
            let diff = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]) / l[2];
            let spec = (n[0] * hv[0] + n[1] * hv[1] + n[2] * hv[2]).max(0.0).powi(26) - hv[2].powi(26);
            let shade = 0.74 + 0.26 * diff;
            let c = p.col[y * wi + x];
            // a little richer, as varnished paint is
            let grey = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
            for k in 0..3 {
                let v = (grey + (c[k] - grey) * 1.2) * shade + spec.max(0.0) * 0.30 * BODY;
                row[x * 3 + k] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    out
}
fn box_blur(src: &[[f32; 3]], r: usize, canvas: [usize; 2]) -> Vec<[f32; 3]> {
    let [wi, hi] = canvas;
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;
    let r = r.max(1);
    let mut a = src.to_vec();
    let mut b = vec![[0.0f32; 3]; src.len()];
    for _ in 0..2 {
        b.par_chunks_mut(wi).enumerate().for_each(|(y, row)| {
            let input = &a[y * wi..(y + 1) * wi];
            let mut sum = [0.0f32; 3];
            for v in &input[..(r + 1).min(wi)] {
                for k in 0..3 {
                    sum[k] += v[k];
                }
            }
            for (x, out) in row.iter_mut().enumerate() {
                if x > 0 {
                    if x + r < wi {
                        for k in 0..3 {
                            sum[k] += input[x + r][k];
                        }
                    }
                    if x > r {
                        for k in 0..3 {
                            sum[k] -= input[x - r - 1][k];
                        }
                    }
                }
                let n = ((x + r + 1).min(wi) - x.saturating_sub(r)) as f32;
                *out = [sum[0] / n, sum[1] / n, sum[2] / n];
            }
        });
        // Independent row stripes permit safe parallel writes. Each stripe owns
        // bounded accumulators; its halo comes from the immutable previous pass.
        const ROWS: usize = 64;
        a.par_chunks_mut(wi * ROWS).enumerate().for_each(|(stripe, rows)| {
            let first = stripe * ROWS;
            let mut sums = vec![[0.0f32; 3]; wi];
            for y in first.saturating_sub(r)..(first + r + 1).min(hi) {
                for (sum, v) in sums.iter_mut().zip(&b[y * wi..(y + 1) * wi]) {
                    for k in 0..3 {
                        sum[k] += v[k];
                    }
                }
            }
            for (dy, row) in rows.chunks_mut(wi).enumerate() {
                let y = first + dy;
                if dy > 0 {
                    if y + r < hi {
                        for (sum, v) in sums.iter_mut().zip(&b[(y + r) * wi..(y + r + 1) * wi]) {
                            for k in 0..3 {
                                sum[k] += v[k];
                            }
                        }
                    }
                    if y > r {
                        for (sum, v) in sums.iter_mut().zip(&b[(y - r - 1) * wi..(y - r) * wi]) {
                            for k in 0..3 {
                                sum[k] -= v[k];
                            }
                        }
                    }
                }
                let n = ((y + r + 1).min(hi) - y.saturating_sub(r)) as f32;
                for (out, sum) in row.iter_mut().zip(&sums) {
                    *out = [sum[0] / n, sum[1] / n, sum[2] / n];
                }
            }
        });
    }
    a
}
fn simplify_photo_path(path: &[[f32; 2]], tolerance: f32) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(path.len());
    if path.len() < 3 {
        return path.to_vec();
    }
    out.push(path[0]);
    let mut start = 0;
    for end in 2..path.len() {
        let a = path[start];
        let b = path[end];
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = d[0].hypot(d[1]).max(1e-6);
        let error = path[start + 1..end].iter().map(|p| ((p[0] - a[0]) * d[1] - (p[1] - a[1]) * d[0]).abs() / len).fold(0.0f32, f32::max);
        if error > tolerance {
            out.push(path[end - 1]);
            start = end - 1;
        }
    }
    out.push(*path.last().unwrap());
    out
}
struct PhotoProxy {
    col: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}
impl PhotoProxy {
    const STEP: usize = 4;
    fn new(col: &[[f32; 3]], canvas: [usize; 2]) -> Self {
        let [wi, hi] = canvas;
        let width = wi.div_ceil(Self::STEP);
        let height = hi.div_ceil(Self::STEP);
        Self {
            width,
            height,
            col: (0..width * height).map(|i| col[((i / width) * Self::STEP).min(hi - 1) * wi + ((i % width) * Self::STEP).min(wi - 1)]).collect(),
        }
    }
    fn at(&self, p: [f32; 2]) -> [f32; 3] {
        let x = (p[0] / Self::STEP as f32).round().clamp(0., (self.width - 1) as f32) as usize;
        let y = (p[1] / Self::STEP as f32).round().clamp(0., (self.height - 1) as f32) as usize;
        self.col[y * self.width + x]
    }
    fn sweep(&mut self, path: &[[f32; 2]], width: f32, col: [f32; 3]) {
        let r = (width * 0.38).max(1.);
        for pair in path.windows(2) {
            let [a, b] = [pair[0], pair[1]];
            let d = [b[0] - a[0], b[1] - a[1]];
            let l2 = (d[0] * d[0] + d[1] * d[1]).max(1e-6);
            let x0 = ((a[0].min(b[0]) - r) / Self::STEP as f32).floor().max(0.) as usize;
            let x1 = ((a[0].max(b[0]) + r) / Self::STEP as f32).ceil().clamp(0., (self.width - 1) as f32) as usize;
            let y0 = ((a[1].min(b[1]) - r) / Self::STEP as f32).floor().max(0.) as usize;
            let y1 = ((a[1].max(b[1]) + r) / Self::STEP as f32).ceil().clamp(0., (self.height - 1) as f32) as usize;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let q = [(x * Self::STEP) as f32 - a[0], (y * Self::STEP) as f32 - a[1]];
                    let t = ((q[0] * d[0] + q[1] * d[1]) / l2).clamp(0., 1.);
                    let v = [q[0] - d[0] * t, q[1] - d[1] * t];
                    if v[0] * v[0] + v[1] * v[1] <= r * r {
                        let old = &mut self.col[y * self.width + x];
                        for k in 0..3 {
                            old[k] += (col[k] - old[k]) * 0.75;
                        }
                    }
                }
            }
        }
    }
}
fn photo_resize(src: &image::RgbImage, w: u32, h: u32) -> image::RgbImage {
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;
    if src.width() > w * 2 || src.height() > h * 2 {
        return image::imageops::resize(src, w, h, image::imageops::FilterType::Triangle);
    }
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let raw = src.as_raw();
    let mut dst = vec![0u8; w as usize * h as usize * 3];
    let xs: Vec<_> = (0..w)
        .map(|x| {
            let x = ((x as f32 + 0.5) * sw as f32 / w as f32 - 0.5).max(0.);
            let a = (x as usize).min(sw - 1);
            (a, (a + 1).min(sw - 1), x - a as f32)
        })
        .collect();
    dst.par_chunks_mut(w as usize * 3).enumerate().for_each(|(y, row)| {
        let y = ((y as f32 + 0.5) * sh as f32 / h as f32 - 0.5).max(0.);
        let ya = (y as usize).min(sh - 1);
        let yb = (ya + 1).min(sh - 1);
        let fy = y - ya as f32;
        for (x, &(xa, xb, fx)) in xs.iter().enumerate() {
            for k in 0..3 {
                let a = raw[(ya * sw + xa) * 3 + k] as f32;
                let b = raw[(ya * sw + xb) * 3 + k] as f32;
                let c = raw[(yb * sw + xa) * 3 + k] as f32;
                let d = raw[(yb * sw + xb) * 3 + k] as f32;
                let top = a + (b - a) * fx;
                let bottom = c + (d - c) * fx;
                row[x * 3 + k] = (top + (bottom - top) * fy + 0.5) as u8;
            }
        }
    });
    image::RgbImage::from_raw(w, h, dst).expect("validated dimensions")
}
pub fn resize_photo_fast(src: &image::RgbImage, ow: u32, oh: u32) -> image::RgbImage {
    if ow != src.width() * 2 || oh != src.height() * 2 {
        return image::imageops::resize(src, ow, oh, image::imageops::FilterType::Triangle);
    }
    let (w, h) = (src.width() as usize, src.height() as usize);
    let mut out = vec![0u8; ow as usize * oh as usize * 3];
    let raw = src.as_raw();
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;
    out.par_chunks_mut(ow as usize * 3).enumerate().for_each(|(y, row)| {
        let ya = (y as isize - 1).div_euclid(2).max(0) as usize;
        let yb = y.div_ceil(2).min(h - 1);
        let fy = if y % 2 == 0 { 3 } else { 1 };
        for x in 0..ow as usize {
            let xa = (x as isize - 1).div_euclid(2).max(0) as usize;
            let xb = x.div_ceil(2).min(w - 1);
            let fx = if x % 2 == 0 { 3 } else { 1 };
            for k in 0..3 {
                let a = raw[(ya * w + xa) * 3 + k] as u32 * (4 - fx) + raw[(ya * w + xb) * 3 + k] as u32 * fx;
                let b = raw[(yb * w + xa) * 3 + k] as u32 * (4 - fx) + raw[(yb * w + xb) * 3 + k] as u32 * fx;
                row[x * 3 + k] = ((a * (4 - fy) + b * fy + 8) / 16) as u8;
            }
        }
    });
    image::RgbImage::from_raw(ow, oh, out).expect("validated 2x dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_releases_job_and_invalid_inputs_are_errors() {
        let img = image::RgbImage::from_pixel(16, 9, image::Rgb([120, 80, 30]));
        let opts = PhotoOptions { attention: false, ..Default::default() };
        for _ in 0..2 {
            let result = pollster::block_on(paint_photo_image_fast_async(
                img.clone(),
                &opts,
                [0.5; 3],
                Dynamics::default(),
                1,
                FastPhotoOptions::default(),
                &mut |_, _, _| Err("cancelled".into()),
            ));
            assert_eq!(result.unwrap_err(), "cancelled");
        }
        let guard = Job::acquire().unwrap();
        assert!(Job::acquire().is_err());
        drop(guard);
        assert!(Job::acquire().is_ok());
        assert!(paint_photo_image_fast_at_size(img, &opts, [0.5; 3], Dynamics::default(), 1, [123, 456], false, true, &mut |_, _, _| Ok(())).is_err());
    }

    #[test]
    #[ignore = "Requires a real GPU; run explicitly with --ignored"]
    fn fast_photo_native_async_roundtrip_and_all_resolutions() {
        let adapter = pollster::block_on(crate::open_gpu()).expect("real GPU");
        eprintln!("fast photo adapter: {adapter}");
        let img = image::RgbImage::from_fn(160, 90, |x, y| image::Rgb([((x * 3 + y) % 256) as u8, (y * 2) as u8, if x > 80 { 180 } else { 40 }]));
        let opts = PhotoOptions { attention: false, ground_subject: true, ..Default::default() };
        let mut last = None;
        for resolution in [FastPhotoResolution::P270, FastPhotoResolution::P540, FastPhotoResolution::P720, FastPhotoResolution::P1080] {
            let t = std::time::Instant::now();
            let painted = pollster::block_on(paint_photo_image_fast_async(
                img.clone(),
                &opts,
                [0.5; 3],
                Dynamics::default(),
                17,
                FastPhotoOptions { resolution, ..Default::default() },
                &mut |_, _, _| Ok(()),
            ))
            .unwrap();
            let [w, h] = resolution.canvas();
            assert_eq!(painted.dimensions(), (w as u32, h as u32));
            assert!(painted.as_raw().windows(3).any(|p| p != &painted.as_raw()[..3]));
            eprintln!("{w}x{h}: {:.3}s", t.elapsed().as_secs_f64());
            if resolution == FastPhotoResolution::P270 {
                last = Some(painted);
            }
        }
        let same =
            paint_photo_image_fast_at_size(img.clone(), &opts, [0.5; 3], Dynamics::default(), 17, [480, 270], false, true, &mut |_, _, _| Ok(())).unwrap();
        assert_eq!(same, last.unwrap(), "blocking and async APIs must agree");
        let changed = paint_photo_image_fast_at_size(
            img.clone(),
            &opts,
            [0.5; 3],
            Dynamics { stroke_scale: 0.5, wetness: 1.5, ..Default::default() },
            17,
            [480, 270],
            false,
            true,
            &mut |_, _, _| Ok(()),
        )
        .unwrap();
        assert_ne!(same, changed, "controls must affect fast paint");
        let portrait = image::imageops::rotate90(&img);
        let painted =
            paint_photo_image_fast_at_size(portrait, &opts, [0.5; 3], Dynamics::default(), 17, [480, 270], true, false, &mut |_, _, _| Ok(())).unwrap();
        assert_eq!(painted.dimensions(), (90, 160));
        if let Ok(dir) = std::env::var("BRUSHWORK_PROOF_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            img.save(format!("{dir}/source.png")).unwrap();
            same.save(format!("{dir}/native.png")).unwrap();
            changed.save(format!("{dir}/native-fine-wet.png")).unwrap();
        }
    }
    const WI: usize = 960;
    const HI: usize = 540;
    #[test]
    fn blur_matches_serial_reference_across_stripe_edges() {
        let source: Vec<_> = (0..WI * HI).map(|i| [hf(i as u32, 1), hf(i as u32, 2), hf(i as u32, 3)]).collect();
        for radius in [1, 3, 24] {
            let expected = serial_blur(&source, radius);
            let actual = box_blur(&source, radius, DEFAULT_CANVAS);
            let max = actual.iter().zip(expected).flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs())).fold(0.0f32, f32::max);
            assert!(max < 0.00002, "radius {radius}: {max}");
        }
    }
    #[test]
    fn photo_sampling_preserves_constants_and_corners() {
        let constant = image::RgbImage::from_pixel(17, 11, image::Rgb([13, 127, 241]));
        for (w, h) in [(17, 11), (10, 7), (4, 3), (31, 22)] {
            let resized = photo_resize(&constant, w, h);
            assert_eq!(resized.dimensions(), (w, h));
            assert!(resized.pixels().all(|v| v.0 == [13, 127, 241]));
        }
        let source = image::RgbImage::from_fn(10, 7, |x, y| image::Rgb([x as u8 * 20, y as u8 * 30, (x + y) as u8]));
        assert_eq!(photo_resize(&source, 10, 7), source);
    }
    #[test]
    fn simplified_paths_keep_endpoints_and_small_turns() {
        let path: Vec<_> = (0..50).map(|i| [i as f32, (i as f32 * 0.14).sin() * 5.]).collect();
        let result = simplify_photo_path(&path, 0.25);
        assert_eq!(result.first(), path.first());
        assert_eq!(result.last(), path.last());
        assert!(result.len() < path.len() / 2);
        for point in &path {
            let distance = result
                .windows(2)
                .map(|pair| {
                    let a = pair[0];
                    let b = pair[1];
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let t = (((point[0] - a[0]) * d[0] + (point[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])).clamp(0., 1.);
                    (point[0] - a[0] - d[0] * t).hypot(point[1] - a[1] - d[1] * t)
                })
                .fold(f32::INFINITY, f32::min);
            assert!(distance <= 0.25001);
        }
    }
    #[test]
    fn proxy_clips_off_canvas_strokes() {
        let mut proxy = PhotoProxy::new(&vec![[0.; 3]; WI * HI], DEFAULT_CANVAS);
        proxy.sweep(&[[-50., -50.], [-30., -30.]], 5., [1.; 3]);
        proxy.sweep(&[[WI as f32 + 20., 0.], [WI as f32 + 40., HI as f32]], 5., [1.; 3]);
        assert!(proxy.col.iter().all(|c| *c == [0.; 3]));
        proxy.sweep(&[[0., 0.], [20., 0.]], 10., [1.; 3]);
        assert!(proxy.at([4., 0.])[0] > 0.);
        assert_eq!(proxy.at([WI as f32, HI as f32]), [0.; 3]);
    }
    fn serial_blur(src: &[[f32; 3]], r: usize) -> Vec<[f32; 3]> {
        let r = r.max(1) as i32;
        let mut a = src.to_vec();
        let mut b = vec![[0.0f32; 3]; src.len()];
        for _ in 0..2 {
            for y in 0..HI {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0;
                for x in -r..WI as i32 {
                    if x + r < WI as i32 {
                        let v = a[y * WI + (x + r) as usize];
                        for k in 0..3 {
                            acc[k] += v[k];
                        }
                        n += 1.0;
                    }
                    if x - r > 0 {
                        let v = a[y * WI + (x - r - 1) as usize];
                        for k in 0..3 {
                            acc[k] -= v[k];
                        }
                        n -= 1.0;
                    }
                    if x >= 0 {
                        b[y * WI + x as usize] = [acc[0] / n, acc[1] / n, acc[2] / n];
                    }
                }
            }
            for x in 0..WI {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0;
                for y in -r..HI as i32 {
                    if y + r < HI as i32 {
                        let v = b[(y + r) as usize * WI + x];
                        for k in 0..3 {
                            acc[k] += v[k];
                        }
                        n += 1.0;
                    }
                    if y - r > 0 {
                        let v = b[(y - r - 1) as usize * WI + x];
                        for k in 0..3 {
                            acc[k] -= v[k];
                        }
                        n -= 1.0;
                    }
                    if y >= 0 {
                        a[y as usize * WI + x] = [acc[0] / n, acc[1] / n, acc[2] / n];
                    }
                }
            }
        }
        a
    }
}
