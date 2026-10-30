//! brushwork: an oil painter. It repaints a film frame from its recorded shapes, or a photo: every shape is
//! covered with bristle strokes that follow its form, each stroke streaked bristle by bristle, picking up the wet
//! paint under it (mixed as pigment) and leaving ridges of thickness that are lit at the end.
//!
//! A film supplies a [`Style`] (which families are its cast, and so on) and its recorded frames
//! ([`brushwork_paint::Film`]); [`run`] paints one frame, [`run_clip`] a stretch, [`paint_frame`] returns pixels.
//! [`run_photo`] / [`run_photo_with`] paint a photograph. [`Painting`] and [`paint_body`] lay strokes directly.
//!
//! Steadiness is the engine's first rule: everything about a stroke comes from the shape it belongs to and its
//! place on that shape (never from list order, the frame, or where the canvas edge falls), and wherever a decision
//! has a threshold (a size, a direction, inside or outside), the two sides are cross-faded, so a shape that changes
//! by a pixel changes the painting by a pixel's worth. The tests at the bottom hold the engine to that.
//!
//! Backends: the CPU is the reference; with the `gpu` feature and `OIL_BACKEND=gpu` (or [`set_backend`]) the same
//! strokes are laid by the GPU (wgpu), see `gpu.rs`. In a browser (WebGPU) open it with [`open_gpu`] and paint with
//! [`paint_frame_async`], since nothing there may block waiting for the GPU.

// Per-channel and per-band loops index several arrays with the same counter and mirror the GPU kernel line for
// line (strokes.wgsl); the painting functions take a painter's many knobs as plain arguments on purpose.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod gpu;

use brushwork_paint::prelude::*;
use brushwork_paint::{Item, Kind, Lighting, Role};

use std::sync::atomic::{AtomicBool, Ordering};

/// Mix picked-up paint as pigment (Kubelka-Munk, 38 bands) instead of averaging RGB.
static PIGMENT: AtomicBool = AtomicBool::new(true);
/// Everything in watercolour, with the ported brush (to check it against the original engine).
static WATER: AtomicBool = AtomicBool::new(false);
/// Mixed media: the scenery in watercolour on paper, the cast in oils over it.
static MIXED: AtomicBool = AtomicBool::new(false);
/// Scene lighting and a point of attention (see `Scene`).
static LIGHT: AtomicBool = AtomicBool::new(true);
/// DEPTH v2: assets declare their own depth; the setting maps depths to brush styles. A shape's BAND says which
/// recipe paints it (one authored table below — the style map); nothing else modulates. Distance never greys:
/// the ONLY colour operator is a pull toward the frame's own veil HUE (a sunset distance goes rose and orange,
/// because the veil is). No blur anywhere; fusion is the Kubelka-Munk wet pickup.
static DEPTH: AtomicBool = AtomicBool::new(false);

/// What is special about one film, for the painter. Everything else is the same for every film.
#[derive(Clone, Copy)]
pub struct Style {
    /// The families (boil keys, matched as prefixes) of the film's characters. The painter models these as forms
    /// in light, lets them cast shadows, paints them with smaller brushes and thicker paint, and looks at them.
    /// Everything else is scenery.
    pub cast: &'static [&'static str],
    /// Families that are far away or have no substance (the sky, the distance): in mixed media they are watercolour.
    pub airy: &'static [&'static str],
    /// Call-site tags of leaves that can have bite holes, and of the holes (painted as openings in the leaf).
    pub leaf_tags: &'static [&'static str],
    pub hole_tags: &'static [&'static str],
    /// The opt-in depth style map (`OIL_DEPTH`): which distance band a family is in. None: everything is Mid.
    pub bands: Option<fn(&str) -> Band>,
    /// The light the film is seen in, unless a shot says otherwise (`Rec::light`). None: a plain light from the
    /// upper left.
    pub lighting: Option<Lighting>,
    /// The film's palette (0-255 colours): shadows are mixed from it, so they harmonise with everything else. Empty:
    /// shadows are mixed with the ambient light's colour.
    pub palette: &'static [Col],
}
impl Style {
    pub const DEFAULT: Style = Style { cast: &[], airy: &["sky", "sun", "far"], leaf_tags: &[], hole_tags: &[], bands: None, lighting: None, palette: &[] };
}
/// Where strokes are laid: on the CPU (the reference) or on the GPU (the `gpu` feature; see gpu.rs).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Backend {
    Cpu,
    Gpu,
}
static BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// Choose the backend for the frames painted from now on. `Gpu` needs the crate's `gpu` feature.
pub fn set_backend(b: Backend) {
    assert!(b == Backend::Cpu || cfg!(feature = "gpu"), "brushwork was built without the `gpu` feature");
    BACKEND.store(b as u8, Ordering::Relaxed);
}
pub fn backend() -> Backend {
    if BACKEND.load(Ordering::Relaxed) == 1 { Backend::Gpu } else { Backend::Cpu }
}
/// `OIL_BACKEND=gpu` picks the GPU (the `oil`/`oilclip` commands read it).
fn backend_from_env() {
    if std::env::var("OIL_BACKEND").is_ok_and(|v| v == "gpu") {
        set_backend(Backend::Gpu);
    }
}

static FILM_STYLE: std::sync::RwLock<Style> = std::sync::RwLock::new(Style::DEFAULT);
fn style() -> Style {
    *FILM_STYLE.read().unwrap_or_else(|e| e.into_inner())
}
/// Set the film's style (the `run` commands do this; call it yourself only to paint frames another way).
pub fn set_style(s: Style) {
    *FILM_STYLE.write().unwrap_or_else(|e| e.into_inner()) = s;
}

/// The five distance bands, nearest to farthest.
#[derive(Clone, Copy, PartialEq)]
pub enum Band {
    Subject,
    Near,
    Mid,
    Far,
    Sky,
}
/// The style map: how each band is brushed. `veil` is the hue pull toward the frame's own early sky (0 = none);
/// `pickup_add` cranks the wet-into-wet fusion; `ridge` scales the paint's relief; `contour` states edges.
#[derive(Clone, Copy)]
struct Recipe {
    detail: f32,
    body: f32,
    pickup_add: f32,
    ridge: f32,
    opacity: f32,
    veil: f32,
    contour: bool,
}
/// One table, authored in one place. MID and NEAR are today's painting exactly (no change); SUBJECT is untouched
/// bristle, ridge and light. SKY and FAR are the distance: few long strokes, thin, fused, no relief, hue-pulled.
const STYLE: [Recipe; 5] = [
    Recipe { detail: 0.8, body: 1.15, pickup_add: 0.0, ridge: 1.0, opacity: 1.0, veil: 0.0, contour: true }, // SUBJECT
    Recipe { detail: 1.25, body: 0.7, pickup_add: 0.0, ridge: 1.0, opacity: 1.0, veil: 0.0, contour: true }, // NEAR (today)
    Recipe { detail: 1.25, body: 0.7, pickup_add: 0.0, ridge: 1.0, opacity: 1.0, veil: 0.0, contour: true }, // MID (today)
    Recipe { detail: 1.7, body: 0.35, pickup_add: 0.5, ridge: 0.0, opacity: 0.82, veil: 0.3, contour: false }, // FAR
    Recipe { detail: 2.4, body: 0.1, pickup_add: 0.62, ridge: 0.0, opacity: 0.62, veil: 0.32, contour: false }, // SKY
];
fn band_of(family: &str) -> Band {
    style().bands.map_or(Band::Mid, |f| f(family))
}

fn is_cast(family: &str) -> bool {
    style().cast.iter().any(|c| family.starts_with(c))
}

/// The softness of scenery with nothing said about its distance.
const MID_SOFT: f32 = 0.3;
/// How far back a family sits, as a painter treats it: 0 for the subjects (crisp edges, lively broken colour),
/// rising through the middle distance to 1 for the far distance and the sky (worked wet into wet, edges lost,
/// strokes close in value). From the style's distance bands when it has them, else from its cast and airy families.
fn softness(family: &str) -> f32 {
    if is_cast(family) {
        return 0.0;
    }
    match style().bands.map(|f| f(family)) {
        Some(Band::Subject) => 0.0,
        Some(Band::Near) => 0.15,
        Some(Band::Mid) => MID_SOFT,
        Some(Band::Far) => 0.7,
        Some(Band::Sky) => 1.0,
        None if style().airy.iter().any(|a| family.starts_with(a)) => 1.0,
        None => MID_SOFT,
    }
}

/// The canvas: every painting is 1920 x 1080.
pub const WI: usize = 1920;
pub const HI: usize = 1080;
/// The smallest brush, in pixels. Nothing in the picture is finer than this.
const SMALLEST: f32 = 7.0;
/// How much of the wet paint under a stroke its brush carries along. What is under a stroke changes from frame to
/// frame as things move, so in a film this is kept modest or the colours flicker.
const STEADY_PICKUP: f32 = 0.18;
/// How much the paint stands up (ridges, lips, gloss). 1 = thick impasto; lower = paint as a means to an end.
const BODY: f32 = 0.4;
/// How much extra pigment a wash leaves along its edge (see `Mask::composite`).
const EDGE_DEPOSIT: f64 = 2.5;

fn h32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^ (x >> 16)
}
fn hf(a: u32, b: u32) -> f32 {
    (h32(a.wrapping_mul(0x9E37_79B1) ^ h32(b.wrapping_add(0x85EB_CA6B))) >> 8) as f32 / 16_777_216.0
}
/// Smooth noise in [0, 1).
fn noise(x: f32, y: f32, seed: u32) -> f32 {
    let (ix, iy) = (x.floor(), y.floor());
    let (tx, ty) = (x - ix, y - iy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let g = |dx: i32, dy: i32| hf((ix as i32 + dx) as u32 ^ seed.wrapping_mul(977), (iy as i32 + dy) as u32);
    let a = g(0, 0) + (g(1, 0) - g(0, 0)) * sx;
    let b = g(0, 1) + (g(1, 1) - g(0, 1)) * sx;
    a + (b - a) * sy
}
fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The canvas as it stood before a leaf was painted (colour, plain picture, thickness), and that leaf's colour:
/// what a bite hole in it shows.
type UnderLeaf = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<f32>, [f32; 3]);
/// A region `run_photo` found: its area in pixels, mean colour, mean texture and outline.
type PhotoShape = (usize, [f32; 3], f32, Vec<[f32; 2]>);

pub struct Painting {
    pub col: Vec<[f32; 3]>,
    /// The picture in plain colour, shape by shape, with no brushwork. STEADINESS: a brush picks up "what is
    /// under it" from here, not from the painted canvas, whose exact state depends on every stroke before and
    /// would make one stroke's small change spread through all the strokes over it.
    pub flat: Vec<[f32; 3]>,
    /// Paint thickness.
    pub thick: Vec<f32>,
    /// How much of the plain picture is paint rather than the bare toned ground (0 to 1): a brush can only work
    /// wet paint into its own, never the canvas's ground.
    wet: Vec<f32>,
    /// Strokes laid so far.
    pub strokes: u32,
    /// With a recorder, strokes are planned and written down for the GPU instead of laid (see gpu.rs).
    rec: Option<Box<gpu::Recorder>>,
    /// The camera's zoom on the picture as a whole (the paint's relief is lit as if seen from that close).
    pub view: f32,
}

/// The painter's tools. Each job gets the one a painter would reach for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tool {
    /// A wide flat: square ends, crisp sides, strong bristle marks. Backgrounds, and (on its chisel edge) cutting
    /// one colour in against another.
    Flat,
    /// A filbert: rounded ends, softer sides. Modelling the forms of the subjects.
    Filbert,
    /// A small round: a few fine hairs, comes to a point, flows. Lines, outlines, small accents.
    Round,
    /// A palette knife: no bristles, a flat slab of thick paint with hard edges that breaks over what is under it.
    /// Highlights.
    Knife,
}

#[derive(Clone, Copy)]
pub struct Stroke {
    pub tool: Tool,
    /// What the brush picked up where the whole stroke landed (pieces of one sweep share it).
    pub picked: Option<[f32; 3]>,
    /// How far along its whole stroke this piece starts, in pixels (so the bristles' marks carry on).
    pub along: f32,
    pub c: [f32; 2],
    pub angle: f32,
    pub len: f32,
    pub width: f32,
    pub bend: f32,
    pub col: [f32; 3],
    /// A second colour some bristles carry (broken colour).
    pub alt: [f32; 3],
    pub opacity: f32,
    /// How much wet paint the brush picks up from the canvas.
    pub pickup: f32,
    pub load: f32,
    /// Part of a longer stroke: the bristles it shares with the rest, and whether the brush lands / lifts here.
    pub part: Option<(u32, bool, bool)>,
    /// Where this piece sits along the whole stroke it belongs to (0..1 at its two ends); (0, 1) on its own.
    pub span: (f32, f32),
}

impl Default for Painting {
    fn default() -> Self {
        Painting::new()
    }
}

impl Painting {
    /// A fresh canvas on the toned ground.
    pub fn new() -> Painting {
        // a warm toned ground
        let mut col = vec![[0.0; 3]; WI * HI];
        for y in 0..HI {
            for x in 0..WI {
                let n = noise(x as f32 / 140.0, y as f32 / 140.0, 7) * 0.06 - 0.03;
                col[y * WI + x] = if MIXED.load(Ordering::Relaxed) || WATER.load(Ordering::Relaxed) {
                    [0.955 + n * 0.3, 0.93 + n * 0.3, 0.875 + n * 0.3]
                } else {
                    [0.60 + n, 0.50 + n, 0.37 + n]
                };
            }
        }
        Painting { flat: col.clone(), col, thick: vec![0.0; WI * HI], wet: vec![0.0; WI * HI], strokes: 0, rec: None, view: 1.0 }
    }

    /// The same canvas with its paint layers replaced (for tests).
    #[cfg(test)]
    fn with_layers(col: Vec<[f32; 3]>, flat: Vec<[f32; 3]>, thick: Vec<f32>) -> Painting {
        let wet = vec![1.0; col.len()];
        Painting { col, flat, thick, wet, strokes: 0, rec: None, view: 1.0 }
    }

    /// One stroke. `mask` (with its origin and size) limits it softly to a shape.
    pub fn stroke(&mut self, s: &Stroke, mask: Option<(&[f32], [i32; 2], [usize; 2])>) {
        self.strokes += 1;
        let (id, _lands, _lifts) = match s.part {
            Some((id, a, b)) => (id, a, b),
            None => (self.strokes, true, true),
        };
        let (sn, cs) = s.angle.sin_cos();
        let reach = s.len / 2.0 + s.width;
        let (x0, x1) = (((s.c[0] - reach).floor() as i32).max(0), ((s.c[0] + reach).ceil() as i32).min(WI as i32 - 1));
        let (y0, y1) = (((s.c[1] - reach).floor() as i32).max(0), ((s.c[1] + reach).ceil() as i32).min(HI as i32 - 1));
        if x0 > x1 || y0 > y1 {
            return;
        }
        // the paint already there, where the brush lands
        let start = [s.c[0] - cs * s.len * 0.4, s.c[1] - sn * s.len * 0.4];
        // a stroke that lands off the canvas has nothing to pick up (not whatever happens to be at the canvas edge)
        let under = s.picked.unwrap_or_else(|| flat_under(&self.flat, start[0], start[1], s.width * 0.3, s.col));
        let premix = PREMIX.with(|p| p.get());
        let fade = FADE.with(|f| f.get());
        if fade <= 0.0 {
            return;
        }
        let s = &if premix < 1.0 || fade < 1.0 {
            let tint = |c: [f32; 3]| -> [f32; 3] { std::array::from_fn(|k| under[k] + (c[k] - under[k]) * premix) };
            Stroke { col: tint(s.col), alt: tint(s.alt), load: s.load * (0.35 + 0.65 * premix), opacity: s.opacity * fade, ..*s }
        } else {
            *s
        };
        // what the brush's two colours become with more and more of the picked-up paint worked into them: as
        // pigments (blue into yellow goes green and stays rich), worked out once per stroke
        let pigment = PIGMENT.load(Ordering::Relaxed);
        let mut worked = [[[0.0f32; 3]; 9]; 2];
        if pigment {
            let d = |c: [f32; 3]| [c[0] as f64, c[1] as f64, c[2] as f64];
            let ku = brushwork_kit::spectral::ks(d(under));
            for (w, own) in [s.col, s.alt].into_iter().enumerate() {
                let ko = brushwork_kit::spectral::ks(d(own));
                for (i, slot) in worked[w].iter_mut().enumerate() {
                    let f = i as f64 / 8.0 * 0.85;
                    let mixed: [f64; 38] = std::array::from_fn(|b| ko[b] * (1.0 - f) + ku[b] * f);
                    let rgb = brushwork_kit::spectral::colour(&mixed).0;
                    *slot = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32];
                }
            }
        }
        let z = zoom();
        let nb = match s.tool {
            Tool::Flat => (s.width / z / 4.5).max(4.0),
            Tool::Filbert => (s.width / z / 5.5).max(3.0),
            Tool::Round => (s.width / z / 3.0).clamp(2.0, 4.0),
            Tool::Knife => 1.0,
        };
        let (marks, lips, edge_soft) = match s.tool {
            // BODY scales the paint's relief: 1 is thick impasto, the default is thin paint that only just shows
            // how it was put on
            // (the sides of a big brush are not razor-cut: its outer hairs splay and carry less paint, so its
            // margin softens with its width, and broad strokes overlap as soft-sided bands, never as rectangles)
            Tool::Flat => (0.26 * BODY, 0.6 * BODY, 1.2 + 0.03 * s.width),
            Tool::Filbert => (0.2 * BODY, 0.5 * BODY, 2.2 + 0.03 * s.width),
            // a fine round loaded with paint (a rigger) leaves a line with clean sides: its soft margin shrinks with
            // the line, so a thin line stays a line and two close ones stay two
            Tool::Round => (0.07 * BODY, 0.2 * BODY, (0.55 + 0.14 * s.width).min(1.4)),
            Tool::Knife => (0.0, 1.1 * BODY, 0.8),
        };
        // the whole stroke's length (a piece of a sweep knows its share of it), and how far in from its ends the
        // bristles touch down and lift off: about a brush width, less on a touch too short to have that much
        let whole_len = (s.len / (s.span.1 - s.span.0).max(1e-3)).max(1.0);
        let end_w = s.width * (whole_len / (5.0 * s.width)).min(1.0);
        // SPEED (no effect on the picture): everything below that is the same for every pixel of the stroke, or
        // for every pixel of one bristle, is worked out once; and each row is only visited across the span where
        // the footprint can be. The footprint is never wider than half_max either side of its (bent) centre line
        // (taper <= 1, the ragged-side noise < 1), and never longer than len, so a pixel outside those bounds would
        // fail the exact tests below anyway; the bounds carry a pixel of margin so rounding can't cut a pixel off.
        let half_max = s.width * 0.53 * (1.0 + 1e-4) + 1e-3;
        let (slant_k, slant_c) = ((hf(id, 991) - 0.5) * 1.7, (hf(id, 992) - 0.5) * 0.25);
        let ridge_k = recipe().ridge;
        let size_k = (s.width / z / 40.0).clamp(0.3, 1.5);
        let knife_k = if s.tool == Tool::Knife { 1.6 } else { 1.0 };
        let bristle =
            |b: i32| -> [f32; 5] { [hf(id, b as u32), hf(id ^ 0x55, b as u32), hf(id ^ 0xAA, b as u32), hf(id ^ 0x3C, b as u32), hf(id ^ 0x77, b as u32)] };
        let nbi = nb.ceil() as i32 + 2;
        let bristles: Vec<[f32; 5]> = (-1..=nbi).map(bristle).collect();
        let along_max = s.len / 2.0 + 1.0;
        let side_max = half_max + s.bend.abs() * along_max * along_max / s.len + 1.0;
        // the x range (in pixels) where |a * dx + c| <= m, for dx = x + 0.5 - cx
        let span = |a: f32, c: f32, m: f32| -> Option<(i32, i32)> {
            if a.abs() < 1e-6 {
                return if c.abs() <= m { Some((x0, x1)) } else { None };
            }
            let (p, q) = ((-m - c) / a, (m - c) / a);
            let (lo, hi) = (p.min(q) + s.c[0] - 0.5, p.max(q) + s.c[0] - 0.5);
            Some(((lo.floor() as i32 - 1).max(x0), (hi.ceil() as i32 + 1).min(x1)))
        };
        if let Some(rec) = self.rec.as_mut() {
            // GPU: write the finished stroke down, with the x range it can touch in each band of 16 rows (for
            // binning). Each strip's x range is linear in dy, so over a band its extremes are at the band's first and
            // last rows; the union over those two rows of each strip, intersected, holds every row's span.
            let mut bands: Vec<(u32, i32, i32)> = vec![];
            let mut ty = y0 as usize / gpu::TILE;
            while ty * gpu::TILE <= y1 as usize {
                let (ra, rb) = ((ty * gpu::TILE).max(y0 as usize) as i32, ((ty + 1) * gpu::TILE - 1).min(y1 as usize) as i32);
                let strip = |a: f32, c_of: &dyn Fn(f32) -> f32, m: f32| -> Option<(i32, i32)> {
                    let (da, db) = (ra as f32 + 0.5 - s.c[1], rb as f32 + 0.5 - s.c[1]);
                    match (span(a, c_of(da), m), span(a, c_of(db), m)) {
                        (Some(p), Some(q)) => Some((p.0.min(q.0), p.1.max(q.1))),
                        (Some(p), None) | (None, Some(p)) => Some(p),
                        (None, None) => {
                            // both ends outside: the strip may still cross the band in between only if the sign changes
                            let (ca, cb) = (c_of(da), c_of(db));
                            if a.abs() < 1e-6 && (ca.signum() != cb.signum()) { Some((x0, x1)) } else { None }
                        }
                    }
                };
                let u_r = strip(cs, &|dy| dy * sn, along_max);
                let v_r = strip(-sn, &|dy| dy * cs, side_max);
                if let (Some(u), Some(v)) = (u_r, v_r) {
                    let (lo, hi) = (u.0.max(v.0), u.1.min(v.1));
                    if lo <= hi {
                        bands.push((ty as u32, lo, hi));
                    }
                }
                ty += 1;
            }
            let (mask_off, mask_ox, mask_oy, mask_w, mask_h) = match mask {
                Some((m, o, sz)) => {
                    let (off, mo, msz) = rec.mask.expect("a masked stroke on the GPU needs its mask handed over first (Recorder::use_mask)");
                    assert!(mo == o && msz == sz && m.len() == sz[0] * sz[1], "the stroke's mask is not the one handed over");
                    (off, o[0], o[1], sz[0] as u32, sz[1] as u32)
                }
                None => (0, 0, 0, 0, 0),
            };
            let cmd = gpu::Cmd {
                cx: s.c[0],
                cy: s.c[1],
                sn,
                cs,
                len: s.len,
                width: s.width,
                bend: s.bend,
                along: s.along,
                span0: s.span.0,
                span1: s.span.1,
                opacity: s.opacity,
                pickup: s.pickup,
                load: s.load,
                col: s.col,
                alt: s.alt,
                under,
                nb,
                marks,
                lips,
                edge_soft,
                whole_len,
                end_w,
                slant_k,
                slant_c,
                ridge_k,
                size_k,
                knife_k,
                z,
                half_max,
                id,
                tool: match s.tool {
                    Tool::Flat => 0,
                    Tool::Filbert => 1,
                    Tool::Round => 2,
                    Tool::Knife => 3,
                },
                pigment: pigment as u32,
                mask_off,
                mask_ox,
                mask_oy,
                mask_w,
                mask_h,
                x0,
                x1,
                y0,
                y1,
                ..Default::default()
            };
            rec.push(cmd, pigment.then_some(&worked), bands);
            return;
        }
        for y in y0..=y1 {
            let dy = y as f32 + 0.5 - s.c[1];
            let Some((ua, ub)) = span(cs, dy * sn, along_max) else { continue };
            let Some((va, vb)) = span(-sn, dy * cs, side_max) else { continue };
            for x in ua.max(va)..=ub.min(vb) {
                let dx = x as f32 + 0.5 - s.c[0];
                let u = dx * cs + dy * sn;
                let pos = u / s.len + 0.5;
                if !(0.0..=1.0).contains(&pos) {
                    continue;
                }
                let v = -dx * sn + dy * cs - s.bend * (u * u) / s.len;
                if v.abs() > half_max {
                    continue;
                }
                // the brush's footprint: a little narrower where it lands and lifts, ragged at the sides
                // position along the whole stroke: pressure, paint and dryness go by it
                let whole = s.span.0 + (s.span.1 - s.span.0) * pos;
                let across = (v / s.width + 0.5) * nb;
                let b = across.floor() as i32;
                let [r1, r2, r3, r4, r5] = if (-1..=nbi).contains(&b) { bristles[(b + 1) as usize] } else { bristle(b) };
                // The brush meets the canvas at a slant and its bristles touch down and lift off one by one, so a
                // stroke's ends are ragged and oblique, and soft with pressure. (Every bristle used to land within
                // the first 2-6 % of the stroke and the longest ones ran into the hard cut at its end: each stroke
                // began and ended on a straight edge square to its direction. In a sky of level strokes those edges
                // lined up into hard vertical seams all over the film.) No bristle reaches either end of the
                // footprint, so the footprint's own cut never shows.
                // the brush is held at a slant to its path (so its ends are oblique, the same way at both ends) and
                // its loaded edge is rounded (so the middle bristles reach a little further than the outer ones)
                // All of it is measured in brush widths from the stroke's ends (not as a share of its length), so a
                // long line still reaches where it was drawn to and only its last width or so is ragged; a short
                // touch keeps its ends in proportion.
                let t = (across / nb).clamp(0.0, 1.0);
                let slant = (slant_k * (t - 0.5) + slant_c) * end_w;
                let round = 0.38 * (2.0 * t - 1.0).powi(2) * end_w;
                let land_at = ((0.06 + 0.38 * r5) * end_w + slant + round).max(0.0) / whole_len;
                // each bristle runs dry at its own point, and skips over the ground as it does
                let dry_at = 1.0 - ((0.30 + 0.85 * (1.0 - r1)) * end_w - slant + round).max(0.18 * end_w) / whole_len;
                let landed = smooth(land_at, land_at + (0.25 + 0.35 * r2) * end_w / whole_len, whole);
                let lift_k = 1.0 - smooth(dry_at - 0.10 * end_w / whole_len, dry_at + 0.12 * end_w / whole_len, whole);
                let m_k = match mask {
                    Some((m, o, sz)) => {
                        let (mx, my) = (x - o[0], y - o[1]);
                        if mx < 0 || my < 0 || mx as usize >= sz[0] || my as usize >= sz[1] {
                            continue;
                        }
                        m[my as usize * sz[0] + mx as usize]
                    }
                    None => 1.0,
                };
                // SPEED: the coverage below is this product times more factors in [0, 1], so (rounding being
                // monotonic) if this is already too little to lay, the full product is too: skip the rest
                if landed * lift_k * m_k <= 0.004 {
                    continue;
                }
                let arc = (whole * std::f32::consts::PI).sin().max(0.0);
                let taper = match s.tool {
                    Tool::Flat | Tool::Knife => 0.94 + 0.06 * arc,
                    Tool::Filbert => 0.70 + 0.30 * arc.powf(0.5),
                    Tool::Round => 0.35 + 0.65 * arc.powf(0.7),
                };
                let u = u + s.along;
                let half = s.width / 2.0 * taper * (0.94 + 0.12 * noise(u / (9.0 * z), 0.0, id));
                if v.abs() > half {
                    continue;
                }
                // (each bristle reads the noise at its own phase: sharing the lattice lined every bristle's skips up in
                // columns, a fine vertical grain across the stroke; outside the dry zone the first factor is exactly
                // 0, and so is skip, so the noise isn't needed there)
                let dry_k = smooth(dry_at - (0.18f32).min(2.5 * s.width / whole_len), dry_at, whole);
                let skip = if dry_k == 0.0 { 0.0 } else { dry_k * (0.35 + 0.65 * noise(u / (5.0 * z) + 37.0 * r3, b as f32 * 3.7, id ^ 9)) };
                let side = smooth(0.0, edge_soft, half - v.abs());
                let mut a = landed * side * (1.0 - smooth(0.45, 0.8, skip)) * lift_k;
                if s.tool == Tool::Knife {
                    a *= smooth(0.25, 0.45, noise(u / (7.0 * z), v / (2.5 * z), id ^ 77) + 0.35 * (1.0 - whole));
                }
                a *= m_k;
                if a <= 0.004 {
                    continue;
                }
                // paint: the brush's colour, streaked bristle by bristle with a second colour and with what it picked up
                let ridge = 1.0 - marks + marks * ((across - b as f32) * std::f32::consts::PI).sin();
                // paint squeezed out to the sides of the brush stands up in a lip
                let lip = (-((half - v.abs()) / (0.10 * s.width + z)).powi(2)).exp();
                let streak = if r3 < 0.22 { 0.6 } else { 0.0 } * (0.5 + 0.5 * r4);
                let pick = (s.pickup * STEADY_PICKUP * (0.4 + 1.2 * r2) * (0.5 + whole)).clamp(0.0, 0.85);
                let tone = 1.0 + 0.07 * (r4 - 0.5) + 0.03 * lip;
                let i = y as usize * WI + x as usize;
                let mut c = [0.0f32; 3];
                if pigment {
                    let at = pick / 0.85 * 8.0;
                    let (i0, f) = ((at.floor() as usize).min(7), at - at.floor().min(7.0));
                    for k in 0..3 {
                        let a = worked[0][i0][k] + (worked[0][i0 + 1][k] - worked[0][i0][k]) * f;
                        let b = worked[1][i0][k] + (worked[1][i0 + 1][k] - worked[1][i0][k]) * f;
                        c[k] = ((a + (b - a) * streak) * tone).clamp(0.0, 1.0);
                    }
                } else {
                    for k in 0..3 {
                        let own = s.col[k] + (s.alt[k] - s.col[k]) * streak;
                        c[k] = ((own + (under[k] - own) * pick) * tone).clamp(0.0, 1.0);
                    }
                }
                let cover = a * s.opacity;
                for k in 0..3 {
                    self.col[i][k] += (c[k] - self.col[i][k]) * cover;
                }
                // thickness: most where the loaded brush lands, least where it lifts; a new stroke flattens what it crosses
                let lay = s.load * ((0.6 + 0.4 * r1) * ridge * (1.0 - 0.6 * whole) + lips * lip) * knife_k * size_k * ridge_k;
                // the brush leaves ITS thickness: going over the same place twice in one sweep adds nothing
                self.thick[i] += (lay - self.thick[i]) * cover;
            }
        }
    }

    /// A glaze over a shape: a thin transparent layer of one colour, the paint below seen through it (multiplied by
    /// `tint`), its texture and relief left as they are. `mask` (origin `o`, size `sz`) is how much of it lies on
    /// each pixel (its softness is the glaze's edge); a faint grain runs along `dir` (sin, cos) from `anchor`, as a
    /// soft brush leaves in a glaze. The GPU does the same (`Recorder::glaze_area`).
    fn glaze_area(&mut self, mask: &[f32], o: [i32; 2], sz: [usize; 2], tint: [f32; 3], anchor: [f32; 2], dir: [f32; 2], id: u32) {
        self.area(mask, o, sz, tint, anchor, dir, id, Layer::Glaze)
    }

    /// A shadow glazed over an area: the paint at each point taken down by its own shadow colour (`shade_paint`),
    /// so every colour it falls on stays itself, a step darker.
    fn shade_area(&mut self, mask: &[f32], o: [i32; 2], sz: [usize; 2], glaze: ([f32; 3], f32, f32), anchor: [f32; 2], dir: [f32; 2], id: u32) {
        self.area(mask, o, sz, glaze.0, anchor, dir, id, Layer::Shade(glaze.1, glaze.2))
    }

    /// A scumble over an area: light paint `col` dragged thinly over what is there (`mask` how much of it), broken by
    /// the canvas's tooth (a coarse, dry grain), the way a painter lays a glow round a light or a veil of haze.
    fn scumble_area(&mut self, mask: &[f32], o: [i32; 2], sz: [usize; 2], col: [f32; 3], anchor: [f32; 2], dir: [f32; 2], id: u32) {
        self.area(mask, o, sz, col, anchor, dir, id, Layer::Scumble)
    }

    fn area(&mut self, mask: &[f32], o: [i32; 2], sz: [usize; 2], tint: [f32; 3], anchor: [f32; 2], dir: [f32; 2], id: u32, layer: Layer) {
        let z = zoom();
        if let Some(r) = self.rec.as_mut() {
            r.use_mask(mask, o, sz);
            let (mode, mix, step) = match layer {
                Layer::Glaze => (0, 0.0, 0.0),
                Layer::Scumble => (1, 0.0, 0.0),
                Layer::Shade(mix, step) => (2, mix, step),
            };
            r.glaze_area(tint, anchor, dir, z, id, mode, mix, step);
            return;
        }
        for my in 0..sz[1] {
            let y = my as i32 + o[1];
            if y < 0 || y >= HI as i32 {
                continue;
            }
            for mx in 0..sz[0] {
                let x = mx as i32 + o[0];
                let m = mask[my * sz[0] + mx];
                if x < 0 || x >= WI as i32 || m <= 0.0 {
                    continue;
                }
                let (gx, gy) = (x as f32 + 0.5 - anchor[0], y as f32 + 0.5 - anchor[1]);
                let (gu, gv) = (gx * dir[1] + gy * dir[0], -gx * dir[0] + gy * dir[1]);
                let c = &mut self.col[y as usize * WI + x as usize];
                match layer {
                    Layer::Scumble => {
                        let g = m * (0.72 + 0.28 * noise(gu / (60.0 * z), gv / (16.0 * z), id));
                        for k in 0..3 {
                            c[k] += (tint[k] - c[k]) * g;
                        }
                    }
                    Layer::Glaze => {
                        let g = m * glaze_grain(gu, gv, z, id);
                        for k in 0..3 {
                            c[k] *= 1.0 - g * (1.0 - tint[k]);
                        }
                    }
                    Layer::Shade(mix, step) => {
                        let g = m * glaze_grain(gu, gv, z, id);
                        let sh = shade_paint(*c, tint, mix, step);
                        for k in 0..3 {
                            c[k] += (sh[k] - c[k]) * g;
                        }
                    }
                }
            }
        }
    }

    /// Light the paint's relief from the upper left and add the canvas weave where it is thin.
    pub fn lit(&self) -> Vec<u8> {
        let mut out = vec![0u8; WI * HI * 3];
        if WATER.load(Ordering::Relaxed) {
            // watercolour alone has no relief to light: the paper, with the kit's grain
            let mut s: u64 = 5;
            let mut rnd = || {
                s = (s * 16807) % 2_147_483_647;
                s as f32 / 2_147_483_647.0
            };
            for (i, c) in self.col.iter().enumerate() {
                let g = if rnd() < 0.55 { 1.0 - rnd() * rnd() * 34.0 / 255.0 } else { 1.0 };
                for k in 0..3 {
                    out[i * 3 + k] = ((c[k] * g).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
            return out;
        }
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
        let mut soft = vec![0.0f32; WI * HI];
        for y in 0..HI {
            for x in 0..WI {
                let mut acc = 0.0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        acc += self.thick[(y as i32 + dy).clamp(0, HI as i32 - 1) as usize * WI + (x as i32 + dx).clamp(0, WI as i32 - 1) as usize];
                    }
                }
                soft[y * WI + x] = acc / 9.0;
            }
        }
        let hgt = |x: usize, y: usize| -> f32 {
            let (xf, yf) = (x as f32, y as f32);
            let weave = 0.02 * ((xf * 1.55).sin() * (yf * 1.55).sin());
            let t = soft[y.min(HI - 1) * WI + x.min(WI - 1)];
            t + weave * (1.0 - t).clamp(0.15, 1.0)
        };
        for y in 0..HI {
            for x in 0..WI {
                let gx = (hgt(x + 1, y) - hgt(x.saturating_sub(1), y)) * 1.9 * BODY * self.view;
                let gy = (hgt(x, y + 1) - hgt(x, y.saturating_sub(1))) * 1.9 * BODY * self.view;
                let inv = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
                let n = [-gx * inv, -gy * inv, inv];
                let diff = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]) / l[2];
                let spec = (n[0] * hv[0] + n[1] * hv[1] + n[2] * hv[2]).max(0.0).powf(26.0) - hv[2].powf(26.0);
                let shade = 0.74 + 0.26 * diff;
                let c = self.col[y * WI + x];
                // a little richer, as varnished paint is
                let grey = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
                for k in 0..3 {
                    let v = (grey + (c[k] - grey) * 1.2) * shade + spec.max(0.0) * 0.30 * BODY;
                    out[(y * WI + x) * 3 + k] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
        out
    }
}

/// What the painter knows about the scene beyond its shapes: where the light comes from and what is being looked at.
#[derive(Clone, Copy)]
struct Scene {
    /// The light the frame is seen in.
    l: Light,
    focus: [f32; 2],
    reach: f32,
}
thread_local! {
    /// The camera's zoom on the shape being painted. STEADINESS under a zoom: a shape's brushwork is planned at the
    /// size the shape has in the film's world and then seen through the camera, like a painting being moved
    /// towards. Brush size, bristle count, stroke spacing and every wobble scale together, so nothing re-plans as
    /// the camera moves in.
    static ZOOM: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
    static PREMIX: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
    static FADE: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
    /// DEPTH v2: the band of the shape being painted, and the frame's atmospheric veil hue (its own early sky).
    static BAND: std::cell::Cell<Band> = const { std::cell::Cell::new(Band::Mid) };
    static VEIL: std::cell::Cell<[f32; 3]> = const { std::cell::Cell::new([0.5, 0.55, 0.65]) };
    /// How far back the shape being painted sits, as a painter treats it (see `softness`): 0 the focal subject,
    /// 1 the far distance and the sky.
    static SOFT: std::cell::Cell<f32> = const { std::cell::Cell::new(MID_SOFT) };
    /// How the shape being painted is to be treated beyond its size and place (from its role; see `Mode`).
    static MODE: std::cell::Cell<Mode> = const { std::cell::Cell::new(Mode { clean: 0.0, model: 0.0, line: 0.0 }) };
    // one frame is painted start to finish on one thread
    static SCENE: std::cell::Cell<Option<Scene>> = const { std::cell::Cell::new(None) };
    /// The frame's light (also when the scene's modelling is switched off: shadows a film draws still need it).
    static FRAME_LIGHT: std::cell::Cell<Option<Light>> = const { std::cell::Cell::new(None) };
}
/// How much of what is painted now is laid (1 = all of it). A painter's decisions about a shape go by its size,
/// and a size threshold crossed by a shape that grows or shrinks by a pixel would switch the treatment in one frame:
/// so near every such threshold the two treatments are cross-faded, each laid at its share (`Fade`), and a shape
/// arrives or leaves by degrees. Fades nest (they multiply).
struct Fade(f32);
impl Fade {
    fn set(k: f32) -> Fade {
        let prev = FADE.with(|f| f.get());
        FADE.with(|f| f.set(prev * k.clamp(0.0, 1.0)));
        Fade(prev)
    }
}
impl Drop for Fade {
    fn drop(&mut self) {
        FADE.with(|f| f.set(self.0));
    }
}

/// How an area layer lies on the paint (`Painting::area`).
#[derive(Clone, Copy)]
enum Layer {
    /// a transparent colour: the paint multiplied by it
    Glaze,
    /// light paint dragged thinly over it
    Scumble,
    /// a shadow: the paint taken down by its own shadow colour (`shade_paint` with this mix and step)
    Shade(f32, f32),
}

/// What a shape's role asks of its painting, each by degrees (0 to 1) so that an inferred role can cross-fade:
/// `clean`, paint that picks up nothing and is laid evenly with no band cut in round it (a light, a highlight);
/// `model`, a light source's soft modelling (lightest in its middle, a little greyer and cooler at its rim);
/// `line`, linework drawn as a shape (one pull of a fine brush, whatever its size).
#[derive(Clone, Copy, Default)]
struct Mode {
    clean: f32,
    model: f32,
    line: f32,
}
struct ModeGuard(Mode);
impl ModeGuard {
    fn set(m: Mode) -> ModeGuard {
        ModeGuard(MODE.with(|c| c.replace(m)))
    }
}
impl Drop for ModeGuard {
    fn drop(&mut self) {
        MODE.with(|c| c.set(self.0));
    }
}

/// The film's opacity for the wash being painted (1 = as is). Paint is mixed down with what it lands on by this
/// much before it is laid, as a painter mixes a thin tint on the palette instead of laying a transparent stroke.
struct Premix;
impl Premix {
    fn set(k: f32) -> Premix {
        PREMIX.with(|p| p.set(k.clamp(0.0, 1.0)));
        Premix
    }
}
impl Drop for Premix {
    fn drop(&mut self) {
        PREMIX.with(|p| p.set(1.0));
    }
}
fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("OIL_TRACE").is_ok())
}
fn zoom() -> f32 {
    ZOOM.with(|z| z.get()).max(0.05)
}
/// the recipe painting the current shape (Mid when the spike is off — today's painting, unchanged)
fn recipe() -> Recipe {
    let b = if DEPTH.load(Ordering::Relaxed) { BAND.with(|d| d.get()) } else { Band::Mid };
    STYLE[b as usize]
}
/// distance moves a colour toward the veil HUE and nowhere else: no grey, no desaturation operator — the valley
/// goes rose at sunset because the veil is rose. Chroma may compress as a consequence, never as an act.
fn towards_veil(c: [f32; 3], pull: f32) -> [f32; 3] {
    if pull <= 0.0 {
        return c;
    }
    let veil = VEIL.with(|v| v.get());
    std::array::from_fn(|k| (c[k] + (veil[k] - c[k]) * pull).clamp(0.0, 1.0))
}
/// The frame's light (a plain one outside a frame: a painting laid stroke by stroke, a photo).
fn frame_light() -> Light {
    FRAME_LIGHT.with(|f| f.get()).unwrap_or_else(|| Light::from(&PLAIN_LIGHT, None, &[]))
}
fn scene() -> Option<Scene> {
    if LIGHT.load(Ordering::Relaxed) { SCENE.with(|s| s.get()) } else { None }
}
impl Scene {
    fn attention(&self, q: [f32; 2]) -> f32 {
        (-((q[0] - self.focus[0]).hypot(q[1] - self.focus[1]) / self.reach).powi(2)).exp()
    }
    /// Restraint: away from what is looked at, colour is greyed and lights and darks are drawn together, a touch
    /// cooler. Full-strength colour is kept for the centre of interest.
    fn quiet(&self, c: [f32; 3], q: [f32; 2]) -> [f32; 3] {
        let away = 1.0 - self.attention(q);
        let grey = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
        std::array::from_fn(|k| {
            // only a little: the scenery should recede, not go grey
            let v = grey + (c[k] - grey) * (1.0 - 0.10 * away);
            (0.55 + (v - 0.55) * (1.0 - 0.07 * away) + [-0.004, 0.0, 0.008][k] * away).clamp(0.0, 1.0)
        })
    }
    /// The side of a form turned from the light, as a painter models it: `side` runs from -1 (turned to the light)
    /// to +1 (turned away). The form turns into shadow over a soft terminator, is darkest just past it (the core
    /// shadow) and picks up a little reflected light at its far rim; the shadow is the form's own colour mixed down
    /// with the scene's shadow pigment (see `Light::shade`, here from the shape's table `tab`), the lit side a
    /// little lighter and towards the key light's colour. (It used to switch to shadow over a narrow band and grey
    /// and darken the colour itself: a cream belly got a hard-edged grey wedge across it.)
    fn turned(&self, tab: &ShadeTable, c: [f32; 3], side: f32, form: f32) -> [f32; 3] {
        let side = side.clamp(-1.0, 1.0);
        let shadow = smooth(0.0, 0.65, side);
        let core = 0.2 * (-((side - 0.7) / 0.22).powi(2)).exp();
        let reflect = 0.3 * smooth(0.75, 1.0, side);
        let depth = (form * (0.55 * shadow + core - reflect)).clamp(0.0, 1.0);
        let lit = form * smooth(0.0, -0.9, side);
        let d = tab.at(depth);
        let key = self.l.key;
        std::array::from_fn(|k| (d[k] + (c[k] * (1.0 + 0.08 * self.l.strength) * (0.9 + 0.1 * key[k]) - c[k]) * lit).clamp(0.0, 1.0))
    }
}

/// The light a frame is seen in, as the painter works with it (0-1 colours, canvas pixels).
#[derive(Clone, Copy)]
struct Light {
    /// Towards the key light (unit), or the key light's place.
    dir: [f32; 2],
    at: Option<[f32; 2]>,
    key: [f32; 3],
    strength: f32,
    /// The paint shadows are mixed down with: the palette's colour nearest the ambient light's, among its darks
    /// (or a deep mix of the ambient light's colour).
    pigment: [f32; 3],
}
/// A plain light from the upper left (what every frame had before lighting could be declared).
const PLAIN_LIGHT: Lighting = Lighting { key_dir: [-0.64, -0.77], key_at: None, key: [255.0, 246.0, 232.0], strength: 0.6, ambient: [96.0, 104.0, 140.0] };

fn lum3(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

impl Light {
    fn from(l: &Lighting, at: Option<[f32; 2]>, palette: &[Col]) -> Light {
        let c = |v: Col| [v[0] as f32 / 255.0, v[1] as f32 / 255.0, v[2] as f32 / 255.0];
        let ambient = c(l.ambient);
        let n = (l.key_dir[0].hypot(l.key_dir[1]) as f32).max(1e-6);
        let dark = |p: [f32; 3]| lum3(p) < 0.4;
        let dist = |p: [f32; 3]| {
            // nearness in hue and in value to a deep version of the ambient light
            let g = [ambient[0] * 0.55, ambient[1] * 0.55, ambient[2] * 0.55];
            (0..3).map(|k| (p[k] - g[k]).powi(2)).sum::<f32>()
        };
        let pigment = palette.iter().map(|v| c(*v)).filter(|p| dark(*p)).min_by(|a, b| dist(*a).total_cmp(&dist(*b))).unwrap_or([
            ambient[0] * 0.55,
            ambient[1] * 0.55,
            ambient[2] * 0.55,
        ]);
        Light { dir: [l.key_dir[0] as f32 / n, l.key_dir[1] as f32 / n], at, key: c(l.key), strength: l.strength.clamp(0.0, 1.0) as f32, pigment }
    }
    /// Towards the key light from `q`.
    fn towards(&self, q: [f32; 2]) -> [f32; 2] {
        match self.at {
            Some(a) => {
                let (dx, dy) = (a[0] - q[0], a[1] - q[1]);
                let l = dx.hypot(dy).max(1e-3);
                // (right at the light, any way is towards it: fade to the declared direction there)
                let k = smooth(0.0, 40.0, l);
                let v = [dx / l * k + self.dir[0] * (1.0 - k), dy / l * k + self.dir[1] * (1.0 - k)];
                let n = v[0].hypot(v[1]).max(1e-6);
                [v[0] / n, v[1] / n]
            }
            None => self.dir,
        }
    }
    /// The colour a painter mixes for paint that looks `local` in the light, where the key light does not reach
    /// (`depth` 1, a full cast shadow) or only partly (`depth` below 1, a form turning away). Not the local colour
    /// gone towards black or grey, nor gone to mud: the local colour with a touch of the scene's shadow colour (from
    /// its palette, nearest its ambient light) worked in, taken down a value step that is as deep as the key light
    /// is strong: cool shadows under a warm light, and so on.
    fn shade(&self, local: [f32; 3], depth: f32) -> [f32; 3] {
        let depth = depth.clamp(0.0, 1.0);
        let v = lum3(local);
        if depth <= 0.0 || v <= 1e-4 {
            return local;
        }
        let target = v * (1.0 - depth * self.strength * 0.72);
        // (a little of the shadow colour, as a painter adds a touch of it to the local colour on the palette:
        // mixed as pigment, a blue into an orange goes green, which is not what a shadow on an orange looks like)
        let m: [f32; 3] = std::array::from_fn(|k| local[k] + (self.pigment[k] - local[k]) * 0.16 * depth);
        let l = lum3(m).max(1e-4);
        m.map(|x| (x * target / l).clamp(0.0, 1.0))
    }
    /// What a shadow glaze of `depth` does to paint that is there: `(pigment, mix, step)` for `shade_paint`.
    fn glaze_of(&self, depth: f32) -> ([f32; 3], f32, f32) {
        let depth = depth.clamp(0.0, 1.0);
        (self.pigment, 0.16 * depth, depth * self.strength * 0.72)
    }
}

/// A shadow glazed over paint `c`: a touch (`mix`) of the shadow colour `pig` worked in, and its value taken down by
/// `step` (the same as `Light::shade`, here for whatever paint is at a pixel; strokes.wgsl has the same).
fn shade_paint(c: [f32; 3], pig: [f32; 3], mix: f32, step: f32) -> [f32; 3] {
    let v = lum3(c);
    let m: [f32; 3] = std::array::from_fn(|k| c[k] + (pig[k] - c[k]) * mix);
    let l = lum3(m).max(1e-4);
    m.map(|x| (x * v * (1.0 - step) / l).clamp(0.0, 1.0))
}

/// `Light::shade` for one local colour at nine depths, interpolated (a form's every stroke needs it).
struct ShadeTable([[f32; 3]; 9]);
impl ShadeTable {
    fn new(l: &Light, local: [f32; 3]) -> ShadeTable {
        ShadeTable(std::array::from_fn(|i| l.shade(local, i as f32 / 8.0)))
    }
    fn at(&self, depth: f32) -> [f32; 3] {
        let x = depth.clamp(0.0, 1.0) * 8.0;
        let i = (x.floor() as usize).min(7);
        let f = x - i as f32;
        std::array::from_fn(|k| self.0[i][k] + (self.0[i + 1][k] - self.0[i][k]) * f)
    }
}

/// The painter's palette. The film's colours are cool and even; a painter would warm them, deepen them and let the
/// strong ones sing: more chroma, a tilt towards warm, darks pulled down a little.
fn palette(c: [f32; 3]) -> [f32; 3] {
    let grey = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let chroma = 1.0 + RICH * 0.34;
    let deep = grey.powf(1.0 + RICH * 0.18) / grey.max(1e-4);
    std::array::from_fn(|k| ((grey + (c[k] - grey) * chroma) * deep * (1.0 + RICH * [0.07, 0.015, -0.085][k])).clamp(0.0, 1.0))
}
/// How far the palette is pushed (0 = the film's own colours).
const RICH: f32 = 0.0;

fn rgb01(c: Col) -> [f32; 3] {
    let c = [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0];
    if WATER.load(Ordering::Relaxed) { c } else { palette(c) }
}
fn shifted(c: [f32; 3], k: f32, warm: f32) -> [f32; 3] {
    [(c[0] * k + warm * 0.04).clamp(0.0, 1.0), (c[1] * k).clamp(0.0, 1.0), (c[2] * k - warm * 0.04).clamp(0.0, 1.0)]
}

/// A shape's soft coverage over its bounding box, grown a little so strokes may cross its edge.
fn shape_mask(pts: &[[f32; 2]], grow: f32) -> Option<(Vec<f32>, [i32; 2], [usize; 2])> {
    shape_mask_with(pts, grow, None)
}

/// As `shape_mask`, with lost and found edges: `lost(i)` says, for the canvas pixel `i` just outside the shape,
/// how far the shape's edge is lost there (0 found: the strokes stop crisply at the edge; 1 lost: they run on
/// past it, over the grown margin, into the paint of the same value beyond).
fn shape_mask_with(pts: &[[f32; 2]], grow: f32, lost: Option<&dyn Fn(usize) -> f32>) -> Option<(Vec<f32>, [i32; 2], [usize; 2])> {
    let (mut m, o, sz) = blurred_cover(pts, grow)?;
    match lost {
        None => {
            for v in &mut m {
                *v = smooth(0.22, 0.5, *v);
            }
        }
        Some(lost) => {
            for y in 0..sz[1] {
                for x in 0..sz[0] {
                    let v = &mut m[y * sz[0] + x];
                    // (inside the shape the blurred coverage is at least one half: only the margin is judged)
                    if *v > 0.0 && *v < 0.5 {
                        let l = lost((y + o[1] as usize) * WI + x + o[0] as usize);
                        *v = smooth(0.44 - 0.3 * l, 0.5, *v);
                    } else {
                        *v = smooth(0.22, 0.5, *v);
                    }
                }
            }
        }
    }
    Some((m, o, sz))
}

/// A shape's coverage over its bounding box (grown by `grow` and a little), blurred twice with a box `grow` wide:
/// one half on its outline, rising inside and falling outside over about `grow`.
fn blurred_cover(pts: &[[f32; 2]], grow: f32) -> Option<(Vec<f32>, [i32; 2], [usize; 2])> {
    let pad = grow.ceil() as i32 + 2;
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let o = [(lo[0].floor() as i32 - pad).max(0), (lo[1].floor() as i32 - pad).max(0)];
    let e = [(hi[0].ceil() as i32 + pad).min(WI as i32), (hi[1].ceil() as i32 + pad).min(HI as i32)];
    if e[0] <= o[0] || e[1] <= o[1] {
        return None;
    }
    let sz = [(e[0] - o[0]) as usize, (e[1] - o[1]) as usize];
    let mut m = vec![0.0f32; sz[0] * sz[1]];
    let n = pts.len();
    let mut xs: Vec<f32> = vec![];
    for y in 0..sz[1] {
        let py = (y as i32 + o[1]) as f32 + 0.5;
        xs.clear();
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            if (a[1] > py) != (b[1] > py) {
                xs.push(a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
            }
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        for pair in xs.chunks(2) {
            if let [a, b] = pair {
                let (xa, xb) = (((*a - o[0] as f32).round() as i32).clamp(0, sz[0] as i32), ((*b - o[0] as f32).round() as i32).clamp(0, sz[0] as i32));
                for x in xa..xb {
                    m[y * sz[0] + x as usize] = 1.0;
                }
            }
        }
    }
    // soften and grow: two box blurs, then a threshold below one half
    let r = (grow.max(1.0)) as usize;
    let mut tmp = vec![0.0f32; m.len()];
    for _ in 0..2 {
        for y in 0..sz[1] {
            let row = &m[y * sz[0]..(y + 1) * sz[0]];
            let mut acc = 0.0;
            let mut cnt = 0usize;
            for x in 0..sz[0].min(r) {
                acc += row[x];
                cnt += 1;
            }
            for x in 0..sz[0] {
                if x + r < sz[0] {
                    acc += row[x + r];
                    cnt += 1;
                }
                if x > r {
                    acc -= row[x - r - 1];
                    cnt -= 1;
                }
                tmp[y * sz[0] + x] = acc / cnt.max(1) as f32;
            }
        }
        // (every column's running sum advanced row by row, side by side: the same additions per column, in the
        // same order, as walking each column alone, but along memory instead of across it)
        let mut acc = vec![0.0f32; sz[0]];
        let mut cnt = 0usize;
        for y in 0..sz[1].min(r) {
            for x in 0..sz[0] {
                acc[x] += tmp[y * sz[0] + x];
            }
            cnt += 1;
        }
        for y in 0..sz[1] {
            if y + r < sz[1] {
                for x in 0..sz[0] {
                    acc[x] += tmp[(y + r) * sz[0] + x];
                }
                cnt += 1;
            }
            if y > r {
                for x in 0..sz[0] {
                    acc[x] -= tmp[(y - r - 1) * sz[0] + x];
                }
                cnt -= 1;
            }
            let d = cnt.max(1) as f32;
            for x in 0..sz[0] {
                m[y * sz[0] + x] = acc[x] / d;
            }
        }
    }
    Some((m, o, sz))
}

#[allow(clippy::too_many_arguments)]
/// The strokes a shape is laid with: (slot centre, stroke id, how much of it is laid 0..1). `c` is the shape's
/// centroid, `lo`/`hi` its bounding box clipped to the canvas, `base` the brushwork's direction.
/// `long` (1 or more) stretches the lattice along the brushwork, for strokes that many times longer.
fn plan_slots(pts: &[[f32; 2]], c: [f32; 2], lo: [f32; 2], hi: [f32; 2], width: f32, base: f32, seed: u32, long: f32) -> Vec<([f32; 2], u32, f32)> {
    let vis = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0];
    let mid = c;
    let (step_u, step_v) = (width * 1.35 * long, width * 0.42);
    let (sn, cs) = base.sin_cos();
    // a stroke reaches up to about 4.3 brush widths (times `long`) from its slot: every slot that could touch the
    // visible part is visited and kept, so none appears or disappears as the canvas edge slides along the shape
    let reach = 4.5 * width * long;
    let rad = (hi[0] - lo[0]).hypot(hi[1] - lo[1]) / 2.0 + reach;
    let (du, dv) = ((vis[0] - c[0]) * cs + (vis[1] - c[1]) * sn, -(vis[0] - c[0]) * sn + (vis[1] - c[1]) * cs);
    let (cu, cv) = ((du / step_u).round() as i32, (dv / step_v).round() as i32);
    let (nu, nv) = ((rad / step_u).ceil() as i32 + 1, (rad / step_v).ceil() as i32 + 1);
    // How far inside the shape a slot's centre is decides how much of the stroke is laid, smoothly. (A hard
    // in/out test on the rasterised mask let a stroke whose centre sat on the edge appear and vanish from one
    // frame to the next as the shape or camera moved a fraction of a pixel; with strokes several brush-widths
    // long, that was the "few strokes that flicker while most are perfect".) Strokes are clipped by the mask
    // anyway, so a slot just outside the edge only contributes the part of it that lands inside.
    let (fade_out, fade_in) = (-0.45 * width, 0.15 * width);
    let mut spots: Vec<([f32; 2], u32, f32)> = vec![];
    for iu in cu - nu..=cu + nu {
        for iv in cv - nv..=cv + nv {
            let id = seed.wrapping_mul(31).wrapping_add((iu * 4001 + iv) as u32);
            let (u, v) = ((iu as f32 + hf(id, 1) - 0.5) * step_u, (iv as f32 + hf(id, 2) - 0.5) * step_v);
            let q = [mid[0] + u * cs - v * sn, mid[1] + u * sn + v * cs];
            if q[0] < lo[0] - reach || q[1] < lo[1] - reach || q[0] > hi[0] + reach || q[1] > hi[1] + reach {
                continue;
            }
            let d = signed_depth(pts, q, fade_in);
            if d <= fade_out {
                continue;
            }
            let e = smooth(fade_out, fade_in, d);
            if e <= 0.002 {
                continue;
            }
            spots.push((q, id, e));
        }
    }
    spots
}

/// The nearest distance from `q` to the outline, and the outline's direction there (a line, so modulo pi), blended
/// over the edges that are nearly as close.
fn edge_dir(pts: &[[f32; 2]], q: [f32; 2], width: f32) -> (f32, f32) {
    let mut ds: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
    let mut near = f32::MAX;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        let ab = [b[0] - a[0], b[1] - a[1]];
        if ab[0].abs() + ab[1].abs() < 1e-3 {
            continue;
        }
        let t = (((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-6)).clamp(0.0, 1.0);
        let d = (q[0] - a[0] - ab[0] * t).hypot(q[1] - a[1] - ab[1] * t);
        near = near.min(d);
        ds.push((d, ab[1].atan2(ab[0])));
    }
    let along = {
        let sigma = (0.5 * width).max(1.0);
        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for (d, th) in &ds {
            let w = (-((d - near) / sigma).powi(2)).exp();
            sx += w * (2.0 * th).cos();
            sy += w * (2.0 * th).sin();
        }
        0.5 * sy.atan2(sx)
    };
    (near, along)
}

/// How far `q` lies inside the polygon (negative outside), in pixels; exact within `cap` of the edge, and `cap`
/// anywhere deeper (the caller only needs the band near the edge).
fn signed_depth(pts: &[[f32; 2]], q: [f32; 2], cap: f32) -> f32 {
    let mut near = f32::MAX;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        let ab = [b[0] - a[0], b[1] - a[1]];
        let t = (((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-6)).clamp(0.0, 1.0);
        near = near.min((q[0] - a[0] - ab[0] * t).hypot(q[1] - a[1] - ab[1] * t));
    }
    if in_poly(pts, q) { near.min(cap) } else { -near }
}

/// The plain picture's colour around (x, y), averaged over a little patch of radius `r`: what a brush landing
/// there picks up. A single pixel would flip between two colours whenever an edge in the plain picture moved
/// across it by a fraction of a pixel, changing the whole stroke's colour from one frame to the next. Off the
/// canvas there is nothing to pick up, so the result slides to `none` over one patch radius past the edge.
fn flat_under(flat: &[[f32; 3]], x: f32, y: f32, r: f32, none: [f32; 3]) -> [f32; 3] {
    let r = r.max(1.0);
    let bil = |x: f32, y: f32| -> [f32; 3] {
        let (x, y) = (x.clamp(0.0, WI as f32 - 1.001), y.clamp(0.0, HI as f32 - 1.001));
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let (a, b, c, d) = (flat[y0 * WI + x0], flat[y0 * WI + x0 + 1], flat[(y0 + 1) * WI + x0], flat[(y0 + 1) * WI + x0 + 1]);
        std::array::from_fn(|k| (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy)
    };
    let mut sum = [0.0f32; 3];
    for j in -1..=1 {
        for i in -1..=1 {
            let q = bil(x + i as f32 * r, y + j as f32 * r);
            for k in 0..3 {
                sum[k] += q[k] / 9.0;
            }
        }
    }
    let outside = (-x).max(x - (WI as f32 - 1.0)).max(-y).max(y - (HI as f32 - 1.0)).max(0.0);
    let w = smooth(0.0, r, outside);
    std::array::from_fn(|k| sum[k] + (none[k] - sum[k]) * w)
}

/// How much of the plain picture around (x, y) is paint (not the bare ground), over the same patch `flat_under`
/// averages.
fn wet_under(wet: &[f32], x: f32, y: f32, r: f32) -> f32 {
    let r = r.max(1.0);
    let at = |x: f32, y: f32| -> f32 {
        let (x, y) = (x.clamp(0.0, WI as f32 - 1.0), y.clamp(0.0, HI as f32 - 1.0));
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(WI - 1), (y0 + 1).min(HI - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        (wet[y0 * WI + x0] * (1.0 - fx) + wet[y0 * WI + x1] * fx) * (1.0 - fy) + (wet[y1 * WI + x0] * (1.0 - fx) + wet[y1 * WI + x1] * fx) * fy
    };
    let mut sum = 0.0;
    for j in -1..=1 {
        for i in -1..=1 {
            sum += at(x + i as f32 * r, y + j as f32 * r) / 9.0;
        }
    }
    sum
}

/// Two paints mixed as pigments (Kubelka-Munk): `f` of `b` worked into `a`.
fn pigment_mix(a: [f32; 3], b: [f32; 3], f: f32) -> [f32; 3] {
    if f <= 0.0 {
        return a;
    }
    let d = |c: [f32; 3]| [c[0] as f64, c[1] as f64, c[2] as f64];
    let (ka, kb) = (brushwork_kit::spectral::ks(d(a)), brushwork_kit::spectral::ks(d(b)));
    let f = f.min(1.0) as f64;
    let mixed: [f64; 38] = std::array::from_fn(|i| ka[i] * (1.0 - f) + kb[i] * f);
    let rgb = brushwork_kit::spectral::colour(&mixed).0;
    [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]
}

/// How different two paints look, as a painter judging an edge does: mostly by value, partly by hue.
fn contrast(a: [f32; 3], b: [f32; 3]) -> f32 {
    let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let hue = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    (lum(a) - lum(b)).abs() + 0.35 * hue
}

/// A shape's breadth, as a painter sizing a brush judges it: the widest part of it, the largest circle that fits
/// inside (found on a grid over its box and refined), never less than its mean breadth (its area over half its
/// outline's length, a strip's width), never more than its bounding box's short side. A thin twig lying on the
/// slant is thin; a crescent is as broad as its belly, not its hair-thin horns.
fn thickness(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    let (mut area, mut perim) = (0.0f32, 0.0f32);
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        for k in 0..2 {
            lo[k] = lo[k].min(a[k]);
            hi[k] = hi[k].max(a[k]);
        }
        area += a[0] * b[1] - b[0] * a[1];
        perim += (b[0] - a[0]).hypot(b[1] - a[1]);
    }
    let small = (hi[0] - lo[0]).min(hi[1] - lo[1]);
    let mean = area.abs() / perim.max(1e-3);
    // (the grid hangs on the shape's own box, so a shape that only moves keeps exactly its breadth)
    const G: usize = 12;
    let mut best = (0.0f32, [0.0f32; 2]);
    for j in 0..G {
        for i in 0..G {
            let q = [lo[0] + (i as f32 + 0.5) / G as f32 * (hi[0] - lo[0]), lo[1] + (j as f32 + 0.5) / G as f32 * (hi[1] - lo[1])];
            let d = signed_depth(pts, q, f32::MAX);
            if d > best.0 {
                best = (d, q);
            }
        }
    }
    if best.0 > 0.0 {
        let mut step = [(hi[0] - lo[0]) / G as f32 * 0.5, (hi[1] - lo[1]) / G as f32 * 0.5];
        for _ in 0..8 {
            let from = best;
            for (dx, dy) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)] {
                let q = [from.1[0] + dx * step[0], from.1[1] + dy * step[1]];
                let d = signed_depth(pts, q, f32::MAX);
                if d > best.0 {
                    best = (d, q);
                }
            }
            step = [step[0] * 0.5, step[1] * 0.5];
        }
    }
    small.min(mean.max(2.0 * best.0))
}

/// Whether a shape covers the whole canvas (a full-frame overlay): the canvas's corners are inside it and none of
/// its outline runs through the canvas.
fn covers_canvas(pts: &[[f32; 2]]) -> bool {
    let (w, h) = (WI as f32, HI as f32);
    let corners = [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]];
    if pts.len() < 3 || !corners.iter().all(|&q| in_poly(pts, q)) {
        return false;
    }
    (0..pts.len()).all(|i| {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        (0..=16).all(|k| {
            let t = k as f32 / 16.0;
            let q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            q[0] <= 0.5 || q[1] <= 0.5 || q[0] >= w - 0.5 || q[1] >= h - 0.5
        })
    })
}

/// A full-frame veil of colour `c` at strength `k` over the painting as it stands. The paint's relief sinks under
/// it in proportion, so at full strength the frame is the colour alone.
fn glaze(p: &mut Painting, c: [f32; 3], k: f32) {
    let k = k.clamp(0.0, 1.0);
    if k <= 0.0 {
        return;
    }
    if let Some(r) = p.rec.as_mut() {
        // GPU: the canvas is glazed there, in order with the strokes; the plain picture stays here
        r.glaze(c, k);
        for i in 0..WI * HI {
            for q in 0..3 {
                p.flat[i][q] += (c[q] - p.flat[i][q]) * k;
            }
            p.wet[i] += (1.0 - p.wet[i]) * k;
        }
        return;
    }
    for i in 0..WI * HI {
        for q in 0..3 {
            p.col[i][q] += (c[q] - p.col[i][q]) * k;
            p.flat[i][q] += (c[q] - p.flat[i][q]) * k;
        }
        p.wet[i] += (1.0 - p.wet[i]) * k;
        p.thick[i] *= 1.0 - k;
    }
}

pub fn paint_body(p: &mut Painting, pts: &[[f32; 2]], col: [f32; 3], alt: [f32; 3], opacity: f32, seed: u32, detail: f32, body: f32, tool: Tool) {
    paint_form(p, pts, col, alt, opacity, seed, detail, body, tool, 0.0)
}

/// `form`: how much the shape is modelled by the scene's light (0 = a flat plane, 1 = a rounded form).
#[allow(clippy::too_many_arguments)]
fn paint_form(p: &mut Painting, pts: &[[f32; 2]], col: [f32; 3], alt: [f32; 3], opacity: f32, seed: u32, detail: f32, body: f32, tool: Tool, form: f32) {
    paint_form_band(p, pts, col, alt, opacity, seed, detail, body, tool, form, 0.0, true)
}
/// as `paint_form`, with the band's two in-stroke knobs: `pickup_add` cranks the wet-into-wet fusion (the KM mix
/// does the merging, no blur), and `contour` says whether the edge is stated at all
fn paint_form_band(
    p: &mut Painting,
    pts: &[[f32; 2]],
    col: [f32; 3],
    alt: [f32; 3],
    opacity: f32,
    seed: u32,
    detail: f32,
    body: f32,
    tool: Tool,
    form: f32,
    pickup_add: f32,
    contour: bool,
) {
    let n = pts.len() as f32;
    let c = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for q in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
        let (dx, dy) = (q[0] - c[0], q[1] - c[1]);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    // STEADINESS: everything about how a shape is painted comes from the shape itself, never from where it happens
    // to be on the canvas or what is under it this frame. Its brush is chosen from its own size (whole, even if
    // part is off the canvas), and in steps, so a shape that swells a little keeps the same brush.
    let small = (hi[0] - lo[0]).min(hi[1] - lo[1]);
    let z = zoom();
    // The brush goes by how broad the shape really is, not its bounding box: a thin twig lying on the slant is
    // painted with a thin brush, not one as wide as its box. (With the box, a 3 px diagonal twig got a 37 px brush
    // spilling 10 px either side, and close twigs, straps and whiskers ran together into one smear.)
    let thick = thickness(pts);
    // how far the shape is a line rather than an area (by degrees: a shape thinning through it changes by degrees)
    let mode = MODE.with(|m| m.get());
    let thin_k = (1.0 - smooth(1.2 * SMALLEST * z, 2.5 * SMALLEST * z, thick)).max(mode.line);
    // The brush follows the shape's size continuously. (It used to snap to steps of 1.22x: a shape that swelled
    // across a step boundary, or a blinking eye, re-planned every stroke in it in one frame, which read as a pop.)
    let width = (thick / z * 0.30 * detail).clamp(5.0, 110.0) * z;
    // How far back the shape sits (see `softness`). A shape that stands out strongly from the paint it is laid on
    // (a moon in the night sky, a light window in a dark wall) is an accent however far away it is, and is painted
    // as crisply and as lively as a subject.
    let far = SOFT.with(|s| s.get());
    let soft = {
        let s = far;
        let r = (thick * 0.3).max(2.0 * z);
        let behind = flat_under(&p.flat, c[0], c[1], r, col);
        s * (1.0 - smooth(0.16, 0.36, contrast(col, behind)) * wet_under(&p.wet, c[0], c[1], r))
    };
    lo = [lo[0].max(-80.0), lo[1].max(-80.0)];
    hi = [hi[0].min(WI as f32 + 80.0), hi[1].min(HI as f32 + 80.0)];
    if hi[0] <= lo[0] || hi[1] <= lo[1] {
        return;
    }
    // strokes follow a long shape's length; a round one takes the painter's diagonal
    let axis = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let stretch = {
        let t = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
        ((sxx + syy + t) / (sxx + syy - t).max(1e-3)).sqrt()
    };
    // a round shape takes the painter's diagonal, a long one its own length; in between, part way (no sudden flip
    // as a shape stretches)
    let base = {
        let diag = -0.6 + 1.2 * (hf(seed, 77) - 0.5);
        let mut to = axis - diag;
        while to > std::f32::consts::FRAC_PI_2 {
            to -= std::f32::consts::PI;
        }
        while to < -std::f32::consts::FRAC_PI_2 {
            to += std::f32::consts::PI;
        }
        // Continuous. (It used to snap to steps of 7.5 degrees; a shape whose axis wobbled across a step boundary
        // turned AND re-laid its whole lattice in one frame: the flicker on wings, eyes and beaks.) A turn of a
        // fraction of a degree per frame is the shape's own motion, which the brushwork should follow.
        diag + to * smooth(1.35, 2.1, stretch)
    };
    // A long thin shape is one pull of the brush along its length, not a row of short dabs: its strokes (and the
    // lattice they hang on) stretch with how long and thin it is.
    // Only a straight one: a curved strap is followed by shorter strokes turning with its edge (one long straight
    // stroke would cut across the curve and leave it half bare). A straight strip's breadth across its axis is
    // its true breadth; a curved one's is much more.
    let straight = {
        let t = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
        let across = 2.0 * ((sxx + syy - t) / 2.0 / n).max(0.0).sqrt();
        smooth(0.45, 0.8, thick / across.max(1e-3))
    };
    let long = 1.0 + thin_k * straight * smooth(2.5, 6.0, stretch) * (stretch / 3.0).min(5.0);
    // The strokes may cross the shape's edge only a little: enough that it is not cut with scissors, never a soft
    // halo of its colour around it (they used to reach a fifth of a brush out, up to 25 px, an airbrushed glow
    // round every big shape), and nothing beyond a thin shape's own breadth.
    // Where the edge is found the strokes stop at it; where it is lost they run on into the neighbouring paint
    // (the margin is judged pixel by pixel against the plain picture, see `shape_mask_with`), further in the
    // distance. (They used to reach a fifth of a brush out everywhere, up to 25 px: an airbrushed glow of its colour
    // round every big shape, however strongly it stood out.)
    let grow = width * (0.22 + 0.12 * soft) * (1.0 - thin_k) + (1.5 - 0.8 * thin_k) * z;
    let (flat, wet) = (&p.flat, &p.wet);
    let (lo_c, hi_c) = (0.05 + 0.06 * soft, 0.22 + 0.1 * soft);
    // (and a light far off in the dark, a moon or a lamp, spreads a little of its light round it: halation)
    let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let lost = |i: usize| {
        let glow = 0.7 * far * smooth(0.2, 0.45, lum(col) - lum(flat[i]));
        ((1.0 - smooth(lo_c, hi_c, contrast(col, flat[i]))) * (0.5 + 0.5 * wet[i])).max(glow)
    };
    let Some((mask, o, sz)) = shape_mask_with(pts, grow.max(1.0), Some(&lost)) else { return };
    // the grid of strokes hangs on the shape's own centre, so it travels with the shape (and a big shape half off
    // the canvas keeps its strokes where they were)
    // A slot's identity is its lattice index (iu, iv) counted from the shape's own centroid, so it never depends
    // on which slots are visited. Only the visiting WINDOW follows the visible part of the shape (a big shape
    // half off the canvas, an iris wedge or a whip streak reaching 4000 px out, has its centroid far away; a
    // window centred on the centroid with the clipped radius missed the visible part entirely).
    let mut spots = plan_slots(pts, c, lo, hi, width, base, seed, long);
    spots.sort_by_key(|(_, id, _)| h32(*id));
    // A light is covered completely, in two layers, the second across the first: luminous paint with the dark
    // showing through a gap reads as a hole, not as paint. (By degrees of how much the shape is a light.)
    if mode.clean > 0.001 {
        let mut more = plan_slots(pts, c, lo, hi, width, base + 0.9, seed ^ 0x5EED_0001, long);
        more.sort_by_key(|(_, id, _)| h32(*id));
        spots.extend(more.into_iter().map(|(q, id, e)| (q, id, e * mode.clean)));
    }
    if trace() {
        eprintln!(
            "  FORM seed={seed:08x} n={} width={width:.3} base={base:.4} c=({:.2},{:.2}) thick={thick:.1} long={long:.2} soft={soft:.2}",
            spots.len(),
            c[0],
            c[1]
        );
        for (q, id, e) in &spots {
            eprintln!("    SPOT {id:08x} ({:.1},{:.1}) e={e:.4}", q[0], q[1]);
        }
    }
    if let Some(r) = p.rec.as_mut() {
        r.use_mask(&mask, o, sz);
    }
    // the colours of the form turning from the light, mixed once for the whole shape
    let shade_tab = scene().filter(|_| form > 0.0).map(|sc| ShadeTable::new(&sc.l, col));
    for (q, id, edge_k) in spots {
        // Brushwork within a plane is deliberate: one direction, drifting slowly. Near where the colour ends it
        // turns to run ALONG that edge, so two colours meet with their strokes side by side, never with one
        // hatched across and chopped off.
        let drift = (noise((q[0] - c[0]) / (420.0 * z), (q[1] - c[1]) / (420.0 * z), seed) - 0.5) * 0.5 + (hf(id, 3) - 0.5) * 0.10;
        // The edge's direction is a blend of the nearest edges' directions (as lines: doubled angles), weighted by
        // how nearly as close each is. Taking only THE nearest edge flipped a stroke's direction outright where two
        // edges were equally close (a corner, or the two sides of a narrow trunk): as the shape or camera moved a
        // fraction of a pixel, a long stroke swung several degrees in one frame.
        let (near, along) = edge_dir(pts, q, width);
        // the edge has no "forwards": take whichever way round is closer to the plane's direction
        let mut turn_to = along - (base + drift);
        while turn_to > std::f32::consts::FRAC_PI_2 {
            turn_to -= std::f32::consts::PI;
        }
        while turn_to < -std::f32::consts::FRAC_PI_2 {
            turn_to += std::f32::consts::PI;
        }
        // A form being modelled is brushed round its edges; a flat plane of scenery is blocked in calmly in one
        // direction and only turns right at its edge. (The turn used to reach 1.6 brush widths in, up to 180 px:
        // every stroke of a big flat shape swung towards some edge and they crossed in a hatch all over it.)
        let calm = 1.0 - form.clamp(0.0, 1.0);
        let pull = (-(near / (width * (1.6 - 1.0 * calm))).powi(2)).exp();
        let swirl = drift + turn_to * pull;
        // Broken colour, the way a painter remixes as they go: value and warmth drift slowly across the area, and
        // each stroke differs a little from its neighbour. Lively on a subject, close in value in the distance (a
        // random jump from every stroke to the next made skies and big shapes a patchwork of dabs).
        let lively = 1.0 - soft;
        let field = |s: u32| noise((q[0] - c[0]) / (2.8 * width), (q[1] - c[1]) / (2.8 * width), seed ^ s) - 0.5;
        // (Gently: a stroke is a long, broad, straight-sided mark, so a stroke-to-stroke jump in temperature reads as
        // a rectangle of another colour. The jumps used to be up to 0.06 in red and blue each way, which turned a
        // dull purple plane into a quilt of brown and blue blocks.) A light source's paint is clean and even.
        let even = 1.0 - 0.75 * mode.clean;
        let k = 1.0 + (0.07 * field(0xB0) + 0.05 * (hf(id, 4) - 0.5) * (0.3 + 0.7 * lively)) * even;
        let warm = (0.8 * field(0xB1) + 0.5 * (hf(id, 8) - 0.5) * (0.25 + 0.75 * lively)) * even;
        // which side of the form this stroke is on
        let lit_col = match scene() {
            Some(sc) => {
                let lt = sc.l.towards(c);
                let across = ((q[0] - c[0]) * -lt[0] + (q[1] - c[1]) * -lt[1]) / (small * 0.5).max(1.0);
                match &shade_tab {
                    Some(tab) => sc.turned(tab, col, across, form),
                    None => col,
                }
            }
            None => col,
        };
        // Wet into wet. Near its edge a stroke works in the wet paint it meets there: where the two colours are
        // close in value the edge is lost in the blend, where they contrast it stays found; in the distance more is
        // blended, so backgrounds fuse. Deep inside a shape the brush carries its own fresh paint: it no longer
        // drags the colour of whatever lies behind the shape through it (an ochre shape on a blue sky came out
        // streaked green), and a thin shape, a rigger's line, takes nothing from its neighbours at all.
        let edge_wet = (-(near / (width * 1.1)).powi(2)).exp() * (1.0 - thin_k);
        let (mut own, mut own_alt) = (shifted(lit_col, k, warm), shifted(alt, 1.0, 0.0));
        if mode.model > 0.0 {
            // a light is lightest and warmest in its middle and turns a little greyer and cooler towards its rim
            // (a moon's limb), softly, with no edge cut in round it
            let rho = ((q[0] - c[0]).hypot(q[1] - c[1]) / (small * 0.5).max(1.0)).min(1.2);
            let limb = shifted(col, 1.0 - 0.07 * rho * rho, -0.6 * rho * rho);
            own = std::array::from_fn(|i| own[i] + (limb[i] - own[i]) * mode.model);
            own_alt = std::array::from_fn(|i| own_alt[i] + (shifted(limb, 1.04, 0.2)[i] - own_alt[i]) * mode.model);
        }
        // what the brush works into is the paint where it lays, at its middle (a stroke laid inwards from just
        // outside a shape used to load up at its start with whatever lay beyond the edge, often the bare ground)
        let picked = flat_under(&p.flat, q[0], q[1], width * 0.3, col);
        let mut lost = 0.0;
        if edge_wet > 0.01 {
            let r = width * 0.5;
            let wet = wet_under(&p.wet, q[0], q[1], r);
            let under = flat_under(&p.flat, q[0], q[1], r, col);
            lost = 1.0 - smooth(0.05 + 0.06 * soft, 0.22 + 0.1 * soft, contrast(col, under));
            let blend = edge_wet * wet * lost * (0.14 + 0.3 * soft) * (1.0 - mode.clean);
            if blend > 0.004 {
                own = pigment_mix(own, under, blend);
                own_alt = pigment_mix(own_alt, under, blend);
            }
        }
        p.stroke(
            &Stroke {
                c: q,
                angle: base + swirl,
                // (a little longer than the paint it lays: its bristles land and lift a way in from its ends)
                len: width * (3.6 + 3.6 * hf(id, 5)) * (1.0 - 0.35 * pull) * long,
                width: width * (0.75 + 0.5 * hf(id, 6)),
                // strokes are arcs from the wrist: mostly bowed the same way, never ruler-straight
                bend: (0.12 + (hf(id, 7) - 0.5) * 0.2) / long,
                col: own,
                alt: own_alt,
                opacity: opacity * (0.86 + 0.12 * hf(id, 9)) * edge_k,
                // (a found edge is cut in cleanly; where it is lost the brush drags the neighbour in; deep inside it
                // carries its own paint. A light's paint is kept clean: the moon used to drag the night sky up through
                // it in dark streaks)
                pickup: ((0.22 + 0.3 * hf(id, 10)) * (0.3 + 0.7 * edge_wet * (0.3 + 0.7 * lost)) * (1.0 - 0.85 * thin_k) * (1.0 - mode.clean) + pickup_add)
                    .min(0.85),
                load: (0.6 + 0.4 * hf(id, 11)) * body,
                // the stroke's bristles belong to its place on its shape, not to how many strokes came before it
                part: Some((h32(id) >> 2, true, true)),
                span: (0.0, 1.0),
                tool,
                picked: Some(picked),
                along: 0.0,
            },
            Some((&mask, o, sz)),
        );
    }
    // the shape now stands in the plain picture, for whatever is painted over it later (a translucent wash as the
    // tint it was mixed to, not as its full colour)
    let premix = PREMIX.with(|p| p.get());
    for y in 0..sz[1] {
        for x in 0..sz[0] {
            if mask[y * sz[0] + x] >= 0.5 {
                let (px, py) = (x as i32 + o[0], y as i32 + o[1]);
                if px >= 0 && py >= 0 && (px as usize) < WI && (py as usize) < HI {
                    let j = py as usize * WI + px as usize;
                    let f = &mut p.flat[j];
                    *f = std::array::from_fn(|k| f[k] + (col[k] - f[k]) * premix);
                    p.wet[j] += (1.0 - p.wet[j]) * premix;
                }
            }
        }
    }
    // the cut-in edge comes in by degrees as a shape grows past the size that has one (by its true breadth: a
    // thin shape has no room for a band cut in along each side)
    let contour_k = smooth(SMALLEST * 3.5 * z, SMALLEST * 4.5 * z, thick);
    let contour_k = contour_k * (1.0 - mode.clean);
    if contour_k > 0.001 && contour {
        let _contour = Fade::set(contour_k);
        // lost and found: a modelled subject keeps much of its cut-in edge everywhere; scenery only where it
        // meets a contrasting value, and less the further back it is
        let keep = 0.6 * smooth(0.5, 1.0, form);
        paint_contour(p, pts, col, alt, (width * 0.45).clamp(SMALLEST * z, 26.0 * z), seed ^ 0xC0, opacity, body, keep, soft);
    }
}

/// How strongly an edge of a shape of colour `col` is stated at `q` (0 lost, 1 found), going by the contrast
/// between the shape and what is just outside it there (the plain picture `reach` px to either side of the edge;
/// one side is the shape itself). `keep` is the least it is stated where the two meet in the same value; the
/// further back (`soft`), the more contrast an edge needs to be found.
fn found_edge(p: &Painting, col: [f32; 3], q: [f32; 2], t: [f32; 2], reach: f32, keep: f32, soft: f32) -> f32 {
    let nrm = [-t[1], t[0]];
    let r = (reach * 0.4).max(1.0);
    let a = flat_under(&p.flat, q[0] + nrm[0] * reach, q[1] + nrm[1] * reach, r, col);
    let b = flat_under(&p.flat, q[0] - nrm[0] * reach, q[1] - nrm[1] * reach, r, col);
    let d = contrast(col, a).max(contrast(col, b));
    keep + (1.0 - keep) * smooth(0.06 + 0.08 * soft, 0.24 + 0.12 * soft, d)
}

/// How the hand moves along a path (see `hand_paths`).
#[derive(Clone, Copy)]
struct Hand {
    /// The least length of path one load of paint lasts (px, seen through the camera); otherwise 15.5 widths.
    reach: f32,
    /// How far the hand wanders either side of the line (a share of the brush's width, at most 14 px of it), and
    /// the least length over which it wanders (px): a slow drift of the arm, not a tremble.
    wander: f32,
    wave: f32,
}
/// Cutting in and outlining a shape: a load per short sweep, the hand drifting about half a brush either side.
const HAND_OUTLINE: Hand = Hand { reach: 0.0, wander: 0.5, wave: 0.0 };
/// A drawn line, with a rigger: long hairs that hold enough paint for one long confident pull, a hand that drifts
/// only slowly and a little. (A 4 px line used to take a fresh load and a fresh taper every 60 px and wobble a px
/// or two every 30 px, so a long line read as a string of beads and close parallel lines kept touching.)
const HAND_LINE: Hand = Hand { reach: 260.0, wander: 0.22, wave: 90.0 };

/// How a hand takes a brush along an outline. Not side by side with a ruler:
/// * the brush lifts at a sharp corner and starts again;
/// * between corners it moves in one smooth curve through the points;
/// * one sweep turns through only so much (about 130 degrees) and lasts only one load of paint, then the
///   painter starts again, overlapping the end of the last stroke;
/// * the hand pulls (left to right, top to bottom) rather than pushes;
/// * it runs a little past where it meant to stop, and wanders a little either side of the line.
///
/// Returns each stroke's path (closely spaced points), the share of it to lay (1 = all), and its stable identity.
fn hand_paths(pts: &[[f32; 2]], closed: bool, width: f32, offset: f32, seed: u32, hand: Hand) -> Vec<(Vec<[f32; 2]>, f32, u32)> {
    let n = pts.len();
    if n < 2 {
        return vec![];
    }
    let at = |i: i64| -> [f32; 2] { if closed { pts[i.rem_euclid(n as i64) as usize] } else { pts[i.clamp(0, n as i64 - 1) as usize] } };
    let turn_at = |i: i64| -> f32 {
        let (a, b, c) = (at(i - 1), at(i), at(i + 1));
        let (u, v) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
        (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1])
    };
    // pieces between sharp corners
    let corners: Vec<usize> = (0..n).filter(|&i| (closed || (i > 0 && i + 1 < n)) && turn_at(i as i64).abs() > 0.95).collect();
    let mut pieces: Vec<(i64, i64)> = vec![];
    if closed {
        if corners.is_empty() {
            let start = (hf(seed, 1) * n as f32) as i64;
            pieces.push((start, start + n as i64));
        } else {
            for (k, &c) in corners.iter().enumerate() {
                let next = corners[(k + 1) % corners.len()] as i64;
                pieces.push((c as i64, if next > c as i64 { next } else { next + n as i64 }));
            }
        }
    } else {
        let mut from = 0i64;
        for &c in &corners {
            pieces.push((from, c as i64));
            from = c as i64;
        }
        pieces.push((from, n as i64 - 1));
    }
    let side = if closed { (0..n).map(|i| pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]).sum::<f32>().signum() } else { 0.0 };
    let step = (width * 0.3).clamp(1.5, 6.0 * zoom().max(1.0));
    let mut out = vec![];
    for (pi, (a, b)) in pieces.into_iter().enumerate() {
        if b <= a {
            continue;
        }
        // one smooth curve through the piece's points (Catmull-Rom), sampled closely
        let mut curve: Vec<[f32; 2]> = vec![];
        for i in a..b {
            let (p0, p1, p2, p3) = (
                at(if i == a && !corners.is_empty() { i } else { i - 1 }),
                at(i),
                at(i + 1),
                at(if i + 1 == b && !corners.is_empty() { i + 1 } else { i + 2 }),
            );
            let l = (p2[0] - p1[0]).hypot(p2[1] - p1[1]);
            let k = (l / step).ceil().max(1.0) as usize;
            for j in 0..k {
                let (u, u2, u3) = (j as f32 / k as f32, (j as f32 / k as f32).powi(2), (j as f32 / k as f32).powi(3));
                curve.push(std::array::from_fn(|d| {
                    0.5 * (2.0 * p1[d]
                        + (p2[d] - p0[d]) * u
                        + (2.0 * p0[d] - 5.0 * p1[d] + 4.0 * p2[d] - p3[d]) * u2
                        + (3.0 * p1[d] - p0[d] - 3.0 * p2[d] + p3[d]) * u3)
                }));
            }
        }
        curve.push(at(b));
        // move it inside the shape (cutting in) and let the hand wander
        let m = curve.len();
        let mut walked = 0.0f32;
        let moved: Vec<[f32; 2]> = (0..m)
            .map(|i| {
                let (p, q) = (curve[i.saturating_sub(1)], curve[(i + 1).min(m - 1)]);
                let l = (q[0] - p[0]).hypot(q[1] - p[1]).max(1e-3);
                if i > 0 {
                    walked += (curve[i][0] - curve[i - 1][0]).hypot(curve[i][1] - curve[i - 1][1]);
                }
                let wave = (width * 8.0).max(hand.wave * zoom());
                let wander = (noise(walked / wave, pi as f32 * 9.1, seed) - 0.5) * hand.wander * width.min(14.0 * zoom());
                let nrm = [-(q[1] - p[1]) / l, (q[0] - p[0]) / l];
                let off = offset * side + wander;
                [curve[i][0] + nrm[0] * off, curve[i][1] + nrm[1] * off]
            })
            .collect();
        // Cut into sweeps: so much turning, so much paint. Each point along the piece has a cost (its length in
        // reaches plus its turning in 130-degree turns), and the sweeps end where the running cost passes a
        // fixed series of marks drawn from the seed. The cuts therefore slide with the shape as it flexes, and a
        // sweep is only ever added or lost at the END of a piece, starting from nothing (it is faded in by how far
        // the piece reaches past its mark). (The sweeps used to be cut greedily, "until this much length or this
        // much turn": a shape that flexed a little re-cut every sweep after the change, and since each sweep was
        // seeded by its place in the list, re-seeded them all; the outline of a breathing owl flickered.)
        let unit = (15.5 * width).max(hand.reach * zoom());
        let mut cost = vec![0.0f32; m];
        let mut arc = vec![0.0f32; m];
        for q in 1..m {
            let d = (moved[q][0] - moved[q - 1][0]).hypot(moved[q][1] - moved[q - 1][1]);
            arc[q] = arc[q - 1] + d;
            let turn = if q >= 2 {
                let (e, f) =
                    ([curve[q - 1][0] - curve[q - 2][0], curve[q - 1][1] - curve[q - 2][1]], [curve[q][0] - curve[q - 1][0], curve[q][1] - curve[q - 1][1]]);
                (e[0] * f[1] - e[1] * f[0]).atan2(e[0] * f[0] + e[1] * f[1]).abs()
            } else {
                0.0
            };
            cost[q] = cost[q - 1] + d / unit + turn / 2.3;
        }
        let total = cost[m - 1];
        // the point at running cost c, and the point `back` pixels of path before it
        let at_cost = |c: f32| -> (usize, [f32; 2], f32) {
            let q = cost.partition_point(|&v| v < c).clamp(1, m - 1);
            let f = ((c - cost[q - 1]) / (cost[q] - cost[q - 1]).max(1e-9)).clamp(0.0, 1.0);
            (q, std::array::from_fn(|k| moved[q - 1][k] + (moved[q][k] - moved[q - 1][k]) * f), arc[q - 1] + (arc[q] - arc[q - 1]) * f)
        };
        let at_arc = |a: f32| -> (usize, [f32; 2]) {
            let q = arc.partition_point(|&v| v < a).clamp(1, m - 1);
            let f = ((a - arc[q - 1]) / (arc[q] - arc[q - 1]).max(1e-9)).clamp(0.0, 1.0);
            (q, std::array::from_fn(|k| moved[q - 1][k] + (moved[q][k] - moved[q - 1][k]) * f))
        };
        let piece_seed = seed ^ (pi as u32).wrapping_mul(131);
        let (mut c0, mut sweep) = (0.0f32, 0u32);
        while c0 < total && sweep < 4096 {
            let c1 = c0 + 0.71 + 0.58 * hf(piece_seed, sweep + 5);
            // a sweep that has only just come into being (the piece reaching past its mark by a hair) is laid by degrees
            let born = if sweep == 0 { 1.0 } else { smooth(0.0, 0.15, total - c0) };
            let (_, _, a0) = at_cost(c0);
            // the next sweep starts a little back, over the end of the last one
            let (qs, ps) = at_arc((a0 - if sweep == 0 { 0.0 } else { width * 0.9 }).max(0.0));
            let (qe, pe, _) = at_cost(c1.min(total));
            let mut path: Vec<[f32; 2]> = vec![ps];
            path.extend_from_slice(&moved[qs..qe]);
            path.push(pe);
            path.dedup_by(|a, b| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-3);
            if path.len() >= 2 && born > 0.001 {
                // run on a little past the end, as a hand does
                let (e, f) = (path[path.len() - 2], path[path.len() - 1]);
                let l = (f[0] - e[0]).hypot(f[1] - e[1]).max(1e-3);
                let over = width * (0.25 + 0.5 * hf(seed, sweep * 7 + pi as u32));
                path.push([f[0] + (f[0] - e[0]) / l * over, f[1] + (f[1] - e[1]) / l * over]);
                // Pull, don't push. Which way a sweep runs is one of two, so near the line between them both are
                // laid, each by its share, instead of the whole sweep turning round in one frame.
                let (s0, s1) = (path[0], path[path.len() - 1]);
                let chord = (s1[0] - s0[0]).hypot(s1[1] - s0[1]).max(1e-3);
                let fwd = smooth(-0.2, 0.2, ((s1[0] - s0[0]) * 0.75 + (s1[1] - s0[1])) / (1.25 * chord));
                // a sweep's identity is its piece and its place on that piece, never its place in the list
                let ident = h32(piece_seed ^ sweep.wrapping_mul(0x9E37_79B9));
                if fwd > 0.001 {
                    out.push((path.clone(), born * fwd, ident));
                }
                if fwd < 0.999 {
                    let mut back = path;
                    back.reverse();
                    out.push((back, born * (1.0 - fwd), ident ^ 0x5bd1_e995));
                }
            }
            c0 = c1;
            sweep += 1;
        }
    }
    out
}

/// How strongly a sweep lays each piece of its path: given the canvas, the piece's middle and its direction.
type SweepWeight<'a> = dyn Fn(&Painting, [f32; 2], [f32; 2]) -> f32 + 'a;

/// One sweep of the brush along a path: the same bristles all the way, pressure and paint by how far along it is.
#[allow(clippy::too_many_arguments)]
fn sweep(
    p: &mut Painting,
    tool: Tool,
    path: &[[f32; 2]],
    width: f32,
    col: [f32; 3],
    alt: [f32; 3],
    opacity: f32,
    pickup: f32,
    load: f32,
    seed: u32,
    weight: &SweepWeight<'_>,
) {
    let total: f32 = path.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
    if total < 1.0 {
        return;
    }
    let ident = 0x4000_0000 | (h32(seed) >> 2);
    if trace() {
        eprintln!(
            "  SWEEP {ident:08x} n={} len={total:.1} from=({:.1},{:.1}) to=({:.1},{:.1}) w={width:.2}",
            path.len(),
            path[0][0],
            path[0][1],
            path[path.len() - 1][0],
            path[path.len() - 1][1]
        );
    }
    let picked = flat_under(&p.flat, path[0][0], path[0][1], width * 0.3, col);
    let mut walked = 0.0;
    for (i, w) in path.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        let l = (b[0] - a[0]).hypot(b[1] - a[1]);
        if l < 0.05 {
            continue;
        }
        let state = weight(p, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0], [(b[0] - a[0]) / l, (b[1] - a[1]) / l]);
        if state > 0.04 {
            p.stroke(
                &Stroke {
                    c: [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0],
                    angle: (b[1] - a[1]).atan2(b[0] - a[0]),
                    len: l + width * 0.12,
                    width: width * (0.6 + 0.4 * state),
                    bend: 0.0,
                    col,
                    alt,
                    opacity: opacity * state,
                    pickup,
                    load,
                    part: Some((ident, i == 0, i + 2 == path.len())),
                    span: (walked / total, (walked + l) / total),
                    tool,
                    picked: Some(picked),
                    along: walked + l / 2.0,
                },
                None,
            );
        }
        walked += l;
    }
}

/// Cutting in: where a colour ends, the brush is drawn along the edge, just inside it.
#[allow(clippy::too_many_arguments)]
fn paint_contour(p: &mut Painting, pts: &[[f32; 2]], col: [f32; 3], alt: [f32; 3], width: f32, seed: u32, opacity: f32, body: f32, keep: f32, soft: f32) {
    for (path, share, ident) in hand_paths(pts, true, width, width * 0.42, seed, HAND_OUTLINE) {
        let id = seed ^ ident;
        sweep(
            p,
            Tool::Flat,
            &path,
            width,
            shifted(col, 0.95 + 0.10 * hf(id, 2), (hf(id, 3) - 0.5) * 0.8),
            alt,
            opacity * 0.88 * share,
            0.34,
            0.85 * body,
            id,
            &|p, q, t| found_edge(p, col, q, t, width * 1.5, keep, soft),
        );
    }
}

/// How much of a glaze a soft brush leaves at a point (`u` along its strokes, `v` across): never quite even, a
/// little more where the strokes overlapped and less where the brush ran thin. (The GPU has the same, strokes.wgsl.)
fn glaze_grain(u: f32, v: f32, z: f32, id: u32) -> f32 {
    0.74 + 0.16 * noise(u / (90.0 * z), v / (28.0 * z), id ^ 5) + 0.1 * noise(u / (16.0 * z), v / (3.4 * z), id)
}

/// The long axis of a set of points, as an angle (for the grain of a glaze).
fn axis_of(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len().max(1) as f32;
    let c = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for q in pts {
        let (dx, dy) = (q[0] - c[0], q[1] - c[1]);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    0.5 * (2.0 * sxy).atan2(sxx - syy)
}

/// A cast shadow, the way a painter indicates one: not a patch of dark paint, but a glaze over what it falls on.
/// `pts` is where it falls; `caster` (if known) what casts it. Its colour is derived from what it falls on and the
/// scene's light: at every point the paint it falls on is taken down by its own shadow colour (`shade_paint`: a touch
/// of the palette's shadow colour worked in, a value step as deep as the light is strong, `depth` 1 for a full
/// shadow), so each colour under it stays itself, darker and cooler. (One glaze colour averaged over everything
/// under it turned a cream belly blue where the shadow also covered the sky.) Its edge is crisp where it leaves the caster and softens with distance
/// from it (the penumbra); `soft` (0 to 1) softens it all over (a form shadow's edges are lost).
fn paint_cast_shadow(p: &mut Painting, pts: &[[f32; 2]], caster: Option<&[[f32; 2]]>, depth: f32, soft: f32, id: u32) {
    let fade = FADE.with(|f| f.get()) * PREMIX.with(|p| p.get());
    if fade <= 0.002 || pts.len() < 3 {
        return;
    }
    let light = frame_light();
    let z = zoom();
    let thick = thickness(pts);
    // the widest the penumbra gets
    // (it widens with how far the shadow reaches from what casts it, as a real penumbra does)
    let ext = {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for q in pts {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        (hi[0] - lo[0]).max(hi[1] - lo[1])
    };
    // (but never so wide that a narrow shadow dissolves: a painter keeps it readable)
    let pen = (((thick * 0.2 + ext * 0.05).min(90.0 * z) + 3.0 * z) * (0.4 + 0.6 * soft.max(0.6))).min(thick * 0.3 + 2.0 * z);
    let Some((mut m, o, sz)) = blurred_cover(pts, pen) else { return };
    let reach = caster.map_or(1.0, |c| (thickness(c) * 1.2).max(30.0 * z));
    // how far each point is from what casts it, measured on a coarse grid and interpolated (it changes slowly)
    const G: usize = 4;
    let (gw, gh) = (sz[0] / G + 2, sz[1] / G + 2);
    let far: Vec<f32> = match caster {
        Some(c) => (0..gw * gh)
            .map(|i| {
                let q = [((i % gw) * G) as f32 + o[0] as f32 + 0.5, ((i / gw) * G) as f32 + o[1] as f32 + 0.5];
                (-signed_depth(c, q, 0.0)).max(0.0)
            })
            .collect(),
        None => vec![],
    };
    for my in 0..sz[1] {
        for mx in 0..sz[0] {
            let v = &mut m[my * sz[0] + mx];
            if *v <= 0.0 {
                continue;
            }
            // how soft the edge is here: crisp next to what casts it, softer the further it falls from it
            let s = match caster {
                Some(_) if *v < 0.98 => {
                    let (gx, gy) = (mx / G, my / G);
                    let (fx, fy) = ((mx % G) as f32 / G as f32, (my % G) as f32 / G as f32);
                    let at = |x: usize, y: usize| far[y * gw + x];
                    let d = (at(gx, gy) * (1.0 - fx) + at(gx + 1, gy) * fx) * (1.0 - fy) + (at(gx, gy + 1) * (1.0 - fx) + at(gx + 1, gy + 1) * fx) * fy;
                    (soft + (1.0 - soft) * smooth(0.0, reach, d)).clamp(0.0, 1.0)
                }
                Some(_) => soft,
                None => soft.max(0.5),
            };
            let w = 0.03 + 0.45 * s;
            *v = smooth(0.5 - w, 0.5 + w, *v);
        }
    }
    for v in &mut m {
        *v *= fade;
    }
    let glaze = light.glaze_of(depth);
    let (sn, cs) = (axis_of(pts) + std::f32::consts::FRAC_PI_2).sin_cos();
    let n = pts.len() as f32;
    let centre = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
    p.shade_area(&m, o, sz, glaze, centre, [sn, cs], h32(id ^ 0x51AD));
    // the plain picture: the shadow now lies on it, for whatever is painted over it later
    for my in 0..sz[1] {
        for mx in 0..sz[0] {
            let (x, y) = (mx as i32 + o[0], my as i32 + o[1]);
            let g = m[my * sz[0] + mx];
            if g > 0.0 && x >= 0 && y >= 0 && x < WI as i32 && y < HI as i32 {
                let f = &mut p.flat[y as usize * WI + x as usize];
                let sh = shade_paint(*f, glaze.0, glaze.1, glaze.2);
                for k in 0..3 {
                    f[k] += (sh[k] - f[k]) * g;
                }
            }
        }
    }
}

/// A transparent wash of colour `col` at strength `k` over what is painted (a tint, a stain, a translucent shade):
/// a glaze, the paint below seen through it, its edges soft.
fn paint_glaze(p: &mut Painting, pts: &[[f32; 2]], col: [f32; 3], k: f32, id: u32) {
    let fade = FADE.with(|f| f.get());
    let k = (k * fade).clamp(0.0, 1.0);
    if k <= 0.002 || pts.len() < 3 {
        return;
    }
    let z = zoom();
    let thick = thickness(pts);
    let Some((mut m, o, sz)) = blurred_cover(pts, (thick * 0.08).min(8.0 * z) + 1.5 * z) else { return };
    for v in &mut m {
        *v = smooth(0.3, 0.6, *v);
    }
    // a thin layer of a colour lets through what its pigment does not absorb
    let tint: [f32; 3] = std::array::from_fn(|c| 1.0 - k * (1.0 - col[c]));
    let (sn, cs) = (axis_of(pts)).sin_cos();
    let n = pts.len() as f32;
    let centre = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
    p.glaze_area(&m, o, sz, tint, centre, [sn, cs], h32(id ^ 0x61A2));
    for my in 0..sz[1] {
        for mx in 0..sz[0] {
            let (x, y) = (mx as i32 + o[0], my as i32 + o[1]);
            let g = m[my * sz[0] + mx];
            if g > 0.0 && x >= 0 && y >= 0 && x < WI as i32 && y < HI as i32 {
                let f = &mut p.flat[y as usize * WI + x as usize];
                for c in 0..3 {
                    f[c] *= 1.0 - g * (1.0 - tint[c]);
                }
            }
        }
    }
}

/// What casts a drawn shadow: the film's named caster (the biggest shape of that family), or, if it names none, the
/// cast's shape nearest to it; none if there is neither.
fn caster_of(items: &[Item], shadow: &Item, cam: Option<Cam>) -> Option<Vec<[f32; 2]>> {
    let centre = |pts: &[[f32; 2]]| {
        let n = pts.len().max(1) as f32;
        [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n]
    };
    let (sp, _) = screen(shadow, cam);
    let sc = centre(&sp);
    let shapes = items.iter().filter(|it| matches!(it.kind, Kind::Shape { wash: Some(_), .. }) && !matches!(it.role, Role::Shadow | Role::FormShadow));
    match shadow.caster {
        Some(f) => shapes
            .filter(|it| it.family.starts_with(f))
            .map(|it| screen(it, cam).0)
            .max_by(|a, b| thickness(a).total_cmp(&thickness(b)).then(a.len().cmp(&b.len()))),
        None => shapes.filter(|it| is_cast(&it.family)).map(|it| screen(it, cam).0).min_by(|a, b| {
            let (ca, cb) = (centre(a), centre(b));
            ((ca[0] - sc[0]).hypot(ca[1] - sc[1])).total_cmp(&(cb[0] - sc[0]).hypot(cb[1] - sc[1]))
        }),
    }
}

/// A glow round a light (`Rec::glows`), as a painter makes one: light paint scumbled thinly over the dark round
/// it, in short strokes that follow the circle, more of it near the light and less and less further out, so the
/// dark shows through broken at the rim. (The film draws its glows as additive light; the painter ignored them,
/// so a moon had no light round it at all.) Rings and strokes are fixed in number and scale with the glow, so a
/// glow that grows or shrinks keeps its strokes; each stroke is its paint let down with what it lands on.
fn paint_glow(p: &mut Painting, centre: [f32; 2], radius: f32, col: [f32; 3], a: f32, id: u32) {
    let z = zoom();
    let fade = FADE.with(|f| f.get()) * smooth(6.0 * z, 14.0 * z, radius);
    if fade <= 0.002 || a <= 0.002 {
        return;
    }
    let r = radius.max(1.0);
    let o = [(centre[0] - r).floor() as i32 - 1, (centre[1] - r).floor() as i32 - 1];
    let e = [(centre[0] + r).ceil() as i32 + 1, (centre[1] + r).ceil() as i32 + 1];
    let (o, e) = ([o[0].max(0), o[1].max(0)], [e[0].min(WI as i32), e[1].min(HI as i32)]);
    if e[0] <= o[0] || e[1] <= o[1] {
        return;
    }
    let sz = [(e[0] - o[0]) as usize, (e[1] - o[1]) as usize];
    let mut m = vec![0.0f32; sz[0] * sz[1]];
    for y in 0..sz[1] {
        for x in 0..sz[0] {
            let t = ((x as i32 + o[0]) as f32 + 0.5 - centre[0]).hypot((y as i32 + o[1]) as f32 + 0.5 - centre[1]) / r;
            if t < 1.0 {
                // the light falls off quickly away from its source
                m[y * sz[0] + x] = (a * (1.0 - t).powf(1.8) * 1.15).min(0.92) * fade;
            }
        }
    }
    p.scumble_area(&m, o, sz, col, centre, [0.0, 1.0], h32(id ^ 0x6170));
}

/// A drawn line or an outline. `lit`: for a closed outline, how strongly to paint the sides that face the light
/// (the far sides are always painted). `against`: for the outline of a shape, the shape's colour and how much of
/// the outline to keep where the shape meets its surroundings in the same value (lost and found; see `found_edge`).
fn paint_line(p: &mut Painting, pts: &[[f32; 2]], closed: bool, col: [f32; 3], width: f32, seed: u32, lit: f32, against: Option<([f32; 3], f32)>) {
    paint_line_with(p, pts, closed, col, width, seed, lit, 0.55, 0.93, against)
}

/// `load` 0 = ink (no body), `opacity` below 1 lets what is under it show.
#[allow(clippy::too_many_arguments)]
fn paint_line_with(
    p: &mut Painting,
    pts: &[[f32; 2]],
    closed: bool,
    col: [f32; 3],
    width: f32,
    seed: u32,
    lit: f32,
    load: f32,
    opacity: f32,
    against: Option<([f32; 3], f32)>,
) {
    let width = width.max(2.5);
    let side = if closed {
        let n = pts.len();
        (0..n).map(|i| pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]).sum::<f32>().signum()
    } else {
        0.0
    };
    // A fine line is one colour, loaded: a few hairs carrying a second, lighter paint broke a 3 px line into
    // stripes that read as grey. Broader lines keep their streaks.
    let fine = smooth(SMALLEST * zoom(), 2.5 * SMALLEST * zoom(), width);
    let streak = [0, 1, 2].map(|k| shifted(col, 1.15, 0.3)[k] + (shifted(col, 1.4, 0.6)[k] - shifted(col, 1.15, 0.3)[k]) * fine);
    // an open line is drawn with a rigger, in long pulls, and takes almost nothing from what it crosses
    let (hand, pickup) = if closed { (HAND_OUTLINE, 0.16 * load / 0.55) } else { (HAND_LINE, 0.05 * load / 0.55) };
    for (path, share, ident) in hand_paths(pts, closed, width, 0.0, seed, hand) {
        let id = seed ^ ident.rotate_left(7);
        let path = &path;
        // the sweep may have been turned round to be pulled: which side is out depends on its direction
        let flip = if closed && path.len() > 2 {
            let (a, b) = (path[0], path[1]);
            let n = pts.len() as f32;
            let c = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
            let left = [-(b[1] - a[1]), b[0] - a[0]];
            if left[0] * (c[0] - a[0]) + left[1] * (c[1] - a[1]) > 0.0 { -1.0 } else { 1.0 }
        } else {
            side
        };
        // lost and found: an edge is stated on the side away from the light and only hinted on the side towards
        // it, and only where the shape stands out from what it meets
        let weight = |p: &Painting, q: [f32; 2], t: [f32; 2]| -> f32 {
            if !closed {
                return 1.0;
            }
            let out = [-t[1] * flip, t[0] * flip];
            let w = lit + (1.0 - lit) * smooth(0.3, 0.6, (out[0] * 0.64 + out[1] * 0.77) * 0.5 + 0.5);
            match against {
                Some((shape, keep)) => w * found_edge(p, shape, q, t, width * 1.5 + 3.0 * zoom(), keep, SOFT.with(|s| s.get())),
                None => w,
            }
        };
        sweep(p, Tool::Round, path, width, shifted(col, 0.95 + 0.2 * hf(id, 4), 0.2), streak, opacity * share, pickup, load, id, &weight);
    }
}

/// `oil <t> <out.png> [seed] [flags]`: one frame. Flags: "plain", "nolight", "nopigment", "mixed", "water".
pub fn run(film: &Film, style: Style, args: &[String]) -> Result<(), String> {
    set_style(style);
    backend_from_env();
    let t: f64 = args.first().and_then(|v| v.parse().ok()).ok_or("oil <t> <out.png> [seed]")?;
    let out = args.get(1).ok_or("oil <t> <out.png> [seed]")?;
    let seed: u32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let flags = args.get(3).map(String::as_str).unwrap_or("");
    // all on unless asked otherwise: "plain" (neither), "nolight", "nopigment"
    LIGHT.store(!(flags.contains("plain") || flags.contains("nolight")), Ordering::Relaxed);
    PIGMENT.store(!(flags.contains("plain") || flags.contains("nopigment")), Ordering::Relaxed);
    MIXED.store(flags.contains("mixed"), Ordering::Relaxed);
    WATER.store(flags.contains("water"), Ordering::Relaxed);
    image::RgbImage::from_raw(WI as u32, HI as u32, paint_frame(film, t, seed)).ok_or("image")?.save(out).map_err(|e| e.to_string())
}

/// `oilclip <t0> <t1> <dir> [N]`: frames f00000.png… of a stretch of the film. Every shape's strokes are
/// seeded by what the shape IS (its name in the film), not by when it is drawn, so its brushwork travels with
/// it from frame to frame. **N = 0 (constant seed) is the house look**: stable painted
/// surfaces, no per-second repaint jumps (the owner rejected the jumps; verified 2026-10-18 — calm scenes
/// hold to ≈1.5 8-bit levels frame-to-frame). N > 0 reseeds the whole picture every N frames, which reads
/// as a once-per-N-frames repaint jump — do not use it for this film.
pub fn run_clip(film: &Film, style: Style, args: &[String]) -> Result<(), String> {
    set_style(style);
    backend_from_env();
    use rayon::prelude::*;
    let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
    let (t0, t1) = (num(0).ok_or("oilclip <t0> <t1> <dir> [N]")?, num(1).ok_or("oilclip <t0> <t1> <dir> [N]")?);
    let dir = args.get(2).ok_or("oilclip <t0> <t1> <dir> [N]")?;
    let every = num(3).unwrap_or(0.0) as u32;
    MIXED.store(args.get(4).is_some_and(|f| f.contains("mixed")), Ordering::Relaxed);
    WATER.store(args.get(4).is_some_and(|f| f.contains("water")), Ordering::Relaxed);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let n = ((t1 - t0) * 24.0).round() as u32;
    let t_start = std::time::Instant::now();
    (0..n).into_par_iter().try_for_each(|i| {
        let abs = ((t0 * 24.0).round() as u32) + i;
        let px = paint_frame(film, abs as f64 / 24.0, clip_seed(abs, every));
        let t0 = std::time::Instant::now();
        let r = image::RgbImage::from_raw(WI as u32, HI as u32, px).ok_or("image")?.save(format!("{dir}/f{i:05}.png")).map_err(|e| e.to_string());
        if timing() {
            eprintln!("TIMING png {:.3} s", t0.elapsed().as_secs_f64());
        }
        r
    })?;
    eprintln!("{n} frames in {:.1} s", t_start.elapsed().as_secs_f64());
    Ok(())
}

/// The seed a clip frame is painted with: one for the whole clip (N = 0, the house look: brushwork holds and only
/// moves with what it paints), or a fresh painting every N frames (a deliberate boil, at 24/N repaints a second).
fn clip_seed(abs_frame: u32, every: u32) -> u32 {
    abs_frame.checked_div(every).map_or(1, |q| 1 + q)
}

/// One frame, painted and lit, as 8-bit RGB (WIDTH x HEIGHT).
pub fn paint_frame(film: &Film, t: f64, seed: u32) -> Vec<u8> {
    // (no clock unless asked for: there is none on wasm32-unknown-unknown)
    let t0 = timing().then(std::time::Instant::now);
    let p = paint_frame_painting(film, t, seed);
    let t1 = timing().then(std::time::Instant::now);
    let px = p.lit();
    if let (Some(t0), Some(t1)) = (t0, t1) {
        eprintln!("TIMING frame {:.3} s painted (planning + strokes), {:.3} s lit", (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
    }
    px
}

/// One frame's paint colour before the relief is lit (linear 0-1 RGB, WIDTH x HEIGHT): for tests of steadiness.
pub fn paint_frame_colour(film: &Film, t: f64, seed: u32) -> Vec<[f32; 3]> {
    paint_frame_painting(film, t, seed).col
}

/// The canvas size every frame is painted at.
pub const WIDTH: usize = WI;
pub const HEIGHT: usize = HI;

/// One frame of the film, painted (before the relief is lit).
fn paint_frame_painting(film: &Film, t: f64, seed: u32) -> Painting {
    // (in a browser the GPU can only be waited for with `await`: there the GPU paints through `paint_frame_async`)
    let mut p = plan_frame(film, t, seed, backend() == Backend::Gpu && cfg!(not(target_arch = "wasm32")));
    finish(&mut p);
    p
}

/// Open the GPU (or a new one, if the last was lost), for [`set_backend`]`(Backend::Gpu)`: what the adapter calls
/// itself (a browser often keeps that private, so it may be empty), or why there is no usable GPU. Natively the
/// first GPU frame opens it by itself; in a browser, await this first (and paint with [`paint_frame_async`]).
#[cfg(feature = "gpu")]
pub async fn open_gpu() -> Result<String, String> {
    gpu::open().await
}

/// [`paint_frame`] that waits for the GPU with `await` instead of blocking, as a browser requires: on the GPU
/// backend (opened with [`open_gpu`]) the strokes are planned here and laid by the GPU; on the CPU backend this is
/// `paint_frame`. An error if the GPU is lost or rejects the work (paint the frame on the CPU instead).
#[cfg(feature = "gpu")]
pub async fn paint_frame_async(film: &Film, t: f64, seed: u32) -> Result<Vec<u8>, String> {
    let mut p = plan_frame(film, t, seed, backend() == Backend::Gpu);
    if let Some(r) = p.rec.take() {
        let (col, thick) = gpu::render_async(&r).await?;
        p.col = col;
        p.thick = thick;
    }
    Ok(p.lit())
}

/// Plan and paint a frame's strokes; with `record`, the strokes are written down for the GPU instead of laid
/// (`finish` lays them).
fn plan_frame(film: &Film, t: f64, seed: u32, record: bool) -> Painting {
    DEPTH.store(std::env::var("OIL_DEPTH").is_ok(), Ordering::Relaxed);
    let rec = film.record(film.shot_at(t), t);
    // test hook: OIL_ZOOM multiplies the camera's zoom (to check that brushwork scales with it and is not re-planned)
    let cam = rec.cam.map(|mut c| {
        c.zoom *= std::env::var("OIL_ZOOM").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
        c
    });
    let mut p = Painting::new();
    // GPU: everything is planned here as usual, and the strokes are laid on the GPU at the end. (Watercolour,
    // mixed media and bite holes paint the canvas directly here, so those frames stay on the CPU.)
    if record && !WATER.load(Ordering::Relaxed) && !MIXED.load(Ordering::Relaxed) && style().leaf_tags.is_empty() {
        p.rec = Some(Box::new(gpu::Recorder::new(p.col.clone())));
    }
    p.view = cam.map_or(1.0, |c| c.zoom as f32).clamp(0.25, 6.0);
    // the light the frame is seen in: the shot's own, else the film's, else a plain light from the upper left (a
    // light's place, drawn in the camera's world, is seen through the camera like any shape)
    let light = {
        let world = |p: [f64; 2]| cam.map_or(p, |c| c.to_screen(p[0], p[1]));
        let (l, at) = match rec.lighting {
            Some((l, on_screen)) => (l, l.key_at.map(|p| if on_screen { p } else { world(p) })),
            None => {
                let l = style().lighting.unwrap_or(PLAIN_LIGHT);
                (l, l.key_at.map(world))
            }
        };
        Light::from(&l, at.map(|p| [p[0] as f32, p[1] as f32]), style().palette)
    };
    FRAME_LIGHT.with(|f| f.set(Some(light)));
    // what is being looked at: the film's subjects (its outlined, smaller shapes), as one group
    if LIGHT.load(Ordering::Relaxed) {
        let (mut sum, mut wsum) = ([0.0f32; 2], 0.0f32);
        for it in &rec.items {
            if let (Kind::Shape { wash: Some(_), .. }, true) = (&it.kind, is_cast(&it.family)) {
                let (pts, _) = screen(it, cam);
                let n = pts.len() as f32;
                let m = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
                let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                for q in &pts {
                    for k in 0..2 {
                        lo[k] = lo[k].min(q[k]);
                        hi[k] = hi[k].max(q[k]);
                    }
                }
                let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]);
                if ext < 900.0 && m[0] > 0.0 && m[1] > 0.0 && m[0] < WI as f32 && m[1] < HI as f32 {
                    let w = (hi[0] - lo[0]) * (hi[1] - lo[1]);
                    sum = [sum[0] + m[0] * w, sum[1] + m[1] * w];
                    wsum += w;
                }
            }
        }
        let focus = if wsum > 0.0 { [sum[0] / wsum, sum[1] / wsum] } else { [WI as f32 / 2.0, HI as f32 / 2.0] };
        SCENE.with(|s| s.set(Some(Scene { l: light, focus, reach: 0.34 * WI as f32 * p.view })));
    }
    // the canvas as it stood before the last leaf was painted, and that leaf's colour (see the bite holes below)
    let bare = p.flat.clone();
    let mut under_leaf: Option<UnderLeaf> = None;
    let (mut holes_end, mut holes_painted) = (0usize, false);
    // DEPTH v2: the atmospheric veil is this frame's own early sky (what everything far sits inside)
    if DEPTH.load(Ordering::Relaxed) {
        let (mut sum, mut n) = ([0.0f32; 3], 0usize);
        for it in rec.items.iter().take(12) {
            if let Kind::Shape { wash: Some((c, _)), .. } = it.kind {
                let q = rgb01(c);
                sum = [sum[0] + q[0], sum[1] + q[1], sum[2] + q[2]];
                n += 1;
            }
        }
        if n > 0 {
            VEIL.with(|v| v.set([sum[0] / n as f32, sum[1] / n as f32, sum[2] / n as f32]));
        }
    }
    let total = rec.items.len().max(1) as f32;
    let progress = PROGRESS.with(|p| p.get());
    let n_items = rec.items.len().max(1) as f32;
    let frame_items: &[Item] = &rec.items;
    for (i, it) in rec.items.iter().enumerate() {
        if let Some(f) = progress {
            f(i as f32 / n_items);
        }
        let (pts, zoom) = screen(it, cam);
        ZOOM.with(|z| z.set(zoom));
        let _ = total;
        // the asset declares its depth; the style map does the rest
        BAND.with(|b| b.set(band_of(&it.family)));
        SOFT.with(|s| s.set(softness(&it.family)));
        // A fade or a flash (fx::flash, world::fade, the whip-pan's cover) is the film's own transition, not a
        // thing in the picture: a sheet of colour over the whole frame at the film's opacity. It is glazed over
        // the painting as it stands, exactly as strong as the film asks, so a fade really fades. (Painted as
        // strokes at 0.97 it hid the whole frame until the very last frame of the fade and then vanished at once
        // — the film's opening fade-in was a hard cut; painted as strokes at the film's opacity, its 1,400
        // translucent strokes smeared streaks over everything.) An iris is a shape and is painted as one.
        if let (true, Kind::Shape { wash: Some((c, op)), fill: None, .. }) = (it.screen && !it.family.starts_with("iris") && covers_canvas(&pts), &it.kind) {
            glaze(&mut p, rgb01(*c), *op as f32);
            continue;
        }
        let tag = it.id.rsplit('.').next().and_then(|s| s.split('#').next()).unwrap_or("");
        if let (true, Kind::Shape { wash: Some((c, _)), .. }) = (style().leaf_tags.contains(&tag), &it.kind) {
            under_leaf = Some((p.col.clone(), p.flat.clone(), p.thick.clone(), rgb01(*c)));
        }
        let is_hole = style().hole_tags.contains(&tag);
        // A bite hole is an opening, not a patch of paint. The film draws it as a shape in the colour of whatever is
        // behind the leaf; the painter instead leaves the leaf's paint off there, so what was painted before the
        // leaf shows through, at its own level, with the leaf's shadow falling into it and its cut edge around it.
        // A leaf's holes are opened together, so that holes run into one another as one opening.
        if let (true, true, Some((col0, flat0, thick0, leaf))) = (is_hole, i >= holes_end, &under_leaf) {
            holes_end = i;
            let mut gap = vec![0.0f32; WI * HI];
            let mut reach = 0.0f32;
            while let Some(h) = rec.items.get(holes_end).filter(|h| h.id.contains("hole")) {
                holes_end += 1;
                let (hp, hz) = screen(h, cam);
                let Some((m, o, sz)) = shape_mask(&hp, 1.0) else { continue };
                if sz[0].max(sz[1]) < (9.0 * hz) as usize + 6 {
                    continue;
                }
                // the cut edge is stated on the leaf first; opening the holes takes away what of it falls in one
                if let Kind::Shape { ink: Some((ic, w)), .. } = h.kind {
                    let ic = rgb01(ic);
                    let edge = [ic[0] * 0.55 + leaf[0] * 0.3, ic[1] * 0.55 + leaf[1] * 0.3, ic[2] * 0.55 + leaf[2] * 0.3];
                    let name = h.family.bytes().chain(h.id.bytes()).fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
                    ZOOM.with(|z| z.set(hz));
                    paint_line(&mut p, &hp, true, edge, (w as f32 * hz * 2.2).max(SMALLEST * hz * 0.8), seed.wrapping_mul(7919) ^ name ^ 0x1234, 0.1, None);
                }
                for y in 0..sz[1] {
                    for x in 0..sz[0] {
                        let j = (y + o[1] as usize) * WI + x + o[0] as usize;
                        gap[j] = gap[j].max(m[y * sz[0] + x]);
                    }
                }
                reach = reach.max((sz[0].min(sz[1]) as f32 * 0.14).min(12.0 * hz));
            }
            ZOOM.with(|z| z.set(zoom));
            let light = scene().map_or([-0.64, -0.77], |sc| sc.l.dir);
            let at = |x: f32, y: f32| -> f32 {
                let (gx, gy) = (x.round() as i32, y.round() as i32);
                if gx < 0 || gy < 0 || gx >= WI as i32 || gy >= HI as i32 { 1.0 } else { gap[gy as usize * WI + gx as usize] }
            };
            let (mut open, mut empty) = (0u32, 0u32);
            for j in 0..WI * HI {
                open += (gap[j] > 0.5) as u32;
                empty += (gap[j] > 0.5 && flat0[j] == bare[j]) as u32;
            }
            // nothing was painted behind this leaf: the hole's own colour is all there is to see through it
            holes_painted = empty * 2 > open;
            for j in 0..WI * HI {
                let k = gap[j];
                if k <= 0.0 {
                    continue;
                }
                // the leaf stands off the ground: towards the light, its edge shades what is seen through
                let (xf, yf) = ((j % WI) as f32, (j / WI) as f32);
                let sh = if holes_painted {
                    0.0
                } else {
                    0.5 * (1.0 - 0.5 * (at(xf + light[0] * reach, yf + light[1] * reach) + at(xf + light[0] * reach * 0.5, yf + light[1] * reach * 0.5)))
                };
                let cool = [1.0 - sh, 1.0 - sh * 0.9, 1.0 - sh * 0.66];
                for q in 0..3 {
                    p.col[j][q] += (col0[j][q] * cool[q] - p.col[j][q]) * k;
                    p.flat[j][q] += (flat0[j][q] - p.flat[j][q]) * k;
                }
                p.thick[j] += (thick0[j] - p.thick[j]) * k;
            }
        }
        if is_hole && under_leaf.is_some() && !holes_painted {
            continue;
        }
        // specks too small to be a brush mark
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for q in &pts {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        // (a speck arrives and leaves by degrees as it crosses the size of a brush mark, never in one frame)
        let speck = smooth(7.0 * zoom, 11.0 * zoom, (hi[0] - lo[0]).max(hi[1] - lo[1]));
        if speck <= 0.002 {
            continue;
        }
        let _speck = Fade::set(speck);
        let _ = i;
        let name = it.family.bytes().chain(it.id.bytes()).fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
        let id = seed.wrapping_mul(7919) ^ name;
        if trace() {
            eprintln!(
                "ITEM {i} {} {} id={id:08x} kind={} lo=({:.1},{:.1}) hi=({:.1},{:.1}) z={zoom:.4}",
                it.family,
                it.id,
                match &it.kind {
                    Kind::Shape { wash, fill, ink, .. } => format!("S w={:?} f={:?} i={:?}", wash.map(|w| w.1), fill.map(|w| w.1), ink.map(|w| w.1)),
                    Kind::Line { width, .. } => format!("L {width:.2}"),
                    _ => "G".into(),
                },
                lo[0],
                lo[1],
                hi[0],
                hi[1]
            );
        }
        let (ext_max, ext_min) = ((hi[0] - lo[0]).max(hi[1] - lo[1]), (hi[0] - lo[0]).min(hi[1] - lo[1]));
        // decisions about a shape go by its size in the film's world, not by how big the camera makes it just now
        let small_brush = SMALLEST * zoom;
        // Mixed media, sparingly: watercolour only for what is far away or has no substance (the sky, the distance,
        // and the film's own soft, unoutlined watercolour shapes). Everything with body is in oil.
        let airy = style().airy.iter().any(|f| it.family.starts_with(f)) || matches!(it.kind, Kind::Shape { wash: None, ink: None, .. });
        if WATER.load(Ordering::Relaxed) || (MIXED.load(Ordering::Relaxed) && !is_cast(&it.family) && airy) {
            // scenery: watercolour on the paper. The flat colour is one even wash; the film's watercolour layer is a
            // second, cloudier one over it; lines and outlines are ink.
            let n0 = pts.len() as f32;
            let centre = [pts.iter().map(|q| q[0]).sum::<f32>() / n0, pts.iter().map(|q| q[1]).sum::<f32>() / n0];
            let quiet = |c: [f32; 3]| scene().map_or(c, |sc| sc.quiet(c, centre));
            match it.kind {
                Kind::Shape { wash, fill, ink, .. } => {
                    if let Some((c, op)) = wash {
                        brush_wash(&mut p, &pts, quiet(rgb01(c)), op as f32);
                        if let Some((m, o, sz)) = shape_mask(&pts, 1.0) {
                            for y in 0..sz[1] {
                                for x in 0..sz[0] {
                                    let (px, py) = (x as i32 + o[0], y as i32 + o[1]);
                                    if m[y * sz[0] + x] >= 0.5 && px >= 0 && py >= 0 && (px as usize) < WI && (py as usize) < HI {
                                        p.flat[py as usize * WI + px as usize] = quiet(rgb01(c));
                                        p.wet[py as usize * WI + px as usize] = 1.0;
                                    }
                                }
                            }
                        }
                    }
                    if let Some((c, op)) = fill {
                        brush_fill(&mut p, &pts, quiet(rgb01(c)), op as f32, it.wc[0] as f32, it.wc[1] as f32, it.wc[2] as f32, id ^ 0x77);
                    }
                    if let Some((c, w)) = ink {
                        brush_stroke(&mut p, &pts, true, rgb01(c), w as f32 * zoom, id ^ 0x1234);
                    }
                }
                Kind::Line { col, width, .. } => brush_stroke(&mut p, &pts, false, rgb01(col), width as f32 * zoom, id ^ 0x4321),
                Kind::Glow { .. } => {}
            }
            continue;
        }
        match it.kind {
            Kind::Shape { wash, fill, ink, .. } => {
                let Some((c, op)) = wash.or(fill) else {
                    if let Some((c, w)) = ink {
                        paint_line(&mut p, &pts, true, rgb01(c), w as f32 * zoom * 1.5, id ^ 0x1234, 0.5, None);
                    }
                    continue;
                };
                let mut c = rgb01(c);
                let n0 = pts.len() as f32;
                let centre = [pts.iter().map(|q| q[0]).sum::<f32>() / n0, pts.iter().map(|q| q[1]).sum::<f32>() / n0];
                if let Some(sc) = scene() {
                    c = sc.quiet(c, centre);
                }
                // DEPTH v2: the band's recipe paints it — hue pulled toward the veil, the brush the recipe gives,
                // the paint as thick as the recipe says, fused as wet as the recipe wants. Subject bands fall
                // through with today's numbers exactly.
                let rec = recipe();
                c = towards_veil(c, rec.veil);
                // the watercolour layer becomes broken colour: some bristles carry it
                let alt = match (wash, fill) {
                    (Some(_), Some((f, _))) => scene().map_or(rgb01(f), |sc| sc.quiet(rgb01(f), centre)),
                    _ => shifted(c, 1.12, 0.5),
                };
                // Oil paint is opaque: a wash is laid at the reference's 0.97 however faint the film drew it. A
                // translucent wash (an eye's shade at 35 %, a sheen on the water, a star twinkling, the sun going
                // down) is instead MIXED on the palette: each stroke's paint is the wash colour let down with what
                // it lands on, by the film's opacity (see PREMIX in Painting::stroke). Laying the strokes
                // themselves at low opacity (glm's change) glazed thin streaks of every stroke over whatever was
                // there, which read as smeared, dirty paint. Only right at the bottom does a wash fade out, so it
                // leaves no pop when it is gone. (Full-frame fades and flashes never get here: they are glazed
                // over the finished picture, above.)
                let op = op as f32;
                if wash.is_some() && op < 0.004 {
                    continue;
                }
                let opacity = if wash.is_some() { 0.97 * smooth(0.0, 0.12, op) } else { (op * 1.1).clamp(0.3, 0.9) };
                // A shape too thin for an outline down each side (a strap, a twig drawn as a shape) has no room for
                // two dark lines and its colour between: it is painted as one line of its own colour gone a little
                // towards the outline's. (Its outline, at least 4 px a side, used to swallow a 5 px shape whole and
                // run close ones together.)
                let thick = thickness(&pts);
                let ink_w = ink.map(|(_, w)| (w as f32 * zoom * 1.5).max(small_brush * 0.6));
                let room = ink_w.map_or(1.0, |iw| smooth(1.5 * iw, 3.0 * iw, thick));
                if let (Some((ic, _)), true) = (ink, room < 1.0) {
                    let ic = rgb01(ic);
                    let edge = [ic[0] * 0.55 + c[0] * 0.3, ic[1] * 0.55 + c[1] * 0.3, ic[2] * 0.55 + c[2] * 0.3];
                    c = std::array::from_fn(|k| c[k] + (edge[k] - c[k]) * 0.35 * (1.0 - room));
                }
                // What the shape is decides how it is painted (see `brushwork_paint::Role`). A shadow or a glaze is
                // not paint laid over what is there but a glaze through which it shows.
                match it.role {
                    Role::Shadow | Role::FormShadow => {
                        let caster = (it.role == Role::Shadow).then(|| caster_of(frame_items, it, cam)).flatten();
                        // (the film's opacity says how present the shadow is, not how dark: it arrives by degrees)
                        let _present = Fade::set((op / 0.3).min(1.0));
                        let (depth, soft) = if it.role == Role::Shadow { (1.0, 0.15) } else { (0.7, 1.0) };
                        paint_cast_shadow(&mut p, &pts, caster.as_deref(), depth, soft, id);
                        continue;
                    }
                    Role::Glaze => {
                        paint_glaze(&mut p, &pts, c, op.min(1.0), id);
                        continue;
                    }
                    _ => {}
                }
                let behind_r = (thick * 0.3).max(2.0 * zoom);
                let behind = flat_under(&p.flat, centre[0], centre[1], behind_r, c);
                let behind_wet = wet_under(&p.wet, centre[0], centre[1], behind_r);
                // Unlabelled, a translucent wash darker than what it lies on is a glaze (an eye's shade, a stain),
                // by degrees of how translucent and how much darker it is.
                let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
                let glaze_k = if it.role == Role::Auto && wash.is_some() && fill.is_none() {
                    smooth(0.85, 0.7, op) * smooth(0.02, 0.08, lum(behind) - lum(c)) * behind_wet
                } else {
                    0.0
                };
                if glaze_k > 0.001 {
                    let _g = Fade::set(glaze_k);
                    paint_glaze(&mut p, &pts, c, op.min(1.0), id);
                }
                if glaze_k > 0.999 {
                    continue;
                }
                let _paint = Fade::set(1.0 - glaze_k);
                // A light source (labelled, or a pale shape on dark paint, which reads as one: a moon, a lit window)
                // is clean, luminous paint, softly modelled, with no band cut in round it. A highlight is clean,
                // loaded paint.
                let light_k = match it.role {
                    Role::Light => 1.0,
                    Role::Auto if !is_cast(&it.family) => smooth(0.76, 0.84, lum(c)) * smooth(0.3, 0.18, lum(behind)) * behind_wet,
                    _ => 0.0,
                };
                let clean = if it.role == Role::Highlight { 1.0 } else { light_k };
                let _mode = ModeGuard::set(Mode { clean, model: light_k, line: if it.role == Role::Line { 1.0 } else { 0.0 } });
                // the role says how far back it sits, where the film's bands do not
                match it.role {
                    Role::Sky | Role::Haze => SOFT.with(|s| s.set(1.0)),
                    Role::Background => SOFT.with(|s| s.set(s.get().max(0.45))),
                    Role::Accent | Role::Light | Role::Highlight => SOFT.with(|s| s.set(0.0)),
                    _ => SOFT.with(|s| s.set(s.get() * (1.0 - light_k))),
                }
                let _premix = Premix::set(if wash.is_some() { op.min(1.0) } else { 1.0 });
                let n = pts.len() as f32;
                let mid = [pts.iter().map(|q| q[0]).sum::<f32>() / n, pts.iter().map(|q| q[1]).sum::<f32>() / n];
                // Small things get their own treatments (below); near the size where a thing stops being small, the
                // small and the full treatment are cross-faded, so a shape that shrinks or grows by a pixel (a moon
                // rising, an eye blinking) changes by degrees, not in one frame. Likewise how much a small thing is an
                // accent, and whether a light accent is a knife touch, go by degrees of contrast.
                let small_k = 1.0 - smooth(small_brush * 3.0, small_brush * 4.0, ext_max);
                // an accent is a small thing the film itself outlines, or one much lighter or darker than mid-tone
                let accent_k = if ink.is_some() { 1.0 } else { smooth(0.26, 0.34, (lum(c) - 0.5).abs()) };
                let knife_k = if ink.is_some() { 0.0 } else { smooth(0.77, 0.83, lum(c)) };
                if small_k < 0.999 {
                    let _big = Fade::set(1.0 - small_k);
                    // 2. Big brushes and thin paint for what is behind; smaller brushes and thicker paint for the
                    //    subjects (here: the outlined, smaller shapes).
                    let subject = is_cast(&it.family);
                    let (detail, body) = if subject { (0.8, 1.15) } else { (1.25, 0.7) };
                    // (the shadow comes in by degrees as a form grows past the size that throws one)
                    let shadow_k = smooth(30.0 * zoom, 42.0 * zoom, ext_max);
                    if let (Some(sc), true) = (scene(), subject && shadow_k > 0.001) {
                        let _shadow = Fade::set(shadow_k);
                        // a form in light throws a shadow onto what is behind it: the same shape, moved away from the
                        // light, glazed over whatever is there in its own shadow colour (it used to be laid as dark
                        // paint, half of what was under it, which read as near-black patches on the owl), crisp where
                        // it leaves the form. It is a shallow shadow (the form stands close to what it falls on).
                        let lt = sc.l.towards(centre);
                        let d = (ext_min * 0.2).min(26.0 * zoom);
                        let cast: Vec<[f32; 2]> = pts.iter().map(|q| [q[0] - lt[0] * d, q[1] - lt[1] * d * 1.15]).collect();
                        paint_cast_shadow(&mut p, &cast, Some(&pts), 0.6, 0.15, id ^ 0x5AD0);
                    }
                    let (detail, body) = match scene() {
                        // brush and paint by attention as well: small and thick where the eye is, broad and thin elsewhere
                        Some(sc) => {
                            let a = sc.attention(centre);
                            (detail * (1.25 - 0.35 * a), body * (0.75 + 0.4 * a))
                        }
                        None => (detail, body),
                    };
                    // DEPTH v2 (opt-in, OIL_DEPTH): the band's recipe replaces the subject/scenery numbers. Without it the
                    // reference painting stands untouched. (recipe() is the MID row when DEPTH is off, which used to scale
                    // every SUBJECT to the background's broad brush and thin paint: detail x1.56, body x0.61.)
                    let (detail, body, opacity) = if DEPTH.load(Ordering::Relaxed) {
                        (detail * rec.detail / if subject { 0.8 } else { 1.25 }, body * rec.body / if subject { 1.15 } else { 0.7 }, opacity * rec.opacity)
                    } else {
                        (detail, body, opacity)
                    };
                    let alt = towards_veil(alt, rec.veil);
                    paint_form_band(
                        &mut p,
                        &pts,
                        c,
                        alt,
                        opacity,
                        id,
                        detail,
                        body,
                        if subject { Tool::Filbert } else { Tool::Flat },
                        // (a plane of scenery is lit nearly evenly: only a hint of turning)
                        if subject { 1.0 } else { 0.15 * (1.0 - smooth(800.0, 1000.0, ext_max / zoom)) },
                        rec.pickup_add,
                        rec.contour,
                    );
                    if let Some((ic, w)) = ink {
                        // 3. An outline is an edge, not a wire: the shape's own colour gone dark, stated on the shadow
                        //    side and lost towards the light. Subjects keep more of it than backgrounds.
                        let ic = rgb01(ic);
                        let edge = [ic[0] * 0.55 + c[0] * 0.3, ic[1] * 0.55 + c[1] * 0.3, ic[2] * 0.55 + c[2] * 0.3];
                        //    Scenery keeps its outline only where it stands out from what it meets.
                        let _room = Fade::set(room);
                        let keep = if subject { 0.6 } else { 0.0 };
                        let _ = w;
                        paint_line(&mut p, &pts, true, edge, ink_w.unwrap_or(small_brush), id ^ 0x1234, if subject { 0.35 } else { 0.1 }, Some((c, keep)));
                    }
                }
                if small_k > 0.001 && accent_k > 0.001 {
                    let _small = Fade::set(small_k * accent_k);
                    // 5. The exception: a small thing that contrasts strongly with what it sits on is an accent (an
                    //    eye, a pupil, a glint). Accents are where the eye goes, so they are painted crisply, within
                    //    their shape, with the smallest brush and little picked-up colour.
                    if knife_k > 0.001 {
                        // a light accent with no outline is a highlight: one touch of the knife, thick
                        let _knife = Fade::set(knife_k);
                        p.stroke(
                            &Stroke {
                                c: mid,
                                angle: -0.5,
                                len: (ext_max * 1.2).max(small_brush * 1.4),
                                width: (ext_min * 1.05).max(small_brush * 0.8),
                                bend: 0.0,
                                col: c,
                                alt: c,
                                opacity: 0.98,
                                pickup: 0.0,
                                load: 1.3,
                                part: Some((h32(id ^ 0x9E37) >> 2, true, true)),
                                span: (0.0, 1.0),
                                tool: Tool::Knife,
                                picked: None,
                                along: 0.0,
                            },
                            None,
                        );
                    }
                    if knife_k < 0.999 {
                        let _round = Fade::set(1.0 - knife_k);
                        paint_body(&mut p, &pts, c, c, 0.98, id, 0.8, 1.2, Tool::Round);
                        if let Some((ic, w)) = ink {
                            paint_line(&mut p, &pts, true, rgb01(ic), (w as f32 * zoom * 1.3).max(small_brush * 0.45), id ^ 0x1234, 0.85, None);
                        }
                    }
                }
                if small_k > 0.001 && accent_k < 0.999 {
                    let _quiet = Fade::set(small_k * (1.0 - accent_k));
                    // 1. Nothing is smaller than the smallest brush. A small thing that sits quietly in its
                    //    surroundings is ONE touch of thick paint, placed along its length, with no outline.
                    let (mut far, mut ang) = (0.0f32, 0.0f32);
                    for q in &pts {
                        let d = (q[0] - mid[0]).hypot(q[1] - mid[1]);
                        if d > far {
                            (far, ang) = (d, (q[1] - mid[1]).atan2(q[0] - mid[0]));
                        }
                    }
                    p.stroke(
                        &Stroke {
                            c: mid,
                            angle: if ext_max > ext_min * 1.4 { ang } else { -0.6 },
                            len: (ext_max * 1.25).max(small_brush * 1.6),
                            width: (ext_min * 1.1).max(small_brush),
                            bend: (hf(id, 3) - 0.5) * 0.4,
                            col: c,
                            alt,
                            opacity: 0.97,
                            pickup: 0.06,
                            load: 1.25,
                            part: Some((h32(id ^ 0x9E37) >> 2, true, true)),
                            span: (0.0, 1.0),
                            tool: Tool::Filbert,
                            picked: None,
                            along: 0.0,
                        },
                        None,
                    );
                }
            }
            Kind::Line { col, width, .. } => {
                // 4. A line is one confident flick of a small brush, or it is left out: nothing finer than the
                //    smallest brush, nothing shorter than a flick.
                let len: f32 = pts.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
                let w = width as f32 * zoom * 1.5;
                // (by degrees near both limits: a line that grows past a flick arrives gradually)
                let flick = smooth(small_brush * 1.6, small_brush * 2.4, len)
                    * (1.0 - (1.0 - smooth(small_brush * 0.3, small_brush * 0.4, w)) * (1.0 - smooth(small_brush * 3.6, small_brush * 4.4, len)));
                if flick <= 0.002 {
                    continue;
                }
                let _flick = Fade::set(flick);
                // (a rigger draws finer than the smallest brush paints an area)
                paint_line(&mut p, &pts, false, rgb01(col), w.max(small_brush * 0.42), id ^ 0x4321, 1.0, None);
            }
            Kind::Glow { col, a } => {
                let r = (pts[1][0] - pts[0][0]).hypot(pts[1][1] - pts[0][1]);
                paint_glow(&mut p, pts[0], r, rgb01(col), a as f32, id ^ 0x6100);
            }
        }
    }
    if std::env::var("OIL_DBG").is_ok() {
        eprintln!("t {t:.3} zoom {:.3} strokes {}", cam.map_or(1.0, |c| c.zoom), p.strokes);
    }
    p
}

thread_local! {
    static PROGRESS: std::cell::Cell<Option<fn(f32)>> = const { std::cell::Cell::new(None) };
}
/// Have `paint_frame` on this thread call `f` with how far through the frame's shapes it is (0 to 1), before each
/// shape: for a progress bar. It doesn't change the painting.
pub fn set_progress(f: Option<fn(f32)>) {
    PROGRESS.with(|p| p.set(f));
}

/// `OIL_TIMING=1` prints where each frame's time goes.
fn timing() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("OIL_TIMING").is_ok())
}

/// Lay whatever strokes a painting has written down for the GPU (no-op on the CPU backend).
fn finish(p: &mut Painting) {
    if let Some(r) = p.rec.take() {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            let t0 = timing().then(std::time::Instant::now);
            let (col, thick) = gpu::render(&r);
            if let Some(t0) = t0 {
                eprintln!("TIMING gpu {:.3} s ({} strokes, {:.0} MB of masks)", t0.elapsed().as_secs_f64(), r.cmds.len(), r.masks.len() as f64 * 4e-6);
            }
            p.col = col;
            p.thick = thick;
        }
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        {
            let _ = r;
            unreachable!("strokes recorded for the GPU are laid by `paint_frame_async` in a browser (and need the `gpu` feature)");
        }
    }
}

fn screen(it: &Item, cam: Option<Cam>) -> (Vec<[f32; 2]>, f32) {
    match (it.screen, cam) {
        (false, Some(c)) => (it.pts.iter().map(|p| c.to_screen(p[0], p[1])).map(|q| [q[0] as f32, q[1] as f32]).collect(), c.zoom as f32),
        _ => (it.pts.iter().map(|p| [p[0] as f32, p[1] as f32]).collect(), 1.0),
    }
}

// ---------------------------------------------------------------- painting from a picture

fn box_blur(src: &[[f32; 3]], r: usize) -> Vec<[f32; 3]> {
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

/// `oilphoto <in> <out.png> [fx fy reach]`: paints a photograph, with its switches read from the environment
/// (`OIL_NOCTURNE`, `OIL_ATTENTION`, `OIL_KEY`, `OIL_GROUND`; see [`PhotoOptions`]).
pub fn run_photo(args: &[String]) -> Result<(), String> {
    run_photo_with(args, &PhotoOptions::from_env())
}

/// The DEEP-SEA TEST switches of [`run_photo_with`] (each one is explained where it is used). [`PhotoOptions::from_env`]
/// reads them from `OIL_NOCTURNE`, `OIL_ATTENTION`, `OIL_KEY` and `OIL_GROUND`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhotoOptions {
    /// `OIL_NOCTURNE=g`
    pub nocturne: Option<f32>,
    /// `OIL_ATTENTION=off` sets this to false
    pub attention: bool,
    /// `OIL_KEY` (default 0.56)
    pub key: f32,
    /// `OIL_GROUND=subject`
    pub ground_subject: bool,
}
impl Default for PhotoOptions {
    fn default() -> Self {
        PhotoOptions { nocturne: None, attention: true, key: 0.56, ground_subject: false }
    }
}
impl PhotoOptions {
    pub fn from_env() -> Self {
        PhotoOptions {
            nocturne: std::env::var("OIL_NOCTURNE").ok().and_then(|v| v.parse::<f32>().ok()),
            attention: !std::env::var("OIL_ATTENTION").is_ok_and(|v| v == "off"),
            key: std::env::var("OIL_KEY").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.56),
            ground_subject: std::env::var("OIL_GROUND").is_ok_and(|v| v == "subject"),
        }
    }
}

/// [`run_photo`] with its switches given rather than read from the environment.
pub fn run_photo_with(args: &[String], opts: &PhotoOptions) -> Result<(), String> {
    let (input, out) = (args.first().ok_or("oilphoto <in> <out.png>")?, args.get(1).ok_or("oilphoto <in> <out.png>")?);
    let img = image::open(input).map_err(|e| e.to_string())?.to_rgb8();
    // the canvas is landscape: a portrait picture is painted on its side and turned back afterwards
    let turned = img.height() > img.width();
    let img = if turned { image::imageops::rotate270(&img) } else { img };
    let scale = (HI as f32 / img.height() as f32).min(WI as f32 / img.width() as f32);
    let (rw, rh) = ((img.width() as f32 * scale) as usize, (img.height() as f32 * scale) as usize);
    let small = image::imageops::resize(&img, rw as u32, rh as u32, image::imageops::FilterType::Triangle);
    let (ox, oy) = ((WI - rw) / 2, (HI - rh) / 2);
    let mut p = Painting::new();
    let mut subject = p.col.clone();
    for y in 0..rh {
        for x in 0..rw {
            let c = small.get_pixel(x as u32, y as u32).0;
            subject[(y + oy) * WI + x + ox] = [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0];
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
    let arg = |i: usize, d: f32| args.get(i).and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let (fu, fv, fr) = (arg(2, 0.5), arg(3, 0.4), arg(4, 0.34));
    let focus =
        if turned { [ox as f32 + fv * rw as f32, oy as f32 + (1.0 - fu) * rh as f32] } else { [ox as f32 + fu * rw as f32, oy as f32 + fv * rh as f32] };
    let reach_px = fr * rw.max(rh) as f32;
    // DEEP-SEA TEST (crabscape): `OIL_ATTENTION=off` paints every part of the picture as if it were looked at: no
    // magnification around the focus, no quieting away from it, small brushes allowed everywhere. For a subject
    // that must keep its exact drawn shape (a game sprite painted on its own).
    let attend = opts.attention;
    let attention = |q: [f32; 2]| if attend { (-(((q[0] - focus[0]).hypot(q[1] - focus[1])) / (1.2 * reach_px)).powi(2)).exp() } else { 1.0 };
    // DEEP-SEA TEST (crabscape): `OIL_KEY` is the value the unattended room relaxes towards (default 0.56, a daylight
    // middle grey). A nocturne's middle value is near black: there the quiet parts must sink, not lift to grey.
    let key = opts.key;
    {
        // what is looked at is seen larger than the lens drew it; the room around it gives way
        let lens = subject.clone();
        let at = |x: f32, y: f32| -> [f32; 3] {
            let (x, y) = (x.clamp(ox as f32, (ox + rw) as f32 - 1.001), y.clamp(oy as f32, (oy + rh) as f32 - 1.001));
            let (x0, y0) = (x.floor() as usize, y.floor() as usize);
            let (tx, ty) = (x - x0 as f32, y - y0 as f32);
            std::array::from_fn(|k| {
                let a = lens[y0 * WI + x0][k] + (lens[y0 * WI + x0 + 1][k] - lens[y0 * WI + x0][k]) * tx;
                let b = lens[(y0 + 1) * WI + x0][k] + (lens[(y0 + 1) * WI + x0 + 1][k] - lens[(y0 + 1) * WI + x0][k]) * tx;
                a + (b - a) * ty
            })
        };
        for y in oy..oy + rh {
            for x in ox..ox + rw {
                let d = [x as f32 - focus[0], y as f32 - focus[1]];
                let near = (-(d[0].hypot(d[1]) / reach_px).powi(2) * 1.1).exp();
                let g = if attend { 1.0 - 0.26 * near } else { 1.0 }; // DEEP-SEA TEST (crabscape): see OIL_ATTENTION
                let mut c = at(focus[0] + d[0] * g, focus[1] + d[1] * g);
                // away from it: colour quieter, lights and darks drawn together, a little cooler and lighter
                let away = 1.0 - attention([x as f32, y as f32]);
                let grey = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
                for (k, v) in c.iter_mut().enumerate() {
                    let quiet = grey + (*v - grey) * (1.0 - 0.38 * away);
                    *v = key + (quiet - key) * (1.0 - 0.30 * away) + [-0.012, 0.0, 0.02][k] * away;
                }
                subject[y * WI + x] = c;
            }
        }
    }
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let inside = |q: [f32; 2]| q[0] >= ox as f32 && q[1] >= oy as f32 && q[0] < (ox + rw) as f32 && q[1] < (oy + rh) as f32;
    let px = |q: [f32; 2]| (q[1].clamp(0.0, HI as f32 - 1.0) as usize) * WI + q[0].clamp(0.0, WI as f32 - 1.0) as usize;
    let t0 = std::time::Instant::now();
    // DEEP-SEA TEST (crabscape): `OIL_GROUND=subject` tones the ground with a broad wash of the subject's own colours
    // (an imprimatura laid in to suit the picture) instead of the default warm mid ochre, so a dark picture's
    // skipped bristles show dark ground, not bright brown.
    if opts.ground_subject {
        let wash = box_blur(&subject, 48);
        p.col.copy_from_slice(&wash);
        p.flat.copy_from_slice(&wash);
        p.wet.fill(1.0);
    }
    // ---- block the whole canvas in with the big flat, squinting
    {
        let seen = box_blur(&subject, 26);
        let w = 70.0f32;
        let mut spots: Vec<([f32; 2], u32)> = vec![];
        for gy in 0..(rh as f32 / (w * 0.5)) as u32 + 1 {
            for gx in 0..(rw as f32 / (w * 0.5)) as u32 + 1 {
                let id = 0x7700_0000 | gy << 12 | gx;
                let q = [ox as f32 + (gx as f32 + hf(id, 1)) * w * 0.5, oy as f32 + (gy as f32 + hf(id, 2)) * w * 0.5];
                if inside(q) {
                    spots.push((q, id));
                }
            }
        }
        spots.sort_by_key(|(_, id)| h32(*id));
        for (q, id) in spots {
            let a = -0.6 + (noise(q[0] / 420.0, q[1] / 420.0, 5) - 0.5) * 0.7;
            let d = [a.cos(), a.sin()];
            let path: Vec<[f32; 2]> = (0..7).map(|i| [q[0] + d[0] * w * 0.45 * (i as f32 - 3.0), q[1] + d[1] * w * 0.45 * (i as f32 - 3.0)]).collect();
            let c = seen[px(q)];
            sweep(&mut p, Tool::Flat, &path, w * (0.85 + 0.3 * hf(id, 4)), c, shifted(c, 1.06, 0.4), 0.97, 0.2, 0.5, id, &|_, _, _| 1.0);
        }
    }
    // ---- structure first. A cellular automaton finds the picture's regions: seeds are scattered (thickly where
    // the picture is busy, thinly where it is calm), and on every step each cell may be taken over by a neighbour's
    // region if that region's colour suits it better than its own does. Regions that end up alike are joined.
    let calm = box_blur(&subject, 2);
    let entropy: Vec<f32> = {
        let mean = box_blur(&calm, 6);
        let sq: Vec<[f32; 3]> = calm.iter().map(|c| [c[0] * c[0], c[1] * c[1], c[2] * c[2]]).collect();
        let msq = box_blur(&sq, 6);
        (0..WI * HI).map(|i| (0..3).map(|k| (msq[i][k] - mean[i][k] * mean[i][k]).max(0.0)).sum::<f32>().sqrt()).collect()
    };
    let mut label = vec![0u32; WI * HI];
    let mut strength = vec![0.0f32; WI * HI];
    let mut seed_col: Vec<[f32; 3]> = vec![[0.0; 3]];
    for gy in 0..(rh / 12) as u32 + 1 {
        for gx in 0..(rw / 12) as u32 + 1 {
            let q = [ox as f32 + (gx as f32 + hf(gx, gy * 7 + 1)) * 12.0, oy as f32 + (gy as f32 + hf(gx, gy * 7 + 2)) * 12.0];
            if !inside(q) {
                continue;
            }
            let busy = (entropy[px(q)] / 0.09).clamp(0.0, 1.0) * attention(q);
            let coarse = gx % 5 == 2 && gy % 5 == 2;
            if coarse || hf(gx, gy * 7 + 3) < busy * busy {
                label[px(q)] = seed_col.len() as u32;
                strength[px(q)] = 1.0;
                seed_col.push(calm[px(q)]);
            }
        }
    }
    {
        use rayon::prelude::*;
        for round in 0..2 {
            let reach = if round == 0 { 0.30 } else { 3.0 };
            for _ in 0..400 {
                let (lab0, str0) = (label.clone(), strength.clone());
                let changed: usize = label
                    .par_chunks_mut(WI)
                    .zip(strength.par_chunks_mut(WI))
                    .enumerate()
                    .map(|(y, (lrow, srow))| {
                        let mut n = 0;
                        if y < oy || y >= oy + rh {
                            return 0;
                        }
                        for x in ox..ox + rw {
                            let me = calm[y * WI + x];
                            let (mut best, mut who) = (srow[x], lrow[x]);
                            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                                let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                                if xx < ox as i32 || yy < oy as i32 || xx >= (ox + rw) as i32 || yy >= (oy + rh) as i32 {
                                    continue;
                                }
                                let j = yy as usize * WI + xx as usize;
                                let l = lab0[j];
                                if l == 0 || l == who {
                                    continue;
                                }
                                let suits = (1.0 - dist(me, seed_col[l as usize]) / reach).max(0.0);
                                let force = str0[j] * suits * if dx != 0 && dy != 0 { 0.978 } else { 0.985 };
                                if force > best + 1e-4 {
                                    (best, who) = (force, l);
                                }
                            }
                            if who != lrow[x] {
                                (lrow[x], srow[x]) = (who, best);
                                n += 1;
                            }
                        }
                        n
                    })
                    .sum();
                if changed == 0 {
                    break;
                }
            }
        }
    }
    // join neighbours that are alike (more readily where the picture is calm): big quiet planes become one shape
    let nl = seed_col.len();
    let (mut sum, mut cnt, mut ent) = (vec![[0.0f64; 3]; nl], vec![0.0f64; nl], vec![0.0f64; nl]);
    let mut looked = vec![0.0f64; nl];
    for y in oy..oy + rh {
        for x in ox..ox + rw {
            let (i, l) = (y * WI + x, label[y * WI + x] as usize);
            for k in 0..3 {
                sum[l][k] += subject[i][k] as f64;
            }
            cnt[l] += 1.0;
            ent[l] += entropy[i] as f64;
            looked[l] += attention([x as f32, y as f32]) as f64;
        }
    }
    let mut pairs: std::collections::HashSet<(u32, u32)> = Default::default();
    for y in oy..oy + rh - 1 {
        for x in ox..ox + rw - 1 {
            let a = label[y * WI + x];
            for b in [label[y * WI + x + 1], label[(y + 1) * WI + x]] {
                if a != b {
                    pairs.insert((a.min(b), a.max(b)));
                }
            }
        }
    }
    let mut parent: Vec<u32> = (0..nl as u32).collect();
    fn root(p: &mut [u32], mut i: u32) -> u32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    let mean_of = |sum: &[[f64; 3]], cnt: &[f64], l: usize| -> [f32; 3] {
        [(sum[l][0] / cnt[l].max(1.0)) as f32, (sum[l][1] / cnt[l].max(1.0)) as f32, (sum[l][2] / cnt[l].max(1.0)) as f32]
    };
    let mut order: Vec<(f32, u32, u32)> = pairs.iter().map(|&(a, b)| (dist(mean_of(&sum, &cnt, a as usize), mean_of(&sum, &cnt, b as usize)), a, b)).collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, a, b) in order {
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra == rb {
            continue;
        }
        let busy = (((ent[ra as usize] + ent[rb as usize]) / (cnt[ra as usize] + cnt[rb as usize]).max(1.0)) as f32 / 0.09).clamp(0.0, 1.0);
        let away = 1.0 - ((looked[ra as usize] + looked[rb as usize]) / (cnt[ra as usize] + cnt[rb as usize]).max(1.0)) as f32;
        if dist(mean_of(&sum, &cnt, ra as usize), mean_of(&sum, &cnt, rb as usize)) < 0.10 - 0.055 * busy + 0.11 * away {
            parent[rb as usize] = ra;
            for k in 0..3 {
                sum[ra as usize][k] += sum[rb as usize][k];
            }
            cnt[ra as usize] += cnt[rb as usize];
            ent[ra as usize] += ent[rb as usize];
            looked[ra as usize] += looked[rb as usize];
        }
    }
    for y in oy..oy + rh {
        for x in ox..ox + rw {
            label[y * WI + x] = root(&mut parent, label[y * WI + x]) + 1;
        }
    }
    // ---- regions to shapes: each connected patch of one region, its outline traced and straightened a little
    let mut comp = vec![0u32; WI * HI];
    let mut shapes: Vec<PhotoShape> = vec![];
    let mut stack: Vec<usize> = vec![];
    for y0 in oy..oy + rh {
        for x0 in ox..ox + rw {
            if comp[y0 * WI + x0] != 0 {
                continue;
            }
            let (id, l) = (shapes.len() as u32 + 1, label[y0 * WI + x0]);
            let (mut area, mut col, mut en) = (0usize, [0.0f64; 3], 0.0f64);
            comp[y0 * WI + x0] = id;
            stack.push(y0 * WI + x0);
            while let Some(i) = stack.pop() {
                area += 1;
                en += entropy[i] as f64;
                for k in 0..3 {
                    col[k] += subject[i][k] as f64;
                }
                let (x, y) = (i % WI, i / WI);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx < ox as i32 || yy < oy as i32 || xx >= (ox + rw) as i32 || yy >= (oy + rh) as i32 {
                        continue;
                    }
                    let j = yy as usize * WI + xx as usize;
                    if comp[j] == 0 && label[j] == l {
                        comp[j] = id;
                        stack.push(j);
                    }
                }
            }
            // walk the outline (the first cell found is the top-left one), keeping the patch on the right hand
            let is = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < WI && (y as usize) < HI && comp[y as usize * WI + x as usize] == id;
            let dirs = [(1i32, 0i32), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
            let (mut cx, mut cy, mut d) = (x0 as i32, y0 as i32, 0usize);
            let mut outline: Vec<[f32; 2]> = vec![];
            for _ in 0..(8 * area + 64).min(400_000) {
                outline.push([cx as f32 + 0.5, cy as f32 + 0.5]);
                let mut moved = false;
                for k in 0..8 {
                    let nd = (d + 6 + k) % 8;
                    if is(cx + dirs[nd].0, cy + dirs[nd].1) {
                        (cx, cy, d, moved) = (cx + dirs[nd].0, cy + dirs[nd].1, nd, true);
                        break;
                    }
                }
                if !moved || (cx == x0 as i32 && cy == y0 as i32 && outline.len() > 2) {
                    break;
                }
            }
            // keep every point that stands more than 1.5 px off the line between the kept ones either side
            let mut kept = vec![false; outline.len()];
            if outline.len() >= 3 {
                let mut todo = vec![(0usize, outline.len() - 1)];
                kept[0] = true;
                let last = outline.len() - 1;
                kept[last] = true;
                while let Some((a, b)) = todo.pop() {
                    let (pa, pb) = (outline[a], outline[b]);
                    let l = (pb[0] - pa[0]).hypot(pb[1] - pa[1]).max(1e-3);
                    let (mut far, mut at) = (0.0f32, a);
                    for i in a + 1..b {
                        let dd = if l < 1.0 {
                            (outline[i][0] - pa[0]).hypot(outline[i][1] - pa[1])
                        } else {
                            ((pb[0] - pa[0]) * (outline[i][1] - pa[1]) - (pb[1] - pa[1]) * (outline[i][0] - pa[0])).abs() / l
                        };
                        if dd > far {
                            (far, at) = (dd, i);
                        }
                    }
                    if far > 1.5 {
                        kept[at] = true;
                        todo.push((a, at));
                        todo.push((at, b));
                    }
                }
            }
            let poly: Vec<[f32; 2]> = outline.iter().zip(&kept).filter(|(_, k)| **k).map(|(q, _)| *q).collect();
            let a = area as f64;
            shapes.push((area, [(col[0] / a) as f32, (col[1] / a) as f32, (col[2] / a) as f32], (en / a) as f32, poly));
        }
    }
    eprintln!("structure: {} regions from {} seeds, {:.1} s", shapes.len(), nl - 1, t0.elapsed().as_secs_f64());
    // ---- paint the shapes as the film's shapes are painted: biggest first, each with the tool for its job
    shapes.sort_by_key(|s| std::cmp::Reverse(s.0));
    for (k, (area, c, en, poly)) in shapes.iter().enumerate() {
        if poly.len() < 3 {
            continue;
        }
        // how much the painter is looking at this shape: at its middle
        let n = poly.len() as f32;
        let a = attention([poly.iter().map(|q| q[0]).sum::<f32>() / n, poly.iter().map(|q| q[1]).sum::<f32>() / n]);
        // away from the subject, small things are simply not painted: the block-in stands for them
        if (*area as f32) < 40.0 + 1400.0 * (1.0 - a).powi(2) {
            continue;
        }
        let quiet = (*en < 0.035 && *area > 9000) || a < 0.4;
        let alt = shifted(*c, 1.1, 0.5);
        paint_body(&mut p, poly, *c, alt, 0.97, 0x5000 + k as u32, 1.7 - 0.95 * a, 0.5 + 0.65 * a, if quiet { Tool::Flat } else { Tool::Filbert });
    }
    eprintln!("shapes painted: {} strokes, {:.1} s", p.strokes, t0.elapsed().as_secs_f64());
    // ---- then look again, and touch in with the small brushes only where it is still wrong
    for (layer, (w, tool, wrong, load)) in [(13.0f32, Tool::Filbert, 0.11f32, 0.9f32), (SMALLEST, Tool::Round, 0.10, 1.0)].into_iter().enumerate() {
        let layer = layer + 2;
        // squint: see the subject no more finely than this brush can paint it
        let seen = box_blur(&subject, (w * 0.38) as usize);
        let grad = |q: [f32; 2]| -> [f32; 2] {
            let d = (w * 0.3).max(1.5);
            [lum(seen[px([q[0] + d, q[1]])]) - lum(seen[px([q[0] - d, q[1]])]), lum(seen[px([q[0], q[1] + d])]) - lum(seen[px([q[0], q[1] - d])])]
        };
        let step = w * 0.55;
        let mut spots: Vec<([f32; 2], u32)> = vec![];
        for gy in 0..(rh as f32 / step) as u32 + 1 {
            for gx in 0..(rw as f32 / step) as u32 + 1 {
                let id = (layer as u32) << 24 | gy << 12 | gx;
                let q = [ox as f32 + (gx as f32 + hf(id, 1)) * step, oy as f32 + (gy as f32 + hf(id, 2)) * step];
                if !inside(q) {
                    continue;
                }
                // is the canvas still wrong here, as far as this brush can tell?
                let mut err = 0.0;
                for (dx, dy) in [(0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)] {
                    let s = [q[0] + dx * step, q[1] + dy * step];
                    err += dist(p.col[px(s)], seen[px(s)]) / 5.0;
                }
                if attention(q) > [0.0, 0.0, 0.42, 0.6][layer.min(3)] && err > wrong {
                    spots.push((q, id));
                }
            }
        }
        spots.sort_by_key(|(_, id)| h32(*id));
        for (q, id) in spots {
            let c = seen[px(q)];
            // along the edge, not across it; where there is no edge, the painter's own diagonal
            let mut dir = {
                let g = grad(q);
                if g[0].hypot(g[1]) > 0.006 {
                    let l = g[0].hypot(g[1]);
                    [-g[1] / l, g[0] / l]
                } else {
                    let a = -0.6 + (noise(q[0] / 380.0, q[1] / 380.0, 5) - 0.5) * 0.9;
                    [a.cos(), a.sin()]
                }
            };
            // pull, don't push
            if dir[0] * 0.75 + dir[1] < 0.0 {
                dir = [-dir[0], -dir[1]];
            }
            let mut path = vec![[q[0] - dir[0] * w * 0.6, q[1] - dir[1] * w * 0.6], q];
            let (seg, most) = (w * 0.45, (7.0 + 9.0 * hf(id, 3)) as usize);
            let mut at = q;
            for i in 0..most {
                let next = [at[0] + dir[0] * seg, at[1] + dir[1] * seg];
                // the stroke ends where its colour stops being right for what is under it
                if !inside(next) || (i >= 2 && dist(seen[px(next)], c) > dist(seen[px(next)], p.col[px(next)]).max(0.06)) {
                    break;
                }
                path.push(next);
                at = next;
                let g = grad(at);
                let l = g[0].hypot(g[1]);
                if l > 0.006 {
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
            let width = w * (0.8 + 0.35 * hf(id, 4));
            sweep(&mut p, tool, &path, width, c, shifted(c, 1.1, 0.5), 0.96, 0.12 + 0.14 * hf(id, 5), load, id, &|_, _, _| 1.0);
        }
        eprintln!("brush {w:>4.0}: {} strokes so far, {:.1} s", p.strokes, t0.elapsed().as_secs_f64());
    }
    // DEEP-SEA TEST (crabscape): the nocturne's values transposed down (see `OIL_NOCTURNE` above)
    if let Some(g) = nocturne {
        for c in p.col.iter_mut() {
            for v in c.iter_mut() {
                *v = v.max(0.0).powf(g);
            }
        }
    }
    let lit = p.lit();
    let mut crop = vec![0u8; rw * rh * 3];
    for y in 0..rh {
        crop[y * rw * 3..(y + 1) * rw * 3].copy_from_slice(&lit[((y + oy) * WI + ox) * 3..((y + oy) * WI + ox + rw) * 3]);
    }
    let done = image::RgbImage::from_raw(rw as u32, rh as u32, crop).ok_or("image")?;
    let done = if turned { image::imageops::rotate90(&done) } else { done };
    done.save(out).map_err(|e| e.to_string())
}

/// `oilmix <out.png>`: the same wet-into-wet strokes painted twice. Top: picked-up paint averaged as RGB. Bottom:
/// mixed as pigment. Each column is a wet field of one colour with a second colour dragged through it.
pub fn run_mix(args: &[String]) -> Result<(), String> {
    let out = args.first().ok_or("oilmix <out.png>")?;
    let pairs: [([f32; 3], [f32; 3]); 5] = [
        ([0.96, 0.82, 0.16], [0.10, 0.25, 0.70]),
        ([0.10, 0.25, 0.70], [0.97, 0.96, 0.92]),
        ([0.86, 0.16, 0.14], [0.96, 0.82, 0.16]),
        ([0.16, 0.50, 0.30], [0.80, 0.20, 0.45]),
        ([0.92, 0.50, 0.12], [0.12, 0.45, 0.60]),
    ];
    let mut p = Painting::new();
    for row in 0..2 {
        PIGMENT.store(row == 1, Ordering::Relaxed);
        let (y0, h) = (40.0 + row as f32 * 520.0, 480.0);
        for (k, (field, drag)) in pairs.iter().enumerate() {
            let (x0, w) = (40.0 + k as f32 * 376.0, 336.0);
            let rect = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]];
            paint_body(&mut p, &rect, *field, *field, 0.98, 100 + k as u32, 1.0, 1.0, Tool::Flat);
            // the second colour dragged through, the brush picking up more of the field as it goes
            for j in 0..5 {
                let y = y0 + 60.0 + j as f32 * 88.0;
                let path: Vec<[f32; 2]> =
                    (0..40).map(|i| [x0 + 26.0 + i as f32 * (w - 52.0) / 39.0, y + 14.0 * ((i as f32) * 0.22 + j as f32).sin()]).collect();
                sweep(&mut p, Tool::Filbert, &path, 46.0, *drag, *drag, 0.97, 0.25 + 0.2 * j as f32, 1.0, 900 + (k * 10 + j) as u32, &|_, _, _| 1.0);
            }
        }
    }
    image::RgbImage::from_raw(WI as u32, HI as u32, p.lit()).ok_or("image")?.save(out).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- the watercolour brush (after p5.brush)
//
// A port of p5.brush's fill, step for step: the shape is copied about ninety times as thin layers, each copy grown
// outwards a little differently (a new point in the middle of every side, pushed out by a random share of the side's
// length), two of every three cut short by a chord, plus a sparse copy through a few of its points and a darker cut
// copy; every copy gets a faint outline; four times along the way several hundred faint circles are lifted back out
// and what has built up is laid onto the picture as pigment, with a little more wherever its density changes
// sharply (tide lines).

struct Roll(u64);
impl Roll {
    fn new(seed: u32) -> Roll {
        let mut r = Roll((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.unit();
        }
        r
    }
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 16_777_216.0
    }
    fn rr(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    fn gauss(&mut self, mean: f32, sd: f32) -> f32 {
        let (u, v) = (self.unit().max(1e-7), self.unit());
        mean + sd * (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn in_poly(v: &[[f32; 2]], p: [f32; 2]) -> bool {
    let mut inside = false;
    for i in 0..v.len() {
        let (a, b) = (v[i], v[(i + 1) % v.len()]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
    }
    inside
}

#[derive(Clone)]
struct FillPoly {
    v: Vec<[f32; 2]>,
    /// How strongly each side grows.
    m: Vec<f32>,
    /// Whether a quarter turn of each side points out of the shape.
    dir: Vec<bool>,
    mid: [f32; 2],
    size: [f32; 2],
}

impl FillPoly {
    fn trim(&self, f: f32, r: &mut Roll) -> (Vec<[f32; 2]>, Vec<f32>, Vec<bool>) {
        let n = self.v.len();
        if !(0.0..1.0).contains(&f) || n <= 8 {
            return (self.v.clone(), self.m.clone(), self.dir.clone());
        }
        let cut = ((1.0 - f) * n as f32) as usize;
        let s = n / 2 - cut / 2;
        let (a, b) = (self.v[(s + n - 1) % n], self.v[(s + cut) % n]);
        let (ev, len) = ([b[0] - a[0], b[1] - a[1]], (b[0] - a[0]).hypot(b[1] - a[1]));
        let at = if s >= 2 {
            r.rr(0.0, s as f32 - 1.0) as usize
        } else if s + cut < n - 1 {
            s + cut
        } else {
            0
        };
        let spacing = (self.v[(at + 1) % n][0] - self.v[at][0]).hypot(self.v[(at + 1) % n][1] - self.v[at][1]).max(1.0);
        let add = ((len / spacing * 0.05).ceil() as usize).max(2);
        let (mut v, mut m, mut d) = (self.v[..s].to_vec(), self.m[..s].to_vec(), self.dir[..s].to_vec());
        let jitter = len * 0.06;
        for k in 0..add {
            let t = (k + 1) as f32 / (add + 1) as f32;
            v.push([a[0] + ev[0] * t + r.rr(-jitter, jitter), a[1] + ev[1] * t + r.rr(-jitter, jitter)]);
            m.push(r.rr(0.3, 0.5));
            d.push(self.dir[s % self.dir.len()]);
        }
        v.extend_from_slice(&self.v[(s + cut).min(n)..]);
        m.extend_from_slice(&self.m[(s + cut).min(n)..]);
        d.extend_from_slice(&self.dir[(s + cut).min(n)..]);
        (v, m, d)
    }

    /// `f` below 1 trims first; 999 and 997 are the library's two texturing passes (a strong even push, then a mild one).
    fn grow(&self, f: f32, bleed: f32, cap: usize, r: &mut Roll) -> FillPoly {
        let (tv, tm, td) = self.trim(f, r);
        let n = tv.len();
        let mut push = if f == 999.0 { r.rr(0.6, 0.8) } else { bleed };
        let pre = if n * 2 > cap { (n * 2).div_ceil(cap) } else { 1 };
        if pre >= 2 && pre % 2 == 0 {
            return FillPoly { v: tv, m: tm, dir: td, mid: self.mid, size: self.size };
        }
        let (mut v, mut m, mut d) = (Vec::with_capacity(n * 2), Vec::with_capacity(n * 2), Vec::with_capacity(n * 2));
        for i in 0..n {
            let (a, b) = (tv[i], tv[(i + 1) % n]);
            if f < 997.0 {
                push = tm[i];
            }
            v.push(a);
            m.push(tm[i]);
            d.push(td[i]);
            let side = [b[0] - a[0], b[1] - a[1]];
            if push < 0.05 {
                v.push([a[0] + side[0] / 2.0, a[1] + side[1] / 2.0]);
                m.push(tm[i]);
                d.push(td[i]);
                continue;
            }
            let ang = (if td[i] { -90.0f32 } else { 90.0 } + r.rr(-1.0, 1.0) * 5.0).to_radians();
            let (s, c) = ang.sin_cos();
            let out = [c * side[0] + s * side[1], c * side[1] - s * side[0]];
            let far = r.gauss(0.5, 0.2) * r.rr(0.65, 1.35) * push;
            v.push([a[0] + side[0] / 2.0 + out[0] * far, a[1] + side[1] / 2.0 + out[1] * far]);
            m.push(tm[i] + r.gauss(0.0, 0.02));
            d.push(td[i]);
        }
        if v.len() > cap {
            let step = v.len().div_ceil(cap);
            v = v.into_iter().step_by(step).collect();
            m = m.into_iter().step_by(step).collect();
            d = d.into_iter().step_by(step).collect();
        }
        FillPoly { v, m, dir: d, mid: self.mid, size: self.size }
    }

    fn scatter(&self, ratio: f32, shape: &[[f32; 2]], r: &mut Roll) -> FillPoly {
        let n = self.v.len();
        let keep = ((n as f32 * ratio) as usize).max(3);
        let step = n as f32 / keep as f32;
        let (mut v, mut m, mut d) = (vec![], vec![], vec![]);
        for i in 0..keep {
            let j = ((i as f32 * step + r.rr(0.0, step * 0.8)) as usize) % n;
            let mut q = self.v[j];
            if !in_poly(shape, q) {
                let k = r.rr(0.3, 0.6);
                q = [self.mid[0] + (q[0] - self.mid[0]) * k, self.mid[1] + (q[1] - self.mid[1]) * r.rr(0.3, 0.6)];
            }
            v.push(q);
            m.push(self.m[j]);
            d.push(!self.dir[j]);
        }
        FillPoly { v, m, dir: d, mid: self.mid, size: self.size }
    }

    fn flipped(mut self) -> FillPoly {
        for d in &mut self.dir {
            *d = !*d;
        }
        self
    }
}

/// What has been painted since the last composite: alpha over the fill's part of the canvas.
struct Mask {
    o: [i32; 2],
    w: usize,
    h: usize,
    a: Vec<f32>,
    /// A stroke's own coverage (so its joints are not laid twice), valid where `stamp` is the current stroke.
    cov: Vec<f32>,
    stamp: Vec<u32>,
    stroke: u32,
    row: Vec<f32>,
}

impl Mask {
    fn new(lo: [f32; 2], hi: [f32; 2]) -> Option<Mask> {
        let o = [(lo[0].floor() as i32).max(0), (lo[1].floor() as i32).max(0)];
        let e = [(hi[0].ceil() as i32).min(WI as i32), (hi[1].ceil() as i32).min(HI as i32)];
        if e[0] <= o[0] || e[1] <= o[1] {
            return None;
        }
        let (w, h) = ((e[0] - o[0]) as usize, (e[1] - o[1]) as usize);
        Some(Mask { o, w, h, a: vec![0.0; w * h], cov: vec![0.0; w * h], stamp: vec![0; w * h], stroke: 0, row: vec![0.0; w] })
    }

    /// A filled polygon (non-zero winding, four sample rows per pixel), laid over what is there.
    fn fill(&mut self, v: &[[f32; 2]], alpha: f32) {
        let n = v.len();
        if n < 3 || alpha <= 0.0 {
            return;
        }
        let (mut y0, mut y1) = (f32::MAX, f32::MIN);
        for q in v {
            y0 = y0.min(q[1]);
            y1 = y1.max(q[1]);
        }
        let (ya, yb) = (((y0 - self.o[1] as f32).floor() as i32).max(0), ((y1 - self.o[1] as f32).ceil() as i32).min(self.h as i32));
        let mut xs: Vec<(f32, i32)> = vec![];
        for y in ya..yb {
            self.row.fill(0.0);
            let mut any = false;
            for sub in 0..4 {
                let py = (y + self.o[1]) as f32 + (sub as f32 + 0.5) / 4.0;
                xs.clear();
                for i in 0..n {
                    let (a, b) = (v[i], v[(i + 1) % n]);
                    if (a[1] > py) != (b[1] > py) {
                        xs.push((a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) - self.o[0] as f32, if b[1] > a[1] { 1 } else { -1 }));
                    }
                }
                if xs.len() < 2 {
                    continue;
                }
                xs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut wind = 0;
                for k in 0..xs.len() - 1 {
                    wind += xs[k].1;
                    if wind == 0 {
                        continue;
                    }
                    let (xa, xb) = (xs[k].0.max(0.0), xs[k + 1].0.min(self.w as f32));
                    if xb <= xa {
                        continue;
                    }
                    any = true;
                    let (ia, ib) = (xa.floor() as usize, (xb.ceil() as usize).min(self.w));
                    for x in ia..ib {
                        let c = (xb.min(x as f32 + 1.0) - xa.max(x as f32)).max(0.0);
                        self.row[x] += 0.25 * c;
                    }
                }
            }
            if any {
                let base = y as usize * self.w;
                for x in 0..self.w {
                    let c = self.row[x];
                    if c > 0.0 {
                        let a = &mut self.a[base + x];
                        *a += (1.0 - *a) * alpha * c.min(1.0);
                    }
                }
            }
        }
    }

    /// The polygon's outline as one stroke with round ends.
    fn outline(&mut self, v: &[[f32; 2]], width: f32, alpha: f32) {
        if width < 0.3 || alpha <= 0.0 {
            return;
        }
        self.stroke += 1;
        let half = width / 2.0;
        let n = v.len();
        for i in 0..n {
            let (a, b) =
                ([v[i][0] - self.o[0] as f32, v[i][1] - self.o[1] as f32], [v[(i + 1) % n][0] - self.o[0] as f32, v[(i + 1) % n][1] - self.o[1] as f32]);
            let (x0, x1) = (((a[0].min(b[0]) - half - 1.0).floor() as i32).max(0), ((a[0].max(b[0]) + half + 1.0).ceil() as i32).min(self.w as i32));
            let (y0, y1) = (((a[1].min(b[1]) - half - 1.0).floor() as i32).max(0), ((a[1].max(b[1]) + half + 1.0).ceil() as i32).min(self.h as i32));
            let ab = [b[0] - a[0], b[1] - a[1]];
            let l2 = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-6);
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = [x as f32 + 0.5 - a[0], y as f32 + 0.5 - a[1]];
                    let t = ((p[0] * ab[0] + p[1] * ab[1]) / l2).clamp(0.0, 1.0);
                    let c = (half + 0.5 - (p[0] - ab[0] * t).hypot(p[1] - ab[1] * t)).clamp(0.0, 1.0);
                    if c <= 0.0 {
                        continue;
                    }
                    let i = y as usize * self.w + x as usize;
                    let before = if self.stamp[i] == self.stroke { self.cov[i] } else { 0.0 };
                    if c > before {
                        self.stamp[i] = self.stroke;
                        self.cov[i] = c;
                        self.a[i] += (1.0 - self.a[i]) * alpha * (c - before);
                    }
                }
            }
        }
    }

    fn lift(&mut self, c: [f32; 2], radius: f32, amount: f32) {
        let (cx, cy) = (c[0] - self.o[0] as f32, c[1] - self.o[1] as f32);
        let (x0, x1) = (((cx - radius - 1.0).floor() as i32).max(0), ((cx + radius + 1.0).ceil() as i32).min(self.w as i32));
        let (y0, y1) = (((cy - radius - 1.0).floor() as i32).max(0), ((cy + radius + 1.0).ceil() as i32).min(self.h as i32));
        for y in y0..y1 {
            for x in x0..x1 {
                let cov = (radius + 0.5 - (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy)).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.a[y as usize * self.w + x as usize] *= 1.0 - amount * cov;
                }
            }
        }
    }

    fn stamp(&mut self, c: [f32; 2], radius: f32, alpha: f32) {
        let (cx, cy) = (c[0] - self.o[0] as f32, c[1] - self.o[1] as f32);
        let (x0, x1) = (((cx - radius - 1.0).floor() as i32).max(0), ((cx + radius + 1.0).ceil() as i32).min(self.w as i32));
        let (y0, y1) = (((cy - radius - 1.0).floor() as i32).max(0), ((cy + radius + 1.0).ceil() as i32).min(self.h as i32));
        for y in y0..y1 {
            for x in x0..x1 {
                let cov = (radius + 0.5 - (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy)).clamp(0.0, 1.0);
                if cov > 0.0 {
                    let a = &mut self.a[y as usize * self.w + x as usize];
                    *a += (1.0 - *a) * alpha * cov;
                }
            }
        }
    }

    /// As `composite`, for brush strokes: no tide lines, and where the tips have piled up the pigment goes darker.
    fn composite_stroke(&mut self, p: &mut Painting, pigment: [f32; 3]) {
        let ks_of = |c: [f32; 3]| brushwork_kit::spectral::ks([c[0] as f64, c[1] as f64, c[2] as f64]);
        // the pigment at each strength of piling-up, worked out once
        let table: Vec<([f64; 38], f64)> = (0..=8)
            .map(|i| {
                let black = 0.5 * 0.3 * i as f32 / 8.0;
                let c = pigment.map(|v| (v * (1.0 - black) - 0.5 * black).max(0.0));
                let k = ks_of(c);
                let l = brushwork_kit::spectral::colour(&k).1;
                (k, l)
            })
            .collect();
        for y in 0..self.h {
            for x in 0..self.w {
                let m = self.a[y * self.w + x];
                if m <= 0.003 {
                    continue;
                }
                let i = (y + self.o[1] as usize) * WI + x + self.o[0] as usize;
                let (kp, lp) = &table[(((m - 0.7).max(0.0) / 0.3 * 8.0).round() as usize).min(8)];
                let bg = p.col[i];
                let kb = ks_of(bg);
                let lb = brushwork_kit::spectral::colour(&kb).1;
                let t = m.min(1.0) as f64;
                let (c1, c2) = ((1.0 - t) * (1.0 - t) * lb, t * t * lp);
                let total = (c1 + c2).max(1e-9);
                let mixed: [f64; 38] = std::array::from_fn(|b| (kb[b] * c1 + kp[b] * c2) / total);
                let rgb = brushwork_kit::spectral::colour(&mixed).0;
                p.col[i] = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32];
            }
        }
        self.a.fill(0.0);
    }

    /// Lay what has built up onto the picture as pigment, and start again. Kubelka-Munk over 38 bands, each paint's
    /// share going with the square of its strength and with its luminance (as the library mixes), so thin dark
    /// paint on a light ground barely shows and colours mix as paints do.
    fn composite(&mut self, p: &mut Painting, pigment: [f32; 3]) {
        use rayon::prelude::*;
        // A careful brush slows where one colour meets another and leaves a little more pigment there. Not in
        // p5.brush: `EDGE_DEPOSIT` is how much more (0 = the library's own faint tide line only).
        let deposit = std::env::var("OIL_EDGE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(EDGE_DEPOSIT);
        let kp = brushwork_kit::spectral::ks([pigment[0] as f64, pigment[1] as f64, pigment[2] as f64]);
        let lp = brushwork_kit::spectral::colour(&kp).1;
        let (w, h, o) = (self.w, self.h, self.o);
        let a = &self.a;
        let at = |x: i32, y: i32| a[(y.clamp(0, h as i32 - 1) as usize) * w + x.clamp(0, w as i32 - 1) as usize];
        let rows: Vec<(usize, Vec<[f32; 3]>)> = (0..h)
            .into_par_iter()
            .filter_map(|y| {
                let base = (y + o[1] as usize) * WI + o[0] as usize;
                let mut out: Vec<[f32; 3]> = vec![];
                let mut touched = false;
                for x in 0..w {
                    let m = a[y * w + x];
                    let bg = p.col[base + x];
                    if m <= 0.0005 {
                        out.push(bg);
                        continue;
                    }
                    let mut tide = 0.0;
                    let mut rim = 0.0f32;
                    // how close this is to the wash's own edge: strongest right at it, gone a dozen pixels in
                    if deposit > 0.0 {
                        let mut out_near = 0.0f32;
                        for (k, reach) in [3i32, 6, 10, 14].into_iter().enumerate() {
                            let mut least = 1.0f32;
                            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)] {
                                least = least.min(at(x as i32 + dx * reach, y as i32 + dy * reach) / m.max(1e-3));
                            }
                            out_near = out_near.max((1.0 - least).clamp(0.0, 1.0) * [1.0, 0.7, 0.4, 0.2][k]);
                        }
                        rim = out_near;
                    }
                    for j in [-2i32, 0, 2] {
                        for i in [-2i32, 0, 2] {
                            let (xx, yy) = (x as i32 + i, y as i32 + j);
                            let g = [(at(xx + 1, yy) - at(xx - 1, yy)) * 7.5, (at(xx, yy + 1) - at(xx, yy - 1)) * 7.5];
                            tide += smooth(0.05, 0.35, g[0].hypot(g[1]));
                        }
                    }
                    let t = ((m + tide / 9.0 * 0.1).clamp(0.0, 1.0)) as f64;
                    let kb = brushwork_kit::spectral::ks([bg[0] as f64, bg[1] as f64, bg[2] as f64]);
                    let lb = brushwork_kit::spectral::colour(&kb).1;
                    let (c1, c2) = ((1.0 - t) * (1.0 - t) * lb, t * t * lp);
                    let total = (c1 + c2).max(1e-9);
                    // more pigment in the same place: its absorption goes up in proportion
                    let more = 1.0 + deposit * (rim * m) as f64;
                    let mixed: [f64; 38] = std::array::from_fn(|b| (kb[b] * c1 + kp[b] * c2 * more) / total);
                    let rgb = brushwork_kit::spectral::colour(&mixed).0;
                    out.push([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]);
                    touched = true;
                }
                touched.then_some((y, out))
            })
            .collect();
        for (y, out) in rows {
            let base = (y + o[1] as usize) * WI + o[0] as usize;
            p.col[base..base + w].copy_from_slice(&out);
        }
        self.a.fill(0.0);
    }
}

/// p5.brush's `fill`. `opacity` 0..1; the kit's usual bleed 0.1, texture 0.4, border 0.35.
#[allow(clippy::too_many_arguments)]
pub fn brush_fill(p: &mut Painting, pts: &[[f32; 2]], col: [f32; 3], opacity: f32, bleed: f32, tex: f32, border: f32, seed: u32) {
    let n = pts.len();
    if n < 3 {
        return;
    }
    let mut r = Roll::new(seed);
    // the library starts the outline at a random point; the first stretch of it hardly grows
    let roll = r.rr(0.0, 75.0);
    let fluid = (n as f32
        * 0.25
        * if roll < 5.0 {
            1.0
        } else if roll < 15.0 {
            2.0
        } else {
            3.0
        }) as usize;
    let m: Vec<f32> = (0..n).map(|i| if i > fluid { 1.0 } else { 0.3 } * r.rr(0.85, 1.4) * bleed).collect();
    let shift = (r.unit() * n as f32) as usize % n;
    let v: Vec<[f32; 2]> = (0..n).map(|i| pts[(i + shift) % n]).collect();
    let centre = {
        let (mut area, mut cx, mut cy) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let (a, b) = (v[i], v[(i + 1) % n]);
            let cr = a[0] * b[1] - b[0] * a[1];
            area += cr;
            cx += (a[0] + b[0]) * cr;
            cy += (a[1] + b[1]) * cr;
        }
        if n < 8 || area.abs() < 1e-3 {
            [v.iter().map(|q| q[0]).sum::<f32>() / n as f32, v.iter().map(|q| q[1]).sum::<f32>() / n as f32]
        } else {
            [cx / (3.0 * area), cy / (3.0 * area)]
        }
    };
    let size = [v.iter().map(|q| (q[0] - centre[0]).abs()).fold(0.0, f32::max), v.iter().map(|q| (q[1] - centre[1]).abs()).fold(0.0, f32::max)];
    let dir: Vec<bool> = (0..n)
        .map(|i| {
            let (a, b) = (v[i], v[(i + 1) % n]);
            let l = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-3);
            !in_poly(&v, [(a[0] + b[0]) / 2.0 - (b[1] - a[1]) / l * 0.75, (a[1] + b[1]) / 2.0 + (b[0] - a[0]) / l * 0.75])
        })
        .collect();
    let mid = [centre[0] + r.rr(-0.6, 0.6) * size[0], centre[1] + r.rr(-0.6, 0.6) * size[1]];
    let first = FillPoly { v, m, dir, mid, size };
    let big = size[0].max(size[1]);
    let reach = big * bleed * 1.8 + big / 25.0 + 4.0;
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for q in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k] - reach);
            hi[k] = hi[k].max(q[k] + reach);
        }
    }
    let Some(mut mask) = Mask::new(lo, hi) else { return };
    let texture = tex * 3.0;
    let layer_alpha = 2.0 * opacity * (1.0 + tex / 2.0) / 100.0;
    let cap = (2024.0 * (2.0 * bleed).max(0.2)) as usize;
    let darker = r.rr(0.15, 0.7);
    let mut pol = first.grow(1.0, bleed, cap, &mut r);
    let sparse = first.scatter(0.1, pts, &mut r).grow(1.0, bleed, cap, &mut r).scatter(0.75, pts, &mut r).flipped();
    let mut pols: Vec<FillPoly> = vec![];
    for i in 0..20 {
        if i % 4 == 0 {
            pol = pol.grow(1.0, bleed, cap, &mut r);
        }
        if i % 2 == 0 {
            pols = [1.0, 0.7, 0.4].iter().map(|f| pol.grow(f - 0.0125 * i as f32, bleed, cap, &mut r)).collect();
        }
        let width = (big / 25.0 + (big / 30.0 - big / 25.0) * (i as f32 / 24.0).clamp(0.0, 1.0)) * border;
        let lay = |shape: FillPoly, alpha: f32, mask: &mut Mask| {
            mask.fill(&shape.v, alpha);
            mask.outline(&shape.v, width, border * 0.01);
        };
        for q in &pols {
            lay(q.grow(999.0, bleed, cap, &mut r).grow(997.0, bleed, cap, &mut r), layer_alpha, &mut mask);
        }
        lay(sparse.grow(999.0, bleed, cap, &mut r).flipped().grow(997.0, bleed, cap, &mut r), layer_alpha * texture, &mut mask);
        if i % 2 == 0 {
            lay(pol.grow(darker, bleed, cap, &mut r).grow(999.0, bleed, cap, &mut r), layer_alpha * 2.0, &mut mask);
        }
        if i % 8 == 0 || i == 19 {
            if texture > 0.0 {
                let count = (r.rr(80.0, 110.0) * (2.0 + 1.5 * texture * 3.0)) as usize;
                let least = size[0].min(size[1]) * 1.3;
                let amount = (5.0 - 0.3) * texture * 3.0 / 255.0;
                for k in 0..count {
                    let c = [mid[0] + r.gauss(0.0, size[0] / 1.3), mid[1] + r.gauss(0.0, size[1] / 1.3)];
                    let d = r.rr(0.03 * least, 0.45 * least);
                    if k % 5 != 0 {
                        mask.lift(c, d / 2.0, amount.min(1.0));
                    }
                }
            }
            mask.composite(p, col);
        }
    }
}

/// p5.brush's `wash`: the shape as one flat layer of pigment.
pub fn brush_wash(p: &mut Painting, pts: &[[f32; 2]], col: [f32; 3], opacity: f32) {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for q in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k] - 2.0);
            hi[k] = hi[k].max(q[k] + 2.0);
        }
    }
    let Some(mut mask) = Mask::new(lo, hi) else { return };
    mask.fill(pts, opacity.clamp(0.0, 1.0));
    mask.composite(p, col);
}

/// p5.brush's default stroke, as the kit's 'ink' brush uses it: round tips stamped every twenty-fifth of the
/// brush's width along the line, each a little off it (more when the pressure is light), their size going with
/// the square of the pressure, which runs from 1.15 at the start of the stroke to 0.75 at its end. `width` is the
/// stroke's width at pressure 1.
pub fn brush_stroke(p: &mut Painting, pts: &[[f32; 2]], closed: bool, col: [f32; 3], width: f32, seed: u32) {
    let n = pts.len();
    if n < 2 || width <= 0.0 {
        return;
    }
    let reach = width * 1.1 + 3.0;
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for q in pts {
        for k in 0..2 {
            lo[k] = lo[k].min(q[k] - reach);
            hi[k] = hi[k].max(q[k] + reach);
        }
    }
    let Some(mut mask) = Mask::new(lo, hi) else { return };
    let mut r = Roll::new(seed);
    let segs = if closed { n } else { n - 1 };
    let total: f32 = (0..segs).map(|i| (pts[(i + 1) % n][0] - pts[i][0]).hypot(pts[(i + 1) % n][1] - pts[i][1])).sum();
    if total < 0.5 {
        return;
    }
    let (scatter, sharp, noise_k) = (0.25 / 5.0 * width, 0.8, 0.015);
    let alpha = (235.0 / 255.0) * (1.0 + r.gauss(0.0, noise_k));
    // the whole stroke's pressure sits a little higher or lower, and its ramp a little steeper or shallower
    let (lift, tilt) = (r.rr(-0.04, 0.04), r.rr(0.9, 1.1));
    let step = (width * 0.04).max(0.35);
    let mut walked = 0.0f32;
    let mut carry = 0.0f32;
    for i in 0..segs {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let l = (b[0] - a[0]).hypot(b[1] - a[1]);
        if l < 1e-3 {
            continue;
        }
        let (dx, dy) = ((b[0] - a[0]) / l, (b[1] - a[1]) / l);
        let mut at = carry;
        while at < l {
            let t = (walked + at) / total;
            let pressure = (0.95 + (1.15 - 0.95) * tilt + ((0.75 - 1.15) * tilt) * t + lift).max(0.2);
            let vib = scatter * (sharp + (1.0 - sharp) * r.gauss(0.0, 1.0) / pressure);
            let (perp, along) = (vib * r.rr(-1.0, 1.0), 0.3 * vib * r.rr(-1.0, 1.0));
            let c = [a[0] + dx * (at + along) - dy * perp, a[1] + dy * (at + along) + dx * perp];
            let d = pressure * pressure * width * r.rr(0.85, 1.15);
            mask.stamp(c, d / 2.0, (pressure.max(0.9) * alpha * r.rr(0.75, 1.1)).min(1.0));
            at += step;
        }
        carry = at - l;
        walked += l;
    }
    mask.composite_stroke(p, col);
}

#[cfg(test)]
mod tests {
    //! Steadiness and wash regressions. The quick ones paint single shapes on a bare canvas; the ones marked
    //! `ignore` paint whole film frames and want a release build:
    //!   cargo test --release -- --include-ignored
    use super::*;

    /// `OIL_TEST_BACKEND=gpu` (with the `gpu` feature) runs the tests on the GPU backend: the test canvas writes its
    /// strokes down and `done` lays them on the GPU before anything is measured.
    fn test_gpu() -> bool {
        cfg!(feature = "gpu") && std::env::var("OIL_TEST_BACKEND").is_ok_and(|v| v == "gpu")
    }
    fn bare() -> Painting {
        ZOOM.with(|z| z.set(1.0));
        let mut p = Painting::new();
        if test_gpu() {
            p.rec = Some(Box::new(gpu::Recorder::new(p.col.clone())));
        }
        p
    }
    fn done(mut p: Painting) -> Painting {
        finish(&mut p);
        p
    }

    fn dusk(r: &mut Rec, _t: f64, _lt: f64) {
        r.boil("sky");
        r.paint("sky", &rect_pts(-40.0, -40.0, 2000.0, 700.0), wash(rgb(0x22325F)).no_ink());
        r.boil("hill");
        r.paint("hill", &[[-40.0, 760.0], [600.0, 640.0], [1300.0, 720.0], [1960.0, 650.0], [1960.0, 1120.0], [-40.0, 1120.0]], wash(rgb(0x3E4A6E)).no_ink());
        r.boil("moon");
        r.paint("moon", &ell_pts(1480.0, 300.0, 130.0, 130.0, 32, 0.0), wash(rgb(0xF2EEDC)).no_ink());
        r.boil("owl");
        r.paint("owl", &ell_pts(760.0, 520.0, 120.0, 150.0, 28, 0.0), wash(rgb(0xB5652F)));
        r.paint("owl", &ell_pts(760.0, 560.0, 72.0, 100.0, 24, 0.0), wash(rgb(0xE8D2A8)).no_ink());
    }

    #[test]
    fn the_gpu_paints_what_the_cpu_paints() {
        // With `OIL_TEST_BACKEND=gpu`: a whole frame laid on the GPU agrees with the CPU's to the engine's GPU
        // tolerance (README, "CPU and GPU"): the GPU's own sin, exp and division move a handful of values by a few
        // levels, and very rarely a pixel right at a threshold falls the other way. At most 0.01 % of the 8-bit
        // values may differ at all, and at most 0.001 % by more than 5. (The canvas page's webcheck --compare holds
        // the browser's WebGPU to the same.)
        if !test_gpu() {
            return;
        }
        let film = Film::new("dusk", 1.0, vec![Shot::new(0.0, "dusk", dusk)], "/nonexistent/dusk/film");
        set_style(Style { cast: &["owl"], ..Style::DEFAULT });
        let cpu = plan_frame(&film, 0.0, 1, false).lit();
        let mut gpu = plan_frame(&film, 0.0, 1, true);
        assert!(gpu.rec.is_some(), "the frame was not recorded for the GPU");
        finish(&mut gpu);
        let gpu = gpu.lit();
        let (mut diff, mut big, mut max) = (0usize, 0usize, 0u8);
        for (a, b) in cpu.iter().zip(&gpu) {
            let d = a.abs_diff(*b);
            diff += (d > 0) as usize;
            big += (d > 5) as usize;
            max = max.max(d);
        }
        eprintln!("GPU vs CPU: {diff} of {} values differ, {big} by more than 5 levels, largest {max}", cpu.len());
        assert!(diff * 10_000 <= cpu.len() && big * 100_000 <= cpu.len(), "{diff} values differ, {big} by more than 5 levels");
    }
    fn band(z: f32, dy: f32) -> Vec<[f32; 2]> {
        // the sky band of the frame that flickered (sky.b#1 at t = 19.125 s): very wide, its top edge on screen,
        // seen through a camera zooming slowly about the frame's centre
        let raw = [[-960.0, 194.0 + dy], [2840.0, 194.0 + dy], [2840.0, 1990.0], [-960.0, 1990.0]];
        raw.iter().map(|q| [960.0 + (q[0] - 960.0) * z, 540.0 + (q[1] - 540.0) * z]).collect()
    }
    /// The largest change of any 24 x 24 block's mean colour between two paintings, within rows y0..y1. A stroke
    /// that pops in or out moves a block by several percent; brushwork that only travels with its shape does not.
    fn block_peak(a: &Painting, b: &Painting, y0: usize, y1: usize) -> f32 {
        let mut peak = 0.0f32;
        for by in (y0..y1 - 23).step_by(24) {
            for bx in (0..WI - 23).step_by(24) {
                let mut d = [0.0f32; 3];
                for y in by..by + 24 {
                    for x in bx..bx + 24 {
                        for k in 0..3 {
                            d[k] += a.col[y * WI + x][k] - b.col[y * WI + x][k];
                        }
                    }
                }
                peak = peak.max(d.iter().map(|v| (v / 576.0).abs()).fold(0.0, f32::max));
            }
        }
        peak
    }

    #[test]
    fn a_slow_zoom_moves_no_stroke_in_or_out_at_an_edge() {
        // Frames 458 -> 459 of the film: the camera zoomed 1.0218 -> 1.0220 and one whole sky stroke (about 600 px
        // long) appeared at the band's edge, because its slot centre crossed the rasterised mask's 0.5 line.
        let (z0, z1) = (1.0218f32, 1.0220f32);
        let col = [0.55, 0.35, 0.6];
        let mut worst = 0.0f32;
        // a few sub-pixel phases, so the test does not hinge on one lucky alignment
        for (k, dy) in [0.0f32, 0.37, 0.71, 1.13].iter().enumerate() {
            let paint = |z: f32| {
                let mut p = bare();
                ZOOM.with(|zz| zz.set(z));
                paint_body(&mut p, &band(z, *dy), col, col, 0.97, 0x8504_0000 ^ k as u32, 1.25, 0.7, Tool::Flat);
                done(p)
            };
            let (a, b) = (paint(z0), paint(z1));
            worst = worst.max(block_peak(&a, &b, 120, 420));
        }
        assert!(worst < 0.02, "a 0.02 % zoom changed a 24 px block of the band's edge by {:.1} %", worst * 100.0);
    }

    #[test]
    fn stroke_slots_fade_in_and_out_at_an_edge_never_pop() {
        // The root cause, at the level of the stroke plan: slide the band's top edge down 120 px in 0.05 px steps
        // (the size of one frame's camera creep) and follow every stroke id. A stroke may only arrive, leave or
        // change its weight a little per step. With the old hard mask test, a slot's weight went 0 -> 1 at once.
        let width = 112.4f32;
        let base = -1.1781f32;
        // a stroke reaches under 4 brush widths from its slot: only those that can reach the canvas count
        let core = |q: [f32; 2]| q[0] > -4.0 * width && q[1] > -4.0 * width && q[0] < WI as f32 + 4.0 * width && q[1] < HI as f32 + 4.0 * width;
        let mut prev: std::collections::HashMap<u32, (f32, [f32; 2])> = Default::default();
        let mut worst = (0.0f32, 0u32, 0.0f32);
        for k in 0..2400 {
            let dy = k as f32 * 0.05;
            let pts = band(1.0, dy);
            let c = [940.0, (194.0 + dy + 1990.0) / 2.0];
            let lo = [-80.0f32, 194.0 + dy];
            let hi = [WI as f32 + 80.0, HI as f32 + 80.0];
            let now: std::collections::HashMap<u32, (f32, [f32; 2])> =
                plan_slots(&pts, c, lo, hi, width, base, 0x8504_0000, 1.0).into_iter().map(|(q, id, e)| (id, (e, q))).collect();
            if k > 0 {
                for (id, (e, q)) in &now {
                    let d = match prev.get(id) {
                        Some((e0, _)) => (e - e0).abs(),
                        None if core(*q) => *e,
                        None => 0.0,
                    };
                    if d > worst.0 {
                        worst = (d, *id, dy);
                    }
                }
                for (id, (e, q)) in &prev {
                    if !now.contains_key(id) && core(*q) && *e > worst.0 {
                        worst = (*e, *id, dy);
                    }
                }
            }
            prev = now;
        }
        assert!(worst.0 < 0.02, "stroke {:08x} jumped by {:.2} of its weight at dy = {:.2}", worst.1, worst.0, worst.2);
    }

    #[test]
    fn a_strokes_edge_direction_never_flips_between_two_equally_near_edges() {
        // A narrow trunk whose two sides lean differently (frames 472 -> 473: one long bark stroke swung as the
        // nearest side changed from left to right). Sliding the shape sideways must turn the direction smoothly.
        let mut worst = 0.0f32;
        let mut last: Option<f32> = None;
        for k in 0..400 {
            let dx = k as f32 * 0.05;
            let pts = vec![[300.0 + dx, 100.0], [420.0 + dx, 100.0], [440.0 + dx, 1000.0], [290.0 + dx, 1000.0]];
            let (_, a) = edge_dir(&pts, [365.0 + 10.0, 550.0], 110.0);
            if let Some(b) = last {
                let mut d = (a - b).abs() % std::f32::consts::PI;
                d = d.min(std::f32::consts::PI - d);
                worst = worst.max(d);
            }
            last = Some(a);
        }
        assert!(worst < 0.01, "the edge direction jumped {:.3} rad for a 0.05 px move", worst);
    }

    /// How strongly the picture has straight seams running at right angles to level brushwork. A seam is many
    /// rows stepping in colour at the SAME column: for every run of `run` rows, each column's median horizontal step
    /// is compared with the mean of its neighbours 4-16 px either side; the score is the largest excess, against
    /// the region's mean step. Ragged, oblique stroke ends spread their steps over many columns and score low; a
    /// stroke that starts or stops on a straight edge square to its direction makes one column stand out.
    fn seam_score(p: &Painting, x0: usize, x1: usize, y0: usize, y1: usize, run: usize) -> f32 {
        let l = |x: usize, y: usize| {
            let c = p.col[y * WI + x];
            0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
        };
        let (w, h) = (x1 - x0, y1 - y0);
        let g: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).map(|(y, x)| (l(x0 + x + 1, y0 + y) - l(x0 + x, y0 + y)).abs()).collect();
        let mean = g.iter().sum::<f32>() / g.len() as f32;
        let mut worst = 0.0f32;
        for ys in (0..h - run).step_by(run / 3) {
            // the median over the run: a seam steps in most of its rows, a speck or a glint in one or two
            let col: Vec<f32> = (0..w)
                .map(|x| {
                    let mut v: Vec<f32> = (ys..ys + run).map(|y| g[y * w + x]).collect();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[run / 2]
                })
                .collect();
            for x in 16..w - 16 {
                let side = (4..=16).map(|d| col[x - d] + col[x + d]).sum::<f32>() / 26.0;
                worst = worst.max(col[x] - side);
            }
        }
        worst / mean.max(1e-6)
    }

    #[test]
    fn level_strokes_end_raggedly_not_on_a_vertical_seam() {
        // The owner's screenshot (the flight shot, t = 71.5 s): a night sky of level flat-brush strokes, cut all
        // over by hard vertical edges where strokes started and stopped square to their direction. Lay level
        // strokes on a flat ground and look, at each end, for one column where most rows step at once.
        let mut worst = 0.0f32;
        for k in 0..8u32 {
            let mut p = bare();
            for c in p.col.iter_mut().chain(p.flat.iter_mut()) {
                *c = [0.10, 0.12, 0.30];
            }
            if let Some(r) = p.rec.as_mut() {
                r.ground = p.col.clone();
            }
            let (len, width) = (520.0f32, 110.0f32);
            p.stroke(
                &Stroke {
                    c: [960.0, 540.0],
                    angle: 0.0,
                    len,
                    width,
                    bend: 0.0,
                    col: [0.25, 0.35, 0.75],
                    alt: [0.3, 0.4, 0.8],
                    opacity: 0.97,
                    pickup: 0.0,
                    load: 1.0,
                    part: Some((h32(0x5EA0 + k) >> 2, true, true)),
                    span: (0.0, 1.0),
                    tool: Tool::Flat,
                    picked: None,
                    along: 0.0,
                },
                None,
            );
            let p = done(p);
            let (x0, x1) = (960 - (len / 2.0) as usize, 960 + (len / 2.0) as usize);
            let (y0, y1) = (540 - 36, 540 + 36);
            let start = seam_score(&p, x0 - 40, x0 + 140, y0, y1, 60);
            let end = seam_score(&p, x1 - 280, x1 + 20, y0, y1, 60);
            if std::env::var("SEAM_PNG").is_ok() {
                eprintln!("stroke {k}: start {start:.2} end {end:.2}");
            }
            worst = worst.max(start).max(end);
        }
        assert!(worst < 6.0, "a stroke end makes a straight vertical seam ({:.1}x the mean step)", worst);
    }

    #[test]
    fn an_outline_that_flexes_keeps_its_sweeps() {
        // Frames 587 -> 588 -> 589: the owl's head outline went from 9 sweeps to 10 and back as it breathed, and
        // since sweeps were seeded by their place in the list every later sweep was re-seeded. Flex an outline in
        // tiny steps: every sweep keeps its identity, and its share may only change a little per step.
        ZOOM.with(|z| z.set(1.0));
        let shape = |sy: f32| -> Vec<[f32; 2]> {
            (0..40)
                .map(|i| {
                    let a = i as f32 / 40.0 * std::f32::consts::TAU;
                    [960.0 + 160.0 * a.cos() * (1.0 + 0.08 * (3.0 * a).sin()), 540.0 + 130.0 * sy * a.sin()]
                })
                .collect()
        };
        let mut prev: std::collections::HashMap<u32, f32> = Default::default();
        let mut worst = (0.0f32, 0usize);
        for k in 0..600 {
            let sy = 1.0 + k as f32 * 0.0005;
            let now: std::collections::HashMap<u32, f32> =
                hand_paths(&shape(sy), true, 9.0, 0.0, 0xC0FFEE, HAND_OUTLINE).into_iter().map(|(_, w, id)| (id, w)).collect();
            if k > 0 {
                for (id, w) in &now {
                    let d = (w - prev.get(id).copied().unwrap_or(0.0)).abs();
                    if d > worst.0 {
                        worst = (d, k);
                    }
                }
                for (id, w) in &prev {
                    if !now.contains_key(id) && *w > worst.0 {
                        worst = (*w, k);
                    }
                }
            }
            prev = now;
        }
        assert!(worst.0 < 0.1, "a sweep's share jumped by {:.2} at step {}", worst.0, worst.1);
    }

    /// A one-shot film: a night sky with one pale disc whose radius grows with t, from a speck to a moon.
    fn disc(r: &mut Rec, t: f64, _lt: f64) {
        r.boil("sky");
        r.paint("sky", &rect_pts(-100.0, -100.0, 2120.0, 1280.0), wash(rgb(0x14204A)).no_ink());
        r.boil("moon");
        let rad = 3.0 + 37.0 * t;
        r.paint("moon", &ell_pts(960.0, 540.0, rad, rad, 28, 0.0), wash(rgb(0xF2EEDC)).no_ink());
    }

    #[test]
    fn a_small_shape_growing_through_every_size_threshold_never_pops() {
        // The owl film's moon (frames 2829-2832) turned into a knife-touch slab for two frames, because a small
        // light shape is painted as a highlight below one size and as a body above it. The same kind of switch
        // sat at the size of a speck and at the size that gets a cut-in edge. Grow a disc from 6 px to 80 px
        // across in small steps: the painting may change only as much as the growth does, never in one jump.
        let film = Film::new("disc", 1.0, vec![Shot::new(0.0, "disc", disc)], "/nonexistent/disc/film");
        set_style(Style::DEFAULT);
        if test_gpu() {
            set_backend(Backend::Gpu);
        }
        let n = 90;
        // per step, the mean absolute change of every pixel around the disc: growth only changes a thin ring at
        // its edge, a change of treatment repaints its inside
        let frames: Vec<Vec<[f32; 3]>> = (0..=n).map(|i| paint_frame_painting(&film, i as f64 / n as f64, 1).col).collect();
        let steps: Vec<f32> = frames
            .windows(2)
            .map(|w| {
                let mut sum = 0.0f32;
                for y in 480..600 {
                    for x in 900..1020 {
                        let (a, b) = (w[0][y * WI + x], w[1][y * WI + x]);
                        sum += (0..3).map(|k| (a[k] - b[k]).abs()).sum::<f32>() / 3.0;
                    }
                }
                sum / (120.0 * 120.0)
            })
            .collect();
        if std::env::var("DISC_STEPS").is_ok() {
            eprintln!("{:?}", steps.iter().map(|s| (s * 1e4).round() as i32).collect::<Vec<_>>());
        }
        // the growth's own steps get bigger with the disc, so each step is held to the steps around it
        let mut worst = (0.0f32, 0usize);
        for i in 3..steps.len() - 3 {
            let around = (steps[i - 3] + steps[i - 2] + steps[i - 1] + steps[i + 1] + steps[i + 2] + steps[i + 3]) / 6.0;
            let r = steps[i] / (around + 0.0005);
            if r > worst.0 {
                worst = (r, i);
            }
        }
        assert!(
            worst.0 < 2.5,
            "radius {:.1} px: one step changed the disc {:.1}x as much as the steps around it",
            3.0 + 37.0 * worst.1 as f32 / n as f32,
            worst.0
        );
    }

    /// A one-shot film: a dusk sky and one long strap (outlined, as films draw them) whose breadth grows with t,
    /// from a line 3 px across to an area 60 px across.
    fn strap(r: &mut Rec, t: f64, _lt: f64) {
        r.boil("sky");
        r.paint("sky", &rect_pts(-100.0, -100.0, 2120.0, 1280.0), wash(rgb(0x2C3A6A)).no_ink());
        r.boil("strap");
        let w = 3.0 + 57.0 * t;
        let (a, b) = ([700.0, 300.0], [1250.0, 800.0]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = f64::hypot(dx, dy);
        let n = [-dy / l * w / 2.0, dx / l * w / 2.0];
        r.paint(
            "strap",
            &[[a[0] + n[0], a[1] + n[1]], [b[0] + n[0], b[1] + n[1]], [b[0] - n[0], b[1] - n[1]], [a[0] - n[0], a[1] - n[1]]],
            wash(rgb(0xC9A27A)),
        );
    }

    #[test]
    fn a_strap_widening_from_a_line_to_an_area_never_pops() {
        // A thin shape is painted as one pull of a fine brush with no room for an outline down each side; a broad
        // one as an area with its outline and its cut-in edge. Widen a slanted strap through all of that in small
        // steps: the painting may change only as much as the widening does, never in one jump.
        let film = Film::new("strap", 1.0, vec![Shot::new(0.0, "strap", strap)], "/nonexistent/strap/film");
        set_style(Style::DEFAULT);
        if test_gpu() {
            set_backend(Backend::Gpu);
        }
        let n = 90;
        let frames: Vec<Vec<[f32; 3]>> = (0..=n).map(|i| paint_frame_painting(&film, i as f64 / n as f64, 1).col).collect();
        let steps: Vec<f32> = frames
            .windows(2)
            .map(|w| {
                let mut sum = 0.0f32;
                for y in (300..800).step_by(2) {
                    for x in (650..1300).step_by(2) {
                        let (a, b) = (w[0][y * WI + x], w[1][y * WI + x]);
                        sum += (0..3).map(|k| (a[k] - b[k]).abs()).sum::<f32>() / 3.0;
                    }
                }
                sum / (250.0 * 325.0)
            })
            .collect();
        let mut worst = (0.0f32, 0usize);
        for i in 3..steps.len() - 3 {
            let around = (steps[i - 3] + steps[i - 2] + steps[i - 1] + steps[i + 1] + steps[i + 2] + steps[i + 3]) / 6.0;
            let r = steps[i] / (around + 0.0005);
            if r > worst.0 {
                worst = (r, i);
            }
        }
        assert!(
            worst.0 < 2.5,
            "breadth {:.1} px: one step changed the strap {:.1}x as much as the steps around it",
            3.0 + 57.0 * worst.1 as f32 / n as f32,
            worst.0
        );
    }

    fn strips(r: &mut Rec, _t: f64, _lt: f64) {
        r.boil("sky");
        r.paint("sky", &rect_pts(-100.0, -100.0, 2120.0, 1280.0), wash(rgb(0x2C3A6A)).no_ink());
        r.boil("strips");
        for i in 0..6 {
            let x = 700.0 + i as f64 * 14.0;
            r.paint("strip", &[[x, 300.0], [x + 6.0, 300.0], [x + 306.0, 800.0], [x + 300.0, 800.0]], wash(rgb(0xC9A27A)));
        }
    }

    #[test]
    fn close_thin_strips_stay_apart() {
        // Twigs, straps and whiskers drawn as thin shapes a few px apart ran together into one smear: each got a
        // brush as wide as its slanted bounding box, spilling far past its sides, and an outline down each side as
        // wide as itself. The gap between two such strips must stay the colour of what is behind them.
        let film = Film::new("strips", 1.0, vec![Shot::new(0.0, "strips", strips)], "/nonexistent/strips/film");
        set_style(Style::DEFAULT);
        if test_gpu() {
            set_backend(Backend::Gpu);
        }
        let p = paint_frame_painting(&film, 0.0, 1);
        let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
        let sky = lum(rgb01(rgb(0x2C3A6A)));
        let (mut gap, mut core, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for y in (400..700).step_by(5) {
            let t = (y as f32 - 300.0) / 500.0;
            for i in 0..5 {
                // the middle of a strip, and the middle of the gap after it
                let x = 700.0 + i as f32 * 14.0 + 300.0 * t;
                core += lum(p.col[y * WI + (x + 3.0) as usize]);
                gap += lum(p.col[y * WI + (x + 10.0) as usize]);
                n += 1.0;
            }
        }
        let (gap, core) = (gap / n, core / n);
        assert!(core - sky > 0.15, "the strips themselves should be painted ({core:.3} against the sky's {sky:.3})");
        assert!((gap - sky).abs() < 0.35 * (core - sky), "the gaps filled in: {gap:.3}, sky {sky:.3}, strips {core:.3}");
    }

    /// A one-shot film under moonlight: a cloth in bands of colour, a post standing on it, and the post's shadow
    /// across the bands (drawn as films draw shadows, a dark translucent shape), at `t` of its full strength; and a
    /// moon with its glow, whose glow grows with `t`.
    fn shadowed(r: &mut Rec, t: f64, _lt: f64) {
        r.light(Lighting::MOONLIGHT.from_dir([0.75, -0.66]));
        r.boil("ground");
        r.paint("ground", &rect_pts(-60.0, -60.0, 2040.0, 1200.0), wash(rgb(0x5A6E4A)).no_ink());
        r.boil("cloth");
        for (i, c) in [0xD9663A, 0xE8B866, 0x3E6E9E, 0xF2EEDC].iter().enumerate() {
            r.paint("band", &rect_pts(300.0 + i as f64 * 150.0, 200.0, 150.0, 700.0), wash(rgb(*c)).no_ink());
        }
        r.boil("post");
        r.paint("post", &[[1000.0, 160.0], [1060.0, 160.0], [1070.0, 760.0], [990.0, 760.0]], wash(rgb(0x6B4A3A)));
        r.boil("post.shadow");
        r.paint("shadow", &[[1000.0, 760.0], [1070.0, 760.0], [420.0, 880.0], [300.0, 860.0]], wash(rgb(0x1D1A1F)).op(255.0 * t).no_ink().shadow_of("post"));
        r.boil("sky");
        r.paint("sky", &rect_pts(1300.0, -60.0, 700.0, 1200.0), wash(rgb(0x14204A)).no_ink());
        r.glows("moon.glow", &[(1650.0, 400.0, 40.0 + 300.0 * t, rgb(0xBFD0FF), 0.4)]);
        r.boil("moon");
        r.paint("moon", &ell_pts(1650.0, 400.0, 90.0, 90.0, 32, 0.0), wash(rgb(0xF5F0DC)).no_ink().role(Role::Light));
    }

    #[test]
    fn a_shadow_is_a_glaze_that_keeps_what_it_falls_on() {
        // The owl's "very deep shadows": a shadow was painted as opaque dark paint (half of whatever was under it),
        // a near-black patch whatever it fell on. A shadow is a glaze: each band it crosses stays its own colour,
        // darker by a step, and the bands stay as different from each other as they were.
        let film = Film::new("shadowed", 1.0, vec![Shot::new(0.0, "s", shadowed)], "/nonexistent/shadowed/film");
        set_style(Style { cast: &["post"], ..Style::DEFAULT });
        if test_gpu() {
            set_backend(Backend::Gpu);
        }
        let lit = paint_frame_painting(&film, 0.0, 1);
        let shaded = paint_frame_painting(&film, 1.0, 1);
        // along the middle of the shadow
        let mean = |p: &Painting, x0: usize| {
            let yc = (760.0 + (1035.0 - (x0 as f32 + 10.0)) * 110.0 / 675.0) as usize;
            let mut m = [0.0f32; 3];
            for y in yc - 5..yc + 5 {
                for x in x0..x0 + 20 {
                    for k in 0..3 {
                        m[k] += p.col[y * WI + x][k] / 200.0;
                    }
                }
            }
            m
        };
        let mut lums = vec![];
        for i in 0..4 {
            let x0 = 365 + i * 150;
            let (a, b) = (mean(&lit, x0), mean(&shaded, x0));
            let (la, lb) = (lum3(a), lum3(b));
            assert!(lb < la * 0.9 && lb > la * 0.35, "band {i}: in shadow {lb:.3}, in light {la:.3}: a shadow is a step down, not black");
            // its hue survives: the strongest channel in the light is still the strongest in shadow
            let top = |c: [f32; 3]| (0..3).max_by(|&p, &q| c[p].total_cmp(&c[q])).unwrap();
            assert!(la - lum3([a[0].min(a[1]).min(a[2]); 3]) < 0.05 || top(a) == top(b), "band {i}: its colour turned from {a:?} to {b:?}");
            lums.push(lb);
        }
        let spread = lums.iter().cloned().fold(0.0f32, f32::max) - lums.iter().cloned().fold(1.0f32, f32::min);
        assert!(spread > 0.15, "the bands ran together in shadow ({lums:?})");
    }

    #[test]
    fn a_shadow_fading_in_and_a_glow_growing_never_pop() {
        let film = Film::new("shadowed", 1.0, vec![Shot::new(0.0, "s", shadowed)], "/nonexistent/shadowed/film");
        set_style(Style { cast: &["post"], ..Style::DEFAULT });
        if test_gpu() {
            set_backend(Backend::Gpu);
        }
        let n = 40;
        let frames: Vec<Vec<[f32; 3]>> = (0..=n).map(|i| paint_frame_painting(&film, i as f64 / n as f64, 1).col).collect();
        for (x0, x1, y0, y1, what) in [(280, 1080, 740, 900, "shadow"), (1300, 1920, 60, 740, "glow")] {
            let steps: Vec<f32> = frames
                .windows(2)
                .map(|w| {
                    let mut sum = 0.0f32;
                    for y in (y0..y1).step_by(2) {
                        for x in (x0..x1).step_by(2) {
                            sum += (0..3).map(|k| (w[0][y * WI + x][k] - w[1][y * WI + x][k]).abs()).sum::<f32>();
                        }
                    }
                    sum / ((y1 - y0) * (x1 - x0)) as f32
                })
                .collect();
            if std::env::var("STEPS").is_ok() {
                eprintln!("{what}: {:?}", steps.iter().map(|s| (s * 1e4).round() as i32).collect::<Vec<_>>());
            }
            for i in 2..steps.len() - 2 {
                let around = (steps[i - 2] + steps[i - 1] + steps[i + 1] + steps[i + 2]) / 4.0;
                assert!(steps[i] < 2.5 * around + 0.002, "the {what}: step {i} changed it {:.1}x as much as the steps around it", steps[i] / around.max(1e-6));
            }
        }
    }

    #[test]
    fn same_shape_same_place_paints_identically() {
        let pts: Vec<[f32; 2]> = (0..24)
            .map(|i| {
                let a = i as f32 / 24.0 * std::f32::consts::TAU;
                [900.0 + 210.0 * a.cos(), 520.0 + 150.0 * a.sin()]
            })
            .collect();
        let paint = || {
            let mut p = bare();
            paint_form(&mut p, &pts, [0.8, 0.45, 0.2], [0.9, 0.7, 0.4], 0.97, 0xBEEF, 0.8, 1.15, Tool::Filbert, 1.0);
            done(p)
        };
        let (a, b) = (paint(), paint());
        assert!(a.col == b.col && a.thick == b.thick, "painting the same frame twice must give the same picture");
    }

    #[test]
    fn a_shape_that_swells_a_little_keeps_its_brushwork() {
        // The brush used to snap to steps of 1.22x and the stroke direction to steps of 7.5 degrees: a shape
        // crossing a step boundary re-planned every stroke at once. Swelling by 0.5 % must change little.
        let shape = |s: f32, turn: f32| -> Vec<[f32; 2]> {
            (0..20)
                .map(|i| {
                    let a = i as f32 / 20.0 * std::f32::consts::TAU;
                    let (x, y) = (60.0 * s * a.cos(), 25.0 * s * a.sin());
                    let (sn, cs) = turn.sin_cos();
                    [960.0 + x * cs - y * sn, 540.0 + x * sn + y * cs]
                })
                .collect()
        };
        let mut worst = 0.0f32;
        for k in 0..12 {
            // sweep sizes and turns across where the old steps sat
            let (s, turn) = (0.9 + k as f32 * 0.07, -1.2 + k as f32 * 0.21);
            let paint = |s: f32, turn: f32| {
                let mut p = bare();
                paint_form(&mut p, &shape(s, turn), [0.8, 0.45, 0.2], [0.9, 0.7, 0.4], 0.97, 0xFACE, 0.8, 1.15, Tool::Filbert, 0.0);
                done(p)
            };
            worst = worst.max(block_peak(&paint(s, turn), &paint(s * 1.005, turn + 0.002), 420, 660));
        }
        assert!(worst < 0.06, "0.5 % of swelling changed a block by {:.1} %", worst * 100.0);
    }

    #[test]
    fn a_shape_reaching_far_off_canvas_is_still_painted_where_it_shows() {
        // An iris wedge / whip streak: its centroid is thousands of pixels off screen, its visible part is not.
        let pts = vec![[300.0, 200.0], [700.0, 260.0], [4300.0, 2000.0], [-3000.0, 3800.0]];
        let mut p = bare();
        let col = [0.1, 0.08, 0.2];
        paint_body(&mut p, &pts, col, col, 0.97, 0x1215, 1.25, 0.7, Tool::Flat);
        let p = done(p);
        let (mut inside, mut dark) = (0, 0);
        for y in (300..1080).step_by(7) {
            for x in (200..900).step_by(7) {
                if in_poly(&pts, [x as f32, y as f32]) && signed_depth(&pts, [x as f32, y as f32], 60.0) >= 60.0 {
                    inside += 1;
                    let c = p.col[y * WI + x];
                    dark += (c[0] + c[1] + c[2] < 0.9) as u32;
                }
            }
        }
        assert!(inside > 100 && dark as f32 > 0.9 * inside as f32, "only {dark} of {inside} samples inside the visible part were painted");
    }

    #[test]
    fn a_fade_fades_and_leaves_without_a_pop() {
        // fx::flash / world::fade: the film's own transition, glazed at exactly the film's strength
        let base = {
            let mut p = bare();
            let col = [0.8, 0.5, 0.3];
            paint_body(&mut p, &[[400.0, 300.0], [1500.0, 300.0], [1500.0, 800.0], [400.0, 800.0]], col, col, 0.97, 7, 1.25, 0.7, Tool::Flat);
            done(p)
        };
        let night = [0.05, 0.07, 0.2];
        let frame = |k: f32| {
            let mut p = Painting::with_layers(base.col.clone(), base.flat.clone(), base.thick.clone());
            glaze(&mut p, night, k);
            done(p)
        };
        let j = 540 * WI + 960;
        for k in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let c = frame(k).col[j];
            for q in 0..3 {
                let want = base.col[j][q] + (night[q] - base.col[j][q]) * k;
                assert!((c[q] - want).abs() < 1e-4, "at strength {k} the fade should be exactly that strong");
            }
        }
        // the last frame of a fade-out (k just above the recorder's 0.01 cut) is all but invisible
        assert!(block_peak(&frame(0.011), &frame(0.0), 0, HI) < 0.012);
        // and the overlay is recognised as one
        assert!(covers_canvas(&[[-60.0, -60.0], [1980.0, -60.0], [1980.0, 1140.0], [-60.0, 1140.0]]));
        // fx::flash draws its rectangle with the kit's rect_pts: eight points, the midpoints included
        let flash = rect_pts(-60.0, -60.0, 1920.0 + 120.0, 1080.0 + 120.0).iter().map(|q| [q[0] as f32, q[1] as f32]).collect::<Vec<_>>();
        assert!(covers_canvas(&flash));
        assert!(!covers_canvas(&[[0.0, 0.0], [1000.0, 0.0], [1000.0, 1080.0], [0.0, 1080.0]]));
    }

    #[test]
    fn a_translucent_wash_is_mixed_down_not_glazed_in_streaks() {
        // an eye's shade at 90/255: the paint is the shade let down with what it lands on, laid opaquely
        let under = [0.85, 0.7, 0.2];
        let shade = [0.4, 0.25, 0.05];
        let sq = [[800.0, 400.0], [1100.0, 400.0], [1100.0, 700.0], [800.0, 700.0]];
        let paint = |op: f32| {
            let mut p = bare();
            paint_body(&mut p, &sq, under, under, 0.97, 11, 1.25, 0.7, Tool::Flat);
            let mut p = done(p);
            let before = p.col.clone();
            {
                let _m = Premix::set(op);
                paint_body(&mut p, &sq, shade, shade, 0.97 * smooth(0.0, 0.12, op), 12, 1.25, 0.7, Tool::Flat);
            }
            (before, p)
        };
        let (_, p) = paint(90.0 / 255.0);
        let mean = |p: &Painting| {
            let mut m = [0.0f32; 3];
            for y in 480..620 {
                for x in 880..1020 {
                    for k in 0..3 {
                        m[k] += p.col[y * WI + x][k] / (140.0 * 140.0);
                    }
                }
            }
            m
        };
        let m = mean(&p);
        let want: [f32; 3] = std::array::from_fn(|k| under[k] + (shade[k] - under[k]) * (90.0 / 255.0));
        for k in 0..3 {
            assert!((m[k] - want[k]).abs() < 0.08, "channel {k}: {:.3} vs the mixed tint {:.3}", m[k], want[k]);
        }
        // as the film's opacity runs out the wash disappears smoothly
        // (just above the 0.004 below which a wash is left out: its last frame must be next to nothing)
        let (before, faint) = paint(0.005);
        let base = Painting::with_layers(before, faint.flat.clone(), faint.thick.clone());
        let pk = block_peak(&faint, &base, 360, 720);
        assert!(pk < 0.01, "a 1 % wash moved a block by {:.2} %", pk * 100.0);
        assert_eq!(PREMIX.with(|p| p.get()), 1.0, "the tint must not leak into the next shape");
    }

    #[test]
    fn the_boil_repaints_at_its_rate_and_only_then() {
        // N = 0: one painting for the whole clip
        assert!((0..500).all(|f| clip_seed(f, 0) == 1));
        // N = 6: a fresh painting exactly every 6 frames (4 a second), held in between
        for f in 0..120u32 {
            assert_eq!(clip_seed(f, 6) != clip_seed(f + 1, 6), (f + 1) % 6 == 0, "frame {f}");
        }
        // and a fresh seed really is a fresh painting, while the same seed is the same one
        let pts = [[700.0, 400.0], [1200.0, 380.0], [1250.0, 700.0], [680.0, 720.0]];
        let paint = |seed: u32| {
            let mut p = bare();
            paint_body(&mut p, &pts, [0.3, 0.5, 0.7], [0.4, 0.6, 0.8], 0.97, seed.wrapping_mul(7919) ^ 0x51, 1.25, 0.7, Tool::Flat);
            done(p)
        };
        let (a, a2, b) = (paint(clip_seed(5, 6)), paint(clip_seed(4, 6)), paint(clip_seed(6, 6)));
        assert_eq!(block_peak(&a, &a2, 360, 720), 0.0);
        assert!(block_peak(&a, &b, 360, 720) > 0.02, "a boil step must visibly repaint");
    }
}
