//! The brush engine's helpers (engines/brush/core.js), ported one to one: timing, easing, motion principles,
//! geometry and colour. Everything is a pure function of its arguments, so every frame is a pure function of t.

use std::f64::consts::PI;

/// The canvas: 1920×1080, origin top-left, y down.
pub const W: f64 = 1920.0;
pub const H: f64 = 1080.0;
pub const TAU: f64 = PI * 2.0;
/// Frames per second of every studio film.
pub const FPS: f64 = 24.0;
/// Boil drawings per second (linework is redrawn on twos).
pub const BOIL: f64 = 12.0;

pub type P = [f64; 2];
/// sRGB, 0–255 per channel.
pub type Col = [f64; 3];

// ---------- numbers ----------
pub fn clamp(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
pub fn clamp_to(x: f64, a: f64, b: f64) -> f64 {
    x.max(a).min(b)
}
pub fn lerp(a: f64, b: f64, x: f64) -> f64 {
    a + (b - a) * x
}
pub fn frac(x: f64) -> f64 {
    x - x.floor()
}
/// A stable pseudo-random value in [0, 1) per input: use it for anything that must not boil (star positions,
/// tuft heights, a patron's colour).
pub fn hash(i: f64) -> f64 {
    let x = (i * 127.1 + 311.7).sin() * 43758.5453;
    x - x.floor()
}
/// 0..1 progress of t through [a, b].
pub fn seg(t: f64, a: f64, b: f64) -> f64 {
    clamp((t - a) / (b - a))
}
/// A sine wobble: `sin((t f + ph) τ)`.
pub fn wob(t: f64, f: f64, ph: f64) -> f64 {
    ((t * f + ph) * TAU).sin()
}

// ---------- easing ----------
pub fn ease(x: f64) -> f64 {
    let x = clamp(x);
    x * x * (3.0 - 2.0 * x)
}
pub fn ease_out(x: f64) -> f64 {
    1.0 - (1.0 - clamp(x)).powi(3)
}
pub fn ease_in(x: f64) -> f64 {
    clamp(x).powi(3)
}
/// Overshoots and settles.
pub fn back_out(x: f64) -> f64 {
    let x = clamp(x);
    let s = 1.9;
    1.0 + (s + 1.0) * (x - 1.0).powi(3) + s * (x - 1.0).powi(2)
}
pub fn elastic_out(x: f64) -> f64 {
    let x = clamp(x);
    if x == 0.0 || x == 1.0 { x } else { 2f64.powf(-10.0 * x) * ((x * 10.0 - 0.75) * (TAU / 3.0)).sin() + 1.0 }
}
pub fn linear(x: f64) -> f64 {
    clamp(x)
}

/// kf(t, [(t0, v0), (t1, v1), …], ease): scalar keyframes, eased between keys and held outside them.
pub fn kf(t: f64, keys: &[(f64, f64)], e: fn(f64) -> f64) -> f64 {
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let ((a, va), (b, vb)) = (w[0], w[1]);
        if t < b {
            return lerp(va, vb, e((t - a) / (b - a)));
        }
    }
    keys[keys.len() - 1].1
}
/// kf for points.
pub fn kf2(t: f64, keys: &[(f64, P)], e: fn(f64) -> f64) -> P {
    let x: Vec<(f64, f64)> = keys.iter().map(|(k, v)| (*k, v[0])).collect();
    let y: Vec<(f64, f64)> = keys.iter().map(|(k, v)| (*k, v[1])).collect();
    [kf(t, &x, e), kf(t, &y, e)]
}
/// kf for 3-vectors (camera keys `[cx, cy, zoom]`), eased with `ease`.
pub fn kf3(t: f64, keys: &[(f64, [f64; 3])]) -> [f64; 3] {
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let ((a, va), (b, vb)) = (w[0], w[1]);
        if t < b {
            let k = ease((t - a) / (b - a));
            return [lerp(va[0], vb[0], k), lerp(va[1], vb[1], k), lerp(va[2], vb[2], k)];
        }
    }
    keys[keys.len() - 1].1
}

// ---------- rhythm ----------
/// The song's beat grid (from `out/analysis/tempo.json`: bpm and offset).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tempo {
    pub bpm: f64,
    pub offset: f64,
}
impl Tempo {
    pub const fn new(bpm: f64, offset: f64) -> Tempo {
        Tempo { bpm, offset }
    }
    /// Seconds per beat.
    pub fn beat(&self) -> f64 {
        60.0 / self.bpm
    }
    /// Beat position: 0 on the first beat, 1 on the second, fractional in between.
    pub fn bp_of(&self, t: f64) -> f64 {
        (t - self.offset) / self.beat()
    }
    /// Integer beat index.
    pub fn beat_n(&self, t: f64) -> f64 {
        self.bp_of(t).floor()
    }
    /// Time of beat n.
    pub fn at(&self, n: f64) -> f64 {
        self.offset + n * self.beat()
    }
    /// 1 exactly on each beat, decaying after (k = 6 is the kit's default).
    pub fn pulse(&self, t: f64, k: f64) -> f64 {
        (-frac(self.bp_of(t)) * k).exp()
    }
    /// The same on eighth notes.
    pub fn pulse2(&self, t: f64, k: f64) -> f64 {
        (-frac(self.bp_of(t) * 2.0) * k).exp()
    }
}

// ---------- motion principles ----------
/// A damped wobble after an event at t0, for settles and follow-through (kit defaults k 6, w 18).
pub fn spring(t: f64, t0: f64, k: f64, w: f64) -> f64 {
    if t < t0 { 0.0 } else { (-k * (t - t0)).exp() * (w * (t - t0)).sin() }
}
/// One spring kick per event time.
pub fn ring(t: f64, evs: &[f64], k: f64, w: f64) -> f64 {
    evs.iter().map(|e| spring(t, *e, k, w)).sum()
}
/// Hold each drawing for two frames (12 drawings a second). Wrap a shot's t in it for a snappier, hand-drawn feel.
pub fn on_twos(t: f64) -> f64 {
    (t * 12.0 + 1e-6).floor() / 12.0
}
/// Point on a thrown or jumping arc from p0 to p1, peaking h px above the straight line; k = 0..1 along the flight.
pub fn arc_pt(p0: P, p1: P, h: f64, k: f64) -> P {
    [lerp(p0[0], p1[0], k), lerp(p0[1], p1[1], k) - h * 4.0 * k * (1.0 - k)]
}
/// Squash and vertical offset: `sq` > 0 squashes (wider, shorter), < 0 stretches; `dy` in the caller's units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Squash {
    pub dy: f64,
    pub sq: f64,
}
/// A hop that takes off at t0 and lands at t1, h units high: crouch (anticipation), stretch on takeoff, squash on
/// landing and spring back.
pub fn jump(t: f64, t0: f64, t1: f64, h: f64) -> Squash {
    if t < t0 - 0.12 {
        return Squash::default();
    }
    if t < t0 {
        return Squash { dy: 0.0, sq: 0.18 * ease(seg(t, t0 - 0.12, t0)) };
    }
    if t < t1 {
        let k = (t - t0) / (t1 - t0);
        return Squash { dy: -h * 4.0 * k * (1.0 - k), sq: -0.16 * (1.0 - 2.0 * k).abs() };
    }
    let a = t - t1;
    Squash { dy: 0.0, sq: 0.22 * (-8.0 * a).exp() * (20.0 * a).cos() }
}
/// A surprise "take" peaking at t0: a quick squash, then a big stretch up that springs back. amt scales it.
pub fn take(t: f64, t0: f64, amt: f64) -> Squash {
    if t < t0 - 0.1 {
        return Squash::default();
    }
    if t < t0 {
        return Squash { dy: 0.0, sq: 0.12 * amt * ease(seg(t, t0 - 0.1, t0)) };
    }
    let a = t - t0;
    Squash { sq: -0.26 * amt * (-6.0 * a).exp() * (16.0 * a).cos(), dy: -1.2 * amt * (-7.0 * a).exp() * (9.0 * a).cos().max(0.0) }
}
/// Just the squash of [`take`].
pub fn take_sq(t: f64, t0: f64, amt: f64) -> f64 {
    take(t, t0, amt).sq
}
/// An eased walk from x0 to x1 between t0 and t1 for a character of unit u.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroll {
    pub x: f64,
    /// Distance walked, in strides (drive a walk cycle with it).
    pub walk: f64,
    pub moving: bool,
    /// Walking toward screen-left.
    pub flip: bool,
    /// Bob, in units of u.
    pub dy: f64,
}
pub fn stroll(t: f64, t0: f64, t1: f64, x0: f64, x1: f64, u: f64) -> Stroll {
    let x = lerp(x0, x1, ease(seg(t, t0, t1)));
    let d = (x - x0).abs() / (4.0 * u);
    let moving = t > t0 && t < t1;
    Stroll { x, walk: d, moving, flip: x1 < x0, dy: if moving { -(d * PI).sin().abs() * 0.5 } else { 0.0 } }
}
/// Camera shake: [dx, dy] to add to the camera on impacts (changes every frame).
pub fn shake_xy(t: f64, amt: f64) -> P {
    let f = (t * 24.0).floor();
    [(hash(f * 1.7) - 0.5) * 2.0 * amt, (hash(f * 2.3 + 9.0) - 0.5) * 2.0 * amt]
}
/// 1 during a blink (about every 3–5 s, staggered by seed), else 0.
pub fn blink_at(t: f64, seed: f64) -> f64 {
    let p = frac(t / (3.1 + hash(seed) * 1.7) + hash(seed + 3.0));
    if p < 0.045 { 1.0 } else { 0.0 }
}
/// Outline weight for a shape of size s (px): thin for small shapes, capped for big ones.
pub fn sw_for(s: f64) -> f64 {
    clamp_to(s / 11.0, 0.25, 1.3)
}

// ---------- colour ----------
pub const fn rgb(h: u32) -> Col {
    [((h >> 16) & 255) as f64, ((h >> 8) & 255) as f64, (h & 255) as f64]
}
/// Mix two colours in RGB (k 0 = a, 1 = b).
pub fn mix(a: Col, b: Col, k: f64) -> Col {
    let k = clamp(k);
    [(lerp(a[0], b[0], k)).round(), (lerp(a[1], b[1], k)).round(), (lerp(a[2], b[2], k)).round()]
}
/// The kit's palette (PAL in core.js). Soft colours only: use INK for black and CREAM or PAPER for white.
/// Spectral (Kubelka-Munk) pigment mixing, after spectral.js.
pub mod spectral;

pub mod pal {
    use super::{Col, rgb};
    pub const PAPER: Col = rgb(0xF3EBDC);
    pub const INK: Col = rgb(0x2B2233);
    pub const CLAY: Col = rgb(0xD97757);
    pub const CLAY_DK: Col = rgb(0xA84D33);
    pub const CLAY_LT: Col = rgb(0xF2A283);
    pub const NIGHT: Col = rgb(0x1F2550);
    pub const INDIGO: Col = rgb(0x2F3C7A);
    pub const ROSE: Col = rgb(0xE27A92);
    pub const OCHRE: Col = rgb(0xE8AA38);
    pub const SAP: Col = rgb(0x6E9F58);
    pub const TEAL: Col = rgb(0x3A9C98);
    pub const VIOLET: Col = rgb(0x7B5CA8);
    pub const CREAM: Col = rgb(0xFFF5E2);
    pub const SKY: Col = rgb(0x8EC3E6);
}

// ---------- geometry ----------
/// A rectangle as 8 points (corners and edge midpoints).
pub fn rect_pts(x: f64, y: f64, w: f64, h: f64) -> Vec<P> {
    vec![[x, y], [x + w / 2.0, y], [x + w, y], [x + w, y + h / 2.0], [x + w, y + h], [x + w / 2.0, y + h], [x, y + h], [x, y + h / 2.0]]
}
/// An ellipse as n points, starting at angle rot.
pub fn ell_pts(cx: f64, cy: f64, rx: f64, ry: f64, n: usize, rot: f64) -> Vec<P> {
    (0..n)
        .map(|i| {
            let a = rot + i as f64 / n as f64 * TAU;
            [cx + a.cos() * rx, cy + a.sin() * ry]
        })
        .collect()
}
/// A rounded rectangle (corner radius r).
pub fn rr_pts(x: f64, y: f64, w: f64, h: f64, r: f64) -> Vec<P> {
    let mut p = Vec::with_capacity(24);
    let mut corner = |cx: f64, cy: f64, a0: f64| {
        for i in 0..=5 {
            let a = a0 + i as f64 / 5.0 * PI / 2.0;
            p.push([cx + a.cos() * r, cy + a.sin() * r]);
        }
    };
    corner(x + w - r, y + r, -PI / 2.0);
    corner(x + w - r, y + h - r, 0.0);
    corner(x + r, y + h - r, PI / 2.0);
    corner(x + r, y + r, PI);
    p
}
/// A star with n points (inner = inner radius / outer), pointing up.
pub fn star_pts(cx: f64, cy: f64, r: f64, inner: f64, n: usize) -> Vec<P> {
    let rot = -PI / 2.0;
    (0..n * 2)
        .map(|i| {
            let a = rot + i as f64 * PI / n as f64;
            let q = if i % 2 == 1 { r * inner } else { r };
            [cx + a.cos() * q, cy + a.sin() * q]
        })
        .collect()
}
fn cr(p0: P, p1: P, p2: P, p3: P, u: f64) -> P {
    let (u2, u3) = (u * u, u * u * u);
    let f = |d: usize| {
        0.5 * (2.0 * p1[d] + (p2[d] - p0[d]) * u + (2.0 * p0[d] - 5.0 * p1[d] + 4.0 * p2[d] - p3[d]) * u2 + (3.0 * p1[d] - p0[d] - 3.0 * p2[d] + p3[d]) * u3)
    };
    [f(0), f(1)]
}
/// Smooth open curve through the points (Catmull-Rom), n samples per span.
pub fn through(p: &[P], n: usize) -> Vec<P> {
    if p.len() < 3 {
        return p.to_vec();
    }
    let mut out = Vec::with_capacity((p.len() - 1) * n + 1);
    let m = p.len();
    for i in 0..m - 1 {
        let (p0, p1, p2, p3) = (p[i.saturating_sub(1)], p[i], p[i + 1], p[(i + 2).min(m - 1)]);
        for k in 0..n {
            out.push(cr(p0, p1, p2, p3, k as f64 / n as f64));
        }
    }
    out.push(p[m - 1]);
    out
}
/// Smooth closed curve through the points.
pub fn ring_pts(p: &[P], n: usize) -> Vec<P> {
    let m = p.len();
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        let (p0, p1, p2, p3) = (p[(i + m - 1) % m], p[i], p[(i + 1) % m], p[(i + 2) % m]);
        for k in 0..n {
            out.push(cr(p0, p1, p2, p3, k as f64 / n as f64));
        }
    }
    out
}
/// A band: a smooth top edge, closed down to y = bottom (hills, soil, water).
pub fn band_pts(top: &[P], bottom: f64) -> Vec<P> {
    let mut v = through(top, 6);
    v.push([top[top.len() - 1][0], bottom]);
    v.push([top[0][0], bottom]);
    v
}
/// Tapered ribbon around a path, as one closed outline (tails, trails, vines, noodly arms).
pub fn ribbon(p: &[P], w0: f64, w1: f64) -> Vec<P> {
    let c = through(p, 6);
    let n = c.len();
    let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let (a, b) = (c[i.saturating_sub(1)], c[(i + 1).min(n - 1)]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let d = dx.hypot(dy).max(1e-9);
        let w = lerp(w0, w1, i as f64 / (n.max(2) - 1) as f64) / 2.0;
        l.push([c[i][0] - dy / d * w, c[i][1] + dx / d * w]);
        r.push([c[i][0] + dy / d * w, c[i][1] - dx / d * w]);
    }
    r.reverse();
    l.extend(r);
    l
}
/// Point and heading `[x, y, angle]` at fraction u along a smoothed polyline.
pub fn path_at(p: &[P], u: f64) -> [f64; 3] {
    let c = through(p, 8);
    let mut l = vec![0.0];
    for i in 1..c.len() {
        l.push(l[i - 1] + (c[i][0] - c[i - 1][0]).hypot(c[i][1] - c[i - 1][1]));
    }
    let d = clamp(u) * l[l.len() - 1];
    let mut i = 1;
    while i < l.len() - 1 && l[i] < d {
        i += 1;
    }
    let span = l[i] - l[i - 1];
    let k = if span == 0.0 { 0.0 } else { (d - l[i - 1]) / span };
    let (a, b) = (c[i - 1], c[i]);
    [lerp(a[0], b[0], k), lerp(a[1], b[1], k), (b[1] - a[1]).atan2(b[0] - a[0])]
}
/// Scale a unit shape by s and move it to (x, y).
pub fn sc(p: &[P], s: f64, x: f64, y: f64) -> Vec<P> {
    p.iter().map(|q| [x + q[0] * s, y + q[1] * s]).collect()
}

#[cfg(test)]
mod tests;
