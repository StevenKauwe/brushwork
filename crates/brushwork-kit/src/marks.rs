//! Lightweight, seeded paint marks, independent of any rasterizer or animation clock.
//!
//! A [`StrokePlan`] resamples a caller's path and gives a renderer only the newly reached line segments.
//! [`Stamp`] plans a complete dab, flick, or spatter in a bounded local rectangle. Lines have **round caps**;
//! ellipse rotations are radians. Composite primitives in their given order using source-over alpha.
//! Colour, lighter/darker mixing targets, paths, timing, easing, placement and particle motion belong to callers.
//! This planner does not use or change the oil painter.
//!
//! Every plan owns its random stream. It uses Mulberry32, initialized with the low and high halves of the explicit
//! `u64` seed XORed together. Each sample adds `0x6d2b79f5` to the state, copies it into `t`, sets
//! `t = (t ^ (t >> 15)) * (t | 1)`,
//! then `t ^= t + (t ^ (t >> 7)) * (t | 61)`, and returns `(t ^ (t >> 14)) / 4294967296`. All integer operations
//! wrap at 32 bits. No frame, canvas, ambient randomness or other plan can change a mark. Floating-point
//! transcendental functions can differ by a few ulps between targets; scheduling does not change any values.
//!
//! ```
//! use brushwork_kit::marks::{Ink, Stamp, StampKind, StrokeOptions, StrokePlan};
//!
//! let ink = Ink { color: 0x3486ab, light: 0xffffff, dark: 0x101010 };
//! let mut stroke = StrokePlan::new(&[[8.0, 12.0], [90.0, 36.0]], 18.0, ink, 42, StrokeOptions::default())?;
//! let first_half = stroke.advance(0.5)?;
//! let remainder = stroke.advance(1.0)?; // only what has not already been drawn
//! assert!(stroke.advance(1.0)?.is_empty());
//! let stamp = Stamp::new(StampKind::Flick, 20.0, ink, 73)?;
//! assert!(!stamp.primitives.is_empty());
//! # Ok::<(), brushwork_kit::marks::MarkError>(())
//! ```

use crate::P;
use std::f64::consts::{PI, TAU};
use std::fmt;

/// Maximum input polyline length. Limits apply before any plan is constructed.
pub const MAX_PATH_POINTS: usize = 16_384;
/// Maximum number of resampled points, including both endpoints.
pub const MAX_RESAMPLED_POINTS: usize = 16_384;
/// Maximum possible draw commands in a stroke, including its body and all bristles.
pub const MAX_STROKE_PRIMITIVES: usize = 262_144;
/// Largest absolute input coordinate. Keeps all geometry safely finite.
pub const MAX_COORDINATE: f64 = 1_000_000_000.0;
/// Largest brush width in the caller's units.
pub const MAX_WIDTH: f64 = 4096.0;
/// Largest stamp size. A flick of this size has a 4096-unit-wide local rectangle.
pub const MAX_STAMP_SIZE: f64 = 1024.0;

/// Invalid input is rejected rather than clamped into a different mark or allowed to allocate unbounded work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarkError {
    /// A scalar, point or colour is outside the documented domain (including NaN and infinity).
    InvalidInput(&'static str),
    /// An input or planned output exceeds a documented work or geometry limit.
    LimitExceeded(&'static str),
}

impl fmt::Display for MarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(field) => write!(f, "invalid mark {field}"),
            Self::LimitExceeded(limit) => write!(f, "mark exceeds {limit} limit"),
        }
    }
}

impl std::error::Error for MarkError {}

/// Packed `0xRRGGBB` paint colour and the caller's lighter and darker mixing targets. No palette is implicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ink {
    pub color: u32,
    pub light: u32,
    pub dark: u32,
}

impl Ink {
    fn validate(self) -> Result<(), MarkError> {
        if (self.color | self.light | self.dark) > 0x00ff_ffff {
            return Err(MarkError::InvalidInput("RGB colour"));
        }
        Ok(())
    }

    fn shade(self, fraction: f64) -> u32 {
        let to = if fraction > 0.0 { self.light } else { self.dark };
        let k = fraction.abs();
        let mut color = 0;
        for shift in [16, 8, 0] {
            let from = ((self.color >> shift) & 255) as f64;
            let to = ((to >> shift) & 255) as f64;
            color |= ((from * (1.0 - k) + to * k).round() as u32) << shift;
        }
        color
    }
}

/// Pressure along a stroke: land/carry/lift, or grow from a hair into a full drop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PressureProfile {
    #[default]
    Lift,
    Flick,
}

impl PressureProfile {
    /// Sample at finite progress, clamped to `0..=1`. NaN and infinity are errors.
    pub fn at(self, progress: f64) -> Result<f64, MarkError> {
        Ok(self.sample(finite_progress(progress)?))
    }

    fn sample(self, progress: f64) -> f64 {
        match self {
            Self::Lift => (PI * progress.clamp(0.04, 1.0)).sin().powf(0.35),
            Self::Flick => 0.12 + 0.88 * progress.powf(0.8),
        }
    }
}

/// Numeric brush settings. The body is thinner than the bristle fan; after `dry` only the bristles remain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeOptions {
    /// Body opacity, `0..=1` (default `0.9`).
    pub body: f64,
    /// Width profile (default [`PressureProfile::Lift`]).
    pub pressure: PressureProfile,
    /// Body cutoff in normalized path progress (default `0.9`); greater than `1` keeps it through the last segment.
    pub dry: f64,
    /// Positive arc-length sampling interval (default `3` caller units).
    pub step: f64,
}

impl Default for StrokeOptions {
    fn default() -> Self {
        Self { body: 0.9, pressure: PressureProfile::Lift, dry: 0.9, step: 3.0 }
    }
}

impl StrokeOptions {
    fn validate(self) -> Result<(), MarkError> {
        if !self.body.is_finite() || !(0.0..=1.0).contains(&self.body) {
            return Err(MarkError::InvalidInput("body opacity"));
        }
        if !self.dry.is_finite() || self.dry < 0.0 {
            return Err(MarkError::InvalidInput("dry cutoff"));
        }
        positive(self.step, "sampling step")
    }
}

/// Renderer-neutral commands. Lines use round caps; strokes are centered on the specified geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Primitive {
    Line { from: P, to: P, width: f64, color: u32, alpha: f64 },
    FillEllipse { center: P, radii: P, rotation: f64, color: u32, alpha: f64 },
    StrokeEllipse { center: P, radii: P, rotation: f64, width: f64, color: u32, alpha: f64 },
}

impl Primitive {
    /// Geometric paint bounds `[left, top, right, bottom]`, including line caps and ellipse outlines.
    /// A rasterizer's antialiasing fringe is not part of these bounds.
    pub fn bounds(&self) -> [f64; 4] {
        match *self {
            Self::Line { from, to, width, .. } => {
                let r = width / 2.0;
                [from[0].min(to[0]) - r, from[1].min(to[1]) - r, from[0].max(to[0]) + r, from[1].max(to[1]) + r]
            }
            Self::FillEllipse { center, radii, rotation, .. } => ellipse_bounds(center, radii, rotation, 0.0),
            Self::StrokeEllipse { center, radii, rotation, width, .. } => ellipse_bounds(center, radii, rotation, width / 2.0),
        }
    }

    fn translate(&mut self, by: P) {
        match self {
            Self::Line { from, to, .. } => {
                for i in 0..2 {
                    from[i] += by[i];
                    to[i] += by[i];
                }
            }
            Self::FillEllipse { center, .. } | Self::StrokeEllipse { center, .. } => {
                center[0] += by[0];
                center[1] += by[1];
            }
        }
    }
}

fn ellipse_bounds(center: P, radii: P, angle: f64, outline: f64) -> [f64; 4] {
    let (s, c) = angle.sin_cos();
    let x = (radii[0] * c).hypot(radii[1] * s) + outline;
    let y = (radii[0] * s).hypot(radii[1] * c) + outline;
    [center[0] - x, center[1] - y, center[0] + x, center[1] + y]
}

/// Resample at even distances along the polyline, preserving the first and final points exactly.
/// The final interval may be shorter. Consecutive duplicate points and an already sampled endpoint are not
/// repeated; an empty path stays empty, and a wholly coincident path becomes one point.
///
/// Points must be finite with absolute coordinates at most [`MAX_COORDINATE`]; `step` must be finite and positive.
/// Input and output lengths are bounded by [`MAX_PATH_POINTS`] and [`MAX_RESAMPLED_POINTS`]. The work is linear
/// in those bounded lengths, even for a subnormal step or a degenerate path.
pub fn resample(points: &[P], step: f64) -> Result<Vec<P>, MarkError> {
    positive(step, "sampling step")?;
    if points.len() > MAX_PATH_POINTS {
        return Err(MarkError::LimitExceeded("input points"));
    }
    for &point in points {
        validate_point(point)?;
    }
    let Some(&first) = points.first() else { return Ok(Vec::new()) };
    let total: f64 = points.windows(2).map(|p| (p[1][0] - p[0][0]).hypot(p[1][1] - p[0][1])).sum();
    if total / step > (MAX_RESAMPLED_POINTS - 1) as f64 {
        return Err(MarkError::LimitExceeded("resampled points"));
    }
    let mut out = Vec::with_capacity(points.len().min(MAX_RESAMPLED_POINTS));
    out.push(first);
    let mut carry = 0.0;
    let mut samples = 1;
    for pair in points.windows(2) {
        let ([x0, y0], [x1, y1]) = (pair[0], pair[1]);
        let len = (x1 - x0).hypot(y1 - y0);
        if len == 0.0 {
            continue;
        }
        let mut d = step - carry;
        while d <= len {
            samples += 1;
            if samples > MAX_RESAMPLED_POINTS {
                return Err(MarkError::LimitExceeded("resampled points"));
            }
            let t = d / len;
            let point = if d == len { pair[1] } else { [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t] };
            if out.last() != Some(&point) {
                out.push(point);
            }
            let next = d + step;
            if next <= d {
                return Err(MarkError::LimitExceeded("resampling precision"));
            }
            d = next;
        }
        carry = len - (d - step);
    }
    let last = *points.last().expect("nonempty path");
    if out.last() != Some(&last) {
        if out.len() == MAX_RESAMPLED_POINTS {
            return Err(MarkError::LimitExceeded("resampled points"));
        }
        out.push(last);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bristle {
    offset: f64,
    wobble: f64,
    phase: f64,
    frequency: f64,
    width: f64,
    alpha: f64,
    color: u32,
    end: f64,
}

/// A deterministic brush stroke and its monotone drawing cursor. Construction does not draw anything.
/// Cloning preserves the cursor; construct a new plan with the same inputs to replay from the beginning.
#[derive(Clone, Debug, PartialEq)]
pub struct StrokePlan {
    path: Vec<P>,
    width: f64,
    ink: Ink,
    options: StrokeOptions,
    bristles: Vec<Bristle>,
    swell: P,
    drawn: usize,
}

impl StrokePlan {
    pub fn new(points: &[P], width: f64, ink: Ink, seed: u64, options: StrokeOptions) -> Result<Self, MarkError> {
        Self::with_rng(points, width, ink, options, &mut Rng::new(seed))
    }

    /// A short, fat touch centered at `center`, at a seeded angle with length `size * [0.5, 0.9)`.
    /// Uses the default stroke settings; the caller owns its animation duration.
    pub fn dab(center: P, size: f64, ink: Ink, seed: u64) -> Result<Self, MarkError> {
        validate_point(center)?;
        validate_width(size)?;
        let mut rng = Rng::new(seed);
        let angle = rng.range(0.0, TAU);
        let len = size * rng.range(0.5, 0.9);
        let points = [
            [center[0] - angle.cos() * len / 2.0, center[1] - angle.sin() * len / 2.0],
            [center[0] + angle.cos() * len / 2.0, center[1] + angle.sin() * len / 2.0],
        ];
        Self::with_rng(&points, size, ink, StrokeOptions::default(), &mut rng)
    }

    fn with_rng(points: &[P], width: f64, ink: Ink, options: StrokeOptions, rng: &mut Rng) -> Result<Self, MarkError> {
        validate_width(width)?;
        ink.validate()?;
        options.validate()?;
        let path = resample(points, options.step)?;
        let count = (14.0 + width * 0.8).min(40.0).round() as usize;
        if path.len().saturating_sub(1) * (count + 1) > MAX_STROKE_PRIMITIVES {
            return Err(MarkError::LimitExceeded("stroke primitives"));
        }
        let mut bristles = Vec::with_capacity(count);
        for _ in 0..count {
            let edge = rng.next() < 0.18;
            let offset = if edge { rng.range(0.9, 1.15) * if rng.next() < 0.5 { -1.0 } else { 1.0 } } else { rng.range(-0.95, 0.95) };
            bristles.push(Bristle {
                offset,
                wobble: rng.range(0.02, 0.08),
                phase: rng.range(0.0, TAU),
                frequency: rng.range(2.0, 7.0),
                width: rng.range(1.2, 3.4),
                alpha: if edge { rng.range(0.2, 0.5) } else { rng.range(0.45, 1.0) },
                color: ink.shade(rng.range(-0.3, 0.34)),
                end: if options.pressure == PressureProfile::Flick {
                    1.0
                } else if rng.next() < 0.45 {
                    rng.range(0.7, 0.96)
                } else {
                    1.0
                },
            });
        }
        let swell = [rng.range(0.0, 6.0), rng.range(1.5, 3.0)];
        Ok(Self { path, width, ink, options, bristles, swell, drawn: 0 })
    }

    /// The validated, evenly resampled centerline, independent of drawing progress.
    pub fn path(&self) -> &[P] {
        &self.path
    }

    pub fn segment_count(&self) -> usize {
        self.path.len().saturating_sub(1)
    }

    /// Number of complete path segments already emitted.
    pub fn drawn_segments(&self) -> usize {
        self.drawn
    }

    /// Emit newly reached segments in order, with the thinner body first and its bristles immediately after it.
    /// Finite progress is clamped to `0..=1`; repeated/backward progress does nothing and allocates nothing.
    /// NaN/infinity return an error without changing the cursor. Progress `1` always reaches the endpoint.
    /// Concatenating any increasing sequence ending at `1` is identical to a single `advance(1)`.
    pub fn advance(&mut self, progress: f64) -> Result<Vec<Primitive>, MarkError> {
        let progress = finite_progress(progress)?;
        let segments = self.segment_count();
        let to = (progress * segments as f64).floor() as usize;
        if to <= self.drawn {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity((to - self.drawn) * (self.bristles.len() + 1));
        for i in self.drawn + 1..=to {
            let (from, to) = (self.path[i - 1], self.path[i]);
            let dx = to[0] - from[0];
            let dy = to[1] - from[1];
            let len = dx.hypot(dy);
            if len == 0.0 {
                continue;
            }
            let normal = [-dy / len, dx / len];
            let f = i as f64 / segments as f64;
            let pressure = 0.88 + 0.12 * (self.swell[0] + f * PI * self.swell[1]).sin();
            let width = self.width * pressure * self.options.pressure.sample(f);
            let body_width = width * 0.62;
            if f < self.options.dry && body_width > 0.0 {
                out.push(Primitive::Line { from, to, width: body_width, color: self.ink.color, alpha: self.options.body });
            }
            for b in &self.bristles {
                if f > b.end {
                    continue;
                }
                let offset = (b.offset + b.wobble * (b.phase + f * b.frequency * PI).sin()) * width / 2.0;
                out.push(Primitive::Line {
                    from: [from[0] + normal[0] * offset, from[1] + normal[1] * offset],
                    to: [to[0] + normal[0] * offset, to[1] + normal[1] * offset],
                    width: b.width.min(width * 0.5 + 0.6),
                    color: b.color,
                    alpha: b.alpha,
                });
            }
        }
        self.drawn = to;
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StampKind {
    Dab,
    Flick,
    Spatter,
}

/// A completed mark, ready to render once and cache. All paint lies inside `[0, width] × [0, height]`.
/// Dimensions are whole caller units, not device pixels. Scale for device pixel ratio in the renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct Stamp {
    pub width: f64,
    pub height: f64,
    pub primitives: Vec<Primitive>,
}

impl Stamp {
    /// Plan a curved fat dab, a hair-to-drop flick, or five elliptical drops with lighter rims.
    /// Size must be finite, positive and at most [`MAX_STAMP_SIZE`]. Ordinary stamps retain their natural
    /// `ceil(length + size * 1.4)` by `ceil(size * 2.8)` footprint. For very small marks only, padding expands
    /// if minimum bristle/rim widths would otherwise leave that footprint. No paint is clipped to fit.
    pub fn new(kind: StampKind, size: f64, ink: Ink, seed: u64) -> Result<Self, MarkError> {
        positive(size, "stamp size")?;
        if size > MAX_STAMP_SIZE {
            return Err(MarkError::LimitExceeded("stamp size"));
        }
        ink.validate()?;
        let mut rng = Rng::new(seed);
        let len = match kind {
            StampKind::Flick => size * 2.6,
            StampKind::Dab => size * 0.8,
            StampKind::Spatter => size,
        };
        let pad = size * 0.7;
        let width = (len + pad * 2.0).ceil();
        let height = (size * 1.4 + pad * 2.0).ceil();
        let y = height / 2.0;
        let x0 = pad;
        let x1 = pad + len;
        let bend = rng.range(-0.4, 0.4) * size;
        let primitives = if kind == StampKind::Spatter {
            let mut primitives = Vec::with_capacity(10);
            for i in 0..5 {
                let r = if i == 0 { size * rng.range(0.28, 0.4) } else { size * rng.range(0.08, 0.2) };
                let cx = if i == 0 { x0 + len / 2.0 } else { x0 + len / 2.0 + rng.range(-0.7, 0.7) * size };
                let cy = if i == 0 { y } else { y + rng.range(-0.6, 0.6) * size };
                let color = ink.shade(rng.range(-0.2, 0.15));
                let radii = [r * rng.range(1.0, 1.35), r];
                let rotation = rng.range(0.0, PI);
                // Extremely small positive inputs can underflow a radius. Do not send an invalid ellipse to a renderer.
                if radii[0] > 0.0 && radii[1] > 0.0 {
                    primitives.push(Primitive::FillEllipse { center: [cx, cy], radii, rotation, color, alpha: 1.0 });
                    primitives.push(Primitive::StrokeEllipse {
                        center: [cx, cy],
                        radii,
                        rotation,
                        width: (r * 0.25).max(0.8),
                        color: ink.shade(0.35),
                        alpha: 0.45,
                    });
                }
            }
            primitives
        } else {
            let (points, brush_width, options) = if kind == StampKind::Flick {
                (
                    cubic([x0, y], [x0 + len * 0.35, y + bend], [x0 + len * 0.7, y - bend * 0.5], [x1, y + bend * 0.3]),
                    size * 0.6,
                    StrokeOptions { body: 0.95, pressure: PressureProfile::Flick, dry: 1.01, step: 2.0 },
                )
            } else {
                (
                    cubic([x0, y + bend * 0.3], [x0 + len * 0.33, y - bend * 0.3], [x0 + len * 0.66, y + bend * 0.3], [x1, y - bend * 0.2]),
                    size,
                    StrokeOptions { body: 0.95, step: 2.0, ..StrokeOptions::default() },
                )
            };
            if brush_width == 0.0 { Vec::new() } else { StrokePlan::with_rng(&points, brush_width, ink, options, &mut rng)?.advance(1.0)? }
        };
        let mut stamp = Self { width, height, primitives };
        stamp.fit_bounds();
        Ok(stamp)
    }

    fn fit_bounds(&mut self) {
        let mut bounds = [0.0, 0.0, self.width, self.height];
        for primitive in &self.primitives {
            let p = primitive.bounds();
            bounds[0] = bounds[0].min(p[0]);
            bounds[1] = bounds[1].min(p[1]);
            bounds[2] = bounds[2].max(p[2]);
            bounds[3] = bounds[3].max(p[3]);
        }
        let shift = [(-bounds[0]).ceil(), (-bounds[1]).ceil()];
        self.width = bounds[2].ceil() + shift[0];
        self.height = bounds[3].ceil() + shift[1];
        if shift != [0.0, 0.0] {
            for primitive in &mut self.primitives {
                primitive.translate(shift);
            }
        }
    }
}

fn cubic(a: P, b: P, c: P, d: P) -> Vec<P> {
    (0..=40)
        .map(|i| {
            let t = i as f64 / 40.0;
            let u = 1.0 - t;
            [
                u * u * u * a[0] + 3.0 * u * u * t * b[0] + 3.0 * u * t * t * c[0] + t * t * t * d[0],
                u * u * u * a[1] + 3.0 * u * u * t * b[1] + 3.0 * u * t * t * c[1] + t * t * t * d[1],
            ]
        })
        .collect()
}

fn positive(value: f64, name: &'static str) -> Result<(), MarkError> {
    if !value.is_finite() || value <= 0.0 { Err(MarkError::InvalidInput(name)) } else { Ok(()) }
}

fn validate_width(width: f64) -> Result<(), MarkError> {
    positive(width, "brush width")?;
    if width > MAX_WIDTH { Err(MarkError::LimitExceeded("brush width")) } else { Ok(()) }
}

fn validate_point(point: P) -> Result<(), MarkError> {
    if !point[0].is_finite() || !point[1].is_finite() {
        return Err(MarkError::InvalidInput("point"));
    }
    if point[0].abs() > MAX_COORDINATE || point[1].abs() > MAX_COORDINATE {
        return Err(MarkError::LimitExceeded("coordinate"));
    }
    Ok(())
}

fn finite_progress(progress: f64) -> Result<f64, MarkError> {
    if progress.is_finite() { Ok(progress.clamp(0.0, 1.0)) } else { Err(MarkError::InvalidInput("progress")) }
}

struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed as u32 ^ (seed >> 32) as u32 }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        (t ^ (t >> 14)) as f64 / 4_294_967_296.0
    }

    fn range(&mut self, from: f64, to: f64) -> f64 {
        from + self.next() * (to - from)
    }
}

#[cfg(test)]
mod tests;
