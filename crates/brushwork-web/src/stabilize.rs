//! Calming a finger- or mouse-drawn outline before it becomes a shape.
//!
//! A hand on glass or a mouse adds jitter, a wobble a few pixels wide, to every line it draws. This takes it out
//! without the two faults smoothing usually has: lag, and shrinking (rounding corners off, pulling a loop in).
//!
//! 1. **Even spacing.** The pointer's samples arrive as fast or slowly as the device sends them, so first the path is
//!    resampled at a fixed distance along it (three screen pixels, closer on a small shape). After that, a number of vertices
//!    means a distance on the canvas, whatever the device, the speed or the zoom.
//! 2. **Smoothing without shrinking.** Each pass moves every vertex part of the way to the midpoint of its two
//!    neighbours, then a slightly larger part of the way back (Taubin's two-step filter). The first step alone is
//!    a diffusion that takes out the jitter but also shrinks every curve; the step back undoes the shrinking for
//!    anything broad (a curve longer than about sixteen vertices keeps its size), while the short wobble, which the
//!    first step flattened far more, stays flat. Repeating it is a low-pass filter on the outline's shape.
//! 3. **Deliberate corners stay put.** How hard each vertex is pulled depends on how fast the pen went past it,
//!    compared with this stroke's own typical speed. A hand that means a corner slows down or stops there, so those
//!    vertices are pulled little or not at all, and the two sides straighten into it instead of rounding it off. The
//!    pull fades in with speed (no threshold where a corner suddenly appears). With no timing at all, everything is
//!    pulled fully.
//! 4. **No lag, closed stays closed.** An open path keeps both ends exactly where the pointer was: the end being
//!    drawn is the pen. A path that comes back to where it began is a loop: it is smoothed all the way round, the
//!    join included, so no gap or kink is left where the hand finished.
//! 5. **Fewer points.** Vertices that sit within a third of a pixel of the line through their neighbours are dropped
//!    (Ramer-Douglas-Peucker), so a straight side is two points, as if it had been placed by hand.

/// Vertices per pass are pulled towards the midpoint of their neighbours by this much, then pushed back by
/// `MU`. With `LAMBDA` = 0.5 and `MU` = -0.52, over `PASSES` passes, the filter passes outline features longer than
/// about 16 vertices (about 50 screen pixels) unchanged and all but flattens wobbles shorter than about 6 (18 px).
const LAMBDA: f64 = 0.5;
const MU: f64 = -0.52;
/// Passes of the two-step filter.
const PASSES: usize = 60;
/// At least this many vertices along any path, so a small shape's own form (a loop is one wave of its whole
/// length, an oval two) stays well inside what the filter keeps.
const MIN_VERTICES: f64 = 48.0;
/// Distance between vertices, in screen (CSS) pixels.
const SPACING: f64 = 3.0;
/// Below this share of the stroke's typical speed a vertex is not pulled at all; above `FREE` fully.
const STILL: f64 = 0.12;
const FREE: f64 = 0.45;
/// How far either side of a vertex its speed is measured over, in vertices (one: only where the pen really paused
/// is held, and the sides run straight into it).
const SPEED_SPAN: usize = 1;
/// How far from the line through its neighbours a vertex may be and still be dropped, in canvas pixels.
const SIMPLIFY: f64 = 0.33;

/// The calmed outline of a drawn path: `raw` is the pointer's samples `[x, y, time in ms]` in canvas pixels, `unit`
/// how many canvas pixels one screen (CSS) pixel covers. Closed (smoothed all round) if it ends near where it began.
pub fn outline(raw: &[[f64; 3]], unit: f64) -> Vec<[f64; 2]> {
    let closed = comes_back(raw, unit);
    path(raw, unit, closed)
}

/// Whether a path ends near where it began: within 14 screen pixels, or a tenth of its length, whichever is more
/// (and it has gone somewhere in between).
pub fn comes_back(raw: &[[f64; 3]], unit: f64) -> bool {
    if raw.len() < 4 {
        return false;
    }
    let (a, b) = (raw[0], raw[raw.len() - 1]);
    let gap = (b[0] - a[0]).hypot(b[1] - a[1]);
    let len: f64 = raw.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
    gap <= (14.0 * unit).max(0.1 * len) && len > 4.0 * gap
}

/// The calmed path, open (both ends pinned where the pointer was) or `closed` (smoothed all round).
pub fn path(raw: &[[f64; 3]], unit: f64, closed: bool) -> Vec<[f64; 2]> {
    let len: f64 = raw.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
    let spacing = (SPACING * unit).clamp(1.5, 12.0).min(len / MIN_VERTICES).max(0.25);
    let (mut pts, pull) = resample(raw, spacing, closed);
    if pts.len() < 3 {
        return pts;
    }
    smooth(&mut pts, &pull, closed);
    simplify(&pts, closed)
}

/// The path at even `spacing` along it, with each vertex's pull (0: stays where it is, 1: smoothed fully).
fn resample(raw: &[[f64; 3]], spacing: f64, closed: bool) -> (Vec<[f64; 2]>, Vec<f64>) {
    // drop repeats: they carry time (a pause) but no direction; the pause stays in the next sample's time
    let mut src: Vec<[f64; 3]> = vec![];
    for &p in raw {
        if src.last().is_none_or(|q: &[f64; 3]| (p[0] - q[0]).hypot(p[1] - q[1]) > 1e-9) {
            src.push(p);
        } else if src.len() == 1 {
            // a pause at the very first point counts from when the pen got moving
            src[0][2] = p[2];
        }
    }
    if src.len() < 2 {
        return (src.iter().map(|p| [p[0], p[1]]).collect(), vec![0.0; src.len()]);
    }
    let seg = |a: [f64; 3], b: [f64; 3]| (b[0] - a[0]).hypot(b[1] - a[1]);
    let open_len: f64 = src.windows(2).map(|w| seg(w[0], w[1])).sum();
    let open_time = src[src.len() - 1][2] - src[0][2];
    if closed {
        // the join, from the last sample back to the first, at the stroke's average pace (it was never drawn, so
        // it is neither a pause nor a dash)
        let (a, b) = (src[src.len() - 1], src[0]);
        let pace = if open_time > 0.0 { open_len / open_time } else { 0.0 };
        let t = a[2] + if pace > 0.0 { seg(a, b) / pace } else { 0.0 };
        src.push([b[0], b[1], t]);
    }
    let total: f64 = src.windows(2).map(|w| seg(w[0], w[1])).sum();
    // vertices at even distances; a closed loop's last one stops short of the first (it wraps round to it)
    let n = if closed { (total / spacing).round().max(3.0) as usize } else { (total / spacing).round().max(1.0) as usize + 1 };
    let step = if closed { total / n as f64 } else { total / (n - 1) as f64 };
    let mut out = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    let (mut k, mut walked) = (0usize, 0.0f64);
    for i in 0..n {
        let want = i as f64 * step;
        while k + 2 < src.len() && walked + seg(src[k], src[k + 1]) < want {
            walked += seg(src[k], src[k + 1]);
            k += 1;
        }
        let (a, b) = (src[k], src[k + 1]);
        let l = seg(a, b);
        let f = if l > 0.0 { ((want - walked) / l).clamp(0.0, 1.0) } else { 0.0 };
        out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
        times.push(a[2] + (b[2] - a[2]) * f);
    }
    if !closed {
        // the ends are exactly the pointer's first and last positions
        out[0] = [src[0][0], src[0][1]];
        let last = src[src.len() - 1];
        out[n - 1] = [last[0], last[1]];
    }
    let period = src[src.len() - 1][2] - src[0][2];
    (out, pull(&times, step, closed, period))
}

/// Each vertex's pull, from the pen's speed past it relative to the stroke's median speed.
fn pull(times: &[f64], step: f64, closed: bool, period: f64) -> Vec<f64> {
    let n = times.len();
    let speed: Vec<Option<f64>> = (0..n)
        .map(|i| {
            let (a, b) =
                if closed { (i as isize - SPEED_SPAN as isize, i + SPEED_SPAN) } else { (i.saturating_sub(SPEED_SPAN) as isize, (i + SPEED_SPAN).min(n - 1)) };
            // time at vertex j, counting on round a loop
            let at = |j: isize| -> f64 {
                let w = j.rem_euclid(n as isize) as usize;
                times[w] + period * (j.div_euclid(n as isize)) as f64
            };
            let dt = at(b as isize) - at(a);
            (dt > 0.0).then(|| (b as isize - a) as f64 * step / dt)
        })
        .collect();
    let mut known: Vec<f64> = speed.iter().flatten().copied().collect();
    if known.len() < n / 2 || known.is_empty() {
        return vec![1.0; n]; // no timing to go by: smooth everything
    }
    known.sort_by(f64::total_cmp);
    let typical = known[known.len() / 2];
    speed
        .iter()
        .map(|v| match v {
            Some(v) => {
                let x = ((v / typical - STILL) / (FREE - STILL)).clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            }
            None => 1.0,
        })
        .collect()
}

/// Taubin's two-step smoothing, each vertex moved by its own pull; an open path's ends stay put.
fn smooth(pts: &mut [[f64; 2]], pull: &[f64], closed: bool) {
    let n = pts.len();
    let mut d = vec![[0.0f64; 2]; n];
    for _ in 0..PASSES {
        for k in [LAMBDA, MU] {
            for i in 0..n {
                if !closed && (i == 0 || i == n - 1) {
                    d[i] = [0.0, 0.0];
                    continue;
                }
                let (a, b) = (pts[(i + n - 1) % n], pts[(i + 1) % n]);
                let w = k * pull[i];
                d[i] = [w * ((a[0] + b[0]) * 0.5 - pts[i][0]), w * ((a[1] + b[1]) * 0.5 - pts[i][1])];
            }
            for i in 0..n {
                pts[i][0] += d[i][0];
                pts[i][1] += d[i][1];
            }
        }
    }
}

/// Drop the vertices within `SIMPLIFY` of the line through the ones kept either side (Ramer-Douglas-Peucker).
fn simplify(pts: &[[f64; 2]], closed: bool) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; n];
    if closed {
        // split the loop at its first vertex and the vertex farthest from it, and simplify each half
        let far = (1..n).max_by(|&a, &b| dist2(pts[0], pts[a]).total_cmp(&dist2(pts[0], pts[b]))).unwrap_or(n / 2);
        let ring: Vec<[f64; 2]> = pts.iter().copied().chain(std::iter::once(pts[0])).collect();
        keep[0] = true;
        keep[far] = true;
        mark(&ring, 0, far, &mut keep);
        let mut tail = vec![false; n + 1];
        mark(&ring, far, n, &mut tail);
        for i in far..n {
            keep[i] |= tail[i];
        }
    } else {
        keep[0] = true;
        keep[n - 1] = true;
        mark(pts, 0, n - 1, &mut keep);
    }
    pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect()
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// Mark which vertices strictly between `a` and `b` must stay for the path to stay within `SIMPLIFY` of itself.
fn mark(pts: &[[f64; 2]], a: usize, b: usize, keep: &mut [bool]) {
    if b <= a + 1 {
        return;
    }
    let (p, q) = (pts[a], pts[b]);
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let len = dx.hypot(dy);
    let off = |r: [f64; 2]| if len > 0.0 { ((r[0] - p[0]) * dy - (r[1] - p[1]) * dx).abs() / len } else { dist2(p, r).sqrt() };
    let (i, d) = (a + 1..b).map(|i| (i, off(pts[i]))).max_by(|x, y| x.1.total_cmp(&y.1)).unwrap_or((a, 0.0));
    if d > SIMPLIFY {
        keep[i] = true;
        mark(pts, a, i, keep);
        mark(pts, i, b, keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// A hand's wobble: a smooth random offset a few pixels wide, changing every few pixels along the path (the same
    /// every run).
    struct Hand(u64);
    impl Hand {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Samples along `curve` (t from 0 to 1, `len` pixels long) as a touch screen sends them: every 8 ms, at `speed`
    /// px per ms, each with the digitiser's and the hand's wobble (up to `jitter` px, across the path) added.
    fn drawn(curve: impl Fn(f64) -> [f64; 2], len: f64, speed: f64, jitter: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut hand = Hand(seed);
        let n = (len / (speed * 8.0)).ceil() as usize;
        (0..=n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let p = curve(t);
                let q = curve((t + 1e-4).min(1.0));
                let r = curve((t - 1e-4).max(0.0));
                let (tx, ty) = (q[0] - r[0], q[1] - r[1]);
                let l = tx.hypot(ty).max(1e-12);
                let j = jitter * hand.next();
                [p[0] - ty / l * j, p[1] + tx / l * j, i as f64 * 8.0]
            })
            .collect()
    }

    /// Points every half pixel along an outline (round to the start if `closed`).
    fn dense(pts: &[[f64; 2]], closed: bool) -> Vec<[f64; 2]> {
        let n = pts.len();
        let segs = if closed { n } else { n - 1 };
        let mut out = vec![];
        for i in 0..segs {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let k = (dist2(a, b).sqrt() / 0.5).ceil().max(1.0) as usize;
            out.extend((0..k).map(|j| {
                let f = j as f64 / k as f64;
                [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]
            }));
        }
        out
    }

    /// How much an outline turns in all (radians, every turn counted whichever way), measured every 4 px along it,
    /// so that outlines with different numbers of points compare. A calm line hardly turns; a loop turns 2 pi.
    fn turning(pts: &[[f64; 2]], closed: bool) -> f64 {
        let d = dense(pts, closed);
        let walk: Vec<[f64; 2]> = d.iter().step_by(8).copied().collect();
        let n = walk.len();
        let heading = |a: [f64; 2], b: [f64; 2]| (b[1] - a[1]).atan2(b[0] - a[0]);
        let (segs, turns) = if closed { (n, n) } else { (n - 1, n - 2) };
        let h: Vec<f64> = (0..segs).map(|i| heading(walk[i], walk[(i + 1) % n])).collect();
        (0..turns)
            .map(|i| {
                let t = h[(i + 1) % segs] - h[i];
                (t + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI
            })
            .map(f64::abs)
            .sum()
    }

    /// The root mean square of `f` over `pts`.
    fn rms(pts: &[[f64; 2]], f: impl Fn([f64; 2]) -> f64) -> f64 {
        (pts.iter().map(|p| f(*p).powi(2)).sum::<f64>() / pts.len() as f64).sqrt()
    }

    /// The distance of `p` from the segment `a`-`b`.
    fn off_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
    }

    #[test]
    fn a_straight_line_stays_straight() {
        // drawn cleanly, a line is exactly its two ends
        let line = drawn(|t| [100.0 + 400.0 * t, 80.0 + 150.0 * t], 427.2, 0.6, 0.0, 1);
        let out = path(&line, 1.0, false);
        assert_eq!(out, vec![[100.0, 80.0], [500.0, 230.0]]);
        // drawn with a wobbly hand, it comes out calm (it hardly turns, where the hand turned this way and that
        // all along), no further from the line than drawn, and ends where the pointer did
        let shaky = drawn(|t| [100.0 + 400.0 * t, 80.0 + 150.0 * t], 427.2, 0.6, 1.5, 2);
        let out = path(&shaky, 1.0, false);
        let (a, b) = ([100.0, 80.0], [500.0, 230.0]);
        let shaky_pts: Vec<[f64; 2]> = shaky.iter().map(|p| [p[0], p[1]]).collect();
        let (turn_in, turn_out) = (turning(&shaky_pts, false), turning(&out, false));
        assert!(turn_out < 0.2 * turn_in, "the line still turns {turn_out:.2} rad in all (drawn {turn_in:.2})");
        let (off_in, off_out) = (rms(&dense(&shaky_pts, false), |p| off_segment(p, a, b)), rms(&dense(&out, false), |p| off_segment(p, a, b)));
        assert!(off_out < off_in, "{off_out:.2} px off the line (drawn {off_in:.2})");
        assert_eq!(out[0], [shaky[0][0], shaky[0][1]]);
        assert_eq!(*out.last().unwrap(), [shaky[shaky.len() - 1][0], shaky[shaky.len() - 1][1]]);
    }

    #[test]
    fn a_closed_loop_stays_closed() {
        // a circle of radius 100, drawn with a wobble, the hand stopping 6 px short of where it began
        let (cx, cy, r) = (300.0, 250.0, 100.0);
        let stop = 1.0 - 6.0 / (TAU * r);
        let raw = drawn(|t| [cx + r * (TAU * t * stop).cos(), cy + r * (TAU * t * stop).sin()], TAU * r * stop, 0.5, 2.0, 3);
        assert!(comes_back(&raw, 1.0));
        let out = outline(&raw, 1.0);
        // round all the way, the join included: smoothed through where the hand stopped, not two loose ends
        // joined by a step (which would keep the hand's wobble at both of them)
        let off = |p: [f64; 2]| (p[0] - cx).hypot(p[1] - cy) - r;
        let ring = dense(&out, true);
        let at_join: Vec<[f64; 2]> = ring.iter().copied().filter(|p| dist2(*p, [raw[0][0], raw[0][1]]) < 20.0 * 20.0).collect();
        let raw_pts: Vec<[f64; 2]> = raw.iter().map(|p| [p[0], p[1]]).collect();
        // calm: it turns once round, near enough, where the hand's loop turned this way and that all along
        let (turn_in, turn_out) = (turning(&raw_pts, true), turning(&out, true));
        assert!(turn_out < 1.25 * TAU && turn_in > 2.0 * TAU, "the loop turns {turn_out:.2} rad in all (drawn {turn_in:.2})");
        // no further from the circle than drawn, and no worse at the join than anywhere else
        let wobble = rms(&ring, off);
        assert!(wobble < rms(&dense(&raw_pts, true), off), "the circle wobbles {wobble:.2} px");
        let join = at_join.iter().map(|p| off(*p).abs()).fold(0.0, f64::max);
        let worst = ring.iter().map(|p| off(*p).abs()).fold(0.0, f64::max);
        assert!(join <= worst && join < 1.5, "a {join:.2} px kink where the loop closes");
        // and the same size: not shrunk by the smoothing
        let mean = ring.iter().map(|p| (p[0] - cx).hypot(p[1] - cy)).sum::<f64>() / ring.len() as f64;
        assert!((mean - r).abs() < 0.01 * r, "mean radius {mean:.2}, drawn as {r}");
    }

    /// A square drawn with a wobbly hand that stops at each corner for a moment, as a hand does when it means one.
    fn square_with_pauses(pause_ms: f64, jitter: f64) -> (Vec<[f64; 3]>, [[f64; 2]; 4]) {
        let corners = [[200.0, 100.0], [400.0, 100.0], [400.0, 300.0], [200.0, 300.0]];
        let mut hand = Hand(4);
        let mut raw: Vec<[f64; 3]> = vec![];
        let mut t = 0.0;
        for side in 0..4 {
            let (a, b) = (corners[side], corners[(side + 1) % 4]);
            // the pause: the pointer reports the corner, then nothing until it moves on
            raw.push([a[0], a[1], t]);
            t += pause_ms;
            let n = 50;
            for i in 1..n {
                let f = i as f64 / n as f64;
                let j = jitter * hand.next();
                let (nx, ny) = ((b[1] - a[1]) / 200.0, -(b[0] - a[0]) / 200.0);
                t += 8.0;
                raw.push([a[0] + (b[0] - a[0]) * f + nx * j, a[1] + (b[1] - a[1]) * f + ny * j, t]);
            }
            t += 8.0;
        }
        raw.push([corners[0][0], corners[0][1], t]);
        (raw, corners)
    }

    #[test]
    fn corners_drawn_deliberately_stay_crisp() {
        let (raw, corners) = square_with_pauses(250.0, 1.2);
        let out = outline(&raw, 1.0);
        // every corner is still there, to within a pixel
        let ring = dense(&out, true);
        for c in corners {
            let near = ring.iter().map(|p| dist2(*p, c).sqrt()).fold(f64::MAX, f64::min);
            assert!(near < 1.0, "the corner at {c:?} was rounded off: nearest outline point {near:.2} px away");
        }
        // and the sides between them are calmer than drawn: what turning there is, is at the corners
        let raw_pts: Vec<[f64; 2]> = raw.iter().map(|p| [p[0], p[1]]).collect();
        let (turn_in, turn_out) = (turning(&raw_pts, true), turning(&out, true));
        assert!(turn_out < 0.2 * turn_in, "the square turns {turn_out:.2} rad in all (drawn {turn_in:.2})");
    }

    #[test]
    fn corners_swept_through_without_a_pause_are_eased_not_cut() {
        // the same square swept round at an even pace: no corner was meant, so it is eased a little, but not cut
        let (raw, corners) = square_with_pauses(8.0, 0.0);
        let out = outline(&raw, 1.0);
        let ring = dense(&out, true);
        let near = ring.iter().map(|p| dist2(*p, corners[1]).sqrt()).fold(f64::MAX, f64::min);
        assert!(near > 0.3 && near < 4.0, "a swept corner {near:.2} px from where it was");
        // still the square's size: its sides stay where they were
        let mid = ring.iter().map(|p| dist2(*p, [400.0, 200.0]).sqrt()).fold(f64::MAX, f64::min);
        assert!(mid < 0.5, "the right side moved {mid:.2} px");
    }

    #[test]
    fn the_end_being_drawn_is_the_pen() {
        // no lag: however the path is smoothed, it ends exactly at the pointer
        let raw = drawn(|t| [100.0 + 300.0 * t, 200.0 + 60.0 * (t * 9.0).sin()], 420.0, 0.7, 2.0, 5);
        for cut in [5, 12, 20, raw.len()] {
            let out = path(&raw[..cut], 1.0, false);
            let pen = raw[cut - 1];
            assert_eq!(*out.last().unwrap(), [pen[0], pen[1]]);
        }
    }

    #[test]
    fn short_and_untimed_paths_are_safe() {
        assert!(outline(&[], 1.0).is_empty());
        assert_eq!(outline(&[[1.0, 2.0, 0.0]], 1.0), vec![[1.0, 2.0]]);
        assert_eq!(outline(&[[1.0, 2.0, 0.0], [1.0, 2.0, 5.0]], 1.0), vec![[1.0, 2.0]]);
        // all samples at the same instant: smoothed without speeds, nothing comes out NaN
        let raw: Vec<[f64; 3]> = (0..40).map(|i| [i as f64 * 3.0, (i as f64 * 0.7).sin() * 4.0, 0.0]).collect();
        assert!(outline(&raw, 1.0).iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }
}
