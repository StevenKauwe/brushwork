//! Full-frame effects and transitions (core.js and timeline.js). Call them after `cam_end()`, in screen space.

use brushwork_kit::*;

use crate::rec::{Brush, Rec, wash};

/// A full-frame flash of colour c at strength k (0..1).
pub fn flash(r: &mut Rec, k: f64, c: Col) {
    if k > 0.01 {
        r.boil("flash");
        r.paint("flash", &rect_pts(-60.0, -60.0, W + 120.0, H + 120.0), wash(c).op(255.0 * clamp(k)).no_ink());
    }
}
/// Paint everything outside a star-shaped hole (pts in screen space).
pub fn iris_shape(r: &mut Rec, pts: &[P], c: Col) {
    let far = 4000.0;
    let n = pts.len() as f64;
    let (cx, cy) = (pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n);
    let out = |p: P| {
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        let d = dx.hypot(dy).max(1e-9);
        [cx + dx / d * far, cy + dy / d * far]
    };
    r.boil("iris");
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        let (ex, ey) = ((b[0] - a[0]) * 0.06, (b[1] - a[1]) * 0.06);
        let (a2, b2) = ([a[0] - ex, a[1] - ey], [b[0] + ex, b[1] + ey]);
        r.paint("iris", &[a2, b2, out(b2), out(a2)], wash(c).op(255.0).no_ink());
    }
}
/// A circular iris of radius rad around a screen point; below 4 px the frame is fully covered.
pub fn iris(r: &mut Rec, cx: f64, cy: f64, rad: f64, c: Col) {
    if rad < 4.0 {
        r.boil("iris");
        r.paint("iris.full", &rect_pts(-60.0, -60.0, W + 120.0, H + 120.0), wash(c).no_ink());
    } else {
        iris_shape(r, &ell_pts(cx, cy, rad, rad, 40, 0.0), c);
    }
}
/// Fat paint strokes sweep across (p 0 → .5) and drag off (p .5 → 1). Cut to the next shot under full cover at
/// p = .5: the outgoing shot calls it with p rising to .5, the incoming one continues from .5.
pub fn brush_wipe(r: &mut Rec, p: f64, cols: [Col; 2]) {
    if p <= 0.0 || p >= 1.0 {
        return;
    }
    let (c1, c2) = (cols[0], cols[1]);
    let n = 5;
    let bh = (H + 420.0) / n as f64 + 40.0;
    r.boil("wipe");
    r.push();
    r.translate(W / 2.0, H / 2.0);
    r.rotate(-0.1);
    r.translate(-W / 2.0, -H / 2.0);
    for i in 0..n {
        let fi = i as f64;
        let y0 = -230.0 + fi * (H + 420.0) / n as f64;
        let d = [0.0, 0.14, 0.06, 0.18, 0.1][i];
        let q = if p < 0.5 { ease_out(clamp((p * 2.0 - d) / (1.0 - d))) } else { ease(clamp(((p - 0.5) * 2.0 - d) / (1.0 - d))) };
        let x0 = if p < 0.5 { -300.0 } else { lerp(-300.0, W + 400.0, q) };
        let x1 = if p < 0.5 { lerp(-300.0, W + 400.0, q) } else { W + 400.0 };
        if x1 - x0 < 30.0 {
            continue;
        }
        let rag = |k: f64| 40.0 + 50.0 * hash(fi * 31.0 + k);
        let mut pts: Vec<P> = vec![];
        for k in 0..=8 {
            let fk = k as f64;
            pts.push([lerp(x0, x1, fk / 8.0), y0 + (fk * 0.9 + fi).sin() * 14.0]);
        }
        for k in 1..9 {
            let fk = k as f64;
            pts.push([x1 + rag(fk) - 40.0, y0 + bh * fk / 9.0]);
        }
        for k in (0..=8).rev() {
            let fk = k as f64;
            pts.push([lerp(x0, x1, fk / 8.0), y0 + bh + (fk * 0.8 + fi * 2.0).sin() * 14.0]);
        }
        if p >= 0.5 {
            for k in (1..=8).rev() {
                let fk = k as f64;
                pts.push([x0 - rag(fk + 20.0) + 40.0, y0 + bh * fk / 9.0]);
            }
        }
        let (wc, fc) = if i % 2 == 1 { (c1, c2) } else { (c2, c1) };
        r.sub(format!("b{i}."));
        r.paint(if p < 0.5 { "wipe.in" } else { "wipe.out" }, &pts, wash(wc).op(255.0).fill(fc, 70.0).no_ink());
    }
    r.sub("");
    r.pop();
}
/// Whip-pan smear: horizontal dry strokes across the frame, k 0..1..0. Pair it with a fast camera move.
pub fn whip_smear(r: &mut Rec, k: f64, cols: [Col; 3]) {
    if k <= 0.02 {
        return;
    }
    r.boil("whip");
    for i in 0..16 {
        let fi = i as f64;
        let (y, l, x) = (hash(fi * 3.0) * H, 500.0 + 900.0 * hash(fi + 1.0), hash(fi + 7.0) * W);
        r.ink_line("whip", &[[x - l / 2.0, y], [x + l / 2.0, y]], 20.0 + 40.0 * hash(fi + 2.0), cols[i % 3], Brush::Dry, 0.0);
    }
    r.paint("whip.veil", &rect_pts(-60.0, -60.0, W + 120.0, H + 120.0), wash(cols[0]).op(150.0 * k).no_ink());
}
