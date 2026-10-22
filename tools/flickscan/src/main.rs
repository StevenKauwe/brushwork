//! flickscan: finds strokes that flicker in a rendered frame sequence.
//!
//!     flickscan <dir> <first> <last> [block=16] [top=40]
//!
//! Reads `<dir>/f00000.png`-style 1920 x 1080 frames from `first` to `last`, averages each frame into
//! `block`-pixel squares, and looks for a square that changes for one or two frames and then goes back, while the
//! squares around it hold still. That is how a flickering stroke looks; real motion moves its surroundings too, and
//! cuts and flashes (a big change of the whole frame) are skipped. Prints the number of events and the `top`
//! clusters, strongest first: frame, time at 24 fps, how many frames the change lasted, the block and its pixel
//! position, the score (how far the block strayed beyond how much its neighbours in time differ) and how many
//! blocks the cluster covers.

use rayon::prelude::*;

const W: usize = 1920;
const H: usize = 1080;
/// A block counts when it strays this much (0-1, the largest channel) beyond its before-and-after difference.
const MIN_SCORE: f32 = 0.03;
/// A frame step above this (the mean block change) is a cut or a flash, not flicker.
const CUT: f32 = 0.04;

type Frame = Vec<[f32; 3]>;

fn usage() -> ! {
    eprintln!("usage: flickscan <dir> <first> <last> [block=16] [top=40]");
    std::process::exit(2);
}

/// One frame as mean colours (0-1) per block.
fn blocks(dir: &str, f: usize, bs: usize) -> Result<Frame, String> {
    let path = format!("{dir}/f{f:05}.png");
    let im = image::open(&path).map_err(|e| format!("{path}: {e}"))?.to_rgb8();
    let (bw, bh) = (W / bs, H / bs);
    let mut b = vec![[0.0f32; 3]; bw * bh];
    for (x, y, p) in im.enumerate_pixels() {
        let (bx, by) = (x as usize / bs, y as usize / bs);
        if bx < bw && by < bh {
            for (acc, v) in b[by * bw + bx].iter_mut().zip(p.0) {
                *acc += v as f32;
            }
        }
    }
    let n = (bs * bs) as f32 * 255.0;
    for v in b.iter_mut().flatten() {
        *v /= n;
    }
    Ok(b)
}

/// How different two block colours are: the largest channel difference.
fn diff(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (Some(dir), Some(f0), Some(f1)) = (a.get(1), a.get(2).and_then(|v| v.parse::<usize>().ok()), a.get(3).and_then(|v| v.parse::<usize>().ok())) else {
        usage()
    };
    let bs: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(16).max(1);
    let top: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(40);
    let (bw, bh) = (W / bs, H / bs);
    let frames: Vec<Frame> = match (f0..=f1).into_par_iter().map(|f| blocks(dir, f, bs)).collect() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("flickscan: {e}");
            std::process::exit(1);
        }
    };
    // whole-frame change, to skip cuts and flashes
    let global: Vec<f32> = frames.windows(2).map(|w| w[1].iter().zip(&w[0]).map(|(a, b)| diff(a, b)).sum::<f32>() / (bw * bh) as f32).collect();
    // (score, frame, frames it lasted, block x, block y)
    let mut ev: Vec<(f32, usize, usize, usize, usize)> = vec![];
    for i in 1..frames.len().saturating_sub(2) {
        for k in 1..=2usize {
            if i + k >= frames.len() || (i - 1..(i + k).min(global.len())).any(|g| global[g] > CUT) {
                continue;
            }
            let (prev, next) = (&frames[i - 1], &frames[i + k]);
            let during = &frames[i..i + k];
            for j in 0..bw * bh {
                let back = diff(&prev[j], &next[j]);
                let strayed = during.iter().map(|f| diff(&f[j], &prev[j]).min(diff(&f[j], &next[j]))).fold(f32::MAX, f32::min);
                let s = strayed - back;
                if s <= MIN_SCORE {
                    continue;
                }
                // the neighbourhood (7 x 7 blocks, leaving out this one and its 8 neighbours) must hold still from
                // before to after: a stroke that flickers in a still area, not something moving through
                let (bx, by) = ((j % bw) as i64, (j / bw) as i64);
                let still = (-3i64..=3).all(|dy| {
                    (-3i64..=3).all(|dx| {
                        let (x, y) = (bx + dx, by + dy);
                        if (dx.abs() <= 1 && dy.abs() <= 1) || x < 0 || y < 0 || x >= bw as i64 || y >= bh as i64 {
                            return true;
                        }
                        let jj = y as usize * bw + x as usize;
                        diff(&prev[jj], &next[jj]) <= 0.02 && during.iter().all(|f| diff(&f[jj], &prev[jj]) <= 0.05)
                    })
                });
                if still {
                    ev.push((s, f0 + i, k, j % bw, j / bw));
                }
            }
        }
    }
    ev.sort_by(|a, b| b.0.total_cmp(&a.0));
    // one entry per cluster: events within 3 frames and 6 blocks of a stronger one join it
    let mut kept: Vec<(f32, usize, usize, usize, usize, usize)> = vec![];
    for e in &ev {
        let near = |c: &&mut (f32, usize, usize, usize, usize, usize)| c.1.abs_diff(e.1) <= 3 && c.3.abs_diff(e.3) <= 6 && c.4.abs_diff(e.4) <= 6;
        match kept.iter_mut().find(near) {
            Some(c) => c.5 += 1,
            None => kept.push((e.0, e.1, e.2, e.3, e.4, 1)),
        }
    }
    println!("events(raw)={} clusters={}", ev.len(), kept.len());
    for c in kept.iter().take(top) {
        println!("frame {} t={:.2}s k={} block=({},{}) px=({},{}) score={:.3} blocks={}", c.1, c.1 as f64 / 24.0, c.2, c.3, c.4, c.3 * bs, c.4 * bs, c.0, c.5);
    }
}
