// The oil painter's stroke, for one pixel at a time: the GPU half of studio-oil (see gpu.rs). Every line here
// mirrors the pixel loop of `Painting::stroke` in lib.rs, in the same order; keep the two in step.

struct Cmd {
    cx: f32, cy: f32, sn: f32, cs: f32, len: f32, width: f32, bend: f32, along: f32,
    span0: f32, span1: f32, opacity: f32, pickup: f32, load: f32,
    col_r: f32, col_g: f32, col_b: f32, alt_r: f32, alt_g: f32, alt_b: f32, und_r: f32, und_g: f32, und_b: f32,
    nb: f32, marks: f32, lips: f32, edge_soft: f32, whole_len: f32, end_w: f32, slant_k: f32, slant_c: f32,
    ridge_k: f32, size_k: f32, knife_k: f32, z: f32, half_max: f32,
    id: u32, tool: u32, pigment: u32, worked: u32,
    mask_off: u32, mask_ox: i32, mask_oy: i32, mask_w: u32, mask_h: u32,
    x0: i32, x1: i32, y0: i32, y1: i32,
    kind: u32,
    pad0: u32, pad1: u32, pad2: u32, pad3: u32, pad4: u32, pad5: u32, pad6: u32, pad7: u32,
    pad8: u32, pad9: u32, pad10: u32, pad11: u32, pad12: u32, pad13: u32, pad14: u32,
};

@group(0) @binding(0) var<storage, read> cmds: array<Cmd>;
@group(0) @binding(1) var<storage, read> masks: array<u32>;
@group(0) @binding(2) var<storage, read> worked: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> entries: array<u32>;
@group(0) @binding(5) var<storage, read_write> col: array<f32>;
@group(0) @binding(6) var<storage, read_write> thick: array<f32>;

const WI: u32 = 1920u;
const HI: u32 = 1080u;
const PI: f32 = 3.14159265358979323846;
const STEADY_PICKUP: f32 = 0.18;

fn h32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7FEB352Du;
    x = x ^ (x >> 15u);
    x = x * 0x846CA68Bu;
    return x ^ (x >> 16u);
}
fn hf(a: u32, b: u32) -> f32 {
    return f32(h32((a * 0x9E3779B1u) ^ h32(b + 0x85EBCA6Bu)) >> 8u) / 16777216.0;
}
fn noise(x: f32, y: f32, seed: u32) -> f32 {
    let ix = floor(x);
    let iy = floor(y);
    let tx = x - ix;
    let ty = y - iy;
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let s = seed * 977u;
    let xi = i32(ix);
    let yi = i32(iy);
    let g00 = hf(u32(xi) ^ s, u32(yi));
    let g10 = hf(u32(xi + 1) ^ s, u32(yi));
    let g01 = hf(u32(xi) ^ s, u32(yi + 1));
    let g11 = hf(u32(xi + 1) ^ s, u32(yi + 1));
    let a = g00 + (g10 - g00) * sx;
    let b = g01 + (g11 - g01) * sx;
    return a + (b - a) * sy;
}
fn smoothk(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// the shape mask's value at a pixel (-1 outside its box); masks are stored in 16 x 16 blocks: 0 all outside, 1 all
// inside, n >= 2 the block's values at n - 2
fn mask_at(s: Cmd, x: i32, y: i32) -> f32 {
    let mx = x - s.mask_ox;
    let my = y - s.mask_oy;
    if (mx < 0 || my < 0 || u32(mx) >= s.mask_w || u32(my) >= s.mask_h) {
        return -1.0;
    }
    let bw = (s.mask_w + 15u) / 16u;
    let blk = masks[s.mask_off + (u32(my) / 16u) * bw + u32(mx) / 16u];
    if (blk == 0u) {
        return 0.0;
    } else if (blk == 1u) {
        return 1.0;
    }
    return bitcast<f32>(masks[blk - 2u + (u32(my) % 16u) * 16u + u32(mx) % 16u]);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    if (gid.x >= WI || gid.y >= HI) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * WI + gid.x;
    var c0 = vec3<f32>(col[i * 3u], col[i * 3u + 1u], col[i * 3u + 2u]);
    var th = thick[i];
    let tile = wid.y * ((WI + 15u) / 16u) + wid.x;
    let e0 = offsets[tile];
    let e1 = offsets[tile + 1u];
    for (var e = e0; e < e1; e = e + 1u) {
        let s = cmds[entries[e]];
        if (s.kind == 1u) {
            // a full-frame glaze
            let k = s.opacity;
            c0 = c0 + (vec3<f32>(s.col_r, s.col_g, s.col_b) - c0) * k;
            th = th * (1.0 - k);
            continue;
        }
        if (x < s.x0 || x > s.x1 || y < s.y0 || y > s.y1) {
            continue;
        }
        if (s.kind == 2u) {
            // a glaze over a shape: the paint below, seen through a thin layer of colour (Painting::glaze_area)
            let m = mask_at(s, x, y);
            if (m <= 0.0) {
                continue;
            }
            let gx = f32(x) + 0.5 - s.cx;
            let gy = f32(y) + 0.5 - s.cy;
            let gu = gx * s.cs + gy * s.sn;
            let gv = -gx * s.sn + gy * s.cs;
            let tex = max(th, 0.0) / (max(th, 0.0) + 0.12);
            if (s.tool == 1u) {
                // light falling on the paint (light_on in lib.rs)
                let g = clamp(s.opacity * m * (0.45 + 1.1 * tex), 0.0, 1.0);
                let inner = vec3<f32>(s.col_r, s.col_g, s.col_b);
                let outer = vec3<f32>(s.alt_r, s.alt_g, s.alt_b);
                c0 = c0 + (outer + (inner - outer) * m - c0) * g;
            } else if (s.tool == 2u) {
                let g = m * (0.82 + 0.25 * tex);
                {
                    // a shadow: the paint here taken down by its own shadow colour (shade_paint in lib.rs)
                    let lw = vec3<f32>(0.3, 0.59, 0.11);
                    let v = dot(c0, lw);
                    let mm = c0 + (vec3<f32>(s.col_r, s.col_g, s.col_b) - c0) * s.bend;
                    let l = max(dot(mm, lw), 1e-4);
                    let sh = clamp(mm * (v * (1.0 - s.along) / l), vec3<f32>(0.0), vec3<f32>(1.0));
                    c0 = c0 + (sh - c0) * g;
                }
            } else {
                let g = m * (0.74 + 0.16 * noise(gu / (90.0 * s.z), gv / (28.0 * s.z), s.id ^ 5u) + 0.1 * noise(gu / (16.0 * s.z), gv / (3.4 * s.z), s.id));
                c0 = c0 * (vec3<f32>(1.0) - g * (vec3<f32>(1.0) - vec3<f32>(s.col_r, s.col_g, s.col_b)));
            }
            continue;
        }
        if (s.kind != 0u) {
            continue;
        }
        let dx = f32(x) + 0.5 - s.cx;
        let dy = f32(y) + 0.5 - s.cy;
        let u = dx * s.cs + dy * s.sn;
        let pos = u / s.len + 0.5;
        if (!(pos >= 0.0 && pos <= 1.0)) {
            continue;
        }
        let v = -dx * s.sn + dy * s.cs - s.bend * (u * u) / s.len;
        if (abs(v) > s.half_max) {
            continue;
        }
        let whole = s.span0 + (s.span1 - s.span0) * pos;
        let across = (v / s.width + 0.5) * s.nb;
        let b = i32(floor(across));
        let bu = u32(b);
        let r1 = hf(s.id, bu);
        let r2 = hf(s.id ^ 0x55u, bu);
        let r3 = hf(s.id ^ 0xAAu, bu);
        let r4 = hf(s.id ^ 0x3Cu, bu);
        let r5 = hf(s.id ^ 0x77u, bu);
        let t = clamp(across / s.nb, 0.0, 1.0);
        let slant = (s.slant_k * (t - 0.5) + s.slant_c) * s.end_w;
        let tt = 2.0 * t - 1.0;
        let big = smoothk(28.0, 80.0, s.width / s.z);
        let round = (0.38 + 0.9 * big) * (tt * tt) * s.end_w;
        let land_at = max((0.06 + 0.38 * r5) * s.end_w + slant + round, 0.0) / s.whole_len;
        let dry_at = 1.0 - max((0.30 + 0.85 * (1.0 - r1)) * s.end_w - slant + round, 0.18 * s.end_w) / s.whole_len;
        let landed = smoothk(land_at, land_at + (0.25 + 0.35 * r2) * s.end_w / s.whole_len, whole) * 1.0;
        let lift_k = 1.0 - smoothk(dry_at - 0.10 * s.end_w / s.whole_len, dry_at + 0.12 * s.end_w / s.whole_len, whole);
        var m_k = 1.0;
        if (s.mask_w > 0u) {
            let mx = x - s.mask_ox;
            let my = y - s.mask_oy;
            if (mx < 0 || my < 0 || u32(mx) >= s.mask_w || u32(my) >= s.mask_h) {
                continue;
            }
            // masks are stored in 16 x 16 blocks: 0 all outside, 1 all inside, n >= 2 the block's values at n - 2
            let bw = (s.mask_w + 15u) / 16u;
            let blk = masks[s.mask_off + (u32(my) / 16u) * bw + u32(mx) / 16u];
            if (blk == 0u) {
                m_k = 0.0;
            } else if (blk == 1u) {
                m_k = 1.0;
            } else {
                m_k = bitcast<f32>(masks[blk - 2u + (u32(my) % 16u) * 16u + u32(mx) % 16u]);
            }
        }
        if (landed * lift_k * m_k <= 0.004) {
            continue;
        }
        let arc = max(sin(whole * PI), 0.0);
        var taper = (0.94 - 0.2 * big) + (0.06 + 0.2 * big) * pow(arc, 0.5);
        if (s.tool == 1u) {
            taper = 0.70 + 0.30 * pow(arc, 0.5);
        } else if (s.tool == 2u) {
            taper = 0.35 + 0.65 * pow(arc, 0.7);
        }
        let ua = u + s.along;
        let half = s.width / 2.0 * taper * (0.94 + 0.12 * noise(ua / (9.0 * s.z), 0.0, s.id));
        if (abs(v) > half) {
            continue;
        }
        let dry_k = smoothk(dry_at - min(0.18, 2.5 * s.width / s.whole_len), dry_at, whole);
        var skip = 0.0;
        if (dry_k != 0.0) {
            skip = dry_k * (0.35 + 0.65 * noise(ua / (5.0 * s.z) + 37.0 * r3, f32(b) * 3.7, s.id ^ 9u));
        }
        let side = smoothk(0.0, s.edge_soft, half - abs(v));
        var a = landed * side * (1.0 - smoothk(0.45, 0.8, skip)) * lift_k * 1.0;
        if (s.tool == 3u) {
            a = a * smoothk(0.25, 0.45, noise(ua / (7.0 * s.z), v / (2.5 * s.z), s.id ^ 77u) + 0.35 * (1.0 - whole));
        }
        a = a * m_k;
        if (a <= 0.004) {
            continue;
        }
        let ridge = 1.0 - s.marks + s.marks * sin((across - f32(b)) * PI);
        let lq = (half - abs(v)) / (0.10 * s.width + s.z);
        let lip = exp(-(lq * lq));
        var streak = 0.0;
        if (r3 < 0.22) {
            streak = 0.6;
        }
        streak = streak * (0.5 + 0.5 * r4);
        let pick = clamp(s.pickup * STEADY_PICKUP * (0.4 + 1.2 * r2) * (0.5 + whole), 0.0, 0.85);
        let tone = 1.0 + 0.07 * (r4 - 0.5) + 0.03 * lip;
        var c = vec3<f32>(0.0);
        if (s.pigment != 0u) {
            let at = pick / 0.85 * 8.0;
            let fl = floor(at);
            let i0 = min(u32(fl), 7u);
            let f = at - min(fl, 7.0);
            let w0 = s.worked * 3u;
            for (var k = 0u; k < 3u; k = k + 1u) {
                let a0 = worked[w0 + i0 * 3u + k];
                let a1 = worked[w0 + (i0 + 1u) * 3u + k];
                let b0 = worked[w0 + 27u + i0 * 3u + k];
                let b1 = worked[w0 + 27u + (i0 + 1u) * 3u + k];
                let aa = a0 + (a1 - a0) * f;
                let bb = b0 + (b1 - b0) * f;
                c[k] = clamp((aa + (bb - aa) * streak) * tone, 0.0, 1.0);
            }
        } else {
            let sc = vec3<f32>(s.col_r, s.col_g, s.col_b);
            let sa = vec3<f32>(s.alt_r, s.alt_g, s.alt_b);
            let su = vec3<f32>(s.und_r, s.und_g, s.und_b);
            let own = sc + (sa - sc) * streak;
            c = clamp((own + (su - own) * pick) * tone, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        let cover = a * s.opacity;
        c0 = c0 + (c - c0) * cover;
        let lay = s.load * ((0.6 + 0.4 * r1) * ridge * (1.0 - 0.6 * whole) + s.lips * lip) * s.knife_k * s.size_k * s.ridge_k;
        th = th + (lay - th) * cover;
    }
    col[i * 3u] = c0.x;
    col[i * 3u + 1u] = c0.y;
    col[i * 3u + 2u] = c0.z;
    thick[i] = th;
}
