//! What the page paints: flat shapes, as a film's shots would record them, and how they travel to the worker.

use brushwork_kit::{Col, ell_pts, rgb};
use brushwork_paint::{Film, Lighting, Rec, Role, Shot, wash};

/// One flat shape laid down on the page, in canvas pixels (the painted region's own coordinates).
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub col: Col,
    /// A subject: modelled in light, smaller brushes and thicker paint, throws a shadow, is looked at.
    pub subject: bool,
    /// Drawn with the film's ink outline (stated on the shadow side, lost towards the light).
    pub outline: bool,
    /// What else the shape is: a cast shadow (where the shadow of the nearest subject falls: painted as a glaze over
    /// what is there, in a colour the painter derives from it and the light), or a glowing light (luminous paint
    /// with a glow round it, and the scene's key light comes from it).
    pub kind: Kind,
    pub pts: Vec<[f64; 2]>,
}

/// What a shape is beyond its colour.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Kind {
    #[default]
    Plain,
    Shadow,
    Glow,
}

/// The lights the page offers.
pub const LIGHTS: [(&str, Lighting); 5] = [
    ("Moonlight", Lighting::MOONLIGHT),
    ("Sunset", Lighting::SUNSET),
    ("Daylight", Lighting::DAYLIGHT),
    ("Firelight", Lighting::FIRELIGHT),
    ("Overcast", Lighting::OVERCAST),
];
/// Where the light comes from, in eight steps round the compass: (name, direction towards the light).
pub const FROM: [(&str, [f64; 2]); 8] = [
    ("upper left", [-0.7, -0.7]),
    ("above", [0.0, -1.0]),
    ("upper right", [0.7, -0.7]),
    ("right", [1.0, 0.0]),
    ("lower right", [0.7, 0.7]),
    ("below", [0.0, 1.0]),
    ("lower left", [-0.7, 0.7]),
    ("left", [-1.0, 0.0]),
];

/// The light the shapes are painted in: one of `LIGHTS`, from one of `FROM` (a glowing shape, if there is one, is
/// the key light instead, from where it is).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Light {
    pub preset: usize,
    pub from: usize,
}

/// The palette the page offers.
pub const PALETTE: [u32; 10] = [0x1B2A5A, 0x3E6E9E, 0xE8B866, 0xD9663A, 0xB5652F, 0x6E8B3D, 0x2F4A2A, 0x6B4A3A, 0xF2EEDC, 0x1D1A1F];

/// Shapes as one flat list of numbers for postMessage: per shape `r g b flags n x0 y0 x1 y1 ...`.
pub fn encode(shapes: &[Shape]) -> Vec<f32> {
    let mut v = vec![];
    for s in shapes {
        v.extend(s.col.iter().map(|c| *c as f32));
        let kind = match s.kind {
            Kind::Plain => 0,
            Kind::Shadow => 1,
            Kind::Glow => 2,
        };
        v.push((s.subject as u8 | (s.outline as u8) << 1 | kind << 2) as f32);
        v.push(s.pts.len() as f32);
        for p in &s.pts {
            v.push(p[0] as f32);
            v.push(p[1] as f32);
        }
    }
    v
}

pub fn decode(v: &[f32]) -> Vec<Shape> {
    let mut out = vec![];
    let mut i = 0;
    while i + 5 <= v.len() {
        let col = [v[i] as f64, v[i + 1] as f64, v[i + 2] as f64];
        let flags = v[i + 3] as u8;
        let n = v[i + 4] as usize;
        i += 5;
        if i + 2 * n > v.len() {
            break;
        }
        let pts = (0..n).map(|k| [v[i + 2 * k] as f64, v[i + 2 * k + 1] as f64]).collect();
        i += 2 * n;
        let kind = match (flags >> 2) & 3 {
            1 => Kind::Shadow,
            2 => Kind::Glow,
            _ => Kind::Plain,
        };
        out.push(Shape { col, subject: flags & 1 != 0, outline: flags & 2 != 0, kind, pts });
    }
    out
}

thread_local! {
    static SCENE: std::cell::RefCell<Vec<Shape>> = const { std::cell::RefCell::new(Vec::new()) };
    static LIGHT: std::cell::Cell<Light> = const { std::cell::Cell::new(Light { preset: 0, from: 0 }) };
}

/// The one shot of the page's film: the shapes, back to front. Subjects are their own family (the painter's cast).
fn shot(r: &mut Rec, _t: f64, _lt: f64) {
    let centre = |pts: &[[f64; 2]]| {
        let n = pts.len().max(1) as f64;
        [pts.iter().map(|q| q[0]).sum::<f64>() / n, pts.iter().map(|q| q[1]).sum::<f64>() / n]
    };
    SCENE.with(|s| {
        let shapes = s.borrow();
        let light = LIGHT.with(|l| l.get());
        let (_, preset) = LIGHTS[light.preset % LIGHTS.len()];
        // the key light: the last glowing shape, else the chosen light from the chosen side
        let lighting = match shapes.iter().rev().find(|s| s.kind == Kind::Glow) {
            Some(g) => Lighting { key: g.col, ..preset.at(centre(&g.pts)) },
            None => preset.from_dir(FROM[light.from % FROM.len()].1),
        };
        r.light(lighting);
        for (i, shape) in shapes.iter().enumerate() {
            r.boil(if shape.subject { format!("subject{i}") } else { format!("ground{i}") });
            let po = if shape.outline { wash(shape.col) } else { wash(shape.col).no_ink() };
            match shape.kind {
                Kind::Plain => r.paint("shape", &shape.pts, po),
                Kind::Shadow => r.paint("shadow", &shape.pts, po.no_ink().role(Role::Shadow)),
                // (a light lights what is round it by itself)
                Kind::Glow => r.paint("light", &shape.pts, po.no_ink().role(Role::Light)),
            }
        }
    });
}

/// A film of the shapes (one still shot) in a light, for `brushwork::paint_frame`. The shapes live on this thread.
pub fn film(shapes: Vec<Shape>, light: Light) -> Film {
    SCENE.with(|s| *s.borrow_mut() = shapes);
    LIGHT.with(|l| l.set(light));
    Film::new("canvas", 1.0, vec![Shot::new(0.0, "canvas", shot)], "/brushwork/canvas/film")
}

/// The style the page paints with: shapes marked as subjects are the cast.
pub const STYLE: brushwork::Style = brushwork::Style { cast: &["subject"], ..brushwork::Style::DEFAULT };

/// The scene that loads with the page, laid out for a canvas `w` x `h` (designed at 1024 x 576): an owl on a branch
/// at dusk, under a big moon.
pub fn sample(w: f64, h: f64) -> Vec<Shape> {
    let (sx, sy) = (w / 1024.0, h / 576.0);
    let at = |pts: Vec<[f64; 2]>| pts.into_iter().map(|[x, y]| [x * sx, y * sy]).collect::<Vec<_>>();
    let rect = |x: f64, y: f64, w: f64, h: f64| at(vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]]);
    let ell = |cx: f64, cy: f64, rx: f64, ry: f64, n: usize, rot: f64| at(ell_pts(cx, cy, rx, ry, n, rot));
    let ground = |col: u32, pts: Vec<[f64; 2]>| Shape { col: rgb(col), subject: false, outline: false, kind: Kind::Plain, pts };
    let subject = |col: u32, outline: bool, pts: Vec<[f64; 2]>| Shape { col: rgb(col), subject: true, outline, kind: Kind::Plain, pts };
    let glow = |col: u32, pts: Vec<[f64; 2]>| Shape { col: rgb(col), subject: false, outline: false, kind: Kind::Glow, pts };
    let shadow = |pts: Vec<[f64; 2]>| Shape { col: rgb(0x1D1A1F), subject: false, outline: false, kind: Kind::Shadow, pts };
    vec![
        ground(0x22325F, rect(-20.0, -20.0, 1064.0, 260.0)),
        ground(0x4A5C8E, rect(-20.0, 220.0, 1064.0, 140.0)),
        ground(0xC98A6A, rect(-20.0, 340.0, 1064.0, 120.0)),
        glow(0xF2EEDC, ell(790.0, 170.0, 70.0, 70.0, 32, 0.0)),
        ground(
            0x3E4A6E,
            at(vec![[-20.0, 420.0], [180.0, 360.0], [380.0, 400.0], [600.0, 350.0], [820.0, 395.0], [1044.0, 370.0], [1044.0, 600.0], [-20.0, 600.0]]),
        ),
        ground(0x2F3A2A, at(vec![[-20.0, 500.0], [300.0, 470.0], [700.0, 490.0], [1044.0, 470.0], [1044.0, 600.0], [-20.0, 600.0]])),
        ground(0x5A4034, at(vec![[60.0, 330.0], [620.0, 300.0], [640.0, 322.0], [70.0, 352.0]])),
        // the owl's shadow along the branch, falling away from the moon
        shadow(ell(330.0, 325.0, 70.0, 11.0, 20, -0.06)),
        subject(0xB5652F, true, ell(380.0, 250.0, 62.0, 78.0, 28, 0.0)),
        subject(0xE8D2A8, false, ell(380.0, 268.0, 38.0, 54.0, 24, 0.0)),
        subject(0xB5652F, true, ell(380.0, 175.0, 52.0, 44.0, 24, 0.0)),
        subject(0xE8B866, true, ell(360.0, 172.0, 15.0, 15.0, 16, 0.0)),
        subject(0xE8B866, true, ell(400.0, 172.0, 15.0, 15.0, 16, 0.0)),
        subject(0x1D1A1F, false, ell(360.0, 173.0, 6.0, 6.0, 12, 0.0)),
        subject(0x1D1A1F, false, ell(400.0, 173.0, 6.0, 6.0, 12, 0.0)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_survive_the_trip_to_the_worker() {
        let s = sample(1024.0, 576.0);
        let back = decode(&encode(&s));
        assert_eq!(back.len(), s.len());
        for (a, b) in s.iter().zip(&back) {
            assert_eq!((a.subject, a.outline, a.pts.len()), (b.subject, b.outline, b.pts.len()));
            assert!(a.pts.iter().zip(&b.pts).all(|(p, q)| (p[0] - q[0]).abs() < 1e-3 && (p[1] - q[1]).abs() < 1e-3));
        }
    }

    #[test]
    fn the_sample_paints_in_its_region() {
        let (w, h) = (256usize, 144usize);
        brushwork::set_style(STYLE);
        let rgb = brushwork::paint_frame(&film(sample(w as f64, h as f64), Light::default()), 0.0, 1);
        let blank = brushwork::Painting::new().lit();
        let px = |img: &[u8], x: usize, y: usize| img[(y * brushwork::WI + x) * 3..(y * brushwork::WI + x) * 3 + 3].to_vec();
        // the owl's middle is painted; far outside the region the canvas is untouched
        assert_ne!(px(&rgb, 95, 62), px(&blank, 95, 62));
        assert_eq!(px(&rgb, 1500, 900), px(&blank, 1500, 900));
    }
}
