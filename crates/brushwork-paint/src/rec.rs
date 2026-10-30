//! The recorder: the stand-in for the p5 canvas. Scene code calls paint / ink_line / glows as the JS kit did,
//! and each call becomes an Item: world-space points plus a style, filed under a layer family (the boil key) and a
//! stable identity (a call-site tag and its occurrence), so the baker can follow it from frame to frame.

use std::collections::HashMap;

use brushwork_kit::{Col, H, P, W};

use crate::semantics::{Lighting, Role};

/// The default outline colour (PAL.ink).
pub const PAL_INK: Col = brushwork_kit::pal::INK;
/// Pixels per unit of p5.brush stroke weight (measured on the original render: a weight-5 'ink' stroke at sw 38 is
/// ~140 px wide).
const BRUSH_PX: f64 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brush {
    /// The kit's 'ink' brush (weight 5): outlines and most lines.
    Ink,
    /// 'inkfine' (weight 2.6): rings, hairs, details.
    InkFine,
    /// 'dry' (weight 14): bristly broad strokes (furrows, whip smears).
    Dry,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ink {
    /// The kit's default outline (PAL.ink), drawn when a paint() call doesn't say `ink: null`.
    Default,
    None,
    Col(Col),
}

/// paint() options: wash (flat colour), fill (watercolour layer), outline. Build with [`wash`] or [`fill_only`] and
/// the chained setters.
#[derive(Clone, Copy, Debug)]
pub struct Po {
    pub wash: Option<Col>,
    pub wash_op: f64,
    pub fill: Option<Col>,
    pub fill_op: f64,
    pub ink: Ink,
    pub sw: f64,
    pub curv: f64,
    /// The watercolour layer's bleed, texture and border (p5.brush's fillBleed and fillTexture; the kit's defaults
    /// are 0.1, 0.4 and 0.35).
    pub wc: [f64; 3],
    /// What the shape is, for the painter (see [`Role`]).
    pub role: Role,
    /// For a shadow: the family (boil key, matched as a prefix) of what casts it.
    pub caster: Option<&'static str>,
}
impl Default for Po {
    fn default() -> Self {
        Po { wash: None, wash_op: 255.0, fill: None, fill_op: 170.0, ink: Ink::Default, sw: 1.0, curv: 0.0, wc: WC, role: Role::Auto, caster: None }
    }
}
/// A flat colour shape with the default ink outline.
pub fn wash(c: Col) -> Po {
    Po { wash: Some(c), ..Po::default() }
}
/// A watercolour fill only (opacity 0–255). Add `.no_ink()` and it bleeds: soft, displaced edges.
pub fn fill_only(c: Col, op: f64) -> Po {
    Po { fill: Some(c), fill_op: op, ..Po::default() }
}
/// The kit's default bleed, texture and border for a watercolour layer.
pub const WC: [f64; 3] = [0.1, 0.4, 0.35];

impl Po {
    /// How far the watercolour layer bleeds past its shape (a share of each side's length; the kit's 0.1).
    pub fn bleed(mut self, v: f64) -> Po {
        self.wc[0] = v;
        self
    }
    /// How broken the watercolour layer is (0–1; the kit's 0.4).
    pub fn tex(mut self, v: f64) -> Po {
        self.wc[1] = v;
        self
    }
    /// How strongly pigment gathers at the watercolour layer's edge (0–1; the kit's 0.35).
    pub fn border(mut self, v: f64) -> Po {
        self.wc[2] = v;
        self
    }
    /// Wash opacity, 0–255.
    pub fn op(mut self, v: f64) -> Po {
        self.wash_op = v;
        self
    }
    /// Add a watercolour layer over the wash (opacity 0–255).
    pub fn fill(mut self, c: Col, op: f64) -> Po {
        self.fill = Some(c);
        self.fill_op = op;
        self
    }
    /// Outline colour.
    pub fn ink(mut self, c: Col) -> Po {
        self.ink = Ink::Col(c);
        self
    }
    /// No outline (`ink: null`).
    pub fn no_ink(mut self) -> Po {
        self.ink = Ink::None;
        self
    }
    /// Outline weight (~0.4–2; scales with the camera's zoom).
    pub fn sw(mut self, v: f64) -> Po {
        self.sw = v;
        self
    }
    /// 0–1: smooth the outline through the points.
    pub fn curv(mut self, v: f64) -> Po {
        self.curv = v;
        self
    }
    /// Say what the shape is (see [`Role`]).
    pub fn role(mut self, r: Role) -> Po {
        self.role = r;
        self
    }
    /// A cast shadow, and the family that casts it (a prefix of its boil key): the shape is a hint of where the
    /// shadow falls, and the painter derives its colour from what it falls on and the scene's light, its softness
    /// from how far it is from what casts it.
    pub fn shadow_of(mut self, caster: &'static str) -> Po {
        self.role = Role::Shadow;
        self.caster = Some(caster);
        self
    }
}

#[derive(Clone, Debug)]
pub enum Kind {
    /// A closed painted shape: optional wash, optional fill, optional outline.
    Shape { wash: Option<(Col, f64)>, fill: Option<(Col, f64)>, ink: Option<(Col, f64)>, curv: f64 },
    /// An open ink line.
    Line { col: Col, width: f64, curv: f64 },
    /// Additive light; pts = [centre, a point on the rim].
    Glow { col: Col, a: f64 },
}

#[derive(Clone, Debug)]
pub struct Item {
    /// Layer family: the boil key (glows get their own family).
    pub family: String,
    /// Identity within the family: call-site tag and its occurrence this frame.
    pub id: String,
    /// Drawn outside the camera (screen space).
    pub screen: bool,
    pub kind: Kind,
    pub pts: Vec<P>,
    /// A shape's watercolour bleed, texture and border.
    pub wc: [f64; 3],
    /// What it is, for the painter (glows are [`Role::Light`], lines [`Role::Line`], shapes as drawn).
    pub role: Role,
    /// For a shadow: the family that casts it.
    pub caster: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    pub cx: f64,
    pub cy: f64,
    pub zoom: f64,
    pub rot: f64,
}
impl Cam {
    /// The resting camera: world = screen.
    pub const REST: Cam = Cam { cx: W / 2.0, cy: H / 2.0, zoom: 1.0, rot: 0.0 };
    pub fn to_screen(&self, x: f64, y: f64) -> P {
        let (c, s) = (self.rot.cos(), self.rot.sin());
        let (dx, dy) = ((x - self.cx) * self.zoom, (y - self.cy) * self.zoom);
        [W / 2.0 + dx * c - dy * s, H / 2.0 + dx * s + dy * c]
    }
}

/// 2D affine [a b c d e f]: x' = a x + c y + e, y' = b x + d y + f.
#[derive(Clone, Copy, Debug)]
struct Mat([f64; 6]);
impl Mat {
    const ID: Mat = Mat([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    fn mul(&self, o: &Mat) -> Mat {
        let [a, b, c, d, e, f] = self.0;
        let [a2, b2, c2, d2, e2, f2] = o.0;
        Mat([a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2, a * e2 + c * f2 + e, b * e2 + d * f2 + f])
    }
    fn apply(&self, p: P) -> P {
        let [a, b, c, d, e, f] = self.0;
        [a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f]
    }
    fn scale_factor(&self) -> f64 {
        let [a, b, c, d, ..] = self.0;
        (a * d - b * c).abs().sqrt()
    }
}

/// One frame being painted. Shots receive it as `r` and call its drawing methods.
pub struct Rec {
    /// Song time of this frame.
    pub t: f64,
    /// Time into the current shot.
    pub lt: f64,
    /// Length of the current shot (0 outside a film).
    pub dur: f64,
    m: Mat,
    stack: Vec<Mat>,
    key: String,
    prefix: String,
    occ: HashMap<(String, &'static str), usize>,
    in_cam: bool,
    pub cam: Option<Cam>,
    pub last_cam: Option<Cam>,
    pub items: Vec<Item>,
    /// The light the frame is seen in (see [`Rec::light`]), and whether its place is on the screen (not in the
    /// camera's world).
    pub lighting: Option<(Lighting, bool)>,
}

impl Rec {
    pub fn new(t: f64) -> Rec {
        Rec::in_shot(t, t, 0.0)
    }
    /// A frame at song time t, lt into a shot of length dur.
    pub fn in_shot(t: f64, lt: f64, dur: f64) -> Rec {
        Rec {
            t,
            lt,
            dur,
            m: Mat::ID,
            stack: vec![],
            key: "frame".into(),
            prefix: String::new(),
            occ: HashMap::new(),
            in_cam: false,
            cam: None,
            last_cam: None,
            items: vec![],
            lighting: None,
        }
    }
    pub fn push(&mut self) {
        self.stack.push(self.m);
    }
    pub fn pop(&mut self) {
        self.m = self.stack.pop().unwrap_or(Mat::ID);
    }
    pub fn translate(&mut self, x: f64, y: f64) {
        self.m = self.m.mul(&Mat([1.0, 0.0, 0.0, 1.0, x, y]));
    }
    pub fn rotate(&mut self, a: f64) {
        let (c, s) = (a.cos(), a.sin());
        self.m = self.m.mul(&Mat([c, s, -s, c, 0.0, 0.0]));
    }
    pub fn scale(&mut self, sx: f64, sy: f64) {
        self.m = self.m.mul(&Mat([sx, 0.0, 0.0, sy, 0.0, 0.0]));
    }
    /// Start a new element (and layer family). Call it before each separate thing: a background layer, a prop, a
    /// character. Its key must be the same every frame.
    pub fn boil(&mut self, key: impl Into<String>) {
        self.key = key.into();
        self.prefix.clear();
    }
    /// World point (cx, cy) lands at screen centre. One level only: always pair with [`Rec::cam_end`].
    pub fn cam_begin(&mut self, cx: f64, cy: f64, zoom: f64, rot: f64) {
        let c = Cam { cx, cy, zoom, rot };
        self.cam = Some(c);
        self.last_cam = Some(c);
        self.in_cam = true;
    }
    pub fn cam_end(&mut self) {
        self.in_cam = false;
    }
    /// World → screen through the active camera (identity outside one). Aim an iris at a character with it.
    pub fn to_screen(&self, x: f64, y: f64) -> P {
        match (self.in_cam, self.cam) {
            (true, Some(c)) => c.to_screen(x, y),
            _ => [x, y],
        }
    }
    /// A sub-element inside the current family (a floret, a bite hole) with its own stable identity, so it stays
    /// in its parent's layer instead of becoming a layer of its own.
    pub fn sub(&mut self, prefix: impl Into<String>) {
        self.prefix = prefix.into();
    }
    fn ident(&mut self, family: &str, tag: &'static str) -> String {
        let n = self.occ.entry((format!("{family}/{}", self.prefix), tag)).or_insert(0);
        let id = format!("{}{tag}#{n}", self.prefix);
        *n += 1;
        id
    }
    fn xform(&self, pts: &[P]) -> Vec<P> {
        pts.iter().map(|p| self.m.apply(*p)).collect()
    }

    /// Paint one closed shape. `tag` names the call site (any string literal, the same every frame).
    pub fn paint(&mut self, tag: &'static str, pts: &[P], o: Po) {
        if pts.len() < 3 {
            return;
        }
        // outlines use the 'ink' brush (weight 5)
        let k = self.m.scale_factor() * 5.0 * BRUSH_PX;
        let ink = match o.ink {
            Ink::None => None,
            Ink::Default => Some((PAL_INK, o.sw * k)),
            Ink::Col(c) => Some((c, o.sw * k)),
        };
        let kind = Kind::Shape { wash: o.wash.map(|c| (c, o.wash_op / 255.0)), fill: o.fill.map(|c| (c, o.fill_op / 255.0)), ink, curv: o.curv };
        // a bare watercolour fill (no wash, no outline) bleeds: it gets its own layer, softened by effects
        let family = if o.wash.is_none() && o.fill.is_some() && o.ink == Ink::None { format!("{}@wc", self.key) } else { self.key.clone() };
        let id = self.ident(&family, tag);
        let pts = self.xform(pts);
        self.items.push(Item { family, id, screen: !self.in_cam, kind, pts, wc: o.wc, role: o.role, caster: o.caster });
    }

    /// The light this frame is seen in: where the key light is, its colour and strength, and the ambient light in
    /// the shadows (see [`Lighting`]). A place (`Lighting::at`) is taken through the current transform, like a
    /// shape's points (and, drawn inside the camera, through the camera). Without it the painter uses the film's
    /// `Style`, or a plain light from the upper left.
    pub fn light(&mut self, mut l: Lighting) {
        if let Some(p) = l.key_at {
            l.key_at = Some(self.m.apply(p));
        }
        self.lighting = Some((l, !self.in_cam));
    }

    /// An open ink line through the points (`curv` 0–1 smooths it). sw is the brush weight multiplier.
    pub fn ink_line(&mut self, tag: &'static str, pts: &[P], sw: f64, col: Col, br: Brush, curv: f64) {
        if pts.len() < 2 {
            return;
        }
        let weight = match br {
            Brush::Ink => 5.0,
            Brush::InkFine => 2.6,
            Brush::Dry => 14.0,
        };
        let width = sw * weight * BRUSH_PX * self.m.scale_factor();
        let family = self.key.clone();
        let id = self.ident(&family, tag);
        let pts = self.xform(pts);
        self.items.push(Item { family, id, screen: !self.in_cam, kind: Kind::Line { col, width, curv }, pts, wc: WC, role: Role::Line, caster: None });
    }

    /// Additive halos: (x, y, r, colour, a). Light adds, so it shows on dark grounds and barely on light ones.
    /// Draw it before the things that sit in front of the light.
    pub fn glows(&mut self, tag: &'static str, list: &[(f64, f64, f64, Col, f64)]) {
        let family = format!("+{}", self.key);
        for &(x, y, r, col, a) in list {
            if a <= 0.01 || r <= 1.0 {
                continue;
            }
            let id = self.ident(&family, tag);
            let pts = self.xform(&[[x, y], [x + r, y]]);
            self.items.push(Item {
                family: family.clone(),
                id,
                screen: !self.in_cam,
                kind: Kind::Glow { col, a: a.min(1.0) },
                pts,
                wc: WC,
                role: Role::Light,
                caster: None,
            });
        }
    }
    /// One additive halo.
    pub fn glow(&mut self, tag: &'static str, x: f64, y: f64, r: f64, col: Col, a: f64) {
        self.glows(tag, &[(x, y, r, col, a)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_stable_per_tag_and_family() {
        let mut r = Rec::new(0.0);
        r.boil("hero");
        r.paint("body", &[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], wash([1.0; 3]));
        r.paint("body", &[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], wash([1.0; 3]));
        r.sub("arm.");
        r.ink_line("line", &[[0.0, 0.0], [5.0, 5.0]], 1.0, [0.0; 3], Brush::Ink, 0.0);
        r.boil("sky");
        r.paint("band", &[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], fill_only([1.0; 3], 100.0).no_ink());
        let ids: Vec<(&str, &str)> = r.items.iter().map(|i| (i.family.as_str(), i.id.as_str())).collect();
        assert_eq!(ids, [("hero", "body#0"), ("hero", "body#1"), ("hero", "arm.line#0"), ("sky@wc", "band#0")]);
    }

    #[test]
    fn camera_marks_world_items_and_transforms_apply() {
        let mut r = Rec::new(0.0);
        r.cam_begin(100.0, 100.0, 2.0, 0.0);
        r.push();
        r.translate(10.0, 20.0);
        r.scale(2.0, 2.0);
        r.paint("a", &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], wash([0.0; 3]));
        r.pop();
        assert_eq!(r.to_screen(100.0, 100.0), [W / 2.0, H / 2.0]);
        r.cam_end();
        r.paint("b", &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], wash([0.0; 3]));
        assert!(!r.items[0].screen && r.items[1].screen);
        assert_eq!(r.items[0].pts, vec![[10.0, 20.0], [12.0, 20.0], [10.0, 22.0]]);
        // the outline scales with the transform: weight 1 × 5 × 0.75 × 2
        let Kind::Shape { ink: Some((_, w)), .. } = r.items[0].kind else { panic!("no ink") };
        assert_eq!(w, 7.5);
    }
}
