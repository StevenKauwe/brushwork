//! What a shape IS, and the light it is seen in: the words a film uses to tell the painter how to paint, instead of
//! leaving it to guess from geometry and colour.
//!
//! A shape's [`Role`] says what it is in the picture (a body, the sky, a cast shadow, a light...). Unlabelled shapes
//! are [`Role::Auto`]: the painter infers what it can, as it always has. A scene's [`Lighting`] says where the key
//! light comes from, its colour and strength, and the colour of the ambient light that fills the shadows; the
//! painter derives every shadow and every turn of a form from it (and from the scene's palette), the way a painter
//! reasons about a shadow rather than reaching for black.

use brushwork_kit::Col;

/// What a shape is, for the painter. Each role chooses a technique:
///
/// | role | what it is | how it is painted |
/// |---|---|---|
/// | `Auto` | not said | inferred (a translucent dark wash is a glaze, a pale disc in a dark sky a light, ...) |
/// | `Body` | a form or object in its local colour | blocked in, modelled by the light if it is in the cast |
/// | `Background` | a plane of scenery | calm, broad, wet into wet with its neighbours |
/// | `Sky` | the sky or the far distance | fused, close in value, edges lost |
/// | `Haze` | atmosphere, mist, a veil | a thin scumble of its colour over what is behind |
/// | `Shadow` | where a caster's shadow falls | derived: the receiver's own colour, darkened and turned towards the ambient light with the palette, glazed, edges crisp at the caster and softening away from it |
/// | `FormShadow` | the side of a body turned from the light | the same derivation, glazed into the body, its edges lost |
/// | `Highlight` | where the light catches | a loaded touch of light paint (a knife on a small one) |
/// | `Glaze` | a transparent tint over what is below | a thin coloured layer: what is below shows through |
/// | `Light` | a light source (moon, lamp, sun, window) | clean luminous paint, softly modelled, a crisp but not cut-out edge; it lights what is round it (see below) |
/// | `Accent` | a small detail that must read | crisp, small brush, clean paint |
/// | `Opening` | a hole in something (a bite in a leaf), and what is seen through it ([`Optics::opening`]) | painted on as that far layer's own paint, calm and clean; its rim lit or dark as the light dictates; strong light through it spills a little of the near surface's own backlit colour round it |
/// | `Translucent` | something light passes through: a leaf lit from behind, a wing, a membrane, frosted glass, mist ([`Optics::translucent`]) | painted on, never overlaid: the material's own colour mixed lighter, warmer and richer where light passes through; frosted, a softened, light, opaque rendering of what is behind it, scumbled on |
/// | `Line` | linework drawn as a shape | one pull of a rigger |
///
/// Light falls on what is already painted: a light (and a glow, `Rec::glows`) lifts the paint round it towards its
/// colour, in a few soft value steps broken by that paint's own brushwork, so nothing travels with the light but
/// the light itself. How far a light's own glow reaches goes with the air's [`Lighting::haze`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Role {
    #[default]
    Auto,
    Body,
    Background,
    Sky,
    Haze,
    Shadow,
    FormShadow,
    Highlight,
    Glaze,
    Light,
    Accent,
    Line,
    Opening,
    Translucent,
}

/// What a shape does with light, for the roles that need it (`Opening`, `Translucent`): the optics of the subject,
/// from which the painter derives the paint. Nothing here is a layer operation: an opening is not erased, a
/// translucent shape is not made transparent; both are painted on, in paint mixed for what the eye sees.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Optics {
    /// What is seen through an opening, or behind a translucent shape: its colour (0-255). None: what was painted
    /// there before the thing the opening is cut in (or the translucent shape itself) was painted over it.
    pub behind: Option<Col>,
    /// How much light comes through (0 to 1): the sun through a bite, a lamp behind a membrane.
    pub light: f64,
    /// How much it scatters what passes through (0 to 1): 0 clear (a leaf lit from behind, a wing), 1 frosted glass
    /// or mist (what is behind is seen only as soft, light shapes of its colour).
    pub diffuse: f64,
}

impl Optics {
    /// An opening through which `behind` is seen (None: whatever was painted behind the thing it is cut in), with
    /// `light` (0 to 1) coming through it.
    pub const fn opening(behind: Option<Col>, light: f64) -> Optics {
        Optics { behind, light, diffuse: 0.0 }
    }
    /// Something light passes through: `light` (0 to 1) of it, scattered by `diffuse` (0 clear .. 1 frosted), with
    /// `behind` (if known) behind it.
    pub const fn translucent(light: f64, diffuse: f64, behind: Option<Col>) -> Optics {
        Optics { behind, light, diffuse }
    }
}

/// The light a scene is seen in. Colours are 0-255 like every other colour a film draws with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Where the key light comes from, as a direction on the canvas towards it (x right, y down; need not be unit
    /// length). Ignored when `key_at` is set.
    pub key_dir: [f64; 2],
    /// A key light at a place in the picture (the moon, a lamp), in the same coordinates as the shapes drawn after
    /// it: forms turn towards it and shadows fall away from it.
    pub key_at: Option<[f64; 2]>,
    /// The key light's colour (its temperature: moonlight is cool, firelight warm).
    pub key: Col,
    /// How strongly the key beats the ambient light, 0 to 1: 0.85 a hard moon or sun (deep, clear shadows), 0.4
    /// overcast (shadows only a little darker than the light).
    pub strength: f64,
    /// The ambient light filling the shadows (the sky's colour): shadows are turned towards it.
    pub ambient: Col,
    /// How hazy the air is, 0 to 1: how far a light's glow spreads into it (a clear night 0.3, mist 0.8).
    pub haze: f64,
}

impl Lighting {
    /// A clear night under a full moon: a cool silver key, a deep blue sky filling the shadows.
    pub const MOONLIGHT: Lighting =
        Lighting { key_dir: [-0.6, -0.8], key_at: None, key: [214.0, 226.0, 255.0], strength: 0.72, ambient: [40.0, 58.0, 120.0], haze: 0.35 };
    /// Firelight or a lamp: a warm orange key, cool dark shadows.
    pub const FIRELIGHT: Lighting =
        Lighting { key_dir: [-0.6, -0.8], key_at: None, key: [255.0, 176.0, 98.0], strength: 0.8, ambient: [52.0, 50.0, 92.0], haze: 0.5 };
    /// Sunlight on a clear day: a warm white key, blue sky in the shadows.
    pub const DAYLIGHT: Lighting =
        Lighting { key_dir: [-0.64, -0.77], key_at: None, key: [255.0, 244.0, 222.0], strength: 0.7, ambient: [120.0, 150.0, 210.0], haze: 0.15 };
    /// An overcast sky: soft, cool, little difference between light and shade.
    pub const OVERCAST: Lighting =
        Lighting { key_dir: [-0.3, -0.95], key_at: None, key: [232.0, 236.0, 240.0], strength: 0.4, ambient: [150.0, 160.0, 175.0], haze: 0.7 };
    /// Sunset: a low orange key, violet shadows.
    pub const SUNSET: Lighting =
        Lighting { key_dir: [0.95, -0.3], key_at: None, key: [255.0, 160.0, 96.0], strength: 0.75, ambient: [92.0, 70.0, 140.0], haze: 0.45 };

    /// The same light coming from `dir` (towards the light, on the canvas).
    pub const fn from_dir(mut self, dir: [f64; 2]) -> Lighting {
        self.key_dir = dir;
        self.key_at = None;
        self
    }
    /// The same light placed at a point in the picture (a moon, a lamp).
    pub const fn at(mut self, p: [f64; 2]) -> Lighting {
        self.key_at = Some(p);
        self
    }
}
