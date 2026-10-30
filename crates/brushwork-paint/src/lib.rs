//! The painting API a film's shots draw with: the recorder ([`Rec`]: paint / ink_line / glows / camera / boil
//! keys) and the full-frame effects and transitions ([`fx`]). Pure Rust; EffectCraft never appears here.

mod film;
pub mod fx;
mod look;
mod rec;
mod semantics;

pub use film::{Film, Shot, ShotFn};
pub use look::Look;
pub use rec::{Brush, Cam, Ink, Item, Kind, PAL_INK, Po, Rec, fill_only, wash};
pub use semantics::{Lighting, Role};

/// Everything a film's shots draw with, EffectCraft-free: the kit, the recorder, films and shots, the look presets
/// and the full-frame effects. (`studio::prelude` is the same set.)
pub mod prelude {
    pub use crate::fx::*;
    pub use crate::{Brush, Cam, Film, Lighting, Look, PAL_INK, Po, Rec, Role, Shot, fill_only, wash};
    pub use brushwork_kit::*;
    pub use std::f64::consts::PI;
}
