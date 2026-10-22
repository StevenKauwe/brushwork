//! How much the paint itself moves between drawings, as four presets. `subtle` is the default
//! (docs/animation-guide.md, rule 1: "the paint breathes, gently and broadly"). Plain data: the look crate's
//! Paper & Boil effect reads it.

/// How much the paint moves between drawings (the brush engine repaints every wash each drawing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    /// Wash breathing: strength of broad tone swells inside washes (0 = still, pinned to the paper).
    pub wash: f64,
    /// Wash breathing drawings per second.
    pub wash_fps: f64,
    /// 1 = each swell fades into the next; 0 = it snaps on every drawing (the brush engine's flicker).
    pub wash_smooth: f64,
    /// Edge creep: Turbulent Displace evolution on bleeding fills, in degrees per second (0 = still).
    pub edge_deg_s: f64,
    /// Edge creep snaps to a new shape on each drawing (12/s) instead of drifting.
    pub edge_snap: bool,
}

impl Look {
    /// No paint movement beyond the linework boil.
    pub const OFF: Look = Look { wash: 0.0, wash_fps: 12.0, wash_smooth: 1.0, edge_deg_s: 0.0, edge_snap: false };
    /// The default: tone swells in blotches 250–520 px across, 4 a second, each fading into the next; fill edges
    /// drift slowly.
    pub const SUBTLE: Look = Look { wash: 0.04, wash_fps: 4.0, wash_smooth: 1.0, edge_deg_s: 40.0, edge_snap: false };
    /// Choruses and party beats: the brush engine's own level.
    pub const MEDIUM: Look = Look { wash: 0.07, wash_fps: 8.0, wash_smooth: 0.6, edge_deg_s: 110.0, edge_snap: false };
    /// Snapping 12 a second: a one-off accent only.
    pub const STRONG: Look = Look { wash: 0.09, wash_fps: 12.0, wash_smooth: 0.0, edge_deg_s: 0.0, edge_snap: true };

    pub const PRESETS: [(&'static str, Look); 4] = [("off", Look::OFF), ("subtle", Look::SUBTLE), ("medium", Look::MEDIUM), ("strong", Look::STRONG)];

    /// A preset by name: off, subtle, medium or strong.
    pub fn by_name(name: &str) -> Option<Look> {
        Look::PRESETS.iter().find(|(n, _)| *n == name).map(|(_, l)| *l)
    }
    pub fn name(&self) -> &'static str {
        Look::PRESETS.iter().find(|(_, l)| l == self).map(|(n, _)| *n).unwrap_or("custom")
    }
}

impl Default for Look {
    fn default() -> Self {
        Look::SUBTLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtle_is_the_default_and_names_round_trip() {
        assert_eq!(Look::default(), Look::SUBTLE);
        for (n, l) in Look::PRESETS {
            assert_eq!(Look::by_name(n), Some(l));
            assert_eq!(l.name(), n);
        }
        assert_eq!(Look::by_name("loud"), None);
    }

    #[test]
    fn presets_get_stronger_in_order() {
        let w: Vec<f64> = Look::PRESETS.iter().map(|(_, l)| l.wash).collect();
        assert!(w.windows(2).all(|p| p[0] < p[1]), "{w:?}");
    }
}
