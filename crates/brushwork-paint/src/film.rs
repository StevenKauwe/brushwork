//! A film: its shots (pure functions of t that paint a whole frame), its length, its look, and where its song and
//! renders live.

use std::path::{Path, PathBuf};

use crate::{Look, Rec};

/// A shot function: `f(r, t, lt)` paints the whole frame at song time t, lt seconds into the shot
/// (`r.dur` is the shot's length).
pub type ShotFn = fn(&mut Rec, f64, f64);

/// One shot: it runs from `t0` until the next shot's `t0` (or the end of the film).
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub t0: f64,
    pub name: &'static str,
    pub f: ShotFn,
    /// This shot's paint movement, when it differs from the film's (a chorus at `medium`).
    pub look: Option<Look>,
}
impl Shot {
    pub const fn new(t0: f64, name: &'static str, f: ShotFn) -> Shot {
        Shot { t0, name, f, look: None }
    }
    /// Give this shot its own look preset.
    pub const fn look(mut self, l: Look) -> Shot {
        self.look = Some(l);
        self
    }
}

pub struct Film {
    pub title: String,
    /// The song folder's name (songs/<slug>).
    pub slug: String,
    /// Seconds: the song's length.
    pub duration: f64,
    /// In time order.
    pub shots: Vec<Shot>,
    /// The film's paint movement (default subtle); `--look` overrides it and every shot's own.
    pub look: Look,
    /// The song, muxed under the picture (None renders silent video).
    pub audio: Option<PathBuf>,
    /// Where renders, sheets and stills go (songs/<slug>/out).
    pub out_dir: PathBuf,
}

impl Film {
    /// A film whose crate lives in `songs/<slug>/film/`: pass `env!("CARGO_MANIFEST_DIR")` as `crate_dir`. The song
    /// is `songs/<slug>/assets/song.m4a` and renders go to `songs/<slug>/out/`.
    pub fn new(title: &str, duration: f64, shots: Vec<Shot>, crate_dir: &str) -> Film {
        let song = Path::new(crate_dir).parent().map(Path::to_path_buf).unwrap_or_default();
        let slug = song.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "film".into());
        Film { title: title.into(), slug, duration, shots, look: Look::SUBTLE, audio: Some(song.join("assets/song.m4a")), out_dir: song.join("out") }
    }
    pub fn look(mut self, l: Look) -> Film {
        self.look = l;
        self
    }
    pub fn slug(mut self, s: &str) -> Film {
        self.slug = s.into();
        self
    }
    pub fn audio(mut self, p: impl Into<PathBuf>) -> Film {
        self.audio = Some(p.into());
        self
    }
    pub fn out_dir(mut self, p: impl Into<PathBuf>) -> Film {
        self.out_dir = p.into();
        self
    }

    /// Shots must start in order, inside the film.
    pub fn check(&self) -> Result<(), String> {
        if self.shots.is_empty() {
            return Err("the film has no shots".into());
        }
        for (i, w) in self.shots.windows(2).enumerate() {
            if w[1].t0 <= w[0].t0 {
                return Err(format!("shot {} ({}) starts at {} s, not after shot {} ({}) at {} s", i + 2, w[1].name, w[1].t0, i + 1, w[0].name, w[0].t0));
            }
        }
        if let Some(s) = self.shots.iter().find(|s| s.t0 < 0.0 || s.t0 >= self.duration) {
            return Err(format!("shot {} starts at {} s, outside the film (0–{} s)", s.name, s.t0, self.duration));
        }
        Ok(())
    }
    /// End time of shot i.
    pub fn end_of(&self, i: usize) -> f64 {
        self.shots.get(i + 1).map(|s| s.t0).unwrap_or(self.duration)
    }
    /// The shot playing at song time t.
    pub fn shot_at(&self, t: f64) -> usize {
        self.shots.iter().rposition(|s| s.t0 <= t).unwrap_or(0)
    }
    /// The look shot i is painted with, given an optional override.
    pub fn look_of(&self, i: usize, over: Option<Look>) -> Look {
        over.or(self.shots[i].look).unwrap_or(self.look)
    }
    /// Record one frame of shot i at song time t.
    pub fn record(&self, i: usize, t: f64) -> Rec {
        let s = &self.shots[i];
        let mut r = Rec::in_shot(t, t - s.t0, self.end_of(i) - s.t0);
        (s.f)(&mut r, t, t - s.t0);
        r
    }
}
