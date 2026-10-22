//! Paint one frame of a tiny film: a night sky with an owl-coloured shape bobbing in it.
//!
//!     cargo run --release --example frame -- out.png        (OIL_BACKEND=gpu with --features gpu for the GPU)

use brushwork::{Style, paint_frame, set_style};
use brushwork_paint::prelude::*;

fn shot(r: &mut Rec, t: f64, _lt: f64) {
    r.boil("sky");
    r.paint("sky", &rect_pts(-60.0, -60.0, 2040.0, 1200.0), wash(rgb(0x1B2A5A)).no_ink());
    r.boil("owl");
    r.paint("owl", &ell_pts(960.0, 540.0 + 30.0 * (t * 3.0).sin(), 140.0, 180.0, 32, 0.0), wash(rgb(0xB5652F)));
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "frame.png".into());
    let film = Film::new("demo", 2.0, vec![Shot::new(0.0, "night", shot)], env!("CARGO_MANIFEST_DIR"));
    set_style(Style { cast: &["owl"], ..Style::DEFAULT });
    let rgb: Vec<u8> = paint_frame(&film, 0.5, 1); // 1920 x 1080, 8-bit RGB
    if let Some(Err(e)) = image::RgbImage::from_raw(1920, 1080, rgb).map(|im| im.save(&out)) {
        eprintln!("could not write {out}: {e}");
    }
}
