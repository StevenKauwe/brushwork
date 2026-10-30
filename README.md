# brushwork

An oil painter in pure Rust. Give it flat shapes, the frames of an animated film or a photograph, and it paints
them the way a painter would. Each shape is covered with bristle strokes that follow its form, with the brush picked
for the job (flat, filbert, round, palette knife). Strokes pick up the wet paint underneath and mix it as pigment
(Kubelka-Munk, 38 bands). The ridges of paint they leave are lit at the end. It is also **steady**: an animated
shape keeps its brushwork from frame to frame, and only moves where the shape moves.

It is the painter behind the oil-painted films in music-to-movie-magic and CrabScape's oil-painted art.

**Try it:** [the canvas page](https://stevenkauwe.github.io/brushwork/). Draw flat shapes with a finger or the
mouse, then press Paint. The engine runs in your browser, compiled to WebAssembly in a Web Worker. Where the browser
offers WebGPU it lays the strokes on the graphics card (the label on the canvas says `GPU`), otherwise on the
processor (`CPU`; `?backend=cpu` asks for it). On a desktop the 1024 x 576 canvas takes about 0.4 s on the GPU and
1 s on the CPU. A drawn outline is calmed before it is painted: the hand's jitter comes out, while its size, its ends
and any corner the hand paused on stay where they were.

## Crates

| crate | what it is |
|---|---|
| [`brushwork`](crates/brushwork) | the engine: strokes, the frame painter (`run`, `run_clip`, `paint_frame`), photo painting (`run_photo`), and the optional GPU backend |
| [`brushwork-paint`](crates/brushwork-paint) | the shape recorder a film's shots draw with (`Rec`: paint, ink lines, glows, camera), `Film`/`Shot`, look presets, transitions |
| [`brushwork-kit`](crates/brushwork-kit) | the brush kit: time, easing, motion, geometry, colour, and the spectral pigment mixer |
| [`tools/flickscan`](tools/flickscan) | finds flickering strokes in a rendered frame sequence |
| [`brushwork-web`](crates/brushwork-web) | the canvas page (wasm-bindgen + web-sys); `tools/build-web.sh` builds it into `dist/` |
| [`tools/webcheck`](tools/webcheck) | opens the page in WebKit or Chromium through WebDriver and checks it paints, says which backend painted, and (`--compare`) that the browser's GPU agrees with its CPU |

## Use

```rust
use brushwork::{Style, paint_frame, set_style};
use brushwork_paint::prelude::*;

fn shot(r: &mut Rec, t: f64, _lt: f64) {
    r.boil("sky");
    r.paint("sky", &rect_pts(-60.0, -60.0, 2040.0, 1200.0), wash(rgb(0x1B2A5A)).no_ink());
    r.boil("owl");
    r.paint("owl", &ell_pts(960.0, 540.0 + 30.0 * (t * 3.0).sin(), 140.0, 180.0, 32, 0.0), wash(rgb(0xB5652F)));
}

let film = Film::new("demo", 2.0, vec![Shot::new(0.0, "night", shot)], env!("CARGO_MANIFEST_DIR"));
set_style(Style { cast: &["owl"], ..Style::DEFAULT }); // the cast is painted as forms in light
let rgb: Vec<u8> = paint_frame(&film, 0.5, 1);         // 1920 x 1080, 8-bit RGB
```

The whole program is `cargo run --release --example frame -- out.png`. A [`Style`](crates/brushwork/src/lib.rs) says
what is special about a film: its cast (painted with smaller brushes and thicker paint, modelled in light, and
throwing shadows), its airy families (watercolour in mixed media), and so on. Everything else is the same for every
film.

A film's `main` usually hands its command line to `brushwork::run` (one frame: `oil <t> <out.png>`) and
`brushwork::run_clip` (a stretch: `oilclip <t0> <t1> <dir>`). `brushwork::run_photo_with` paints a photograph.
`Painting`, `paint_body` and the p5.brush-style `brush_fill` / `brush_wash` / `brush_stroke` lay strokes directly.

## Saying what a shape is, and the light it is in

Unlabelled shapes are painted from what the painter can infer from their size, colour and place. A film can instead
say what a shape **is** (`brushwork_paint::Role`, with `Po::role`) and the light the scene is in (`Lighting`, on the
`Style` or per shot with `Rec::light`), and the painter picks the technique a painter would:

| role | painted as |
|---|---|
| `Body` | a form or object: blocked in, and for the cast modelled by the light |
| `Background`, `Sky`, `Haze` | calm planes, wet into wet; the sky and haze fused, edges lost |
| `Shadow` (`Po::shadow_of(caster)`) | a hint of where a caster's shadow falls, not paint: a glaze over whatever it falls on, each colour under it taken down a value step as deep as the key light is strong, with a touch of the palette's shadow colour; crisp where it leaves the caster, softening with distance |
| `FormShadow` | the same derivation glazed into a body, its edges lost |
| `Glaze` | a transparent tint: the paint below shows through |
| `Light` | a light source (moon, lamp, window): clean luminous paint, softly modelled, fully covered, no cut-in edge |
| `Highlight`, `Accent`, `Line` | a clean loaded touch; a crisp small detail; one pull of a rigger |

Light falls on what is already painted. A `Light` lights the air and the paint round it (as far as the scene's
`Lighting::haze` carries it, and a brighter aureole close in), and so do the film's glows (`Rec::glows`): the strokes
there are lifted towards the light's colour, mixed with theirs as pigment, more on the ridges of their paint that
catch it and less in the hollows between. The light has no texture of its own: as it moves (a moon rising, a lamp
carried past a wall) the brushwork stays where it is and each stroke only grows lighter or darker
(`a_light_gliding_over_a_still_sky_moves_no_texture`). Forms in the cast turn from the declared key light over a soft
terminator into a core shadow, with reflected light at the far rim, in their own colour mixed down with the palette's
darks (`Style::palette`), never grey or black. `Lighting::MOONLIGHT`, `FIRELIGHT`, `DAYLIGHT`, `OVERCAST` and `SUNSET`
are ready-made; `.at(point)` places the key light in the picture (shadows fall away from it), `.from_dir(dir)` aims it.

```rust
r.light(Lighting::MOONLIGHT.at([1480.0, 300.0]));
r.paint("moon", &ell_pts(1480.0, 300.0, 130.0, 130.0, 32, 0.0), wash(rgb(0xF5F0DC)).no_ink().role(Role::Light));
r.paint("owl", &owl_body, wash(rgb(0xB5652F)));
r.paint("owl.shadow", &where_it_falls, wash(rgb(0x1D1A1F)).no_ink().shadow_of("owl"));
```

## CPU and GPU

The CPU engine is the reference. A frame takes about 3-6 s on one core, and `run_clip` paints frames in parallel.
With the `gpu` feature and `OIL_BACKEND=gpu` (or `brushwork::set_backend(Backend::Gpu)`), the CPU still plans
every stroke, and the GPU (wgpu) lays them, one thread per pixel walking its tile's strokes in the CPU's order. On an
RTX 3090 that is about 14 times all 16 CPU threads. A handful of colour values per frame differ from the CPU's, by 1
to 5 levels out of 255, from the GPU's own `sin`, `exp` and division (and very rarely a pixel right at a threshold
falls the other way: at most 0.01 % of values may differ, 0.001 % by more than 5 levels; `the_gpu_paints_what_the_cpu_paints`
and `webcheck --compare` hold it to that). Frames with watercolour or bite holes stay on the CPU.

In a browser nothing may block waiting for the GPU, so there the backend is opened with `brushwork::open_gpu().await`
and frames painted with `brushwork::paint_frame_async(...).await` (the canvas page does this in its Web Worker,
compute only, no canvas). Natively `paint_frame` is unchanged.

## Steadiness, and how it is tested

Everything about a stroke comes from the shape it belongs to and its place on that shape, never from list order, the
frame number or where the canvas edge falls. Wherever a decision has a threshold (a size, a direction, inside or
outside), the two sides are cross-faded, so a shape that changes by a pixel changes the painting by a pixel's worth.

The tests in [`crates/brushwork/src/lib.rs`](crates/brushwork/src/lib.rs) hold the engine to that. Each one comes
from a flicker that was once seen in a film:

- `stroke_slots_fade_in_and_out_at_an_edge_never_pop`, `a_slow_zoom_moves_no_stroke_in_or_out_at_an_edge`
- `a_strokes_edge_direction_never_flips_between_two_equally_near_edges`, `a_shape_that_swells_a_little_keeps_its_brushwork`
- `an_outline_that_flexes_keeps_its_sweeps`, `a_small_shape_growing_through_every_size_threshold_never_pops`
- `level_strokes_end_raggedly_not_on_a_vertical_seam`, `same_shape_same_place_paints_identically`
- `a_fade_fades_and_leaves_without_a_pop`, `a_translucent_wash_is_mixed_down_not_glazed_in_streaks`
- `a_strap_widening_from_a_line_to_an_area_never_pops`, `close_thin_strips_stay_apart`
- `a_shadow_is_a_glaze_that_keeps_what_it_falls_on`, `a_shadow_fading_in_and_a_glow_growing_never_pop`
- `a_light_gliding_over_a_still_sky_moves_no_texture`, ...

`cargo test --release` runs them on the CPU; `OIL_TEST_BACKEND=gpu cargo test --release --features brushwork/gpu`
runs them on the GPU. For a rendered film, `flickscan <dir> <first> <last>` lists any block that changes for a frame
or two and reverts while its surroundings hold still.

## Switches

`OIL_BACKEND=gpu` picks the GPU. Photo painting reads `OIL_NOCTURNE`, `OIL_ATTENTION`, `OIL_KEY` and `OIL_GROUND`
(or take a `PhotoOptions`). `OIL_DEPTH` turns on the depth style map. `OIL_TIMING`, `OIL_TRACE`, `OIL_DBG` and
`OIL_ZOOM` are for debugging.

## Licence

MIT, see [LICENSE](LICENSE). The spectral mixing tables come from [spectral.js](https://github.com/rvanwijnen/spectral.js)
(MIT, Ronald van Wijnen); its notice is at the top of
[`crates/brushwork-kit/src/spectral.rs`](crates/brushwork-kit/src/spectral.rs).
