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

### A canvas of any size, and the paint as a layer

`Painting::new()` is the film's canvas, 1920 x 1080 on a toned ground. For a strip, a tile or a sprite, make a canvas
the size of what is painted (it is quicker to make and to light, and much smaller) and lift the paint off it as a
straight-alpha layer to put over anything:

```rust
let mut p = Painting::with_size(1468, 220, [1.0, 0.988, 0.969]); // a strip on the page's colour
p.keep_coverage();                                               // opt in: how much of each pixel the strokes cover
for s in &strokes {
    p.stroke(s, None);
}
let rgba: Vec<u8> = p.layer_rgba().unwrap(); // 1468 x 220 RGBA: the paint's own colour, alpha its coverage
```

The layer's relief is lit as `lit()` lights it, with the canvas weave only under paint, so bare canvas is fully
transparent; where the paint covers the canvas completely its colour is exactly `lit()`'s. `stroke`, `lit`,
`keep_coverage` and `layer_rgba` work at any size; the scene painters (`paint_frame`, `paint_body`, `brush_fill` and
the rest, the photo painters) frame their shapes on 1920 x 1080 and want `new()`. Coverage is kept by the CPU's
strokes (a painting recorded for the GPU does not keep it). To show a painting as it is being painted, lift only what
changed: `p.layer_rgba_in([x, y], [w, h])` is exactly that rectangle of `layer_rgba()`, for the price of lighting it.
To paint only the part of a long strip that shows, make a canvas that size `.with_origin([x, y])` (where it sits on the
strip) and lay the strokes that reach it, moved by `-x, -y`: it lights exactly as the whole strip does there, but for
its outermost two pixels.

### One sweep along a path

A long stroke (a stripe, a ribbon, a signature) is one sweep of the brush laid as straight pieces that share its
bristles, each knowing where along the whole stroke it is: the brush lands once, its paint thins and dries as it goes,
it lifts once, and its marks run unbroken through the joins. `sweep_path` turns a path into those pieces, ready to
lay; the width is even, given at each point, a broad nib's (wide across its edge, thin along it:
`width * |sin(dir - angle)|`), or anything else piece by piece:

```rust
use brushwork::{Nib, Painting, Picked, Sweep, Tool, Width, sweep_path};

let (terracotta, marigold) = ([0.85, 0.33, 0.18], [0.94, 0.66, 0.23]);
// a stripe that runs low, rises 120 px through an S and runs on high (canvas pixels)
let path: Vec<[f32; 2]> = (0..=300)
    .map(|i| {
        let x = i as f32 * 4.0;
        let t = ((x - 380.0) / 160.0).clamp(0.0, 1.0);
        [x, 170.0 - 120.0 * t * t * (3.0 - 2.0 * t)]
    })
    .collect();
let nib = Nib { width: 27.0, angle: 20f32.to_radians(), edge: 0.3 }; // a flat held at 20 degrees
// the brush carries more of its neighbour's wet paint the further it goes
let drag = |t: f32| std::array::from_fn(|k| terracotta[k] + (marigold[k] - terracotta[k]) * 0.3 * t);
let stripe = Sweep { load: 1.35, pickup: 1.6, picked: Picked::Along(&drag), ..Sweep::new(Tool::Flat, Width::Nib(nib), terracotta, 0x51A7) };

let mut p = Painting::with_size(1200, 220, [1.0, 0.988, 0.969]);
p.keep_coverage();
for s in sweep_path(&path, &stripe) {
    p.stroke(&s, None);
}
let rgba = p.layer_rgba().unwrap();
```

A nib whose pressure comes and goes is `Width::By(&|i, dir| nib.mark(dir) * press[i])`. The strokes depend only on the
path and the sweep, never on the canvas, so the same sweep paints the same anywhere (moved by whole pixels, it lays the
same paint, moved). A stroke laid as several paths gives each its stretch of the whole (`span`, and `along`), so it
lands once and lifts once.

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
| `Opening` (`Po::opening(behind, light)`) | a hole (a bite in a leaf): painted on as the far layer's own paint (what was painted behind the thing it is cut in, unless the film says what), lighter where light comes through; its rim dark or backlit as the light dictates; strong light spills the near surface's own backlit colour round it. Nothing is erased |
| `Translucent` (`Po::translucent(light, diffuse, behind)`) | lit from behind (a leaf, a wing, a membrane): the material's own colour mixed lighter, richer and warmer; frosted (`diffuse` 1): an opaque, light, softened rendering of what is behind it. Never a transparency or an overlay |

Light falls on what is already painted. A `Light` lights the air and the paint round it (as far as the scene's
`Lighting::haze` carries it, and a brighter aureole close in), and so do the film's glows (`Rec::glows`): the strokes
there are mixed lighter in their own colour with a little of the light's, more on the ridges of their paint that
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

## Fast photo painting (native and browser)

The fast photo painter developed for ShapeStage is now in the engine. It is an **opt-in approximation**:
a small image proxy guides strokes, paths are simplified, pigment tables are reused, and blur/relief work is
parallel on native CPUs. The GPU lays batches of strokes. The detailed photo painter, semantic shape renderer,
and 0.5.0's openings, translucent materials and lighting remain available.

With the `gpu` feature, `paint_photo_image_fast_async` runs on native wgpu (Metal, Vulkan or DX12 as available)
and browser WebGPU. Native callers can also use the blocking `paint_photo_image_fast`,
`paint_photo_image_steady`, or `paint_photo_image_fast_at_size` wrappers. No unsafe Rust is used in the engine;
the GPU command layout is checked by `bytemuck` derives and a size assertion.

```rust,ignore
use brushwork::{Dynamics, FastPhotoOptions, FastPhotoResolution, PhotoOptions};
let options = PhotoOptions { attention: false, ground_subject: true, ..Default::default() };
brushwork::open_gpu().await?; // required in the browser; also works natively
let painted = brushwork::paint_photo_image_fast_async(
    input_rgb, &options, [0.5, 0.5, 0.34], Dynamics::default(), 1,
    FastPhotoOptions { resolution: FastPhotoResolution::P720, steady: true, upscale: true },
    &mut |phase, completed, total| Ok(()), // return Err to cancel between work units
).await?;
```

Working canvases are 480×270, 960×540, 1280×720 and 1920×1080. Output is cropped to the input aspect ratio;
portrait images are turned back upright. `upscale: false` prevents enlargement of a small source. Resolution
changes the actual painting work, rather than just resizing the finished pixels. `steady` keeps a seeded stroke
layout across photos; it does not track moving objects. `Dynamics` supplies per-painting stroke size, wetness,
pickup, thickness, bristles, soft edges and contour controls. Neutral values retain the detailed painter's defaults.

The async API returns GPU errors, including an explicit busy error for overlapping fast-photo calls on the same
engine instance. Queue jobs in a worker. Native blocking wrappers serialize their calls. A browser worker can be
terminated to cancel even CPU work; the demo's Cancel button does that and recreates the worker.
`paint_photo_image` is the in-memory detailed CPU alternative, with a progress callback. It is slower and has a
different stroke plan, so callers should label a fallback rather than implying the fast GPU path ran.

**Browser demo:** [Fast photo painter](https://stevenkauwe.github.io/brushwork/photo.html).
`brushwork-web` exports `paint_photo_rgba`, returning pixels, actual backend, fallback reason and elapsed time.
Input is straight RGBA; output preserves the source alpha (it does not generate missing texture under transparent
pixels). Run it in a Web Worker. Source photos stay on the device.

`tools/build-web.sh` builds a portable scalar module and a separate **plain SIMD128** module. A tiny capability
probe selects SIMD before importing the module, with scalar retry if initialization fails. `?engine=scalar` and
`?engine=simd` allow testing either build. Neither uses relaxed SIMD. `?backend=cpu` forces the CPU photo fallback.
This follows [BrowserTabICL's capability-first loading and asynchronous GPU pattern](https://github.com/StevenKauwe/tabicl-anywhere/tree/6290de3664520becce649c1ca8cf94f0c26ecbcf);
its model kernels were not copied. SIMD accelerates suitable CPU loops; WebGPU accelerates stroke application.
Stroke planning, image preparation, and final relief are still CPU work. Browser CPU loops run in the worker;
they do not start a Rayon thread pool.

On an M2 Max, the included synthetic native GPU test took 26/44/85/155 ms at 270p/540p/720p/1080p after device
initialization. Chrome's 540p demo took approximately 160 ms using SIMD + real Apple WebGPU, versus 4.45 s for
its detailed CPU fallback. These are single smoke-test timings on different stroke plans, not a general benchmark
or an image-equivalence claim. Browser scalar and SIMD outputs were also exercised, along with cancel/retry.

Validation:

```sh
cargo test --workspace --release --locked
cargo test -p brushwork --release --features gpu fast_photo -- --include-ignored --test-threads=1
OIL_TEST_BACKEND=gpu cargo test -p brushwork --release --features gpu -- --test-threads=2
node --test web/engine-loader.test.mjs
tools/build-web.sh
# WebDriver/browser installed; --engine simd also exercises the SIMD build:
target/release/webcheck --dir dist --photo --engine scalar --expect-backend cpu
```

CI checks CPU/GPU/wasm linting, portable and SIMD build validity (no relaxed SIMD), the shape page in desktop
and phone WebKit, and cancellation/repainting in the photo worker with both WASM builds. GPU tests need actual
hardware and are run separately. ShapeStage's frame scheduling, camera crop, paint-ahead cache, and editor UI
remain application concerns; the reusable painter no longer needs its vendored copy to get these engine APIs.

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
- `a_light_gliding_over_a_still_sky_moves_no_texture`, `a_shape_cut_by_the_frame_is_the_same_shape_cut_off`, ...

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
