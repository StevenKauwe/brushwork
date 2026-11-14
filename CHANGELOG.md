# Changelog

Versions before 0.6.0 are described in their `brushwork 0.X.Y:` commits.

## 0.6.0

A canvas of any size, and the paint lifted off it as a layer. **No pixel changes:** `Painting::new()` is still the
film's 1920 x 1080 canvas, and its strokes, light and frames are byte for byte what 0.5.0 painted
(`a_new_canvas_paints_as_it_always_has` holds fingerprints taken on 0.5.0; the example frame and the photo painters'
outputs hash the same before and after).

- `Painting::with_size(width, height, ground)`: a blank canvas of any size, all one ground colour, for laying strokes
  directly. `Painting::size()` says how big a canvas is.
- `Painting::lit()` lights the canvas at its own size (it assumed 1920 x 1080).
- `Painting::keep_coverage()` (opt in) keeps a coverage channel: how much of each pixel the strokes laid since cover,
  composited as their paint is. `Painting::coverage()` reads it. Off, nothing is kept and nothing changes.
- `Painting::layer_rgba()`: the paint laid since `keep_coverage` as straight-alpha RGBA, its colour unmixed from the
  canvas it was laid on, its alpha the coverage, its relief lit as `lit()` lights it with the canvas weave only under
  paint (bare canvas is 0, 0, 0, 0). Where the paint covers the canvas completely its colour is exactly `lit()`'s.
- The scene painters (`paint_frame`, `paint_body`, `brush_fill` and the rest, the photo painters) still frame their
  shapes on 1920 x 1080 and want `Painting::new()`.
- Coverage is kept by the CPU's strokes only; the GPU recorder does not keep it, and `strokes.wgsl` is unchanged.
