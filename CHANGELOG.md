# Changelog

Versions before 0.6.0 are described in their `brushwork 0.X.Y:` commits.

## 0.6.1

One sweep along a path as a public API, and a long strip painted and lifted only where it shows. **No pixel changes:** every
existing stroke, light, layer and frame is byte for byte what 0.6.0 painted (`a_new_canvas_paints_as_it_always_has`
holds; the example frame and the photo painters' outputs hash the same before and after).

- `sweep_path(path, &Sweep) -> Vec<Stroke>`: one sweep of the brush along a path, as the strokes to lay in order. A
  piece from each point to the next, all sharing the sweep's bristles (`ident`), each knowing its stretch of the
  whole stroke (`span`) and how far along it is (`along`), so the brush lands once, thins and dries as it goes, and
  lifts once. `Sweep::new(tool, width, col, ident)` and struct update for the rest (`alt`, `picked`, `opacity`,
  `pickup`, `load`, `bend`, `span`, `along`).
- `Width`: `Even`, `Points` (one per point; a piece is the mean of its ends), `Nib`, or `By` (any closure of the
  piece and its direction). `Nib { width, angle, edge }` is a broad nib: `mark(dir)` is
  `width * |sin(dir - angle)|`, never under `width * edge`.
- `Picked`: `Under` (each piece picks up what is under it, as a lone stroke does), `Same(colour)`, or `Along` (a
  colour by how far along the whole stroke each piece is).
- A stroke laid as several paths: give each its stretch (`span`) and `along`; it lands only at the start of the first
  and lifts only at the end of the last.
- `Painting::layer_rgba_in(origin, size)`: exactly that rectangle of `layer_rgba()`, lighting only it (and the relief a
  pixel round it), for showing a painting as it is being painted.
- `Painting::with_origin([x, y])`: a canvas as a window onto a larger one, so its light shows the larger canvas's
  weave there: a window lights exactly as the whole does (but for its outermost two pixels), given the strokes that
  reach it, moved by -origin. With `with_size`, only the part of a long strip that shows need be painted. A canvas's
  origin is [0, 0] unless set, and nothing else changes.
- The painters' own sweeps (outlines, the photo painters' strokes) walk their paths with the same code as
  `sweep_path`; what they lay is unchanged.

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
