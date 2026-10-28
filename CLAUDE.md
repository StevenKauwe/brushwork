# brushwork: notes for Claude

A pure-Rust oil painter (see README.md). Used by music-to-movie-magic (the films' `oil` commands) and CrabScape
(`crates/painter`, the oil-painted atlas), both pinned to a tag here.

## Rules
- **Steadiness first.** Everything about a stroke comes from its shape and its place on it (stable per-stroke and
  per-bristle seeds), never list order, frame or canvas edge. A threshold is a cross-fade (`Fade`), never a switch.
  A flicker fix comes with a test in `crates/brushwork/src/lib.rs` built from the case that flickered.
- **Pixels don't change by accident.** A speed or tidy change must give byte-identical frames: hash frames
  before and after. A change that is meant to change the look says so in its commit and in the consumers' PRs
  (CrabScape's CI repaints its atlas and fails on any byte of drift).
- **CPU is the reference; the GPU mirrors it.** `strokes.wgsl` follows `Painting::stroke` line by line: change both.
- **No film-specific code.** Films pass a `Style`. The `OIL_*` switches are engine features: keep them working.
- **100% Rust.** The CPU path builds for wasm32.

## Before every commit
`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` (also with
`--features brushwork/gpu`), `cargo test --workspace --release`. With a GPU, also
`OIL_TEST_BACKEND=gpu cargo test --release --features brushwork/gpu`.

## The canvas page
`crates/brushwork-web` (UI + Web Worker, pure Rust; `web/` holds the page and the 8-line worker bootstrap).
`tools/build-web.sh` builds `dist/`; `cargo run -p webcheck -- --dir dist [--phone]` checks it paints (CI runs it in
WebKit and deploys `dist/` to https://stevenkauwe.github.io/brushwork/ from main). Keep `RUSTFLAGS` empty: iPhone
WebKit rejects relaxed SIMD.

## Releasing
Tag `vX.Y.Z` on a green main, then bump the pin in music-to-movie-magic (`studio/Cargo.toml`) and CrabScape
(`crates/painter/Cargo.toml`). CrabScape needs its atlas repainted (`tools/paint-oil-crab.sh`) if pixels changed.
