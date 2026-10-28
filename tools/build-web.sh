#!/usr/bin/env bash
# Build the canvas page into dist/: brushwork-web as WebAssembly, its wasm-bindgen glue, the page and the worker.
# Needs: the wasm32-unknown-unknown target and wasm-bindgen-cli at the version in Cargo.lock. RUSTFLAGS stays empty:
# no extra CPU features (iPhone WebKit rejects relaxed SIMD).
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="" cargo build --release --locked --target wasm32-unknown-unknown -p brushwork-web
rm -rf dist && mkdir -p dist
wasm-bindgen --target web --no-typescript --out-dir dist target/wasm32-unknown-unknown/release/brushwork_web.wasm
OPT="${WASM_OPT:-$(command -v wasm-opt || true)}"
if [ -n "$OPT" ]; then
  "$OPT" -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
    -o dist/brushwork_web_bg.wasm dist/brushwork_web_bg.wasm
fi
cp web/index.html web/worker.js dist/
ls -l dist
