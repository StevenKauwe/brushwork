#!/usr/bin/env bash
# Portable WASM plus optional plain SIMD. Neither build uses relaxed SIMD.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist && mkdir -p dist
RUSTFLAGS="" cargo build --release --locked --target wasm32-unknown-unknown -p brushwork-web
wasm-bindgen --target web --no-typescript --out-dir dist target/wasm32-unknown-unknown/release/brushwork_web.wasm
RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR=target/wasm-simd cargo build --release --locked --target wasm32-unknown-unknown -p brushwork-web
mkdir -p target/wasm-simd/glue
wasm-bindgen --target web --no-typescript --out-dir target/wasm-simd/glue target/wasm-simd/wasm32-unknown-unknown/release/brushwork_web.wasm
# Both variants must have the same JS ABI before sharing one loader.
cmp dist/brushwork_web.js target/wasm-simd/glue/brushwork_web.js
cp target/wasm-simd/glue/brushwork_web_bg.wasm dist/brushwork_web_simd.wasm
OPT="${WASM_OPT:-$(command -v wasm-opt || true)}"
if [ -n "$OPT" ]; then
  for name in bg simd; do
    extra=()
    if [ "$name" = simd ]; then extra+=(--enable-simd); fi
    "$OPT" -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals "${extra[@]}" -o "dist/brushwork_web_$name.wasm" "dist/brushwork_web_$name.wasm"
  done
fi
cp web/index.html web/worker.js web/engine-loader.mjs web/photo.html web/photo-worker.js dist/
ls -lh dist
