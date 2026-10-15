#!/usr/bin/env bash
# Build the viewer's encoder modules (web/codecs, a workspace of its own) into
# $CHECKPOINT_OUT/codecs/<name>/. Single-threaded WebAssembly with SIMD128; the
# worker imports each module the first time its encoder is used.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CARGO_TARGET_DIR:?set an isolated target directory}"
: "${CHECKPOINT_OUT:?set the browser output directory}"
[ "$(wasm-bindgen --version)" = 'wasm-bindgen 0.2.129' ] || { echo 'wasm-bindgen-cli 0.2.129 required' >&2; exit 2; }
target="$CARGO_TARGET_DIR/codecs"
RUSTFLAGS='-C target-feature=+simd128' CARGO_TARGET_DIR="$target" CARGO_NET_GIT_FETCH_WITH_CLI=true \
  cargo build --locked --release --manifest-path web/codecs/Cargo.toml --target wasm32-unknown-unknown --workspace
for name in jxl avif webp; do
  wasm-bindgen --target web --no-typescript --out-dir "$CHECKPOINT_OUT/codecs/$name" \
    "$target/wasm32-unknown-unknown/release/zenmetrics_codec_$name.wasm"
done
