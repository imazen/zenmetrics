#!/usr/bin/env bash
# Run inside the pinned CI snapshot. Build shared-memory browser assets only.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CARGO_TARGET_DIR:?set an isolated target directory}"
: "${CHECKPOINT_OUT:?set a new browser output directory}"
[ ! -e "$CHECKPOINT_OUT" ] || { echo 'Output already exists; refusing overwrite' >&2; exit 2; }
[ "$(wasm-bindgen --version)" = 'wasm-bindgen 0.2.129' ] || { echo 'wasm-bindgen-cli 0.2.129 required' >&2; exit 2; }
export RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128,+relaxed-simd -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=2147483648 -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base'
# Two builds of one crate: `pkg/` is the first-load core module (decoders,
# JPEG encoder, Zensim); `pkg-full/` scores every CPU metric and is imported
# the first time another metric is chosen. The builds run in turn in one
# target directory; each is bound to its package directory before the next.
build_module() { # <features> <pkg dir>
  cargo +nightly-2026-10-06 build --locked --release -p zenmetrics-browser \
    --no-default-features --features "$1" --target wasm32-unknown-unknown -Z build-std=std,panic_abort
  mkdir -p "$CHECKPOINT_OUT/$2"
  wasm-bindgen --target web --out-dir "$CHECKPOINT_OUT/$2" \
    "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/zenmetrics_browser.wasm"
}
build_module core pkg
build_module full pkg-full
cp web/metrics/*.html web/metrics/*.js web/metrics/nn-models.json "$CHECKPOINT_OUT/"
# onnxruntime-web for the learned metrics, pinned by version and npm integrity
# hash. Only the JSPI build (WebGPU and WASM providers) is shipped: its wasm
# fits Cloudflare Pages' 25 MiB file limit, the asyncify and JSEP builds do not.
python3 scripts/checkpoint/vendor-ort.py "$CHECKPOINT_OUT/ort"
# The encoder modules: a separate, single-threaded build (web/codecs).
bash scripts/checkpoint/build-codecs.sh
cp Cargo.lock ci/sibling-pins.tsv LICENSE-AGPL3 "$CHECKPOINT_OUT/"
printf '%s\n' "$RUSTFLAGS" > "$CHECKPOINT_OUT/rustflags.txt"
cp scripts/checkpoint/build-browser.sh "$CHECKPOINT_OUT/build-browser.sh"
