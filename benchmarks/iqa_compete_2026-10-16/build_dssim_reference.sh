#!/usr/bin/env bash
# Caller supplies a frozen original-source tree and owns admission/resource limits.
set -euo pipefail
source_dir=$(realpath "$1")
output=$2
mkdir "$output"
output=$(realpath "$output")
export TMPDIR="$output/scratch"
mkdir "$TMPDIR"
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR="$output/target" \
  cargo build --locked --release --manifest-path "$source_dir/Cargo.toml" -p dssim-core
cc -O3 -Wall -Wextra -I "$source_dir/dssim-core" \
  benchmarks/iqa_compete_2026-10-16/dssim_reference.c \
  "$output/target/release/libdssim_core.a" -ldl -pthread -lm -o "$output/dssim-reference"
sha256sum "$output/dssim-reference" "$output/target/release/libdssim_core.a" \
  "$source_dir/dssim-core/src/lib.rs" "$source_dir/Cargo.lock" > "$output/build.sha256"
