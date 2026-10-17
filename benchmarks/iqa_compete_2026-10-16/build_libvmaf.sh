#!/usr/bin/env bash
# Caller owns run-heavy and the bounded shared lock. Cache source is read-only.
set -euo pipefail
export TMPDIR="$HOME/tmp/iqa-compete/build-scratch"
mkdir -p "$TMPDIR"
source_dir=$1
output=$2
mkdir "$output"
avx512=false
if [ "$(uname -m)" = x86_64 ]; then avx512=true; fi
meson setup "$output/build" "$source_dir/vendored/vmaf/libvmaf" --buildtype=release -Denable_tests=false -Denable_docs=false -Denable_float=true -Dbuilt_in_models=true -Denable_avx512="$avx512"
ninja -C "$output/build" -j "${CARGO_BUILD_JOBS:?}" src/libvmaf.a
c++ -O3 -std=c++17 -I "$source_dir/vendored/vmaf/libvmaf/include" benchmarks/iqa_compete_2026-10-16/libvmaf_compute.cpp "$output/build/src/libvmaf.a" -pthread -lm -o "$output/libvmaf-compute"
sha256sum "$output/libvmaf-compute" "$output/build/src/libvmaf.a"
