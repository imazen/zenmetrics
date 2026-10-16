#!/usr/bin/env bash
set -euo pipefail
source_dir=$1 output=$2
test ! -e "$output/iqa-jxl" # Never overwrite a retained reference executable.
cmake -S benchmarks/iqa_compete_2026-10-16 -B "$output" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIBJXL_SOURCE="$source_dir"
cmake --build "$output" --target iqa-jxl --parallel "${CARGO_BUILD_JOBS:?}"
sha256sum "$output/iqa-jxl"
