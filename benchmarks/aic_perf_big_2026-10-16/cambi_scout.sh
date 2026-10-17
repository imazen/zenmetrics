#!/usr/bin/env bash
set -euo pipefail
output=$1
mkdir "$output"
just snap-cargo @ build --release -p vmaf --features parallel,simd,avx512 --example cambi_wall
cp "$LOCK_SNAP_DIR/target/release/examples/cambi_wall" "$output/cambi-new"
sha256sum "$output/cambi-new" crates/vmaf/src/cambi.rs
for size in '1024 1024' '3355 2516'; do
 read -r width height <<< "$size"
 for threads in 1 16; do
  just aic-perf-big-measure "$output/cambi-new" "$width" "$height" "$threads" 5
 done
done
