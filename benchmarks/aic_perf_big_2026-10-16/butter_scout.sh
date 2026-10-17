#!/usr/bin/env bash
set -euo pipefail
output=$1
mkdir "$output"
rustc --edition 2024 -O -g --test benchmarks/aic_perf_big_2026-10-16/butter_wall.rs -o "$output/butter-tests"
"$output/butter-tests" --nocapture
rustc --edition 2024 -O -g benchmarks/aic_perf_big_2026-10-16/butter_wall.rs -o "$output/butter-new"
sha256sum "$output/butter-new" crates/zenmetrics-cli/src/metrics/butteraugli_original.rs
for size in '1024 1024' '3355 2516'; do
 read -r width height <<< "$size"
 for threads in 1 16; do
  just aic-perf-big-measure "$output/butter-new" "$width" "$height" "$threads" 5
 done
done
sudo -n taskset -c 2 perf record -e cpu_core/cycles/ -F 199 --call-graph dwarf,8192 -o "$output/butter-new.data" -- env RAYON_NUM_THREADS=1 "$output/butter-new" 3355 2516 5
sudo -n perf report -i "$output/butter-new.data" --stdio --no-children --percent-limit 1 > "$output/butter-new.report"
cat "$output/butter-new.report"
