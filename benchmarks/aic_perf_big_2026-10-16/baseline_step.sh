#!/usr/bin/env bash
# Called under one shared heavy lock and resource scope.
set -euo pipefail
phase=$1 output=$2 metric=$3 width=$4 height=$5 threads=$6
rev=${7:-@} label=${8:-base}
binary="$output/$metric-$label"
meta="$output/baseline.meta"
if [ "$label" != base ]; then meta="$output/$label.meta"; fi
case "$phase" in
 build)
  test ! -e "$output/cambi-$label"
  test ! -e "$output/butter-$label"
  just snap-cargo "$rev" build --release -p vmaf --features parallel,simd,avx512 --example cambi_wall
  cp "$LOCK_SNAP_DIR/target/release/examples/cambi_wall" "$output/cambi-$label"
  rustc --edition 2024 -O -g "$LOCK_SNAP_DIR/work/zenmetrics/benchmarks/aic_perf_big_2026-10-16/butter_wall.rs" -o "$output/butter-$label"
  { date -u; hostname; rustc -vV; jj log -r "$rev" -T commit_id --no-graph; echo; sha256sum "$LOCK_SNAP_DIR/work/zenmetrics/crates/vmaf/src/cambi.rs" "$LOCK_SNAP_DIR/work/zenmetrics/crates/zenmetrics-cli/src/metrics/butteraugli_original.rs" "$output/cambi-$label" "$output/butter-$label"; } > "$meta"
  cat "$meta"
  ;;
 wall)
  export RAYON_NUM_THREADS="$threads"
  echo "metric=$metric threads=$threads dimensions=${width}x${height} load=$(cat /proc/loadavg)"
  if [ "$threads" = 1 ]; then
   sudo -n taskset -c 2 perf stat -e cpu_core/instructions/,cpu_core/cycles/ -- env RAYON_NUM_THREADS=1 /usr/bin/time -v "$binary" "$width" "$height" 5
  else
   /usr/bin/time -v "$binary" "$width" "$height" 5
  fi
  ;;
 profile)
  export RAYON_NUM_THREADS=1
  echo "metric=$metric profile dimensions=${width}x${height} load=$(cat /proc/loadavg)"
  sudo -n taskset -c 2 perf record -e cpu_core/cycles/ -F 199 --call-graph dwarf,8192 -o "$output/$metric-$label.data" -- env RAYON_NUM_THREADS=1 "$binary" "$width" "$height" 5
  sudo -n perf report -i "$output/$metric-$label.data" --stdio --no-children --percent-limit 1 > "$output/$metric-$label.report"
  cat "$output/$metric-$label.report"
  ;;
 *) echo "Unknown baseline phase $phase" >&2; exit 2 ;;
esac
