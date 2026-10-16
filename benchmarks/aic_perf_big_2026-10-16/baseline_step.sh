#!/usr/bin/env bash
# Called under one shared heavy lock and resource scope.
set -euo pipefail
phase=$1 output=$2 metric=$3 width=$4 height=$5 threads=$6
binary="$output/$metric-base"
case "$phase" in
 build)
  test ! -e "$output/cambi-base"
  test ! -e "$output/butter-base"
  just snap-cargo @ build --release -p vmaf --features parallel,simd,avx512 --example cambi_wall
  cp "$LOCK_SNAP_DIR/target/release/examples/cambi_wall" "$output/cambi-base"
  rustc --edition 2024 -O -g benchmarks/aic_perf_big_2026-10-16/butter_wall.rs -o "$output/butter-base"
  { date -u; hostname; rustc -vV; jj log -r @ -T commit_id --no-graph; echo; sha256sum crates/vmaf/src/cambi.rs crates/zenmetrics-cli/src/metrics/butteraugli_original.rs "$output/cambi-base" "$output/butter-base"; } > "$output/baseline.meta"
  cat "$output/baseline.meta"
  ;;
 wall)
  export RAYON_NUM_THREADS="$threads"
  echo "metric=$metric threads=$threads dimensions=${width}x${height} load=$(cat /proc/loadavg)"
  if [ "$threads" = 1 ]; then
   sudo -n perf stat -e instructions,cycles -- taskset -c 2 env RAYON_NUM_THREADS=1 /usr/bin/time -v "$binary" "$width" "$height" 5
  else
   /usr/bin/time -v "$binary" "$width" "$height" 5
  fi
  ;;
 profile)
  export RAYON_NUM_THREADS=1
  echo "metric=$metric profile dimensions=${width}x${height} load=$(cat /proc/loadavg)"
  sudo -n perf record -F 199 --call-graph dwarf,8192 -o "$output/$metric-base.data" -- taskset -c 2 env RAYON_NUM_THREADS=1 "$binary" "$width" "$height" 5
  sudo -n perf report -i "$output/$metric-base.data" --stdio --no-children --percent-limit 1 > "$output/$metric-base.report"
  cat "$output/$metric-base.report"
  ;;
 *) echo "Unknown baseline phase $phase" >&2; exit 2 ;;
esac
