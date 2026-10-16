#!/usr/bin/env bash
set -euo pipefail
export TMPDIR="$HOME/tmp/aic-variants-scratch" LOCK_SNAP_DIR="$HOME/tmp/zenmetrics-lock-snap-aicperfbig"
mkdir -p "$TMPDIR/round14"
just snap-cargo @ build --release -p vmaf --features parallel,simd,avx512 --example cambi_wall
cp "$LOCK_SNAP_DIR/target/release/examples/cambi_wall" "$TMPDIR/round14/cambi-base"
rustc --edition 2024 -O -g benchmarks/aic_perf_big_2026-10-16/butter_wall.rs -o "$TMPDIR/round14/butter-base"
for metric in cambi butter; do
 for size in '1024 1024' '3355 2516'; do
  read -r w h <<< "$size"
  for threads in 1 16; do
   export RAYON_NUM_THREADS="$threads"
   echo "metric=$metric threads=$threads dimensions=$size load=$(cat /proc/loadavg)"
   if [ "$threads" = 1 ]; then
    perf stat -e instructions,cycles -- taskset -c 2 /usr/bin/time -v "$TMPDIR/round14/$metric-base" "$w" "$h" 5
   else
    /usr/bin/time -v "$TMPDIR/round14/$metric-base" "$w" "$h" 5
   fi
  done
 done
 perf record -F 199 --call-graph dwarf,8192 -o "$TMPDIR/round14/$metric-base.data" -- taskset -c 2 "$TMPDIR/round14/$metric-base" 3355 2516 5
 perf report -i "$TMPDIR/round14/$metric-base.data" --stdio --no-children --percent-limit 1 > "$TMPDIR/round14/$metric-base.report"
done
