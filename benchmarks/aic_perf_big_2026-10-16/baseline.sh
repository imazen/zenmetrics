#!/usr/bin/env bash
# Run OUTSIDE flock: every bounded step acquires/releases the shared lock.
set -euo pipefail
output=$1
: "${LOCK_SNAP_DIR:?set a dedicated pinned snapshot}"
mkdir -p "$output"
mkdir "$output/baseline-steps"
run_step() {
 local name=$1
 shift
 local status=0
 echo "Queueing baseline $name"
 flock "$HOME/tmp/i265-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 16 -- timeout --kill-after=20s 1150s just aic-perf-big-baseline-step "$@" > "$output/baseline-steps/$name.log" 2>&1 || status=$?
 printf '%s\n' "$status" > "$output/baseline-steps/$name.exit"
 echo "Completed baseline $name rc=$status"
 if [ "$status" -ne 0 ]; then cat "$output/baseline-steps/$name.log"; exit "$status"; fi
}
run_step build build "$output" cambi 1024 1024 1
for metric in cambi butter; do
 for size in '1024 1024' '3355 2516'; do
  read -r width height <<< "$size"
  for threads in 1 16; do
   run_step "$metric-${width}x${height}-${threads}t" wall "$output" "$metric" "$width" "$height" "$threads"
  done
 done
 run_step "$metric-profile" profile "$output" "$metric" 3355 2516 1
done
