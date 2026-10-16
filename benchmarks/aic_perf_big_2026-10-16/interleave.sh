#!/usr/bin/env bash
# Runs outside the shared lock; each configuration is one bounded ABAB step.
set -euo pipefail
cambi_base=$1 butter_base=$2 candidate=$3 output=$4
mkdir "$output"
for metric in cambi butter; do
 if [ "$metric" = cambi ]; then base="$cambi_base/cambi-base"; else base="$butter_base/butter-base"; fi
 for size in '1024 1024' '3355 2516'; do
  read -r width height <<< "$size"
  for threads in 1 16; do
   key="$metric-${width}x${height}-${threads}t"
   echo "Queueing ABAB $key"
   status=0
   flock "$HOME/tmp/i265-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 16 -- timeout --kill-after=20s 1150s just aic-perf-big-interleave-step "$base" "$candidate/$metric-candidate" "$output/$key" "$width" "$height" "$threads" > "$output/$key.log" 2>&1 || status=$?
   printf '%s\n' "$status" > "$output/$key.exit"
   echo "Completed ABAB $key rc=$status"
   if [ "$status" -ne 0 ]; then cat "$output/$key.log"; exit "$status"; fi
  done
 done
done
