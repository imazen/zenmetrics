#!/usr/bin/env bash
set -euo pipefail
binary=$1 width=$2 height=$3 threads=$4 reps=$5
export RAYON_NUM_THREADS="$threads"
echo "binary=$binary threads=$threads dimensions=${width}x${height} load=$(cat /proc/loadavg)"
if [ "$threads" = 1 ]; then
 sudo -n taskset -c 2 perf stat -e cpu_core/instructions/,cpu_core/cycles/ -- env RAYON_NUM_THREADS=1 /usr/bin/time -v "$binary" "$width" "$height" "$reps"
else
 /usr/bin/time -v "$binary" "$width" "$height" "$reps"
fi
