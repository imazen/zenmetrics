#!/usr/bin/env bash
set -euo pipefail
base=$1 candidate=$2 output=$3 width=$4 height=$5 threads=$6
mkdir "$output"
sha256sum "$base" "$candidate"
for name in base1 new1 base2 new2; do
 case "$name" in base*) binary=$base ;; new*) binary=$candidate ;; esac
 just aic-perf-big-measure "$binary" "$width" "$height" "$threads" 5 > "$output/$name.log" 2>&1
 cat "$output/$name.log"
done
