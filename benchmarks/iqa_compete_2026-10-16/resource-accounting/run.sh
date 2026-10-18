#!/usr/bin/env bash
# Caller holds the shared heavy lock and runs this through run-heavy.
set -euo pipefail
binary=$1
output=$2
mkdir "$output"
for mode in idle stable churn; do
  if [ "$(uname -s)" = Darwin ]; then
    /usr/sbin/iostat -c 12 -w 1 > "$output/$mode.iostat.txt" 2>&1 &
    accounting_pid=$!
    /usr/bin/time -lp "$binary" "$mode" 8 40 > "$output/$mode.jsonl" 2> "$output/$mode.time.txt"
    wait "$accounting_pid"
  else
    /usr/bin/time -v "$binary" "$mode" 8 40 > "$output/$mode.jsonl" 2> "$output/$mode.time.txt"
  fi
done
