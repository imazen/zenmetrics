#!/usr/bin/env bash
# Diagnostic only. Caller holds the shared lock under the resource wrapper.
set -u
diag=$1
worker=$2
config=$3
output=$4
mkdir "$output" || exit 2
cp "$config" "$output/config.json" || exit 2
shasum -a 256 "$diag" "$worker" "$config" "$(dirname "$0")/worker.py" "$(dirname "$0")/../durable.py" > "$output/source-hashes.txt"
if [ "$(uname -s)" = Darwin ]; then
  /usr/sbin/iostat -c 12 -w 1 > "$output/iostat.txt" 2>&1 &
  system_pid=$!
fi
python3 "$(dirname "$0")/worker.py" "$worker" "$output/config.json" "$output/scratch" > "$output/worker.stdout" 2> "$output/worker.stderr" &
worker_pid=$!
"$diag" external "$worker_pid" 300 > "$output/accounting.jsonl" 2> "$output/accounting.stderr" &
sampler_pid=$!
wait "$worker_pid"
worker_rc=$?
wait "$sampler_pid"
sampler_rc=$?
if [ "$(uname -s)" = Darwin ]; then wait "$system_pid"; fi
printf '{"worker_pid":%s,"worker_rc":%s,"sampler_rc":%s,"qualification":false}\n' "$worker_pid" "$worker_rc" "$sampler_rc" > "$output/exit.json"
if [ "$worker_rc" != 0 ]; then exit "$worker_rc"; fi
exit "$sampler_rc"
