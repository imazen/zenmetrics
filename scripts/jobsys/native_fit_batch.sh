#!/usr/bin/env bash
# Bounded native worker admission under ONE caller-owned run-heavy scope.
# GNU parallel reads the slot count file again when its native passes finish;
# operators may change that file after checking the enclosing scope's memory.
set -euo pipefail
native_root=$(realpath "${1:?runtime directory}")
native_script=$(cd "$(dirname "$0")" && pwd)/native_fit_pass.sh
cat /proc/self/cgroup > "$native_root/scope.cgroup"
printf '%s\n' "$$" > "$native_root/batch.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$native_root/STARTED_AT"
exec parallel --line-buffer --jobs "$native_root/slots" --halt soon,fail=1 \
  --joblog "$native_root/native-passes.tsv" --colsep '\t' \
  bash "$native_script" "$native_root" '{1}' '{2}' :::: "$native_root/worker-plan.tsv"
