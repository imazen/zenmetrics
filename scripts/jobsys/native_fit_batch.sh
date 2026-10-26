#!/usr/bin/env bash
# Bounded native worker admission under ONE caller-owned run-heavy scope.
# GNU parallel reads the slot count file again when its native passes finish;
# operators may change that file after checking the enclosing scope's memory.
set -euo pipefail
native_root=$(realpath "${1:?runtime directory}")
native_script=$(cd "$(dirname "$0")" && pwd)/native_fit_pass.sh
native_mode=${2:-initial}
if [[ "$native_mode" == initial ]]; then
  cat /proc/self/cgroup > "$native_root/scope.cgroup"
  printf '%s\n' "$$" > "$native_root/batch.pid"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$native_root/STARTED_AT"
  native_slots="$native_root/slots"
  native_plan="$native_root/worker-plan.tsv"
  native_log="$native_root/native-passes.tsv"
else
  [[ "$native_mode" =~ ^recovery[0-9]+$ ]]
  native_slots="$native_root/$native_mode-slots"
  native_plan="$native_root/$native_mode-plan.tsv"
  native_log="$native_root/$native_mode-passes.tsv"
fi
exec parallel --line-buffer --jobs "$native_slots" --halt soon,fail=1 \
  --joblog "$native_log" --colsep '\t' \
  bash "$native_script" "$native_root" '{1}' '{2}' :::: "$native_plan"
