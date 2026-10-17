#!/usr/bin/env bash
# Orchestrator runs OUTSIDE the heavy lock; each step takes/releases it.
# The interrupted 3000-square RTen cap is deliberately never rerun here.
set -euo pipefail
models=$1
ladder=$2
output=$3
: "${LOCK_SNAP_DIR:?set a dedicated dependency snapshot}"
export RTEN_NUM_THREADS=16
: "${ORT_DYLIB_PATH:?set ONNX Runtime library}"
: "${LD_LIBRARY_PATH:?set CUDA and cuDNN library paths}"
: "${HEAVY_LOCK:?set shared heavy lock}"
: "${RUN_HEAVY:?set resource-limited command runner}"
mkdir "$output/steps"
for phase in cli inventory debug release fmt clippy nn-clippy gpu-features nn-features rten-1415 rten-2207 rten-2406 cuda cuda-cap konjnd-0 konjnd-1 konjnd-2 konjnd-3 routing cli-build cli-smoke; do
 echo "Queueing review step $phase"
 status=0
 flock "$HEAVY_LOCK" "$RUN_HEAVY" --mem 16G --jobs 16 -- timeout --kill-after=10s 900s just nn-pieapp-review-step "$phase" "$models" "$ladder" "$output/shards" "$output" > "$output/steps/$phase.log" 2>&1 || status=$?
 printf '%s\n' "$status" > "$output/steps/$phase.exit"
 echo "Completed review step $phase rc=$status"
 if [ "$status" -ne 0 ]; then cat "$output/steps/$phase.log"; exit "$status"; fi
done
