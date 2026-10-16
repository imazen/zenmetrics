#!/usr/bin/env bash
# Orchestrator runs OUTSIDE the heavy lock; each step takes/releases it.
# The interrupted 3000-square RTen cap is deliberately never rerun here.
set -euo pipefail
models=$1
ladder=$2
output=$3
: "${LOCK_SNAP_DIR:?set a dedicated dependency snapshot}"
export RTEN_NUM_THREADS=16
export ORT_DYLIB_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0"
export LD_LIBRARY_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir "$output/steps"
for phase in cli inventory debug release fmt clippy nn-clippy gpu-features nn-features rten-1415 rten-2207 rten-2406 cuda routing cli-build cli-smoke; do
 echo "Queueing review step $phase"
 status=0
 flock "$HOME/tmp/i265-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 16 -- timeout --kill-after=20s 1150s just nn-pieapp-review-step "$phase" "$models" "$ladder" "$output/shards" > "$output/steps/$phase.log" 2>&1 || status=$?
 printf '%s\n' "$status" > "$output/steps/$phase.exit"
 echo "Completed review step $phase rc=$status"
 if [ "$status" -ne 0 ]; then cat "$output/steps/$phase.log"; exit "$status"; fi
done
