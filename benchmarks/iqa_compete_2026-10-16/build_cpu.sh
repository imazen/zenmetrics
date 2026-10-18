#!/usr/bin/env bash
set -euo pipefail
repo=$PWD
: "${LOCK_SNAP_DIR:?dedicated benchmark snapshot required}"
export TMPDIR="$HOME/tmp/iqa-compete/build-scratch"
mkdir -p "$TMPDIR"
# Export and lock-check FIRST, then modify only the owned snapshot, not the repo.
bash scripts/ci/lock.sh --check
mode=${1:-cached}
args=()
if [ "$mode" = reference ]; then args+=(--reference); fi
if [ "$mode" != durable ]; then
  python3 benchmarks/iqa_compete_2026-10-16/patch_model_load.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/model-preload.meta.json" "${args[@]}"
fi
python3 benchmarks/iqa_compete_2026-10-16/patch_ort_threads.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/ort-threads.meta.json"
if [ "$mode" != durable ]; then
  python3 benchmarks/iqa_compete_2026-10-16/patch_luma_maps.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/luma-maps.meta.json"
fi
cd "$LOCK_SNAP_DIR/work/zenmetrics"
if [ "$mode" != durable ]; then export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg iqa_compete_ingress"; fi
if [ "$mode" = cached ]; then RUSTFLAGS+=" --cfg iqa_compete_cached"; fi
features=compete,compete-nn,metrics-parallel
if [ "${IQA_GPU_RUNTIME:-}" = cuda ]; then features+=,study-gpu-cuda; fi
if [ "${IQA_GPU_RUNTIME:-}" = wgpu ]; then features+=,study-gpu-wgpu; fi
action=${2:-build}
command=(cargo --locked "$action" --release -p cpu-profile --features "$features" --bin iqa-compete)
if [ "$action" = clippy ]; then command+=(-- -D warnings); fi
CARGO_TARGET_DIR="$LOCK_SNAP_DIR/target" "${command[@]}"
