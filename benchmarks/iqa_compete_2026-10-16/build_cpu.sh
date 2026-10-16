#!/usr/bin/env bash
set -euo pipefail
repo=$PWD
: "${LOCK_SNAP_DIR:?dedicated benchmark snapshot required}"
# Export and lock-check FIRST, then modify only the owned snapshot, not the repo.
bash scripts/ci/lock.sh --check
mode=${1:-cached}
args=()
if [ "$mode" = reference ]; then args+=(--reference); fi
python3 benchmarks/iqa_compete_2026-10-16/patch_model_load.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/model-preload.meta.json" "${args[@]}"
cd "$LOCK_SNAP_DIR/work/zenmetrics"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg iqa_compete_ingress"
if [ "$mode" = cached ]; then RUSTFLAGS+=" --cfg iqa_compete_cached"; fi
CARGO_TARGET_DIR="$LOCK_SNAP_DIR/target" cargo --locked build --release -p cpu-profile --features compete,metrics-parallel --bin iqa-compete
