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
if [ "${IQA_DVIFMISH:-0}" = 1 ]; then
  python3 benchmarks/iqa_compete_2026-10-16/patch_dvifmish.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/dvifmish-patch.meta.json"
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg iqa_compete_dvifmish"
fi
if [ "$mode" != durable ]; then
  python3 benchmarks/iqa_compete_2026-10-16/patch_luma_maps.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/luma-maps.meta.json"
fi
if [ "${IQA_MAC_CPU_COUNTERS:-0}" = 1 ]; then
  python3 benchmarks/iqa_compete_2026-10-16/patch_mac_cpu_counters.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/mac-cpu-counters.meta.json"
fi
cd "$LOCK_SNAP_DIR/work/zenmetrics"
if [ "$mode" != durable ]; then export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg iqa_compete_ingress"; fi
if [ "$mode" = cached ]; then RUSTFLAGS+=" --cfg iqa_compete_cached"; fi
features=compete,compete-nn,metrics-parallel
if [ "${IQA_DVIFMISH:-0}" = 1 ]; then features+=,compete-dvifmish; fi
# Neural Metal uses its own feature, independent of classical GPU kernels.
case "${IQA_NN_ENGINE:-}" in
  ort-webgpu) features+=,zenmetrics-cli/nn-webgpu ;;
  ""|rten|ort-cuda) ;;
  *) echo "unknown IQA_NN_ENGINE" >&2; exit 2 ;;
esac
if [ "${IQA_GPU_RUNTIME:-}" = cuda ]; then features+=,study-gpu-cuda; fi
if [ "${IQA_GPU_RUNTIME:-}" = wgpu ]; then features+=,study-gpu-wgpu; fi
action=${2:-build}
command=(cargo --locked "$action" --release -p cpu-profile --features "$features" --bin iqa-compete)
if [ "$action" = clippy ]; then command+=(-- -D warnings); fi
CARGO_TARGET_DIR="$LOCK_SNAP_DIR/target" "${command[@]}"
