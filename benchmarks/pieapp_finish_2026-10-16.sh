#!/usr/bin/env bash
# Caller holds flock and run-heavy; launcher snapshots this command text.
set -euo pipefail
export LOCK_SNAP_DIR="$HOME/tmp/zenmetrics-lock-snap-pieapp"
export RTEN_NUM_THREADS=16
just nn-pieapp-install-calibration
export ORT_DYLIB_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0"
export LD_LIBRARY_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
just nn-pieapp-gates "$HOME/tmp/pieapp-port/models-final" "$HOME/tmp/nn-phase0/ladder"
