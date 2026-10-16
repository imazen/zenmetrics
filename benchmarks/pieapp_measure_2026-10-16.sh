#!/usr/bin/env bash
# Caller holds flock and run-heavy; serialize all GPU and build work.
set -euo pipefail
export LOCK_SNAP_DIR="$HOME/tmp/zenmetrics-lock-snap-pieapp"
export RTEN_NUM_THREADS=16
models="$HOME/tmp/pieapp-port/models-final"
pairs="$HOME/tmp/pieapp-port/ladder-pairs.tsv"
python="$HOME/tmp/pieapp-oracle/venv-gpu/bin/python"
just nn-pieapp-export "$HOME/tmp/pieapp-oracle/venv/bin/python" "$HOME/tmp/pieapp-oracle/upstream/weights/PieAPPv0.1.pth" "$models"
just nn-pieapp-entry-oracle "$python" "$pairs" "$HOME/tmp/pieapp-port/oracle-ladder.tsv" cuda
just nn-pieapp-goldens "$pairs" "$HOME/tmp/pieapp-port/oracle-ladder.tsv" "$models/pieapp_goldens.tsv"
just nn-pieapp-entry-oracle "$python" "$HOME/tmp/pieapp-port/identity-pairs.tsv" "$HOME/tmp/pieapp-port/oracle-identity.tsv" cuda
export ORT_DYLIB_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0"
export LD_LIBRARY_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
just nn-pieapp-score "$models" "$HOME/tmp/pieapp-port/pilot-pairs.tsv" "$HOME/tmp/pieapp-port/native-pilot-cuda.tsv" ort-cuda
just nn-pieapp-score "$models" "$HOME/tmp/pieapp-port/pilot-pairs.tsv" "$HOME/tmp/pieapp-port/native-pilot-rten.tsv" rten
just nn-pieapp-score "$models" "$HOME/tmp/pieapp-port/identity-pairs.tsv" "$HOME/tmp/pieapp-port/native-identity.tsv" ort-cuda
just nn-pieapp-score "$models" "$HOME/tmp/konjnd-pairs/pairs_abs.tsv" "$HOME/tmp/pieapp-port/konjnd.tsv" ort-cuda

just nn-pieapp-calibration-summary "$HOME/tmp/konjnd-pairs/pairs_abs.tsv" "$HOME/tmp/pieapp-port/konjnd.tsv" "$HOME/tmp/pieapp-port/konjnd-summary.json"
