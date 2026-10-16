#!/usr/bin/env bash
# Caller owns shared heavy lock and run-heavy; load this script before launch.
set -euo pipefail
models=$1
ladder=$2
output=$3
mkdir -p "$output"
export RTEN_NUM_THREADS=16
export ORT_DYLIB_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0"
export LD_LIBRARY_PATH="$HOME/tmp/nnrt-research/venv-cu/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
just snap-cargo @ build --release -p zenmetrics-nn --features ort --example pieapp_cap
for engine in ort-cuda rten; do
 echo "cap engine=$engine load=$(cat /proc/loadavg)"
 /usr/bin/time -v "$LOCK_SNAP_DIR/target/release/examples/pieapp_cap" "$models" "$engine" 3000 3000 > "$output/cap-$engine.log" 2>&1
 cat "$output/cap-$engine.log"
done
root="$HOME/tmp/pieapp-oracle"
just nn-pieapp-review-head "$root/venv-gpu/bin/python" "$root/upstream" "$root/upstream/weights/PieAPPv0.1.pth" "$HOME/tmp/pieapp-port/ladder-pairs.tsv" "$output/head.json"
PIEAPP_LEGACY_NUM_PATCHES=1024 just nn-pieapp-review-cli "$root/venv-gpu/bin/python" "$root/upstream" "$HOME/tmp/pieapp-port/ladder-pairs.tsv" "$HOME/tmp/pieapp-port/oracle-ladder.tsv" "$output/cli.json" "$output/cli-logs"
just nn-pieapp-gates "$models" "$ladder"
