#!/usr/bin/env bash
# Sequential (GPU-serialized) ssim2 sweeps for the remaining picker codecs.
set -u
PPDIR=/mnt/v/output/picker-pipeline-2026-06-22
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
BIN=./target/release/zenmetrics
Q="5,10,20,30,40,50,60,70,80,90,95"
sweep() { # codec  plan  budget_args...
  local codec=$1 plan=$2; shift 2
  echo "=== $(date -u +%H:%M:%S) sweep $codec ($plan $*) ==="
  nice -n 19 "$BIN" sweep --codec "$codec" --sources "$PPDIR/corpus" \
    --q-grid "$Q" --plan "$plan" "$@" --metric ssim2-gpu \
    --output "$PPDIR/sweeps/$codec.tsv"
  echo "=== $(date -u +%H:%M:%S) $codec done: $(wc -l < "$PPDIR/sweeps/$codec.tsv") rows ==="
}
sweep zenavif rd_core
sweep zenjxl  rd_core
sweep zenwebp modes_full --plan-budget 60
echo "ALL SWEEPS DONE"
