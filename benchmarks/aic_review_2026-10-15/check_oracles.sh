#!/usr/bin/env bash
# Caller holds the shared heavy lock and run-heavy scope. Results are exclusive.
set -euo pipefail
review_python=$1
export TMPDIR="$HOME/tmp/aic-variants-scratch"
export AIC_RESULTS_DIR="${AIC_RESULTS_DIR:-$TMPDIR/round5/final}"
mkdir -p "$AIC_RESULTS_DIR"
export AIC_BUTTER_PROBE="$TMPDIR/round5/butter-probe"
export AIC_SSIM1_PROBE="$TMPDIR/round5/ssim1-probe"
export AIC_WS_PROBE="$TMPDIR/round5/ws-probe"
export AIC_CAMBI_PROBE="$TMPDIR/round5/aic-more-probe"
rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/butter_original_probe.rs -o "$AIC_BUTTER_PROBE"
rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ssimulacra1_probe.rs -o "$AIC_SSIM1_PROBE"
rustc --edition 2024 -O benchmarks/aic_more_2026-10-15/ws_hv_probe.rs -o "$AIC_WS_PROBE"
cp "$LOCK_SNAP_DIR/target/release/examples/aic_more_probe" "$AIC_CAMBI_PROBE"
for metric in butter ssim1 ws cambi; do
    "$review_python" "benchmarks/aic_more_2026-10-15/verify_$metric.py"
done
