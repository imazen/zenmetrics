#!/usr/bin/env bash
set -euo pipefail
output=$1
mkdir "$output"
export AIC_RESULTS_DIR="$output"
export AIC_BUTTER_PROBE="$output/butter-probe"
export AIC_SSIM1_PROBE="$output/ssim1-probe"
export AIC_WS_PROBE="$output/ws-probe"
export AIC_CAMBI_PROBE="$output/aic-more-probe"
export AIC_CAMBI_ORACLE_DIR="$output"
just snap-cargo @ build --release -p zenmetrics-cli --example aic_more_probe
cp "$LOCK_SNAP_DIR/target/release/examples/aic_more_probe" "$AIC_CAMBI_PROBE"
just aic-more-verify python3
