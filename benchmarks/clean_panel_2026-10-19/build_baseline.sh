#!/usr/bin/env bash
# Caller owns host admission locks and run-heavy. Source checkout must be frozen.
set -euo pipefail
: "${LOCK_SNAP_DIR:?new dedicated snapshot required}"
repo=$PWD
mkdir -p "$LOCK_SNAP_DIR" "$HOME/tmp/clean-panel-build"
export TMPDIR="$HOME/tmp/clean-panel-build"
export CARGO_INCREMENTAL=0
bash scripts/ci/lock.sh --check
python3 benchmarks/iqa_compete_2026-10-16/patch_model_load.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/model-preload.meta.json"
python3 benchmarks/iqa_compete_2026-10-16/patch_luma_maps.py "$LOCK_SNAP_DIR/work/zenmetrics" "$repo" "$LOCK_SNAP_DIR/luma-maps.meta.json"
cd "$LOCK_SNAP_DIR/work/zenmetrics"
export RUSTFLAGS='--cfg iqa_compete_ingress --cfg iqa_compete_cached'
export CARGO_TARGET_DIR="$LOCK_SNAP_DIR/target"
rustc -vV > "$LOCK_SNAP_DIR/rustc.txt"
cargo -vV > "$LOCK_SNAP_DIR/cargo.txt"
cargo tree --locked -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr -e features > "$LOCK_SNAP_DIR/features.txt"
cargo metadata --locked --format-version 1 > "$LOCK_SNAP_DIR/metadata.json"
printf '%s\n' 'cargo build --locked --release -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr --bin iqa-compete' > "$LOCK_SNAP_DIR/build-command.txt"
cargo build --locked --release -p cpu-profile --features compete,metrics-parallel,zenmetrics-cli/nr --bin iqa-compete
sha256sum "$CARGO_TARGET_DIR/release/iqa-compete" > "$LOCK_SNAP_DIR/binary.sha256"
