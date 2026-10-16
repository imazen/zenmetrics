#!/usr/bin/env bash
# Run from the repository root, under the caller's run-heavy / shared lock.
set -euo pipefail
export TMPDIR="$HOME/tmp/aic-variants-scratch"
export LOCK_SNAP_DIR="${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap-aicvariants}"
just snap-cargo @ test -p zenmetrics-cli -p vmaf --lib
# Use precisely CI's workspace-member selection for formatting.
packages=$(cd "$LOCK_SNAP_DIR/work/zenmetrics" && cargo metadata --locked --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')
# Intentional word splitting: packages is a sequence of -p NAME arguments.
# shellcheck disable=SC2086
just snap-cargo @ fmt --check $packages
just snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings \
    -A deprecated \
    -A clippy::excessive_precision -A clippy::approx_constant \
    -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments \
    -A clippy::doc_lazy_continuation -A clippy::if_same_then_else \
    -A clippy::doc_overindented_list_items
TMPDIR="$HOME/tmp/aic-variants-scratch" python3 -m unittest discover -s scripts/jobsys -p test_benchmark_study.py
