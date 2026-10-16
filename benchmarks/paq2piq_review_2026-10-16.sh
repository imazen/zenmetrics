#!/usr/bin/env bash
set -euo pipefail
just snap-cargo @ test -p zenmetrics-nn
just snap-cargo @ test --release -p zenmetrics-nn
bash scripts/ci/lock.sh --check
snap=${LOCK_SNAP_DIR:?}/work/zenmetrics
(cd "$snap" && cargo fmt --check $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .'))
just snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items
just snap-cargo @ check -p zenmetrics-cli --features sweep,png,gpu,gpu-cuda
just snap-cargo @ check -p zenmetrics-cli --features sweep,orchestrator,nn,nr
just snap-cargo @ test -p zenmetrics-cli --features sweep,orchestrator,nn,nr --lib paq2piq_uses_direct_no_reference_route
just nn-parity "$HOME/tmp/nr-learned-scratch/models" "$HOME/tmp/nn-phase0/ladder" rten model-tests paq2piq,musiq,hyperiqa
