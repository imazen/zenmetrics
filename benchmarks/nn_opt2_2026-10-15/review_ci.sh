#!/usr/bin/env bash
set -euo pipefail
just nn-opt2-review-tests
bash scripts/ci/lock.sh --check
snap=${LOCK_SNAP_DIR:?}/work/zenmetrics
(cd "$snap" && cargo fmt --check $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .'))
just snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items
just nn-parity "$HOME/tmp/nn-opt2/models-all" "$HOME/tmp/nn-phase0/ladder" rten model-tests
