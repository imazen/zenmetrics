#!/usr/bin/env bash
set -euo pipefail
: "${LOCK_SNAP_DIR:?owned pinned snapshot required}"
just iqa-compete-fmt-check
just snap-cargo @ clippy -p zenmetrics-cli --no-default-features --features png,sweep,cpu-metrics,nn-ort,nn-webgpu,nr --bin zenmetrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items
just snap-cargo @ clippy -p cpu-profile --features compete,metrics-parallel --bin bench-iqa --bin iqa-compete -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items
just snap-cargo @ test -p cpu-profile --features compete,metrics-parallel --bin bench-iqa
just cli-feature-matrix
python3 -m unittest discover -s benchmarks/iqa_compete_2026-10-16 -p 'test_*.py'
python3 scripts/ci/check_hygiene.py
