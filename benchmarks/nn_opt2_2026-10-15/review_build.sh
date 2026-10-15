#!/usr/bin/env bash
# Build a frozen algorithm revision with the current full-precision benchmark
# harness. Only the exported snapshot's example is instrumented, never the rev.
set -euo pipefail
rev=$1
out=$2
root=$(pwd)
snap=${LOCK_SNAP_DIR:?}/work/zenmetrics
test ! -e "$out"
if [[ $rev == working ]]; then
    bash scripts/ci/lock.sh --check
else
    bash scripts/ci/lock.sh --check --rev "$rev"
fi
cp "$root/crates/zenmetrics-nn/examples/nn_bench.rs" "$snap/crates/zenmetrics-nn/examples/nn_bench.rs"
(cd "$snap" && CARGO_TARGET_DIR="${LOCK_SNAP_DIR}/target" cargo build --locked --release -p zenmetrics-nn --example nn_bench)
cp "${LOCK_SNAP_DIR}/target/release/examples/nn_bench" "$out"
sha256sum "$out"
