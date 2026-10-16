#!/usr/bin/env bash
# One bounded step; caller acquires/releases the shared lock around each call.
set -euo pipefail
phase=$1
models=$2
ladder=$3
shards=$4
case "$phase" in
 cli) PIEAPP_LEGACY_NUM_PATCHES=1024 just nn-pieapp-review-cli "$HOME/tmp/pieapp-oracle/venv-gpu/bin/python" "$HOME/tmp/pieapp-oracle/upstream" "$HOME/tmp/pieapp-port/ladder-pairs.tsv" "$HOME/tmp/pieapp-port/oracle-ladder.tsv" "$HOME/tmp/pieapp-port/review13b/cli.json" "$HOME/tmp/pieapp-port/review13b/cli-logs" ;;
 inventory) python3 scripts/jobsys/test_benchmark_study.py ;;
 debug) just snap-cargo @ test -p zenmetrics-nn ;;
 release) just snap-cargo @ test --release -p zenmetrics-nn ;;
 fmt) bash scripts/ci/lock.sh --check; (cd "$LOCK_SNAP_DIR/work/zenmetrics" && cargo fmt --check $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')) ;;
 clippy) just snap-cargo @ clippy --workspace --no-default-features --features wgpu,all-metrics -- -D warnings -A deprecated -A clippy::excessive_precision -A clippy::approx_constant -A clippy::erasing_op -A clippy::identity_op -A clippy::too_many_arguments -A clippy::doc_lazy_continuation -A clippy::if_same_then_else -A clippy::doc_overindented_list_items ;;
 nn-clippy) just snap-cargo @ clippy -p zenmetrics-nn --all-targets --features model-tests,ort -- -D warnings ;;
 gpu-features) just snap-cargo @ check -p zenmetrics-cli --features sweep,png,gpu,gpu-cuda ;;
 nn-features) just snap-cargo @ check -p zenmetrics-cli --features sweep,orchestrator,nn,nr ;;
 rten-*) just nn-parity "$shards/${phase#rten-}" "$ladder" rten model-tests,ort pieapp ;;
 cuda) just nn-parity "$models" "$ladder" ort-cuda model-tests,ort pieapp ;;
 routing) just snap-cargo @ test -p zenmetrics-cli --features sweep,orchestrator,nn,nr --lib pieapp_uses_direct_learned_route ;;
 cli-build) just snap-cargo @ build -p zenmetrics-cli --features sweep,orchestrator,nn,nr --bin zenmetrics ;;
 cli-smoke) just nn-pieapp-cli-smoke "$LOCK_SNAP_DIR/target/debug/zenmetrics" "$models" "$ladder" ;;
 *) echo "unknown review step: $phase" >&2; exit 2 ;;
esac
