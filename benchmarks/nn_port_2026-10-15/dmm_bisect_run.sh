#!/usr/bin/env bash
# Serialize immutable 48-row panels; CUDA observations have a full repeat.
set -uo pipefail
models=$1
ladder=$2
output=$3
mkdir "$output" || exit 2
export TMPDIR="${TMPDIR:-$HOME/tmp/nn-port-scratch/tmp}"
export LOCK_SNAP_DIR="${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap-nnport}"
mkdir -p "$TMPDIR"
test -n "${ORT_DYLIB_PATH:-}" || { echo 'ORT_DYLIB_PATH is required' >&2; exit 2; }
failed=0
while IFS=$'\t' read -r change revision; do
    [ "$change" = change ] && continue
    prefix="$output/$change"
    echo "BISECT_START $change $revision"
    if flock "$HOME/tmp/i265-heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G -- \
        timeout --kill-after=10 1150 just nn-dmm-bisect-panel "$revision" "$prefix" "$models" "$ladder" \
        > "$prefix.runner.log" 2>&1; then
        echo "BISECT_GATE_PASS $change"
    else
        echo "BISECT_GATE_FAIL $change"
    fi
    for engine in rten ort-cuda; do
        if ! just nn-dmm-fix-summary "$prefix-$engine.log" "$models" "$engine" "$prefix-$engine.tsv" > "$prefix-$engine.summary" 2>&1; then
            failed=1
        fi
    done
    if ! just nn-dmm-fix-summary "$prefix-ort-cuda-repeat.log" "$models" ort-cuda "$prefix-ort-cuda-repeat.tsv" > "$prefix-ort-cuda-repeat.summary" 2>&1; then
        failed=1
    fi
    echo "BISECT_ROWS_RECORDED $change"
done < benchmarks/nn_port_2026-10-15/dmm_bisect_sources_2026-10-17.tsv
exit "$failed"
