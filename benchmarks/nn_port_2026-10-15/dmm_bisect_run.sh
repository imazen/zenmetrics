#!/usr/bin/env bash
# Serialize immutable 48-row panels; CUDA observations have a full repeat.
# An explicit fourth argument queues those cases in an existing study directory.
set -uo pipefail
models=$1
ladder=$2
output=$3
requested=${4:-}
if [ -n "$requested" ]; then
    test -d "$output" || exit 2
else
    mkdir "$output" || exit 2
fi
export TMPDIR="${TMPDIR:-$HOME/tmp/nn-port-scratch/tmp}"
export LOCK_SNAP_DIR="${LOCK_SNAP_DIR:-$HOME/tmp/zenmetrics-lock-snap-nnport}"
mkdir -p "$TMPDIR"
test -n "${ORT_DYLIB_PATH:-}" || { echo 'ORT_DYLIB_PATH is required' >&2; exit 2; }
declare -A wanted=()
if [ -n "$requested" ]; then
    IFS=, read -ra requested_names <<< "$requested"
    for name in "${requested_names[@]}"; do
        [[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || exit 2
        test -z "${wanted[$name]+present}" || exit 2
        wanted[$name]=1
    done
fi
names=()
revisions=()
while IFS=$'\t' read -r change revision; do
    [ "$change" = change ] && continue
    if [ -n "$requested" ] && [ -z "${wanted[$change]+present}" ]; then
        continue
    fi
    names+=("$change")
    revisions+=("$revision")
done < benchmarks/nn_port_2026-10-15/dmm_bisect_sources_2026-10-17.tsv
if [ -n "$requested" ]; then
    test "${#names[@]}" -eq "${#wanted[@]}" || exit 2
fi
# Refuse the entire submission before starting anything if it would overwrite data.
for change in "${names[@]}"; do
    for suffix in runner.log rten.log rten.tsv rten.summary ort-cuda.log ort-cuda.tsv ort-cuda.summary ort-cuda-repeat.log ort-cuda-repeat.tsv ort-cuda-repeat.summary; do
        if [ -e "$output/$change.$suffix" ] || [ -e "$output/$change-$suffix" ]; then
            echo "Refusing existing measurement: $output/$change ($suffix)" >&2
            exit 2
        fi
    done
done

run_case() {
    local change=$1 revision=$2 prefix="$output/$1" case_failed=0
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
            case_failed=1
        fi
    done
    if ! just nn-dmm-fix-summary "$prefix-ort-cuda-repeat.log" "$models" ort-cuda "$prefix-ort-cuda-repeat.tsv" > "$prefix-ort-cuda-repeat.summary" 2>&1; then
        case_failed=1
    fi
    if [ "$case_failed" -eq 0 ]; then
        echo "BISECT_ROWS_RECORDED $change"
    else
        echo "BISECT_ROWS_MISSING $change"
    fi
    return "$case_failed"
}

failed=0
pids=()
for index in "${!names[@]}"; do
    if [ -n "$requested" ]; then
        # Only flock waiters overlap. Each complete heavy panel owns the same lock.
        run_case "${names[$index]}" "${revisions[$index]}" &
        pids+=("$!")
    elif ! run_case "${names[$index]}" "${revisions[$index]}"; then
        failed=1
    fi
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
