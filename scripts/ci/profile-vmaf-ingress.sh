#!/usr/bin/env bash
# Compare two prebuilt stride_cost examples using retained perf/heaptrack data.
# Run through run-heavy. Profiling excludes no setup, so no kernel-only or
# warm-only timing claim follows. Arguments: before-binary after-binary output.
set -euo pipefail
before=$(realpath "$1")
after=$(realpath "$2")
output=$(realpath -m "$3")
mkdir -p "$(dirname "$output")"
# Refuse to replace any retained attempt, even a previously failed one.
mkdir "$output"
sha256sum "$before" "$after" > "$output/binaries.sha256"
for model in v0 v1; do
    for round in 1 2 3; do
        # Alternate order to expose drift; retain every run and exact score bits.
        order=(before after)
        if (( round % 2 == 0 )); then order=(after before); fi
        for side in "${order[@]}"; do
            binary=${!side}
            perf stat -x, -e instructions,cycles,branches,branch-misses \
                -o "$output/$model-$round-$side.perf" \
                "$binary" "$model" 320 240 100 \
                > "$output/$model-$round-$side.stdout"
        done
        cmp "$output/$model-$round-before.stdout" "$output/$model-$round-after.stdout"
    done
    for side in before after; do
        binary=${!side}
        heaptrack --record-only -o "$output/$model-$side.heaptrack" \
            "$binary" "$model" 320 240 10 \
            > "$output/$model-$side.heaptrack.stdout" 2> "$output/$model-$side.heaptrack.stderr"
        trace="$output/$model-$side.heaptrack.zst"
        if [[ ! -f "$trace" ]]; then trace="$output/$model-$side.heaptrack.gz"; fi
        heaptrack_print -f "$trace" \
            > "$output/$model-$side.allocations.txt"
    done
done
