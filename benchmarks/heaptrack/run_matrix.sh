#!/usr/bin/env bash
# Phase 9x — drive heaptrack over the (metric, mode, size) matrix.
# Output: benchmarks/heaptrack/<metric>_<mode>_<size_label>.zst (heaptrack
# native), plus a summary TSV.
#
# Sizes are passed as W H label triplets so we can use rectangular shapes
# at 40MP (smallest dimension is what bounds the algorithm caps; 7000×5728
# = 40.10 MP for example).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/benchmarks/heaptrack"
BIN="${REPO_ROOT}/target/release/cpu-profile"
SUMMARY="${OUT_DIR}/summary_$(date -u +%Y%m%dT%H%M%SZ).tsv"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found — run \`cargo build --release -p cpu-profile\`" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo -e "metric\tmode\tw\th\tsize_label\toutcome\twall_s\tscore\theaptrack_file" > "$SUMMARY"

# Sizes — square except 40MP where we use 7000x5728 = 40.10 MP rectangle
# to land in the user-spec'd "40 MP target" without overshooting RAM.
# Square 6336² = 40.14 MP would also work; 7000×5728 keeps total under
# 6500² for any algorithm that caps the smaller dimension.
SIZES=(
    "1024 1024 1MP"
    "4096 4096 16MP"
    "7000 5728 40MP"
)

METRICS=( cvvdp ssim2 dssim butter iwssim zensim )
MODES=( full warm_ref strip warm_ref_strip )

run_cell() {
    local metric="$1" mode="$2" w="$3" h="$4" label="$5"
    local out="${OUT_DIR}/${metric}_${mode}_${label}"
    # heaptrack appends `.zst` to whatever --output we pass. We pass `$out`
    # and the actual file lands at `$out.zst`.
    local ht_out="${out}.zst"
    local log_out="${out}.log"

    echo -e "\n=== ${metric} ${mode} ${w}x${h} (${label}) ===" | tee -a "$SUMMARY".log
    rm -f "$ht_out"

    local t0=$(date +%s.%N)
    set +e
    heaptrack --output "$out" \
        "$BIN" "$metric" "$mode" "$w" "$h" \
        > "$log_out" 2>&1
    local rc=$?
    set -e
    local t1=$(date +%s.%N)
    local wall
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b-a }')

    local actual_ht=""
    [[ -f "$ht_out" ]] && actual_ht="$ht_out"

    local outcome score
    if grep -q '^GAP:' "$log_out" 2>/dev/null; then
        outcome="GAP"
        score=""
    elif grep -q '^OK ' "$log_out" 2>/dev/null; then
        outcome="OK"
        score=$(grep '^OK ' "$log_out" | sed -n 's/.*score=\([^ ]*\).*/\1/p')
    elif [[ "$rc" -eq 0 ]]; then
        outcome="OK_no_marker"
        score=""
    else
        outcome="FAIL_rc${rc}"
        score=""
    fi

    echo -e "${metric}\t${mode}\t${w}\t${h}\t${label}\t${outcome}\t${wall}\t${score}\t${actual_ht}" >> "$SUMMARY"
    echo "outcome=${outcome} wall=${wall}s heaptrack=${actual_ht:-none}" | tee -a "$SUMMARY".log
}

for metric in "${METRICS[@]}"; do
    for mode in "${MODES[@]}"; do
        for size in "${SIZES[@]}"; do
            read -r w h label <<<"$size"
            # Refresh the workongoing marker every cell.
            date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts9x
            printf '%s %s %s\n' "$(cat /tmp/ts9x)" "claude-phase9x" \
                "heaptrack ${metric}/${mode}/${label}" \
                > "${REPO_ROOT}/.workongoing"
            run_cell "$metric" "$mode" "$w" "$h" "$label"
        done
    done
done

echo ""
echo "Summary: $SUMMARY"
