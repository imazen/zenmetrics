#!/usr/bin/env bash
# Prioritized Gate A pipeline:
#   Phase 1: val corpora with --acumen-mode-a   (~60 min)
#   Phase 2: val corpora baseline               (~60 min)
#   Phase 3: safesyn with --acumen-mode-a       (~100 min, for trainer)
#
# Gate A panel becomes computable after Phase 2.
set -uo pipefail

cd ~/work/zen/zenmetrics--acumen-gpu
BIN=./target/release/zen-metrics
OUT_DIR=/mnt/v/output/zensim/synthetic-v2
mkdir -p "$OUT_DIR"

run_score() {
    local name="$1" pairs="$2" out="$3" extra_args="${4:-}"
    [ -f "$out" ] && [ "$(stat -c %s "$out")" -gt 100 ] && {
        echo "[$(date -u +%H:%M:%SZ)] skip $name (already done: $out)" >&2
        return 0
    }
    local n; n=$(wc -l < "$pairs")
    echo "[$(date -u +%H:%M:%SZ)] starting $name ($n pairs, args='$extra_args')" >&2
    "$BIN" score-pairs \
        --metric zensim-gpu \
        --pairs-tsv "$pairs" \
        --out-parquet "$out" \
        --gpu-runtime cuda \
        $extra_args 2>&1 | tee "/tmp/run_${name}.log" \
        | grep --line-buffered -E "scored|FAIL|wrote" | tail -3
    echo "[$(date -u +%H:%M:%SZ)] $name done" >&2
}

echo "=== PHASE 1: val corpora with --acumen-mode-a ===" >&2
for corpus in kadid tid aic3 cid22; do
    run_score "${corpus}_acumen" "/tmp/${corpus}_pairs.tsv" \
        "$OUT_DIR/${corpus}_acumen_modea_2026-05-21.parquet" "--acumen-mode-a"
done

echo "=== PHASE 2: val corpora baseline ===" >&2
for corpus in kadid tid aic3 cid22; do
    run_score "${corpus}_baseline" "/tmp/${corpus}_pairs.tsv" \
        "$OUT_DIR/${corpus}_baseline_2026-05-21.parquet" ""
done

echo "[$(date -u +%H:%M:%SZ)] === Gate A panel data ready — compute via /tmp/gate_a_compare.py ===" >&2

echo "=== PHASE 3: safesyn with --acumen-mode-a (for future trainer) ===" >&2
run_score "safesyn_acumen" "/tmp/safesyn_pairs.tsv" \
    "$OUT_DIR/safesyn_acumen_modea_2026-05-21.parquet" "--acumen-mode-a"

echo "[$(date -u +%H:%M:%SZ)] ALL PHASES DONE" >&2
