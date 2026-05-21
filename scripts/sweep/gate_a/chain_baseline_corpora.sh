#!/usr/bin/env bash
# After the acumen chain completes, run the same val corpora
# WITHOUT --acumen-mode-a to provide the Gate A baseline scores.
# Same MLP (V_22 ship weights), same features, just no castleCSF
# modulation on the HF band-energy features.
set -uo pipefail

cd ~/work/zen/zenmetrics--acumen-gpu
BIN=./target/release/zen-metrics
OUT_DIR=/mnt/v/output/zensim/synthetic-v2
mkdir -p "$OUT_DIR"

wait_for_chain() {
    while kill -0 3078550 2>/dev/null; do
        sleep 30
    done
}

run() {
    local name="$1" pairs="$2" out="$3"
    echo "[$(date -u +%H:%M:%SZ)] starting baseline $name ($(wc -l < "$pairs") pairs)" >&2
    "$BIN" score-pairs \
        --metric zensim-gpu \
        --pairs-tsv "$pairs" \
        --out-parquet "$out" \
        --gpu-runtime cuda \
        2>&1 | tee "/tmp/baseline_${name}.log" \
        | grep --line-buffered -E "pairs scored|FAIL|wrote" | tail -3
    echo "[$(date -u +%H:%M:%SZ)] baseline $name done → $out" >&2
}

wait_for_chain
echo "[$(date -u +%H:%M:%SZ)] acumen chain done, starting baseline runs" >&2

run kadid /tmp/kadid_pairs.tsv "$OUT_DIR/kadid_baseline_2026-05-21.parquet"
run tid /tmp/tid_pairs.tsv "$OUT_DIR/tid_baseline_2026-05-21.parquet"
run aic3 /tmp/aic3_pairs.tsv "$OUT_DIR/aic3_baseline_2026-05-21.parquet"
run cid22 /tmp/cid22_pairs.tsv "$OUT_DIR/cid22_baseline_2026-05-21.parquet"
run konjnd /tmp/konjnd_pairs.tsv "$OUT_DIR/konjnd_baseline_2026-05-21.parquet"

echo "[$(date -u +%H:%M:%SZ)] ALL baseline corpora done — Gate A panel data ready"
