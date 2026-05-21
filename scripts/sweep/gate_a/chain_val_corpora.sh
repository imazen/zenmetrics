#!/usr/bin/env bash
# Run val-corpora score-pairs sequentially after safesyn completes.
# Single GPU — can't parallelize.
set -uo pipefail

cd ~/work/zen/zenmetrics--acumen-gpu
BIN=./target/release/zen-metrics
OUT_DIR=/mnt/v/output/zensim/synthetic-v2
mkdir -p "$OUT_DIR"

wait_for_safesyn() {
    while kill -0 3074152 2>/dev/null; do
        sleep 30
    done
}

run() {
    local name="$1" pairs="$2" out="$3"
    echo "[$(date -u +%H:%M:%SZ)] starting $name ($(wc -l < "$pairs") pairs)" >&2
    "$BIN" score-pairs \
        --metric zensim-gpu \
        --acumen-mode-a \
        --pairs-tsv "$pairs" \
        --out-parquet "$out" \
        --gpu-runtime cuda \
        2>&1 | tee "/tmp/acumen_${name}.log" \
        | grep --line-buffered -E "pairs scored|FAIL|wrote" | tail -3
    echo "[$(date -u +%H:%M:%SZ)] $name done → $out" >&2
}

wait_for_safesyn
echo "[$(date -u +%H:%M:%SZ)] safesyn complete, starting val corpora" >&2

run kadid /tmp/kadid_pairs.tsv "$OUT_DIR/kadid_acumen_modea_2026-05-21.parquet"
run tid /tmp/tid_pairs.tsv "$OUT_DIR/tid_acumen_modea_2026-05-21.parquet"
run aic3 /tmp/aic3_pairs.tsv "$OUT_DIR/aic3_acumen_modea_2026-05-21.parquet"
run cid22 /tmp/cid22_pairs.tsv "$OUT_DIR/cid22_acumen_modea_2026-05-21.parquet"
run konjnd /tmp/konjnd_pairs.tsv "$OUT_DIR/konjnd_acumen_modea_2026-05-21.parquet"

echo "[$(date -u +%H:%M:%SZ)] ALL val corpora done"
