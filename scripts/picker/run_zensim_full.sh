#!/usr/bin/env bash
set -u
SRC=/mnt/v/output/picker-pipeline-2026-06-22
OUT=/home/lilith/picker-pp
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
BIN=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
Q="5,10,20,30,40,50,60,70,80,90,95"
z() { local c=$1 corp=$2 plan=$3; shift 3
  echo "=== $(date -u +%H:%M:%S) $c zensim(CPU,fixed,persist) $corp ==="
  timeout 1800 nice -n19 "$BIN" sweep --codec "$c" --sources "$SRC/$corp" --q-grid "$Q" --plan "$plan" "$@" \
    --metric zensim --encoded-out-dir "$OUT/enc/$c" --output "$OUT/sweeps/${c}.zensim.tsv"
  echo "=== $(date -u +%H:%M:%S) $c rc=$? rows=$(wc -l < "$OUT/sweeps/${c}.zensim.tsv" 2>/dev/null) files=$(ls "$OUT/enc/$c" 2>/dev/null|wc -l) ==="
}
z zenjpeg corpus   rd_core
z zenjxl  corpus   rd_core
z zenwebp corpus64 modes_full --plan-budget 60
z zenavif corpus24 rd_core
echo "ZENSIM FULL DONE"
