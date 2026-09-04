#!/usr/bin/env bash
# avifdoe_chroma_harvest.sh — pairs -> sizes -> harvest -> the unconfounded re-cut,
# for the chroma-split arm registered in benchmarks/avif_chroma_split_2026-09-04.md.
#
# ONE command, so a late wake-up is free: the artifacts are already on disk when
# anyone next looks (LATENCY DISCIPLINE, zensim CLAUDE.md). Re-runnable at any
# fill level -- it reports coverage as a FRACTION and never pretends a partial
# read is complete.
#
# Usage: [OUT=dir] avifdoe_chroma_harvest.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
. scripts/lib/s3env.sh
CTL="${ZFC_BIN:-$REPO/target/release/zenfleet-ctl}"
OUT="${OUT:-/mnt/v/output/avif-chroma-2026-09-04}"
PAIRDIR="${ZEN_DOE_PAIRDIR:-$HOME/tmp/chromasplit_pairs}"
REFS=s3://codec-corpus/avif-doe-1024-2026-09-01/
R420=avifdoe-rav-br420-20260904
R444=avifdoe-rav-br444-20260904
SFRUN=avifdoe-cs-sf-cpu-20260904
EP="${ZEN_S3_ENDPOINT:?}"
mkdir -p "$OUT/tables" "$PAIRDIR" "$OUT/scoreblobs"

echo "== 1. pairs (the LEDGER is the cell set; a blob count is not a coverage number)"
for R in $R420 $R444; do
  "$CTL" pairs --ledger "s3://zentrain/jobs/$R/ledger/" --refs-prefix "$REFS" \
      --blobs-prefix "s3://zentrain/jobs/$R/blobs/" --out "$PAIRDIR/${R}_pairs" --endpoint "$EP"
done

echo "== 2. sizes (encode_sha -> bytes, from both arms' blob listings)"
: > "$OUT/sizes.tsv"
for R in $R420 $R444; do
  aws s3 ls --endpoint-url "$EP" "s3://zentrain/jobs/$R/blobs/" --recursive \
    | awk -v OFS='\t' '{n=split($4,p,"/"); print p[n], $3}' >> "$OUT/sizes.tsv"
done
sort -u -o "$OUT/sizes.tsv" "$OUT/sizes.tsv"
echo "   sizes rows: $(wc -l < "$OUT/sizes.tsv")"

echo "== 3. score blobs"
aws s3 sync --endpoint-url "$EP" "s3://zentrain/jobs/$SFRUN/blobs/" "$OUT/scoreblobs/" --only-show-errors
echo "   score blobs: $(find "$OUT/scoreblobs" -type f | wc -l)"

echo "== 4. harvest"
python3 scripts/jobsys/avifdoe_harvest.py --score-dir "$OUT/scoreblobs" --sizes "$OUT/sizes.tsv" \
    --pairs "br420=$PAIRDIR/${R420}_pairs.tsv" --pairs "br444=$PAIRDIR/${R444}_pairs.tsv" \
    --out "$OUT/chroma_scored_2026-09-04.parquet"

echo "== 5. the re-cut"
python3 scripts/jobsys/avifdoe_chroma_analyze.py \
    --scored "$OUT/chroma_scored_2026-09-04.parquet" \
    --stagea-scored /mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet \
    --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
    --br444-pairs "$PAIRDIR/${R444}_pairs.tsv" \
    --brsdr-pairs "$HOME/tmp/avifdoe_pairs/avifdoe-rav-brsdr-20260903_pairs.tsv" \
    --outdir "$OUT/tables"
echo "== done -> $OUT"
