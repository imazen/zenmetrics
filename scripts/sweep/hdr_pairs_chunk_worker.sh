#!/usr/bin/env bash
#
# hdr_pairs_chunk_worker.sh — persisted-pairs shape of the metric-backfill
# chunk worker (`metric_backfill_chunk_worker.sh`): the variants ALREADY
# exist in R2 (e.g. the kadis-hdr distortion corpus), so there is no
# re-encode step — the worker syncs the chunk's refs + dists and runs
# `score-pairs --hdr` for EVERY metric in one pass (one data sync per chunk,
# all metrics; unlike the SDR backfill's one-metric-per-invocation shape).
#
# Chunk JSON (from generate_hdr_pairs_chunks.py):
#   {"chunk_id","run_id","pairs_r2","row_range":[start,end),
#    "data_prefix" (s3://... with refs/ + dist/ under it),
#    "out_prefix"  (s3://... sidecar destination)}
#
# Per chunk:
#   1. fetch pairs.tsv, awk-slice [start,end) data rows
#   2. s5cmd-download the slice's unique refs + dists (relative paths in
#      the TSV resolve against data_prefix)
#   3. for each metric: score-pairs --hdr --hdr-transfer pu-rescale
#      (zensim-gpu additionally emits the 372-feature with-iw sidecar)
#   4. upload out_prefix/<chunk_id>/<metric>.parquet (+ zensim_features)
#
# Env (same conventions as the backfill worker / onstart_unified):
#   R2_ACCOUNT_ID + [r2] profile in ~/.aws/credentials  (creds)
#   METRICS      comma-list; default
#                zensim-gpu,ssim2-gpu,cvvdp,iwssim-gpu,butteraugli-gpu
#                (dssim is HDR-Unsupported by design)
#   GPU_RUNTIME  default cuda
#   ZEN_BIN      zenmetrics binary; default /usr/local/bin/zenmetrics
#   WORK_ROOT    scratch; default /tmp/hdrpairs
#   SKIP_UPLOAD / KEEP_WORK — debug knobs, default 0
set -euo pipefail

METRICS="${METRICS:-zensim-gpu,ssim2-gpu,cvvdp,iwssim-gpu,butteraugli-gpu}"
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"
ZEN_BIN="${ZEN_BIN:-/usr/local/bin/zenmetrics}"
WORK_ROOT="${WORK_ROOT:-/tmp/hdrpairs}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
KEEP_WORK="${KEEP_WORK:-0}"
CHUNK_JSON="${CHUNK_JSON:-$(cat)}"

for tool in jq s5cmd awk; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() { s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 "$@"; }

CHUNK_ID=$(echo "$CHUNK_JSON" | jq -r '.chunk_id')
PAIRS_R2=$(echo "$CHUNK_JSON" | jq -r '.pairs_r2')
ROW_START=$(echo "$CHUNK_JSON" | jq -r '.row_range[0]')
ROW_END=$(echo "$CHUNK_JSON" | jq -r '.row_range[1]')
DATA_PREFIX=$(echo "$CHUNK_JSON" | jq -r '.data_prefix')
OUT_PREFIX=$(echo "$CHUNK_JSON" | jq -r '.out_prefix')
[[ -n "$CHUNK_ID" && "$CHUNK_ID" != null ]] || { echo "chunk_id missing" >&2; exit 1; }

W="$WORK_ROOT/$CHUNK_ID"
mkdir -p "$W/refs" "$W/dist" "$W/out"
cleanup() { [[ "$KEEP_WORK" = 1 ]] || rm -rf "$W"; }
trap cleanup EXIT
LOG="[hdr-pairs $CHUNK_ID]"

echo "$LOG fetch pairs + slice rows [$ROW_START,$ROW_END)" >&2
R2 cp "$PAIRS_R2" "$W/pairs_full.tsv" >&2
# data rows are 1-indexed after the header: rows [start,end) = lines start+2..end+1
awk -v s="$ROW_START" -v e="$ROW_END" 'NR==1 || (NR-2>=s && NR-2<e)' "$W/pairs_full.tsv" > "$W/slice_rel.tsv"
N=$(($(wc -l < "$W/slice_rel.tsv") - 1))
echo "$LOG $N pairs in slice" >&2
[[ "$N" -gt 0 ]] || { echo "$LOG empty slice" >&2; exit 1; }

echo "$LOG sync refs + dists" >&2
awk -F'\t' -v p="$DATA_PREFIX" 'NR>1 {print "cp " p "/" $5 " refs/"; print "cp " p "/" $6 " dist/"}' \
    "$W/slice_rel.tsv" | sort -u > "$W/_dl.run"
( cd "$W" && R2 run "$W/_dl.run" >&2 ) || { echo "$LOG data sync FAILED" >&2; exit 2; }

# Absolute-path pairs slice for score-pairs (ref_path/dist_path columns).
awk -F'\t' -v OFS='\t' -v w="$W" \
    'NR==1 {print; next} {n5=split($5,a,"/"); n6=split($6,b,"/"); $5=w"/refs/"a[n5]; $6=w"/dist/"b[n6]; print}' \
    "$W/slice_rel.tsv" > "$W/slice.tsv"

rc_any=0
IFS=',' read -ra MLIST <<< "$METRICS"
for m in "${MLIST[@]}"; do
    extra=()
    [[ "$m" == zensim* ]] && extra=(--feature-output "$W/out/zensim_features.parquet" \
                                    --zensim-features-regime with-iw)
    echo "$LOG score-pairs --hdr $m" >&2
    if ! "$ZEN_BIN" score-pairs --metric "$m" --hdr --hdr-transfer pu-rescale \
        --pairs-tsv "$W/slice.tsv" --out-parquet "$W/out/$m.parquet" \
        --gpu-runtime "$GPU_RUNTIME" "${extra[@]}" >"$W/out/$m.log" 2>&1; then
        echo "$LOG $m FAILED:" >&2; tail -3 "$W/out/$m.log" >&2
        rc_any=1
        [[ "$SKIP_UPLOAD" = 1 ]] || R2 cp "$W/out/$m.log" "$OUT_PREFIX/$CHUNK_ID/$m.FAILED.log" >&2 || true
        continue
    fi
    grep -m1 -oE 'wrote [0-9]+ rows \([0-9]+ NaN-failures\)' "$W/out/$m.log" | sed "s/^/$LOG $m: /" >&2 || true
    [[ "$SKIP_UPLOAD" = 1 ]] || R2 cp "$W/out/$m.parquet" "$OUT_PREFIX/$CHUNK_ID/$m.parquet" >&2
done
if [[ -f "$W/out/zensim_features.parquet" && "$SKIP_UPLOAD" != 1 ]]; then
    R2 cp "$W/out/zensim_features.parquet" "$OUT_PREFIX/$CHUNK_ID/zensim_features.parquet" >&2
fi
# _DONE sentinel LAST, and only on full success — it is the idempotency
# marker onstart_hdr_pairs.sh checks, so it must postdate every upload.
if [[ "$rc_any" = 0 && "$SKIP_UPLOAD" != 1 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$W/out/_DONE"
    R2 cp "$W/out/_DONE" "$OUT_PREFIX/$CHUNK_ID/_DONE" >&2
fi
echo "$LOG done rc=$rc_any" >&2
exit "$rc_any"
