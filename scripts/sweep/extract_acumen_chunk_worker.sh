#!/usr/bin/env bash
# extract_acumen_chunk_worker.sh — per-chunk feature extraction
# worker for the acumen Mode B-lite production-recipe experiment.
#
# Each chunk = a JSON line of (chunk_id, source_keys[], pairs_tsv_key).
# Worker:
#   1. atomically claims the chunk via S3 If-None-Match emulation
#   2. downloads source PNGs from R2 to local tmp
#   3. downloads the chunk's pairs.tsv from R2
#   4. runs extract_acumen_features (372 features, Mode B-lite band_3)
#   5. uploads the resulting parquet to R2
#   6. cleans up local tmp + claim
#
# Required env vars (forwarded by the onstart script):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID
#   WORKER_ID
#   CHUNKS_R2     (e.g. s3://coefficient/jobs/<run>/chunks/)
#   SIDECARS_R2   (e.g. s3://coefficient/jobs/<run>/sidecars/)
#   ACUMEN_BAND_IDX   (default 3 — sweep winner)
#   ACUMEN_BLUR_SIGMA (default 8)
#   ACUMEN_CLAMP_LO   (default 0.1)
#   ACUMEN_CLAMP_HI   (default 4.0)
#   ACUMEN_PPD        (default 56)
#   ACUMEN_PEAK_NITS  (default 100)
#   ACUMEN_AMBIENT_NITS (default 5)
#   REGIME            (default with_iw — 372 features)
#   ACUMEN_MODE_A     (default 1 — set to 0 to extract baseline)

set -uo pipefail

CHUNK_JSON_LINE="$1"
WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
ACUMEN_BAND_IDX="${ACUMEN_BAND_IDX:-3}"
ACUMEN_BLUR_SIGMA="${ACUMEN_BLUR_SIGMA:-8}"
ACUMEN_CLAMP_LO="${ACUMEN_CLAMP_LO:-0.1}"
ACUMEN_CLAMP_HI="${ACUMEN_CLAMP_HI:-4.0}"
ACUMEN_PPD="${ACUMEN_PPD:-56}"
ACUMEN_PEAK_NITS="${ACUMEN_PEAK_NITS:-100}"
ACUMEN_AMBIENT_NITS="${ACUMEN_AMBIENT_NITS:-5}"
REGIME="${REGIME:-with_iw}"
ACUMEN_MODE_A="${ACUMEN_MODE_A:-1}"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2 ; }

CHUNK_ID=$(echo "$CHUNK_JSON_LINE" | jq -r '.chunk_id')
PAIRS_TSV_KEY=$(echo "$CHUNK_JSON_LINE" | jq -r '.pairs_tsv_key')

LOG_PREFIX="[$WORKER_ID/$CHUNK_ID]"
log "$LOG_PREFIX starting chunk band=$ACUMEN_BAND_IDX regime=$REGIME mode_a=$ACUMEN_MODE_A"

# 1. Atomic claim
CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${CHUNK_ID}.claim"
echo "$WORKER_ID $(date -u +%s)" > /tmp/claim-${CHUNK_ID}.txt
if ! aws --endpoint-url "$R2_ENDPOINT" s3 cp /tmp/claim-${CHUNK_ID}.txt "$CLAIM_KEY" \
        --no-progress --only-show-errors 2>/dev/null; then
    log "$LOG_PREFIX claim already taken — skipping"
    exit 0
fi

# Skip-if-done: check if the sidecar already exists.
SIDECAR_KEY="${SIDECARS_R2%/}/${CHUNK_ID}.parquet"
if aws --endpoint-url "$R2_ENDPOINT" s3 ls "$SIDECAR_KEY" 2>/dev/null | grep -q parquet; then
    log "$LOG_PREFIX sidecar already exists — skipping"
    exit 0
fi

WORK_DIR=/tmp/work-${CHUNK_ID}
mkdir -p "$WORK_DIR"
cleanup() { rm -rf "$WORK_DIR" /tmp/claim-${CHUNK_ID}.txt /tmp/pairs-${CHUNK_ID}.tsv; }
trap cleanup EXIT

# 2. Download pairs.tsv (chunk-specific)
log "$LOG_PREFIX downloading pairs.tsv"
aws --endpoint-url "$R2_ENDPOINT" s3 cp "$PAIRS_TSV_KEY" /tmp/pairs-${CHUNK_ID}.tsv \
    --no-progress --only-show-errors || { log "pairs.tsv download FAIL"; exit 1; }

# 3. Download all referenced PNGs (deduped). The pairs.tsv has
#    columns: ref_path, dist_path, codec, q. Both are R2 keys
#    relative to s3://coefficient/datasets/safesyn-images/.
log "$LOG_PREFIX downloading PNGs"
tail -n +2 /tmp/pairs-${CHUNK_ID}.tsv | awk -F'\t' '{print $1; print $2}' | sort -u > /tmp/pngs-${CHUNK_ID}.txt
PNG_COUNT=$(wc -l < /tmp/pngs-${CHUNK_ID}.txt)
log "$LOG_PREFIX $PNG_COUNT unique PNGs"
# Parallel s5cmd download — same source bucket.
while IFS= read -r relpath; do
    src="s3://coefficient/datasets/safesyn-images/$relpath"
    dst="$WORK_DIR/$relpath"
    mkdir -p "$(dirname "$dst")"
    echo "cp $src $dst"
done < /tmp/pngs-${CHUNK_ID}.txt > /tmp/s5cmd-${CHUNK_ID}.txt
s5cmd --endpoint-url "$R2_ENDPOINT" run /tmp/s5cmd-${CHUNK_ID}.txt > /tmp/s5cmd-${CHUNK_ID}.log 2>&1 \
    || { log "PNG download FAIL"; tail -3 /tmp/s5cmd-${CHUNK_ID}.log >&2; exit 1; }

# Rewrite the pairs.tsv to use local paths.
LOCAL_PAIRS=/tmp/pairs-local-${CHUNK_ID}.tsv
{
    echo -e "ref_path\tdist_path"
    tail -n +2 /tmp/pairs-${CHUNK_ID}.tsv | awk -F'\t' -v root="$WORK_DIR" '{print root "/" $1 "\t" root "/" $2}'
} > "$LOCAL_PAIRS"

# 4. Extract
OUT_PARQUET=$WORK_DIR/${CHUNK_ID}.parquet
EXTRACT_ARGS=(
    --pairs-tsv "$LOCAL_PAIRS"
    --out "$OUT_PARQUET"
    --regime "$REGIME"
)
if [[ "$ACUMEN_MODE_A" == "1" ]]; then
    EXTRACT_ARGS+=(
        --acumen-mode-a
        --acumen-arch mode_b
        --acumen-ppd "$ACUMEN_PPD"
        --acumen-peak-nits "$ACUMEN_PEAK_NITS"
        --acumen-ambient-nits "$ACUMEN_AMBIENT_NITS"
        --mode-b-band-idx "$ACUMEN_BAND_IDX"
        --mode-b-blur-sigma "$ACUMEN_BLUR_SIGMA"
        --mode-b-clamp-lo "$ACUMEN_CLAMP_LO"
        --mode-b-clamp-hi "$ACUMEN_CLAMP_HI"
    )
fi
log "$LOG_PREFIX extracting (args: ${EXTRACT_ARGS[*]})"
/usr/local/bin/extract_acumen_features "${EXTRACT_ARGS[@]}" 2>&1 | tail -3 >&2 \
    || { log "$LOG_PREFIX extraction FAIL"; exit 1; }

# 5. Upload
log "$LOG_PREFIX uploading sidecar to $SIDECAR_KEY"
aws --endpoint-url "$R2_ENDPOINT" s3 cp "$OUT_PARQUET" "$SIDECAR_KEY" \
    --no-progress --only-show-errors || { log "upload FAIL"; exit 1; }

log "$LOG_PREFIX done"
