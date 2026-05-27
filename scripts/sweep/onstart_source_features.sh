#!/usr/bin/env bash
# onstart_source_features.sh — execs `zen-sweep-worker worker --backend vastai --mode source-features`.
# Computes zenanalyze 62-feature vectors for each unique source PNG
# referenced by a chunk's image_basenames, writes parquet to
# s3://zentrain/<run>/source_features/<chunk>.parquet.
# No scoring, no encoded variants, no GPU.
set -euo pipefail
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL*|GPU_*|METRICS|CHUNKS_*|SKIP_*|ADAPT_*|CONTAINER_*)
                export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID missing}"
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS
CHUNKS_R2="${CHUNKS_R2:-s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl}"
echo "[onstart-source-features] worker=${WORKER_ID:-$(hostname)} run=${SWEEP_RUN_ID}" >&2
export RUST_LOG="${RUST_LOG:-info}"
export WORKER_MODE=source-features
exec /usr/local/bin/zen-sweep-worker worker --backend vastai \
    --run-id "${SWEEP_RUN_ID}" \
    --chunks-r2 "${CHUNKS_R2}" \
    --mode source-features
