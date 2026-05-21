#!/usr/bin/env bash
# onstart_feature_backfill.sh — entry point for the unified Rust worker
# in `feature-backfill` mode.
#
# Reads existing omni sidecars from R2, downloads the already-saved
# encoded variants, computes zensim 300-feature vectors per cell
# WITHOUT re-encoding, writes a feature parquet to
# s3://zentrain/<run>/zensim_features/<chunk>.parquet.
#
# Same env contract as onstart_unified.sh; the only new variable
# is WORKER_MODE (set here to feature-backfill).

set -euo pipefail

if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL*|GPU_*|METRICS|CHUNKS_*|SKIP_*|ADAPT_*|CONTAINER_*)
                export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID env missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID env missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY env missing}"
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID env missing}"

mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS

CHUNKS_R2="${CHUNKS_R2:-s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl}"

echo "[onstart-feature-backfill] worker=${WORKER_ID:-$(hostname)} run=${SWEEP_RUN_ID} mode=feature-backfill" >&2

export RUST_LOG="${RUST_LOG:-debug}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-full}"
export WORKER_MODE=feature-backfill

exec /usr/local/bin/vastai-fleet worker \
    --run-id "${SWEEP_RUN_ID}" \
    --chunks-r2 "${CHUNKS_R2}" \
    --mode feature-backfill
