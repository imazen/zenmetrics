#!/usr/bin/env bash
# onstart_extract_acumen.sh — vast.ai entrypoint for the production-
# recipe Mode B-lite feature extraction at 372 features.
#
# Reads chunks.jsonl from R2, runs PARALLEL workers extracting
# (per chunk) the safesyn ~200-pair slice's Mode B-lite features.
# Outputs are 372-col parquets named by chunk_id, uploaded back to
# R2.
#
# Required env (passed via vast.ai --env):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID
# Optional:
#   WORKER_ID, PARALLEL, ACUMEN_BAND_IDX, ACUMEN_MODE_A
#   CHUNKS_R2 (default s3://coefficient/jobs/<run>/chunks.jsonl)
#   SIDECARS_R2 (default s3://coefficient/jobs/<run>/sidecars/)

set -uo pipefail

ONSTART_LOG="/tmp/onstart_extract_acumen.log"
exec > >(tee -a "$ONSTART_LOG") 2> >(tee -a "$ONSTART_LOG" >&2)

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2 ; }

# Hydrate env from PID 1
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL|ACUMEN_*|CHUNKS_R2|SIDECARS_R2|REGIME|CONTAINER_*)
                export "$k=$v"
                ;;
        esac
    done < /proc/1/environ
fi

# EXIT trap with self-destroy (matches v14/acumen pattern).
on_exit() {
    local rc=$?
    if [[ -n "${HEARTBEAT_PID:-}" ]]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi
    if (( rc == 0 )); then log "[on_exit] clean exit"; return; fi

    local run_id="${SWEEP_RUN_ID:-unknown}"
    local worker_id="${WORKER_ID:-$(hostname)-$$}"
    local container_id="${CONTAINER_ID:-unknown}"

    aws --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        s3 cp "$ONSTART_LOG" \
        "s3://coefficient/jobs/${run_id}/worker-logs/${worker_id}-failure.log" \
        --no-progress --only-show-errors 2>/dev/null || true

    if [[ -n "${R2_ACCOUNT_ID:-}" && "$container_id" != "unknown" ]]; then
        log "[on_exit] self-destroying $container_id"
        curl -s -X DELETE \
            "https://console.vast.ai/api/v0/instances/${container_id}/" \
            -H "Authorization: Bearer ${VASTAI_API_KEY:-}" >/dev/null 2>&1 || true
    fi
}
trap on_exit EXIT

WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
PARALLEL="${PARALLEL:-2}"
CHUNKS_R2="${CHUNKS_R2:-s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl}"
SIDECARS_R2="${SIDECARS_R2:-s3://coefficient/jobs/${SWEEP_RUN_ID}/sidecars/}"
export CHUNKS_R2 SIDECARS_R2 WORKER_ID

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

log "worker=$WORKER_ID parallel=$PARALLEL band=${ACUMEN_BAND_IDX:-3} mode_a=${ACUMEN_MODE_A:-1}"

# Heartbeat loop (background)
heartbeat() {
    while true; do
        local hb=$(printf '{"worker": "%s", "ts": "%s", "host": "%s"}' \
            "$WORKER_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)")
        echo "$hb" | aws --endpoint-url "$R2_ENDPOINT" s3 cp - \
            "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.json" \
            --no-progress --only-show-errors 2>/dev/null || true
        sleep 60
    done
}
heartbeat &
HEARTBEAT_PID=$!

# Download chunks.jsonl
log "downloading chunks.jsonl"
aws --endpoint-url "$R2_ENDPOINT" s3 cp "$CHUNKS_R2" /tmp/chunks.jsonl \
    --no-progress --only-show-errors

N_CHUNKS=$(wc -l < /tmp/chunks.jsonl)
log "$N_CHUNKS chunks total"

# Dispatch xargs -P
xargs -d '\n' -P "$PARALLEL" -I {} \
    /usr/local/bin/extract_acumen_chunk_worker.sh "{}" \
    < /tmp/chunks.jsonl
xargs_rc=$?

if (( xargs_rc != 0 )); then
    log "xargs returned $xargs_rc — propagating"
    exit "$xargs_rc"
fi

log "all chunks done"
exit 0
