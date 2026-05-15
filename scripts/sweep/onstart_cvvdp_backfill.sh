#!/usr/bin/env bash
# shellcheck disable=SC2086
#
# onstart_cvvdp_backfill.sh — vast.ai worker entrypoint for the
# cvvdp-backfill fleet (PINNED TASK).
#
# Sibling to onstart_v3.sh. Reuses the same R2 + atomic-claim
# pattern but consumes the chunk format produced by
# scripts/sweep/generate_cvvdp_backfill_chunks.py (input_parquet +
# row_range + image_basenames + out_sidecar_imazen +
# out_sidecar_pycvvdp) and delegates per-chunk work to
# scripts/sweep/cvvdp_backfill_chunk_worker.sh.
#
# Required env vars (passed via vast.ai --env):
#   R2_ACCOUNT_ID
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID                 e.g. cvvdp-backfill-2026-05-15
#   ZEN_METRICS_IMAGE            ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-76854e8
#   PYCVVDP_IMAGE                ghcr.io/imazen/pycvvdp-scorer:0.5.4
# (use the 0.6.4-cvvdp-<short> tag — earlier 0.6.4-aba984c images
#  lacked gpu-cvvdp features and the cvvdp metric was disabled at
#  runtime; the cvvdp_backfill/launch.sh default tracks the current
#  build)
#
# Optional:
#   WORKER_ID                    defaults to $(hostname)-$$
#   PARALLEL                     concurrent chunk workers per box (default 2)
#   WORKDIR                      defaults to /workspace/cvvdp-backfill
#   SCRIPTS_R2_PREFIX            override; default uses SWEEP_RUN_ID

set -uo pipefail
# Don't set -e at script level — we want the worker loop to continue
# on per-chunk failures rather than crashing the box.

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

# Pull R2 env from /proc/1/environ if running inside vast.ai (it sets
# env vars on PID 1 but not necessarily in `bash`'s environment for
# --onstart-cmd).
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|STATS_*|ZEN_METRICS_IMAGE|PYCVVDP_IMAGE|PARALLEL|WORKDIR|SCRIPTS_R2_PREFIX)
                export "$k=$v"
                ;;
        esac
    done < /proc/1/environ
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing — pass via --env}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing — pass via --env}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing — pass via --env}"
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID missing — pass via --env}"
: "${ZEN_METRICS_IMAGE:?ZEN_METRICS_IMAGE missing — pass via --env}"
: "${PYCVVDP_IMAGE:?PYCVVDP_IMAGE missing — pass via --env}"

WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
PARALLEL="${PARALLEL:-2}"
WORKDIR="${WORKDIR:-/workspace/cvvdp-backfill}"
SCRIPTS_R2_PREFIX="${SCRIPTS_R2_PREFIX:-s3://coefficient/jobs/${SWEEP_RUN_ID}}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# R2 wrapper using s5cmd-compatible env.
R2() {
    s5cmd \
        --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        --profile r2 \
        "$@"
}

# s5cmd auth via env (~/.aws/credentials profile=r2).
mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
EOF

# ── Step 1: install s5cmd + jq + minio mc statically ──────────────────
log "installing s5cmd / jq / docker"
if ! command -v s5cmd >/dev/null; then
    curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz" \
        -o /tmp/s5cmd.tgz
    tar xzf /tmp/s5cmd.tgz -C /usr/local/bin s5cmd
    chmod +x /usr/local/bin/s5cmd
fi
if ! command -v jq >/dev/null; then
    curl -fsSL "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64" \
        -o /usr/local/bin/jq
    chmod +x /usr/local/bin/jq
fi
if ! command -v docker >/dev/null; then
    log "ERROR: docker not on PATH and we can't reliably install it as non-root"
    log "use a vast.ai template that already includes docker (most CUDA templates do)"
    exit 2
fi
if ! command -v python3 >/dev/null; then
    log "installing python3 + pyarrow"
    apt-get update -q && apt-get install -yq python3 python3-pip
    pip3 install --quiet pyarrow
fi

# ── Step 2: docker login + image pre-pull (kills cold-pull jitter) ────
log "docker login + pre-pull images"
if [[ -n "${GHCR_TOKEN:-}" ]]; then
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:-imazen}" --password-stdin
fi
docker pull "$ZEN_METRICS_IMAGE" || { log "FAIL pull $ZEN_METRICS_IMAGE"; exit 3; }
docker pull "$PYCVVDP_IMAGE"    || { log "FAIL pull $PYCVVDP_IMAGE";    exit 3; }

# ── Step 3: heartbeat ─────────────────────────────────────────────────
heartbeat() {
    local kind="$1"
    cat > /tmp/hb.json <<EOF
{
  "worker_id": "$WORKER_ID",
  "sweep_run_id": "$SWEEP_RUN_ID",
  "kind": "$kind",
  "epoch": $(date +%s),
  "hostname": "$(hostname)",
  "zen_metrics_image": "$ZEN_METRICS_IMAGE",
  "pycvvdp_image": "$PYCVVDP_IMAGE",
  "parallel": $PARALLEL
}
EOF
    R2 cp /tmp/hb.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.${kind}" \
        >/dev/null 2>&1 || true
}
heartbeat boot

# ── Step 4: pull the worker script + chunks.jsonl from R2 ─────────────
log "pulling chunk worker script + chunks.jsonl"
R2 cp "${SCRIPTS_R2_PREFIX}/cvvdp_backfill_chunk_worker.sh" "$WORKDIR/chunk_worker.sh" \
    || { log "FAIL pull chunk_worker.sh"; exit 4; }
chmod +x "$WORKDIR/chunk_worker.sh"
R2 cp "${SCRIPTS_R2_PREFIX}/chunks.jsonl" "$WORKDIR/chunks.jsonl" \
    || { log "FAIL pull chunks.jsonl"; exit 4; }
log "loaded $(wc -l < "$WORKDIR/chunks.jsonl") chunks"

# ── Step 5: atomic-claim + invoke chunk_worker.sh per chunk ───────────
process_chunk() {
    local line="$1"
    local chunk_id out_imazen out_pycvvdp

    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    out_imazen=$(printf '%s' "$line" | jq -r '.out_sidecar_imazen')
    out_pycvvdp=$(printf '%s' "$line" | jq -r '.out_sidecar_pycvvdp')

    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Skip if BOTH sidecars already exist (idempotent re-runs).
    local have_imazen=0 have_pycvvdp=0
    R2 ls "$out_imazen"  2>/dev/null | grep -q "${chunk_id}\.parquet" && have_imazen=1
    R2 ls "$out_pycvvdp" 2>/dev/null | grep -q "${chunk_id}\.parquet" && have_pycvvdp=1
    if (( have_imazen == 1 && have_pycvvdp == 1 )); then
        log "[skip] $chunk_id already complete"
        return 0
    fi

    # Token-based claim (read-back verification, same pattern as
    # onstart_v3.sh::process_chunk).
    local claim_body=/tmp/claim-${chunk_id}.txt
    local token="${WORKER_ID}-$$-$(date +%s%N)"
    local now_epoch; now_epoch=$(date +%s)
    printf '%s\t%s\t%s' "$token" "$now_epoch" "$WORKER_ID" > "$claim_body"

    local existing
    existing=$(R2 cat "$CLAIM_KEY" 2>/dev/null) || existing=""
    if [[ -n "$existing" ]]; then
        local existing_epoch existing_worker
        existing_epoch=$(printf '%s' "$existing" | awk -F'\t' '{print $2}')
        existing_worker=$(printf '%s' "$existing" | awk -F'\t' '{print $3}')
        if [[ -n "$existing_epoch" ]] \
                && (( now_epoch - existing_epoch < 600 )) \
                && [[ "$existing_worker" != "$WORKER_ID" ]]; then
            log "[skip-claim-fresh] $chunk_id (held by $existing_worker)"
            return 0
        fi
    fi

    R2 cp "$claim_body" "$CLAIM_KEY" 2>/dev/null || return 1
    sleep 1.5
    local verified
    verified=$(R2 cat "$CLAIM_KEY" 2>/dev/null | awk -F'\t' '{print $1}')
    if [[ "$verified" != "$token" ]]; then
        log "[lost-claim] $chunk_id"
        return 0
    fi

    log "[claim-ok] $chunk_id — starting"
    local start_t; start_t=$(date +%s)
    local LOG="/tmp/chunk-${chunk_id}.log"

    R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
    R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    "$WORKDIR/chunk_worker.sh" \
        --chunk-json "$line" \
        --work-dir "$WORKDIR/work-${chunk_id}" \
        --zen-metrics-image "$ZEN_METRICS_IMAGE" \
        --pycvvdp-image "$PYCVVDP_IMAGE" \
        > "$LOG" 2>&1

    local rc=$?
    local elapsed=$(( $(date +%s) - start_t ))
    if [[ $rc == 0 ]]; then
        log "[ok] $chunk_id (${elapsed}s)"
        # Best-effort log upload for post-mortem.
        R2 cp "$LOG" \
            "s3://coefficient/logs/${SWEEP_RUN_ID}/${chunk_id}.log" \
            2>/dev/null || true
    else
        log "[FAIL rc=$rc] $chunk_id (${elapsed}s); see $LOG"
        # Always upload failure logs.
        R2 cp "$LOG" \
            "s3://coefficient/logs/${SWEEP_RUN_ID}/${chunk_id}.fail.log" \
            2>/dev/null || true
        # Don't release the claim — let it expire after 600s so another
        # worker can retry without overlap. This avoids thundering-herd
        # retries on a chunk that's failing for everyone.
    fi
}
export -f process_chunk R2 log heartbeat
export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY \
    SWEEP_RUN_ID WORKER_ID WORKDIR ZEN_METRICS_IMAGE PYCVVDP_IMAGE

# ── Step 6: main loop ─────────────────────────────────────────────────
heartbeat run
log "starting main loop (parallel=$PARALLEL)"

# Use xargs with NUL-delimited input. Each chunk-json line is processed
# by a separate process_chunk invocation. PARALLEL=2 by default; vast.ai
# A100 boxes can usually handle PARALLEL=4 if the chunk-worker's docker
# runs share the GPU sensibly.
shuf "$WORKDIR/chunks.jsonl" | tr '\n' '\0' | \
    xargs -0 -P "$PARALLEL" -I {} bash -c 'process_chunk "$@"' _ {} || true

heartbeat done
log "main loop exited; uploading final heartbeat"
