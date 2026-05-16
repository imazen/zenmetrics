#!/usr/bin/env bash
#
# onstart_cvvdp_backfill_imazen.sh — vast.ai instance entry point
# for the IMAZEN-ONLY variant of the cvvdp-backfill fleet.
#
# Background: standard vast.ai SSH instances don't allow Docker-in-
# Docker (no privileged mode for iptables/nftables), so the
# dual-image flow in onstart_cvvdp_backfill.sh fails at dockerd
# init. This variant trades the cvvdp_pycvvdp_v054 column for a
# working single-image flow: boot the zen-metrics-sweep image
# directly, run zen-metrics from it, skip pycvvdp entirely. The
# finalize.sh path tolerates missing pycvvdp sidecars (parity=null).
#
# Expected boot image:
#   ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-<short>
# which has zen-metrics + s5cmd + jq at /usr/local/bin/, no python.
#
# Required env vars (passed via vast.ai --env):
#   R2_ACCOUNT_ID
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID                 e.g. cvvdp-backfill-2026-05-15-imazen
# Optional:
#   WORKER_ID                    defaults to $(hostname)-$$
#   PARALLEL                     concurrent chunk workers per box (default 2)
#   WORKDIR                      defaults to /workspace/cvvdp-backfill
#   SCRIPTS_R2_PREFIX            override; default uses SWEEP_RUN_ID
#   GPU_RUNTIME                  cvvdp-gpu backend (auto/cuda/wgpu/cpu); default auto

set -uo pipefail

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

# Pull env from PID 1.
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL|WORKDIR|SCRIPTS_R2_PREFIX|GPU_RUNTIME)
                export "$k=$v"
                ;;
        esac
    done < /proc/1/environ
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID missing}"

WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
PARALLEL="${PARALLEL:-2}"
WORKDIR="${WORKDIR:-/workspace/cvvdp-backfill}"
SCRIPTS_R2_PREFIX="${SCRIPTS_R2_PREFIX:-s3://coefficient/jobs/${SWEEP_RUN_ID}}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

R2() {
    s5cmd \
        --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        --profile r2 \
        "$@"
}

# R2 auth (~/.aws/credentials profile=r2). Fresh vast.ai instance,
# overwrite is safe here.
mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
EOF

# ── Step 1: install python3 + pyarrow (chunk_worker.sh slices parquets) ──
log "checking tools: zen-metrics s5cmd jq python3"
for tool in zen-metrics s5cmd jq; do
    if ! command -v "$tool" >/dev/null; then
        log "FAIL: $tool not on PATH; wrong boot image?"
        exit 2
    fi
done
if ! command -v python3 >/dev/null || ! command -v pip3 >/dev/null; then
    log "installing python3 + python3-pip via apt (boot image missing one or both)"
    apt-get update -q
    apt-get install -yq --no-install-recommends python3 python3-pip \
        || { log "FAIL apt-get install python3 python3-pip"; exit 3; }
fi
if ! python3 -c "import pyarrow" 2>/dev/null; then
    log "installing pyarrow"
    pip3 install --quiet --break-system-packages pyarrow 2>/dev/null \
        || pip3 install --quiet pyarrow \
        || { log "FAIL pip install pyarrow"; exit 3; }
fi

# ── Step 2: heartbeat ──────────────────────────────────────────────────
heartbeat() {
    local phase="$1"
    local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    cat > /tmp/hb.json <<EOF
{"ts":"$ts","worker":"$WORKER_ID","phase":"$phase","gpu_runtime":"$GPU_RUNTIME"}
EOF
    R2 cp /tmp/hb.json "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.${phase}" \
        >/dev/null 2>&1 || true
}
heartbeat boot

# ── Step 3: pull worker script + chunks.jsonl ──────────────────────────
log "pulling chunk_worker.sh + chunks.jsonl from $SCRIPTS_R2_PREFIX"
R2 cp "${SCRIPTS_R2_PREFIX}/cvvdp_backfill_chunk_worker.sh" "$WORKDIR/chunk_worker.sh" \
    || { log "FAIL pull chunk_worker.sh"; exit 4; }
chmod +x "$WORKDIR/chunk_worker.sh"
R2 cp "${SCRIPTS_R2_PREFIX}/chunks.jsonl" "$WORKDIR/chunks.jsonl" \
    || { log "FAIL pull chunks.jsonl"; exit 4; }
N_CHUNKS=$(wc -l < "$WORKDIR/chunks.jsonl")
log "loaded $N_CHUNKS chunks"

# ── Step 4: atomic-claim + invoke chunk_worker.sh (imazen-only) ────────
process_chunk() {
    local line="$1"
    local chunk_id out_imazen

    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    out_imazen=$(printf '%s' "$line" | jq -r '.out_sidecar_imazen')

    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Skip if imazen sidecar already exists (idempotent re-runs).
    if R2 ls "$out_imazen" 2>/dev/null | grep -q "${chunk_id}\.parquet"; then
        log "[skip] $chunk_id already complete"
        return 0
    fi

    # Token-based claim with read-back verification (same pattern as
    # onstart_v3.sh + onstart_cvvdp_backfill.sh).
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

    log "[claim-ok] $chunk_id — starting (imazen-only)"
    local start_t; start_t=$(date +%s)
    local LOG="/tmp/chunk-${chunk_id}.log"

    # Run chunk_worker.sh in host-binary mode (no --zen-metrics-image
    # since we're already inside the zen-metrics-sweep container) and
    # with --skip-pycvvdp.
    R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
    R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    GPU_RUNTIME="$GPU_RUNTIME" \
    "$WORKDIR/chunk_worker.sh" \
        --chunk-json "$line" \
        --work-dir "$WORKDIR/work-${chunk_id}" \
        --skip-pycvvdp \
        > "$LOG" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - start_t ))

    if (( rc != 0 )); then
        log "[fail] $chunk_id rc=$rc dt=${dt}s — uploading log"
        R2 cp "$LOG" "s3://coefficient/logs/${SWEEP_RUN_ID}/${chunk_id}.fail.log" \
            >/dev/null 2>&1 || true
        return 1
    fi

    log "[done] $chunk_id dt=${dt}s"
    rm -f "$LOG" /tmp/claim-${chunk_id}.txt
    rm -rf "$WORKDIR/work-${chunk_id}"
}

# LD_LIBRARY_PATH inherited from Dockerfile ENV is already
# correct (nvidia mount paths only, no compat). The cuda124
# image builds cudarc against CUDA 12.4 SDK so the binary's
# cudart matches what most vast.ai hosts run (driver 550+).
log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

heartbeat run

export -f process_chunk log R2
export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY \
    SWEEP_RUN_ID WORKER_ID WORKDIR GPU_RUNTIME

# Shuffle chunks so parallel workers on the same box don't all
# claim-race on the same chunk_id at startup.
shuf "$WORKDIR/chunks.jsonl" > "$WORKDIR/chunks.shuf.jsonl"

xargs -I {} -P "$PARALLEL" -d '\n' bash -c 'process_chunk "$@"' _ {} \
    < "$WORKDIR/chunks.shuf.jsonl"

heartbeat done
log "all chunks processed"
