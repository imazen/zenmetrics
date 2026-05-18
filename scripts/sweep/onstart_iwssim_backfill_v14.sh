#!/usr/bin/env bash
#
# onstart_iwssim_backfill_v14.sh — vast.ai entrypoint for the
# iwssim-backfill fleet, paired with `Dockerfile.sweep.v14`.
#
# Key difference from v3 (`onstart_iwssim_backfill.sh`): the docker
# image already bakes python3 + python3-pyarrow + cuda-nvrtc-12-6 +
# cuda-cudart-12-6 + s5cmd + jq + zen-metrics. This script does NO
# apt installs and NO pip installs — it only fetches the per-sweep
# scripts (chunks.jsonl + chunk worker) and runs the chunk loop.
#
# Boot-to-first-heartbeat: ~5 s (docker pull cached + this script's
# fetch latency). Compare to v3 which spent 5-15 min on apt before
# even reaching the heartbeat line.
#
# Required env vars (passed via vast.ai --env):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID
# Optional:
#   WORKER_ID                    defaults to $(hostname)-$$
#   PARALLEL                     concurrent chunk workers per box (default auto)
#   WORKDIR                      defaults to /workspace/sweep
#   SCRIPTS_R2_PREFIX            override; default uses SWEEP_RUN_ID
#   GPU_RUNTIME                  iwssim-gpu backend (auto/cuda/wgpu/cpu); default auto

set -uo pipefail

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

# Hydrate env from PID 1 (where vast.ai injects worker env vars).
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL|WORKDIR|SCRIPTS_R2_PREFIX|GPU_RUNTIME|CUDA_PATH)
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
PARALLEL="${PARALLEL:-0}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
WORKDIR="${WORKDIR:-/workspace/sweep}"
SCRIPTS_R2_PREFIX="${SCRIPTS_R2_PREFIX:-s3://coefficient/jobs/${SWEEP_RUN_ID}}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Auto-detect PARALLEL from cgroup if not set.
if [[ "$PARALLEL" == "0" ]]; then
    cores_from_cgroup() {
        if [[ -r /sys/fs/cgroup/cpu.max ]]; then
            read -r q p < /sys/fs/cgroup/cpu.max
            [[ "$q" == "max" || -z "$q" ]] && return 1
            echo $(( (q + p - 1) / p ))
            return 0
        fi
        return 1
    }
    ram_gb_from_cgroup() {
        if [[ -r /sys/fs/cgroup/memory.max ]]; then
            local m; m=$(cat /sys/fs/cgroup/memory.max)
            [[ "$m" == "max" || -z "$m" ]] && return 1
            echo $(( m / 1024 / 1024 / 1024 ))
            return 0
        fi
        return 1
    }
    cgroup_cpu=$(cores_from_cgroup || nproc)
    ram_cap=$(ram_gb_from_cgroup || echo 16)
    cpu_cap=$(( cgroup_cpu > 2 ? cgroup_cpu - 2 : 1 ))
    ram_cap=$(( ram_cap * 2 / 3 ))
    PARALLEL=$(( cpu_cap < ram_cap ? cpu_cap : ram_cap ))
    PARALLEL=$(( PARALLEL > 0 ? PARALLEL : 1 ))
    log "auto-detect PARALLEL=$PARALLEL (cgroup_cpu=$cgroup_cpu cpu_cap=$cpu_cap ram_cap=$ram_cap)"
fi

# Tools sanity check: every binary must already exist in the image.
# If any are missing the image is broken — fail loud, don't try to
# install at runtime (that's what we're escaping).
log "checking baked tools: zen-metrics s5cmd jq python3 pyarrow libnvrtc12"
MISSING=()
for tool in zen-metrics s5cmd jq python3; do
    command -v "$tool" >/dev/null || MISSING+=("$tool")
done
python3 -c "import pyarrow" 2>/dev/null || MISSING+=("python3-pyarrow")
ldconfig -p | grep -q libnvrtc.so.12 || MISSING+=("libnvrtc12")
if (( ${#MISSING[@]} > 0 )); then
    log "FATAL: image missing baked dependencies: ${MISSING[*]}"
    log "       this onstart MUST run inside an image built from Dockerfile.sweep.v14+"
    log "       v3 onstart paths would apt-install these; v14 does NOT."
    exit 10
fi
log "baked tools OK"
log "zen-metrics version: $(zen-metrics --version 2>&1 | head -1)"

# R2 credentials wired to s5cmd.
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = $R2_ACCESS_KEY_ID
aws_secret_access_key = $R2_SECRET_ACCESS_KEY
CREDS

R2() {
    s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        --profile r2 "$@"
}

heartbeat() {
    local phase="$1"
    local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    cat > /tmp/hb.json <<EOF
{"ts":"$ts","worker":"$WORKER_ID","phase":"$phase","gpu_runtime":"$GPU_RUNTIME"}
EOF
    R2 cp /tmp/hb.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.${phase}" \
        >/dev/null 2>&1 || true
}
heartbeat boot

log "pulling chunks.jsonl + iwssim_backfill_chunk_worker.sh from $SCRIPTS_R2_PREFIX"
R2 cp "${SCRIPTS_R2_PREFIX}/iwssim_backfill_chunk_worker.sh" "$WORKDIR/chunk_worker.sh" \
    || { log "FAIL pull chunk_worker.sh"; exit 4; }
chmod +x "$WORKDIR/chunk_worker.sh"
R2 cp "${SCRIPTS_R2_PREFIX}/chunks.jsonl" "$WORKDIR/chunks.jsonl" \
    || { log "FAIL pull chunks.jsonl"; exit 4; }
N_CHUNKS=$(wc -l < "$WORKDIR/chunks.jsonl")
log "loaded $N_CHUNKS chunks"

process_chunk() {
    local line="$1"
    local chunk_id out_iwssim
    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    out_iwssim=$(printf '%s' "$line" | jq -r '.out_sidecar_iwssim')

    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Idempotent skip if sidecar already in R2.
    if R2 ls "$out_iwssim" 2>/dev/null | grep -q "${chunk_id}\.parquet"; then
        log "[skip] $chunk_id already complete"
        return 0
    fi

    local claim_body=/tmp/claim-${chunk_id}.txt
    local token="${WORKER_ID}-$$-$(date +%s%N)"
    local now_epoch; now_epoch=$(date +%s)
    printf '%s\t%s\t%s' "$token" "$now_epoch" "$WORKER_ID" > "$claim_body"

    # Race-tolerant claim: if another worker won, skip; if claim is
    # fresh (< 1 h) bail; if stale, steal it.
    local existing
    existing=$(R2 cat "$CLAIM_KEY" 2>/dev/null) || existing=""
    if [[ -n "$existing" ]]; then
        local existing_epoch existing_worker
        existing_epoch=$(printf '%s' "$existing" | awk -F'\t' '{print $2}')
        existing_worker=$(printf '%s' "$existing" | awk -F'\t' '{print $3}')
        if [[ -n "$existing_epoch" ]] \
            && (( now_epoch - existing_epoch < 3600 )) \
            && [[ "$existing_worker" != "$WORKER_ID" ]]; then
            log "[skip-claim-fresh] $chunk_id (held by $existing_worker)"
            rm -f "$claim_body"
            return 0
        fi
    fi
    R2 cp "$claim_body" "$CLAIM_KEY" 2>/dev/null || return 1

    sleep 0.2
    local recheck; recheck=$(R2 cat "$CLAIM_KEY" 2>/dev/null) || recheck=""
    if [[ "$(printf '%s' "$recheck" | awk -F'\t' '{print $1}')" != "$token" ]]; then
        log "[lost-claim] $chunk_id"
        rm -f "$claim_body"
        return 0
    fi
    log "[claim-ok] $chunk_id — starting"

    local LOG="/tmp/iwssim-chunk-${chunk_id}.log"
    if printf '%s' "$line" \
        | "$WORKDIR/chunk_worker.sh" \
            --work-dir "/tmp/iwssim-${chunk_id}-$$" \
            > "$LOG" 2>&1; then
        local dt; dt=$(( $(date +%s) - now_epoch ))
        log "[done] $chunk_id dt=${dt}s"
    else
        local rc=$?
        log "[FAIL] $chunk_id rc=$rc — last 20 lines:"
        tail -20 "$LOG" | sed 's/^/    /' >&2
    fi
    rm -f "$LOG" /tmp/claim-${chunk_id}.txt
}

heartbeat run

# Heartbeat loop in background — phase=alive, every 60s, so the
# launcher can tell we're not dead even mid-chunk.
(
    while true; do
        sleep 60
        heartbeat alive 2>/dev/null || true
    done
) &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null || true' EXIT

# Fan out chunks across PARALLEL workers via xargs.
log "running $N_CHUNKS chunks at parallel=$PARALLEL"
< "$WORKDIR/chunks.jsonl" xargs -P "$PARALLEL" -d '\n' -I {} bash -c 'process_chunk "$@"' _ {}

log "all chunks processed"
heartbeat done
