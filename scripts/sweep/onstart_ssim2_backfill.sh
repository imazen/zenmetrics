#!/usr/bin/env bash
#
# onstart_ssim2_backfill.sh — vast.ai instance entry point for the
# ssim2-backfill fleet (V_24-mix-with-ssim2 retrain dependency).
#
# Adapted from onstart_iwssim_backfill.sh — single-image flow,
# zen-metrics-sweep:0.6.4-ssim2-<sha> is the boot image. Per-chunk
# work is delegated to ssim2_backfill_chunk_worker.sh (uploaded to
# the same S3 prefix as chunks.jsonl).
#
# Required env vars (passed via vast.ai --env):
#   R2_ACCOUNT_ID
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID                 e.g. ssim2-backfill-2026-05-18
# Optional:
#   WORKER_ID                    defaults to $(hostname)-$$
#   PARALLEL                     concurrent chunk workers per box (default auto)
#   WORKDIR                      defaults to /workspace/ssim2-backfill
#   SCRIPTS_R2_PREFIX            override; default uses SWEEP_RUN_ID
#   GPU_RUNTIME                  ssim2-gpu backend (auto/cuda/wgpu/cpu); default auto

set -uo pipefail

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

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
if [[ "$PARALLEL" == "0" ]]; then
    cores_from_cgroup() {
        if [[ -r /sys/fs/cgroup/cpu.max ]]; then
            read -r q p < /sys/fs/cgroup/cpu.max
            [[ "$q" == "max" || -z "$q" ]] && return 1
            echo $(( (q + p / 2) / p ))
            return 0
        fi
        if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
            local q p
            q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
            p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
            (( q > 0 && p > 0 )) && {
                echo $(( (q + p / 2) / p ))
                return 0
            }
        fi
        return 1
    }
    ram_gb_from_cgroup() {
        if [[ -r /sys/fs/cgroup/memory.max ]]; then
            local m
            m=$(cat /sys/fs/cgroup/memory.max)
            [[ "$m" == "max" ]] && return 1
            echo $(( m / 1024 / 1024 / 1024 ))
            return 0
        fi
        return 1
    }
    nc=$(cores_from_cgroup) || nc=$(nproc 2>/dev/null || echo 8)
    cpu_cap=$(( nc > 6 ? nc - 2 : (nc > 2 ? nc - 1 : 2) ))
    ram_cap="$cpu_cap"
    if rg=$(ram_gb_from_cgroup); then
        ram_cap=$(( rg * 2 / 3 ))
    fi
    # SSIMULACRA2 GPU footprint is similar to IW-SSIM (~50 MiB at 1024²).
    # Use the same gpu_cap heuristic as iwssim — 1 instance per ~256 MiB free
    # GPU mem, including intermediate XYB / multiscale pyramid buffers.
    gpu_cap=$cpu_cap
    free_gpu_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$free_gpu_mib" && "$free_gpu_mib" -gt 512 ]]; then
        gpu_cap=$(( free_gpu_mib / 256 ))
    fi
    PARALLEL=$cpu_cap
    [[ "$ram_cap" -lt "$PARALLEL" ]] && PARALLEL=$ram_cap
    [[ "$gpu_cap" -lt "$PARALLEL" ]] && PARALLEL=$gpu_cap
    [[ "$PARALLEL" -lt 1 ]] && PARALLEL=1
    [[ "$PARALLEL" -gt 8 ]] && PARALLEL=8
    log "auto-detect PARALLEL=$PARALLEL (cgroup_cpu=$nc → cpu_cap=$cpu_cap, ram_cap=$ram_cap, gpu_cap=$gpu_cap, free_gpu=${free_gpu_mib:-?}MiB)"
fi
WORKDIR="${WORKDIR:-/workspace/ssim2-backfill}"
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

mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
EOF

log "checking tools: zen-metrics s5cmd jq"
for tool in zen-metrics s5cmd jq; do
    if ! command -v "$tool" >/dev/null; then
        log "FAIL: $tool not on PATH; wrong boot image?"
        exit 2
    fi
done

if [[ -n "${SWEEP_BIN_OVERRIDE:-}" ]]; then
    log "fetching zen-metrics override from $SWEEP_BIN_OVERRIDE"
    if [[ "$SWEEP_BIN_OVERRIDE" == s3://* ]]; then
        R2 cp "$SWEEP_BIN_OVERRIDE" /tmp/zen-metrics.override \
            || { log "FAIL fetch SWEEP_BIN_OVERRIDE"; exit 5; }
    else
        curl -fsSL "$SWEEP_BIN_OVERRIDE" -o /tmp/zen-metrics.override \
            || { log "FAIL fetch SWEEP_BIN_OVERRIDE"; exit 5; }
    fi
    cp /tmp/zen-metrics.override /usr/local/bin/zen-metrics
    chmod +x /usr/local/bin/zen-metrics
    rm /tmp/zen-metrics.override
    log "zen-metrics override installed; version: $(/usr/local/bin/zen-metrics --version 2>&1 | head -1)"
fi
if ! command -v python3 >/dev/null || ! command -v pip3 >/dev/null; then
    log "installing python3 + python3-pip via apt"
    apt-get update -q
    apt-get install -yq --no-install-recommends python3 python3-pip \
        || { log "FAIL apt-get install python3 python3-pip"; exit 3; }
fi
if ! python3 -c "import pyarrow" 2>/dev/null; then
    log "installing pyarrow (apt python3-pyarrow first, pip fallback)"
    apt-get install -yq --no-install-recommends python3-pyarrow 2>/dev/null \
        || pip3 install --quiet --break-system-packages pyarrow 2>/dev/null \
        || pip3 install --quiet pyarrow \
        || { log "FAIL install pyarrow (apt + pip both failed)"; exit 3; }
fi
python3 -c "import pyarrow.parquet as pq; print('pyarrow import OK')" \
    || { log "FAIL pyarrow import segfaults/errors on this host"; exit 3; }

# libnvrtc12: same dance as cvvdp/iwssim — cubecl-cuda needs runtime NVRTC.
if ! ldconfig -p | grep -q libnvrtc.so.12; then
    log "installing libnvrtc12 (NVRTC runtime for cubecl-cuda kernel compilation)"
    if ! command -v gpg >/dev/null; then
        apt-get update -q && apt-get install -yq --no-install-recommends \
            gnupg ca-certificates >/dev/null \
            || { log "FAIL apt install gnupg"; exit 6; }
    fi
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -o /tmp/cuda-keyring.deb \
        || { log "FAIL fetch cuda-keyring"; exit 6; }
    dpkg -i /tmp/cuda-keyring.deb \
        || { log "FAIL dpkg cuda-keyring"; exit 6; }
    rm /tmp/cuda-keyring.deb
    apt-get update -q
    apt-get install -yq --no-install-recommends \
        cuda-nvrtc-12-6 cuda-cudart-12-6 cuda-cudart-dev-12-6 \
        >/dev/null \
        || { log "FAIL apt install cuda-nvrtc"; exit 6; }
    echo "/usr/local/cuda-12.6/lib64" > /etc/ld.so.conf.d/cuda-12.6.conf
    ldconfig
    if [[ ! -L /usr/local/cuda || $(readlink /usr/local/cuda) == "cuda-12.6" ]]; then
        rm -rf /usr/local/cuda
        ln -s /usr/local/cuda-12.6 /usr/local/cuda
    fi
    log "libnvrtc12 installed: $(ldconfig -p | grep libnvrtc.so.12 | head -1)"
fi

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

log "pulling ssim2_backfill_chunk_worker.sh + chunks.jsonl from $SCRIPTS_R2_PREFIX"
R2 cp "${SCRIPTS_R2_PREFIX}/ssim2_backfill_chunk_worker.sh" "$WORKDIR/chunk_worker.sh" \
    || { log "FAIL pull chunk_worker.sh"; exit 4; }
chmod +x "$WORKDIR/chunk_worker.sh"
R2 cp "${SCRIPTS_R2_PREFIX}/chunks.jsonl" "$WORKDIR/chunks.jsonl" \
    || { log "FAIL pull chunks.jsonl"; exit 4; }
N_CHUNKS=$(wc -l < "$WORKDIR/chunks.jsonl")
log "loaded $N_CHUNKS chunks"

process_chunk() {
    local line="$1"
    local chunk_id out_ssim2

    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    out_ssim2=$(printf '%s' "$line" | jq -r '.out_sidecar_ssim2')

    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Idempotent skip: if the ssim2 sidecar already exists, we're done.
    if R2 ls "$out_ssim2" 2>/dev/null | grep -q "${chunk_id}\.parquet"; then
        log "[skip] $chunk_id already complete"
        return 0
    fi

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
    GPU_RUNTIME="$GPU_RUNTIME" \
    "$WORKDIR/chunk_worker.sh" \
        --chunk-json "$line" \
        --work-dir "$WORKDIR/work-${chunk_id}" \
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

log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

if [[ ! -d /usr/local/cuda ]]; then
    log "creating stub /usr/local/cuda to satisfy cubecl-cuda runtime check"
    mkdir -p /usr/local/cuda/include /usr/local/cuda/include/cccl
fi
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
log "CUDA_PATH=${CUDA_PATH}"

heartbeat run

export -f process_chunk log R2
export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY \
    SWEEP_RUN_ID WORKER_ID WORKDIR GPU_RUNTIME

shuf "$WORKDIR/chunks.jsonl" > "$WORKDIR/chunks.shuf.jsonl"

xargs -I {} -P "$PARALLEL" -d '\n' bash -c 'process_chunk "$@"' _ {} \
    < "$WORKDIR/chunks.shuf.jsonl"

heartbeat done
log "all chunks processed"
