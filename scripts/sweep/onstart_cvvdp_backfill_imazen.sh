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
# PARALLEL=0 → auto-detect. Mirrors cvvdp-gpu's
# `recommend_parallel(free_gpu_bytes, w, h)` heuristic (the Rust
# predictor's `PARALLEL_SAFETY_FACTOR = 1.5` const) but in shell so
# the chunk_worker xargs -P can use it before zen-metrics starts.
# Profiling on a healthy worker (RTX 2060 SUPER, 12 cores, 64 GB)
# showed GPU at 3% time-averaged + 1 of 12 CPU cores in use with
# PARALLEL=1 — massive under-utilization. Auto-detect picks N
# concurrent chunk workers based on free GPU and CPU caps.
if [[ "$PARALLEL" == "0" ]]; then
    # Free GPU memory in MiB (works on every nvidia-smi 470+).
    FREE_GPU_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    NCPU=$(nproc)
    if [[ -z "$FREE_GPU_MIB" || "$FREE_GPU_MIB" -lt 1024 ]]; then
        log "auto-detect: nvidia-smi unavailable / free<1GiB; defaulting PARALLEL=2"
        PARALLEL=2
    else
        # Assume per-instance cost ≈ 250 MiB GPU (cvvdp-gpu's
        # estimate_gpu_memory_bytes returns ~208 MiB at 1024²;
        # rounded up for transient kernel buffers). Apply 1.5×
        # safety factor (matches PARALLEL_SAFETY_FACTOR).
        PER_INST_MIB=375  # 250 × 1.5
        GPU_CAP=$((FREE_GPU_MIB / PER_INST_MIB))
        # CPU cap: leave half the cores for ssim2-gpu sweep work +
        # encode threads + system. Floor at 1.
        CPU_CAP=$((NCPU / 2))
        [[ "$CPU_CAP" -lt 1 ]] && CPU_CAP=1
        # Hard ceiling at 8 — beyond that diminishing returns vs
        # disk I/O / R2 upload contention.
        PARALLEL=$GPU_CAP
        [[ "$PARALLEL" -gt "$CPU_CAP" ]] && PARALLEL=$CPU_CAP
        [[ "$PARALLEL" -gt 8 ]] && PARALLEL=8
        [[ "$PARALLEL" -lt 1 ]] && PARALLEL=1
        log "auto-detect PARALLEL=$PARALLEL (free_gpu=${FREE_GPU_MIB}MiB, nproc=$NCPU, gpu_cap=$GPU_CAP, cpu_cap=$CPU_CAP)"
    fi
fi
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

# ── Step 1a: optional binary override (v15 pattern) ────────────────────
# When the docker-image-baked zen-metrics has the wrong cudarc feature
# set (cuda-13020 dlsym DlSym panic), the operator can replace it by
# setting SWEEP_BIN_OVERRIDE to an R2 (s3://…) or HTTPS URL of a
# locally-built binary. We swap /usr/local/bin/zen-metrics with it.
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
    log "installing python3 + python3-pip via apt (boot image missing one or both)"
    apt-get update -q
    apt-get install -yq --no-install-recommends python3 python3-pip \
        || { log "FAIL apt-get install python3 python3-pip"; exit 3; }
fi
if ! python3 -c "import pyarrow" 2>/dev/null; then
    log "installing pyarrow (apt python3-pyarrow first, pip fallback)"
    # Prefer apt's python3-pyarrow build: it's compiled against the
    # exact glibc on the host image and is more stable than pip's
    # wheels, which segfault on certain vast.ai hosts when reading
    # the unified parquet sidecars (v25 lesson: 17% of boxes hit
    # `Segmentation fault` in `pq.read_table` with the pip wheel).
    apt-get install -yq --no-install-recommends python3-pyarrow 2>/dev/null \
        || pip3 install --quiet --break-system-packages pyarrow 2>/dev/null \
        || pip3 install --quiet pyarrow \
        || { log "FAIL install pyarrow (apt + pip both failed)"; exit 3; }
fi
# Verify pyarrow import succeeds without segfault before claiming
# any chunks. If python segfaults loading pyarrow, the worker is
# unusable for this sweep — fail loudly so the operator can
# destroy + replace rather than burning money on retry loops.
python3 -c "import pyarrow.parquet as pq; print('pyarrow import OK')" \
    || { log "FAIL pyarrow import segfaults/errors on this host"; exit 3; }

# libnvrtc12: cubecl-cuda uses NVRTC at runtime to compile PTX from
# kernel source. nvidia-container-toolkit only mounts libcuda; nvrtc
# is a CUDA *runtime* library and must be installed in the container.
# Ubuntu 24.04 ships libnvrtc12 via NVIDIA's CUDA repo; cuda-nvrtc-12-6
# matches our binary's cudart (compiled against CUDA 12.6 SDK).
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
    # Install cuda-nvrtc-12-6 (matches binary's cuda-12060 cudarc).
    # libcudart-12-6: cudart shared library.
    # cuda-cudart-dev-12-6: CUDA runtime HEADERS — needed because
    # cubecl emits kernels with `#include <cuda_runtime.h>` and NVRTC
    # compiles them at launch time. Without the dev headers, every
    # GPU kernel launch fails with "catastrophic error: cannot open
    # source file 'cuda_runtime.h'" surfaced as `InvalidImageSize`
    # → 100 NaN rows per chunk (v24 lesson).
    apt-get install -yq --no-install-recommends \
        cuda-nvrtc-12-6 cuda-cudart-12-6 cuda-cudart-dev-12-6 \
        >/dev/null \
        || { log "FAIL apt install cuda-nvrtc"; exit 6; }
    # apt installs to /usr/local/cuda-12.6/lib64/. Register with
    # the dynamic linker so libnvrtc.so.12 resolves via the system
    # search path (LD_LIBRARY_PATH alone won't propagate through
    # zen-metrics subprocesses if shell vars don't survive).
    echo "/usr/local/cuda-12.6/lib64" > /etc/ld.so.conf.d/cuda-12.6.conf
    ldconfig
    # Symlink /usr/local/cuda → cuda-12.6 so cubecl-cuda's
    # install::cuda_path() finds the real toolkit instead of the
    # stub directory created above.
    if [[ ! -L /usr/local/cuda || $(readlink /usr/local/cuda) == "cuda-12.6" ]]; then
        rm -rf /usr/local/cuda
        ln -s /usr/local/cuda-12.6 /usr/local/cuda
    fi
    log "libnvrtc12 installed: $(ldconfig -p | grep libnvrtc.so.12 | head -1)"
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

# cubecl-cuda's runtime calls install::cuda_path().expect(...) which
# panics with 'CUDA installation not found' when /usr/local/cuda
# doesn't exist (vast.ai's ubuntu:24.04 base does not include the
# CUDA toolkit — only nvidia-container-toolkit mounts libcuda).
# Setting CUDA_PATH or creating /usr/local/cuda satisfies the
# existence check. NVRTC has bundled headers so a bogus include
# path is harmless — the --include-path=<path>/include flag is
# additive, not the primary header source.
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

# Shuffle chunks so parallel workers on the same box don't all
# claim-race on the same chunk_id at startup.
shuf "$WORKDIR/chunks.jsonl" > "$WORKDIR/chunks.shuf.jsonl"

xargs -I {} -P "$PARALLEL" -d '\n' bash -c 'process_chunk "$@"' _ {} \
    < "$WORKDIR/chunks.shuf.jsonl"

heartbeat done
log "all chunks processed"
