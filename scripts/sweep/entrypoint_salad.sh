#!/usr/bin/env bash
# entrypoint_salad.sh — SaladCloud deploy-image entrypoint.
#
# Per SALAD.md § "Deploy image" + the upstream salad-cloud-job-queue-worker
# `with-shell-script` mandelbrot sample: launch BOTH the baked-in sidecar
# (`salad-http-job-queue-worker`) and the app (`zenfleet-sweep worker
# --backend salad`) concurrently, then `wait -n` so the container exits
# when either dies.
#
# Architecture (SALAD.md § "why the app speaks HTTP, not gRPC"):
#   Salad managed queue
#     → salad-http-job-queue-worker (gRPC client to the queue)
#       → POST http://localhost:<SALAD_JOB_PORT><job.path>  (job input)
#         → zenfleet-sweep's local HTTP receiver (SaladJobQueue)
#       ← HTTP response body  (job output, returned to the queue)
#
# BAKE-EVERYTHING (zensim CLAUDE.md): this script runs the JOB, never an
# installer. NOTHING is apt/pip/curl-installed at boot. The sidecar, both
# binaries, and every runtime dep are baked into the image at build time.
# If a baked tool is missing, we FAIL LOUD and exit nonzero so the
# operator rebuilds the image — we do NOT silently install anything.
#
# Run-time env contract (set in the Salad container-group env):
#
#   SWEEP_RUN_ID    REQUIRED. Scopes the R2 run namespace. The worker reads
#                   it via clap env=SWEEP_RUN_ID; the queue jobs the
#                   sidecar forwards carry the per-chunk payload.
#   CHUNKS_R2       Optional. R2 URI of the chunks manifest; defaults to
#                   s3://coefficient/jobs/<SWEEP_RUN_ID>/chunks.jsonl.
#   SALAD_JOB_PORT  Optional but RECOMMENDED. The container group's
#                   queue_connection.port — the port the sidecar POSTs to
#                   and the worker's HTTP receiver binds. Worker defaults
#                   to :80 when unset (matches the upstream sample).
#   WORKER_ID       Optional. Distinguishes peers; defaults to hostname /
#                   SALAD_MACHINE_ID.
#   R2_* / AWS_*    REQUIRED. BYO object-store credentials (R2_ACCOUNT_ID +
#                   R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY). The worker's
#                   SaladEnvCredentials reads the container-group env.
#                   AWS_SESSION_TOKEN (or R2_SESSION_TOKEN) is REQUIRED when
#                   the launcher injected a SCOPED/temporary R2 cred — it is
#                   written to ~/.aws/credentials as aws_session_token (a
#                   temp key+secret without it 403s). Absent for permanent /
#                   root-key use (back-compatible).
#   SALAD_LOG_LEVEL Optional. Sidecar log verbosity (debug/info/warn/error;
#                   default error). RUST_LOG controls the worker.
#
# Salad injects SALAD_MACHINE_ID / SALAD_CONTAINER_GROUP_ID into the
# container env and provides IMDS at the link-local address; the sidecar
# auto-discovers the queue endpoint via IMDS (no config needed on a Salad
# node).

set -euo pipefail

log() { echo "[entrypoint-salad] $*" >&2; }

# ── Hydrate env from /proc/1/environ ────────────────────────────────────
# Salad injects credentials/ids into the container's pid-1 environment.
# A non-pid-1 child does not always inherit them, so copy the relevant
# ones out (same pattern as the vast.ai onstart scripts). The worker also
# does its own env reads, but hydrating here lets us fail loud early with
# a useful message if the box is misconfigured.
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|AWS_*|S5CMD_*|SWEEP_*|WORKER_*|SALAD_*|CHUNKS_*|PARALLEL*|METRICS|RUST_LOG)
                export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi

# ── Fail loud if any baked tool is missing ──────────────────────────────
# A missing tool means the image is broken. Do NOT install at boot — exit
# nonzero so the operator rebuilds (BAKE-EVERYTHING). nvidia-container-
# toolkit mounts libcuda.so.1; libnvrtc.so.12 is baked (L4).
missing=0
for tool in zenfleet-sweep salad-http-job-queue-worker s5cmd; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log "FATAL: baked tool '$tool' not found on PATH — image is broken, rebuild it"
        missing=1
    fi
done
if ! ldconfig -p 2>/dev/null | grep -q libnvrtc.so.12; then
    log "FATAL: libnvrtc.so.12 not found via ldconfig — CUDA NVRTC layer missing, rebuild the image"
    missing=1
fi
if [[ -e /usr/local/lib/cuda_dlsym_stub.so ]]; then
    log "cuda_dlsym_stub.so present (LD_PRELOAD=${LD_PRELOAD:-unset})"
else
    log "FATAL: cuda_dlsym_stub.so missing — rebuild the image"
    missing=1
fi
if (( missing )); then
    log "aborting: one or more baked tools missing; this is a broken image, not a boot-time-installable condition"
    exit 1
fi

# ── Required runtime env ────────────────────────────────────────────────
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID env missing (set it in the Salad container-group env)}"
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID env missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID env missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY env missing}"

# CHUNKS_R2 explicit-or-derived from SWEEP_RUN_ID.
export CHUNKS_R2="${CHUNKS_R2:-s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl}"
export RUST_LOG="${RUST_LOG:-info,zenfleet_salad=info}"

# s5cmd credentials file (the worker shells to s5cmd for R2 ops, matching
# the vast.ai path).
#
# Scoped/temporary R2 creds (minted per-sweep by the launcher) carry a
# session token that the S3 client MUST send — key+secret ALONE 403.
# When AWS_SESSION_TOKEN (or R2_SESSION_TOKEN) is present, write
# `aws_session_token` into the profile. When absent, behave exactly as
# before (permanent-token / root-key use stays back-compatible).
SESSION_TOKEN="${AWS_SESSION_TOKEN:-${R2_SESSION_TOKEN:-}}"
mkdir -p ~/.aws
{
    echo "[r2]"
    echo "aws_access_key_id = ${R2_ACCESS_KEY_ID}"
    echo "aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}"
    if [[ -n "${SESSION_TOKEN}" ]]; then
        echo "aws_session_token = ${SESSION_TOKEN}"
    fi
} > ~/.aws/credentials

JOB_PORT="${SALAD_JOB_PORT:-80}"
log "worker=${WORKER_ID:-${SALAD_MACHINE_ID:-$(hostname)}} run=${SWEEP_RUN_ID} chunks=${CHUNKS_R2} job_port=${JOB_PORT}"

# ── Kernel-cache warmup (NEW in :v4-kernel-cache) ────────────────────────
# Pre-compile every GPU metric's cubecl kernels BEFORE the sidecar starts
# POSTing jobs, so the first real job doesn't pay the ~10-90s NVRTC compile
# burst. The warmup script:
#   1. Logs the GPU arch (nvidia-smi --query-gpu name,compute_cap).
#   2. Runs `zenmetrics score-pairs --metric <X>` for each default
#      GPU metric on 64x64 (256x256 for iwssim-gpu) fixtures.
#   3. cubecl-cuda persists compiled PTX to /var/cache/cubecl (configured
#      via the baked cubecl.toml in WORKDIR).
#   4. Reports per-metric wall time + total + cache-dir size.
# Fail-soft: a warmup failure does NOT abort the boot. If the GPU is
# genuinely broken the real job's durable error-sidecar will surface it.
#
# WORKDIR must be set so cubecl picks up the cubecl.toml in /workspace/
# salad-sweep/. The image's WORKDIR directive is set; verify in case
# something stripped it.
cd "${WORKDIR:-/workspace/salad-sweep}" 2>/dev/null || true
if [[ -f cubecl.toml ]]; then
    log "cubecl.toml present in $(pwd); PTX cache enabled at /var/cache/cubecl"
else
    log "WARN: cubecl.toml not in cwd ($(pwd)); PTX cache may be disabled"
fi
# Per :v6-visibility iter1: write a boot-info file the worker reads + uploads
# to R2 under the run's scoped prefix. Workers self-report their GPU class +
# warmup duration so the launcher / operator can attribute throughput.
BOOT_INFO_FILE="${BOOT_INFO_FILE:-/var/run/zen-boot.txt}"
mkdir -p "$(dirname "${BOOT_INFO_FILE}")" 2>/dev/null || true

# Capture GPU info BEFORE warmup so a warmup OOM/crash still leaves the gpu
# class in the boot record.
gpu_info_csv=""
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_info_csv=$(nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '\r')
fi
gpu_name=$(echo "${gpu_info_csv}" | awk -F', ' '{print $1}')
gpu_uuid=$(echo "${gpu_info_csv}" | awk -F', ' '{print $2}')
gpu_driver=$(echo "${gpu_info_csv}" | awk -F', ' '{print $3}')
gpu_vram=$(echo "${gpu_info_csv}" | awk -F', ' '{print $4}')

warmup_t0=$(date +%s)
if command -v warmup_kernels.sh >/dev/null 2>&1; then
    log "starting kernel-cache warmup pass…"
    warmup_kernels.sh || log "warmup script returned non-zero (continuing per fail-soft policy)"
else
    log "WARN: warmup_kernels.sh not on PATH; skipping warmup"
fi
warmup_elapsed=$(( $(date +%s) - warmup_t0 ))
log "warmup phase: ${warmup_elapsed}s total wall (per-metric breakdown in [warmup] log lines above)"

# Boot record — plain key:value, the worker reads + uploads to
# <scoped-prefix>/boot/<machine_id>.txt. machine_id resolution mirrors
# the worker (HOSTNAME → SALAD_MACHINE_ID → "unknown"); we record both
# so the upload key matches whichever the worker chose.
{
    echo "machine_id: ${SALAD_MACHINE_ID:-${HOSTNAME:-unknown}}"
    echo "hostname: ${HOSTNAME:-unknown}"
    echo "salad_machine_id: ${SALAD_MACHINE_ID:-}"
    echo "salad_container_group_id: ${SALAD_CONTAINER_GROUP_ID:-}"
    echo "gpu_class: ${gpu_name:-unknown}"
    echo "gpu_uuid: ${gpu_uuid:-}"
    echo "driver: ${gpu_driver:-}"
    echo "vram_mib: ${gpu_vram:-}"
    echo "warmup_seconds: ${warmup_elapsed}"
    echo "warmup_begin_unix: ${warmup_t0}"
    echo "warmup_end_unix: $(date +%s)"
    echo "boot_unix_ts: $(date +%s)"
    echo "image_tag: ${IMAGE_TAG:-unknown}"
    echo "run_id: ${SWEEP_RUN_ID:-unknown}"
} > "${BOOT_INFO_FILE}" 2>/dev/null || log "WARN: could not write boot info to ${BOOT_INFO_FILE}"
log "boot info written to ${BOOT_INFO_FILE} (gpu_class=${gpu_name:-unknown} warmup=${warmup_elapsed}s)"
# Also export for the worker process to pick up without re-reading the file.
export ZEN_BOOT_INFO_FILE="${BOOT_INFO_FILE}"
export ZEN_BOOT_GPU_CLASS="${gpu_name:-unknown}"
export ZEN_BOOT_WARMUP_SECONDS="${warmup_elapsed}"

log "launching salad-http-job-queue-worker sidecar + zenfleet-sweep (--backend salad)"

# ── Launch both concurrently (upstream with-shell-script pattern) ───────
# The sidecar is the gRPC client to Salad's queue; the worker serves the
# local HTTP receiver the sidecar POSTs to. Run both in the background and
# wait for whichever exits first; propagate its exit code so Salad sees a
# crashed instance and re-schedules.
salad-http-job-queue-worker &
SIDECAR_PID=$!

# The worker reads SALAD_JOB_PORT itself (defaults to :80). run_id +
# chunks_r2 come from SWEEP_RUN_ID / CHUNKS_R2 (clap env=). The chunk
# payloads arrive via the sidecar POST, not chunks.jsonl, but the worker
# still needs a valid run-id + R2 scope for artifact upload.
zenfleet-sweep worker --backend salad \
    --run-id "${SWEEP_RUN_ID}" \
    --chunks-r2 "${CHUNKS_R2}" &
WORKER_PID=$!

log "sidecar pid=${SIDECAR_PID} worker pid=${WORKER_PID}; waiting for first exit"

# wait -n returns when EITHER exits; capture its status and propagate.
# `|| rc=$?` keeps `set -e` from aborting the script before we run the
# cleanup + diagnostic below when the first-exiting process returns
# nonzero (the common crash case).
rc=0
wait -n || rc=$?
log "a process exited (rc=${rc}); shutting down the other and exiting"
# Best-effort: stop the surviving process so the container exits cleanly.
kill "${SIDECAR_PID}" "${WORKER_PID}" 2>/dev/null || true
exit "${rc}"
