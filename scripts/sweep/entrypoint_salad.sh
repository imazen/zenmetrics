#!/usr/bin/env bash
# entrypoint_salad.sh — SaladCloud deploy-image entrypoint.
#
# Per SALAD.md § "Deploy image" + the upstream salad-cloud-job-queue-worker
# `with-shell-script` mandelbrot sample: launch BOTH the baked-in sidecar
# (`salad-http-job-queue-worker`) and the app (`zen-sweep-worker worker
# --backend salad`) concurrently, then `wait -n` so the container exits
# when either dies.
#
# Architecture (SALAD.md § "why the app speaks HTTP, not gRPC"):
#   Salad managed queue
#     → salad-http-job-queue-worker (gRPC client to the queue)
#       → POST http://localhost:<SALAD_JOB_PORT><job.path>  (job input)
#         → zen-sweep-worker's local HTTP receiver (SaladJobQueue)
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
for tool in zen-sweep-worker salad-http-job-queue-worker s5cmd; do
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
export RUST_LOG="${RUST_LOG:-info,zen_cloud_salad=info}"

# s5cmd credentials file (the worker shells to s5cmd for R2 ops, matching
# the vast.ai path).
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS

JOB_PORT="${SALAD_JOB_PORT:-80}"
log "worker=${WORKER_ID:-${SALAD_MACHINE_ID:-$(hostname)}} run=${SWEEP_RUN_ID} chunks=${CHUNKS_R2} job_port=${JOB_PORT}"
log "launching salad-http-job-queue-worker sidecar + zen-sweep-worker (--backend salad)"

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
zen-sweep-worker worker --backend salad \
    --run-id "${SWEEP_RUN_ID}" \
    --chunks-r2 "${CHUNKS_R2}" &
WORKER_PID=$!

log "sidecar pid=${SIDECAR_PID} worker pid=${WORKER_PID}; waiting for first exit"

# wait -n returns when EITHER exits; capture its status and propagate.
wait -n
rc=$?
log "a process exited (rc=${rc}); shutting down the other and exiting"
# Best-effort: stop the surviving process so the container exits cleanly.
kill "${SIDECAR_PID}" "${WORKER_PID}" 2>/dev/null || true
exit "${rc}"
