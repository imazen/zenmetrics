#!/usr/bin/env bash
# onstart_unified.sh — v22+ onstart for the unified Rust worker.
#
# The bash dispatcher chain (onstart_omni_backfill.sh +
# omni_backfill_chunk_worker.sh) is replaced by a single
# `vastai-fleet worker` invocation that does everything in one
# process:
#
#   - Claim loop (token-race + sidecar idempotency + stale recovery)
#   - Bounded adaptive concurrency (AIMD on nvidia-smi util)
#   - In-process run_sweep per group (cubecl shared, one init total)
#   - arrow-rs parquet IO (no pyarrow / no python3)
#
# Compatibility: this script consumes the same env vars the old
# bash chain did, so the existing launchers + chunks.jsonl shape
# work unchanged. Defaults match the legacy behaviour.
#
# Run-time env contract:
#
#   SWEEP_RUN_ID         REQUIRED. The chunks.jsonl + sidecar path scope.
#   CHUNKS_R2            Optional. Defaults to
#                        s3://coefficient/jobs/<SWEEP_RUN_ID>/chunks.jsonl.
#   WORKER_ID            Optional. Distinguishes peers; defaults to
#                        hostname.
#   PARALLEL_CHUNKS      Optional. Initial in-flight chunk count;
#                        unset → auto-detect from host specs.
#   METRICS              Optional. Comma-list of metric names.
#                        Default: zensim-gpu,ssim2-gpu,butteraugli-gpu,
#                                 cvvdp,dssim-gpu. iwssim-gpu is
#                                 OFF by default (176-px min fails on
#                                 gif/wikimedia + ~16% pool pressure).
#   SKIP_CLAIMS          Optional. Set to 1 for single-instance smoke.
#   R2_*                 REQUIRED. R2_ACCOUNT_ID + access keys.
#   ADAPT_INTERVAL_SEC   Optional. AIMD sample period; default 60.
#   MAX_CHUNKS_PER_PROCESS Optional. After N chunks the worker exits 0
#                        and the outer loop respawns it (fresh cubecl
#                        pool). Default 20. Set 0 to disable.
#   MAX_RESPAWNS         Optional. Hard ceiling on respawn loop
#                        iterations. Default 200 (4000 chunks at the
#                        default chunk cap of 20). Hit only if the
#                        chunk corpus is huge AND each respawn lands
#                        the cap.
#
# Launcher invocation expectations are unchanged — point any of the
# existing launchers at this onstart instead of onstart_omni_backfill.sh.

set -euo pipefail

# Hydrate env from /proc/1/environ. The Rust worker also does this
# but having it in bash too means we can early-fail with a useful
# message if the box is misconfigured.
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

# Set up s5cmd credentials file (the Rust worker shells to s5cmd
# for R2 ops — phase C will move to native aws-sdk-s3).
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS

# CHUNKS_R2 explicit-or-derived from SWEEP_RUN_ID. CHUNKS_PATH is
# the legacy bash var name; honour it too.
CHUNKS_R2="${CHUNKS_R2:-${CHUNKS_PATH:-s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl}}"

echo "[onstart-unified] worker=${WORKER_ID:-$(hostname)} run=${SWEEP_RUN_ID} chunks=${CHUNKS_R2}" >&2

# Tracing level. The Rust binary respects RUST_LOG; the bash
# default was info, so match.
export RUST_LOG="${RUST_LOG:-info}"

# Respawn loop. The Rust worker has a per-process chunk cap
# (MAX_CHUNKS_PER_PROCESS, default 20) that causes it to exit 0
# after dispatching N chunks. This loop respawns it so the
# cubecl-cuda pool footprint resets to zero. Non-zero exit
# breaks the loop and falls through to the trap wrapper which
# self-destroys the vast.ai instance (billing protection).
#
# MAX_RESPAWNS (default 200) bounds the loop so a runaway respawn
# can't burn a box indefinitely. At 20 chunks/respawn that's 4000
# chunks per box — well above any single-box realistic workload.
MAX_RESPAWNS="${MAX_RESPAWNS:-200}"
for ((i=1; i<=MAX_RESPAWNS; i++)); do
    echo "[onstart-unified] worker process #${i} start" >&2
    /usr/local/bin/vastai-fleet worker \
        --run-id "${SWEEP_RUN_ID}" \
        --chunks-r2 "${CHUNKS_R2}"
    rc=$?
    if (( rc != 0 )); then
        echo "[onstart-unified] worker exited nonzero rc=${rc}; aborting respawn" >&2
        exit "${rc}"
    fi
    echo "[onstart-unified] worker process #${i} exited cleanly (rc=0); respawning" >&2
done

echo "[onstart-unified] hit MAX_RESPAWNS=${MAX_RESPAWNS}; declaring success" >&2
exit 0
