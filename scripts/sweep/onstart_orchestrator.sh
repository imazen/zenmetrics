#!/usr/bin/env bash
# scripts/sweep/onstart_orchestrator.sh
#
# Orchestrator-aware sweep onstart. Pairs with Dockerfile.sweep.v27.
#
# Flow per CLAUDE.md "Sweep runner discipline" + "Remote / Fleet Images":
#
#   1. Hydrate env vars from /proc/1/environ (vast.ai injects them
#      there before exec'ing PID 1).
#   2. (Optional) Fetch a fleet-shared capability_<hash>.toml from R2
#      into $ZENMETRICS_CACHE_DIR. When absent, the orchestrator's
#      `warm()` runs the local bench at startup (~30 s) — same
#      behaviour as a fresh box.
#   3. Drive `zen-metrics sweep --use-orchestrator ...` on the claimed
#      chunk. Identical chunk-claim contract as `onstart_v3.sh`.
#   4. On rc≠0 the EXIT trap (installed via run_with_error_trap.sh)
#      uploads the tail log to R2 + invokes `vastai-fleet self-destroy`
#      so a broken worker doesn't burn billable hours.
#
# Operator override pattern:
#
#   vastai create instance ... \
#       --image ghcr.io/imazen/zen-metrics-sweep:v27 \
#       --entrypoint /usr/local/bin/run_with_error_trap.sh \
#       --args "/usr/local/bin/onstart_orchestrator.sh"
#
# Required env vars (all sourced from /proc/1/environ on vast.ai):
#
#   R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY  — credentials.
#   RUN_ID                                       — sweep run identifier.
#   CHUNKS_R2_PREFIX                             — s3 prefix to chunks.jsonl.
#   RESULTS_R2_PREFIX                            — s3 prefix for upload.
#   CONTAINER_ID                                 — vast.ai instance id
#                                                  (used by self-destroy).
#
# Optional env vars:
#
#   ZENMETRICS_CACHE_R2_KEY                      — when set, fetch the
#       given key into $ZENMETRICS_CACHE_DIR before running. Lets a
#       fleet share a pre-built capability profile across boxes with
#       identical (gpu_model, driver_version, cpu_brand, cpu_features).
#   ZENMETRICS_BENCH_ON_START                    — "auto" (default),
#       "yes", or "no". Forwarded to --bench-on-start.
#   ZENMETRICS_CPU_FEATURES                      — comma list forwarded
#       to --cpu-features (e.g. "ssim2,dssim,zensim" to opt out of
#       cvvdp / butter CPU on a build that ships all five).

set -euo pipefail
shopt -s extglob

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

# 1. Hydrate env from PID 1. vast.ai injects credentials this way.
log "hydrating env from /proc/1/environ"
if [[ -r /proc/1/environ ]]; then
    while IFS= read -r -d '' kv; do
        # Only forward vars matching our expected prefixes — avoid
        # leaking unrelated PID-1 env into the worker shell.
        case "$kv" in
            R2_*|RUN_ID=*|CHUNKS_R2_PREFIX=*|RESULTS_R2_PREFIX=*|\
            CONTAINER_ID=*|ZENMETRICS_*|AWS_*|PYTHONUNBUFFERED=*)
                export "$kv"
                ;;
        esac
    done < /proc/1/environ
fi

# 2. Verify baked tools are present (per CLAUDE.md "Remote/Fleet Images"
# — image is broken if any of these are missing; fail loud).
log "verifying baked tools"
for cmd in zen-metrics s5cmd jq python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log "FATAL: required baked binary '$cmd' not on PATH; image is broken"
        exit 99
    fi
done
python3 -c 'import pyarrow.parquet' >/dev/null 2>&1 || {
    log "FATAL: pyarrow.parquet import failed; image is broken"
    exit 99
}
ldconfig -p | grep -q libnvrtc.so.12 || {
    log "FATAL: libnvrtc.so.12 not in linker cache; image is broken"
    exit 99
}

# 3. Cache directory. Honour an R2-shared profile if requested.
: "${ZENMETRICS_CACHE_DIR:=/root/.cache/zenmetrics}"
mkdir -p "$ZENMETRICS_CACHE_DIR"

if [[ -n "${ZENMETRICS_CACHE_R2_KEY:-}" ]]; then
    log "fetching shared capability profile from R2: ${ZENMETRICS_CACHE_R2_KEY}"
    # `|| true` — a missing or unreadable profile is non-fatal; the
    # orchestrator's warm() will rebuild from local detection.
    s5cmd cp "${ZENMETRICS_CACHE_R2_KEY}" "${ZENMETRICS_CACHE_DIR}/" \
        >&2 2>&1 || log "shared profile fetch failed; falling through to local warm()"
fi

# 4. Drive the sweep. The chunk-claim contract is the SAME as
# onstart_v3.sh — workers pick up chunks.jsonl, atomic-claim a chunk,
# run zen-metrics sweep, upload results. We only swap the per-cell
# scoring driver from "legacy direct dispatch" to the orchestrator.
#
# Real onstart wrappers (onstart_unified.sh, onstart_v3.sh, etc.) live
# in this directory and already implement the full chunk-claim ←→
# results-upload loop. We delegate to onstart_unified.sh and just
# forward the orchestrator flags through the env so the
# zen-metrics-cli library calls inside zen-sweep-worker pick them up.
log "configuring orchestrator env for downstream worker"
export ZENMETRICS_USE_ORCHESTRATOR=1
export ZENMETRICS_BENCH_ON_START="${ZENMETRICS_BENCH_ON_START:-auto}"
export ZENMETRICS_CPU_FEATURES="${ZENMETRICS_CPU_FEATURES:-all}"
log "  ZENMETRICS_USE_ORCHESTRATOR=$ZENMETRICS_USE_ORCHESTRATOR"
log "  ZENMETRICS_CACHE_DIR=$ZENMETRICS_CACHE_DIR"
log "  ZENMETRICS_BENCH_ON_START=$ZENMETRICS_BENCH_ON_START"
log "  ZENMETRICS_CPU_FEATURES=$ZENMETRICS_CPU_FEATURES"

# Smoke: the binary recognises --use-orchestrator. If this fails the
# image was built without the orchestrator feature — surface a clear
# error before burning chunk time.
if ! zen-metrics --help 2>&1 | grep -q -- '--use-orchestrator'; then
    log "FATAL: zen-metrics binary does not expose --use-orchestrator"
    log "       (image was built without orchestrator feature; rebuild)"
    exit 98
fi

# Print the orchestrator's resolved capability profile so the worker
# log captures which machine fingerprint scored this chunk. The
# `--use-orchestrator` flag plus `score-pairs` with a tiny synthetic
# input would also work, but a no-op list-metrics keeps the smoke fast.
log "orchestrator preflight (list-metrics, capability cache will be written/loaded)"
zen-metrics --use-orchestrator \
    --orchestrator-cache "$ZENMETRICS_CACHE_DIR" \
    --bench-on-start "$ZENMETRICS_BENCH_ON_START" \
    --cpu-features "$ZENMETRICS_CPU_FEATURES" \
    list-metrics 2>&1 | head -20 || {
    log "WARN: orchestrator preflight returned non-zero; continuing anyway"
}

# Delegate to the existing unified-worker entrypoint. Its chunk-claim
# loop calls `zen-metrics sweep` for every chunk; we've exported
# ZENMETRICS_USE_ORCHESTRATOR=1 so each invocation routes through the
# orchestrator path.
log "delegating to /usr/local/bin/onstart_unified.sh"
exec /usr/local/bin/onstart_unified.sh "$@"
