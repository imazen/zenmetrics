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
#   3. Drive `zenmetrics sweep --use-orchestrator ...` on the claimed
#      chunk. Identical chunk-claim contract as `onstart_unified.sh`.
#   4. On rc≠0 the EXIT trap (installed via run_with_error_trap.sh)
#      uploads the tail log to R2 + invokes `zenfleet-vastai self-destroy`
#      so a broken worker doesn't burn billable hours.
#
# Operator override pattern:
#
#   vastai create instance ... \
#       --image ghcr.io/imazen/zenmetrics-sweep:v27 \
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
for cmd in zenmetrics s5cmd jq python3; do
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
# onstart_unified.sh — workers pick up chunks.jsonl, atomic-claim a chunk,
# run zenmetrics sweep, upload results. We only swap the per-cell
# scoring driver from "legacy direct dispatch" to the orchestrator.
#
# The real onstart wrapper (onstart_unified.sh) lives in this directory
# and already implements the full chunk-claim ←→ results-upload loop.
# We delegate to onstart_unified.sh and just
# forward the orchestrator flags through the env so the
# zenmetrics-cli library calls inside zenfleet-sweep pick them up.
log "configuring orchestrator env for downstream worker"
# Phase 7.7.1 (2026-05-27): the orchestrator is now the CLI default;
# `ZENMETRICS_USE_ORCHESTRATOR=1` is no longer required (and is a
# deprecated no-op the binary will warn about). Set
# `ZENMETRICS_USE_LEGACY_SCHEDULER=1` to opt OUT — needed only when
# this worker is intentionally exercising the legacy direct-dispatch
# path for a parity comparison.
export ZENMETRICS_BENCH_ON_START="${ZENMETRICS_BENCH_ON_START:-auto}"
export ZENMETRICS_CPU_FEATURES="${ZENMETRICS_CPU_FEATURES:-all}"
log "  ZENMETRICS_CACHE_DIR=$ZENMETRICS_CACHE_DIR"
log "  ZENMETRICS_BENCH_ON_START=$ZENMETRICS_BENCH_ON_START"
log "  ZENMETRICS_CPU_FEATURES=$ZENMETRICS_CPU_FEATURES"
log "  orchestrator: default (set ZENMETRICS_USE_LEGACY_SCHEDULER=1 to opt out)"

# Smoke: the binary recognises the orchestrator flag surface. We check
# `--use-legacy-scheduler` (the new opt-OUT flag) so this works on
# Phase 7.7.1+ binaries; older binaries that predate the flip still
# expose `--use-orchestrator` and we accept either.
if ! zenmetrics --help 2>&1 | grep -q -E -- '--use-(orchestrator|legacy-scheduler)'; then
    log "FATAL: zenmetrics binary does not expose orchestrator flags"
    log "       (image was built without orchestrator feature; rebuild)"
    exit 98
fi

# Print the orchestrator's resolved capability profile so the worker
# log captures which machine fingerprint scored this chunk. A no-op
# list-metrics call exercises the orchestrator's startup path
# (capability detection + cache load) without burning chunk time.
log "orchestrator preflight (list-metrics, capability cache will be written/loaded)"
zenmetrics \
    --orchestrator-cache "$ZENMETRICS_CACHE_DIR" \
    --bench-on-start "$ZENMETRICS_BENCH_ON_START" \
    --cpu-features "$ZENMETRICS_CPU_FEATURES" \
    list-metrics 2>&1 | head -20 || {
    log "WARN: orchestrator preflight returned non-zero; continuing anyway"
}

# Delegate to the existing unified-worker entrypoint. Its chunk-claim
# loop calls `zenmetrics sweep` for every chunk; with the Phase
# 7.7.1 default flip, every invocation routes through the
# orchestrator without needing an env-var opt-in.
log "delegating to /usr/local/bin/onstart_unified.sh"
exec /usr/local/bin/onstart_unified.sh "$@"
