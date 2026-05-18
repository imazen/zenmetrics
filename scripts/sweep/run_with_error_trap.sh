#!/usr/bin/env bash
#
# run_with_error_trap.sh — thin EXIT-trap wrapper for sweep onstart
# scripts. Runs the underlying onstart and, on non-zero exit, uploads
# the captured stderr to R2 then self-destroys the vast.ai instance.
#
# Usage:
#   run_with_error_trap.sh <onstart-script> [args...]
#
# Or via Dockerfile ENTRYPOINT:
#   ENTRYPOINT ["/usr/local/bin/run_with_error_trap.sh",
#               "/usr/local/bin/onstart_cvvdp_picker_corpus.sh"]
#
# Required env vars (passed through by vast.ai or set by onstart):
#   CONTAINER_ID            — vast.ai-set; identifies the instance
#   CONTAINER_API_KEY       — vast.ai-set; container-scoped API key
#   R2_ACCOUNT_ID + creds   — for s5cmd upload
#   SWEEP_RUN_ID            — sweep identifier (used in R2 error path)
#
# Behavior:
#   - rc=0:   exit cleanly. No upload, no destroy. (Watch loop on the
#             controller handles end-of-work destroy at fleet level.)
#   - rc!=0:  upload <tee-log> to s3://zentrain/<run>/errors/<id>.log,
#             then call `vastai-fleet self-destroy`.
#   - SIGTERM during shutdown: same as rc!=0 if the script has done
#             ANY work (avoid destroying an idle box that vast.ai is
#             rebooting for a maintenance reason).
#
# Idempotent + defensive: if any step in the trap fails, the others
# still attempt to run. The box ultimately destroys even if the log
# upload bombs.

set -uo pipefail

# Hydrate env from PID 1 (vast.ai injects credentials into pid-1 env).
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            CONTAINER_*|R2_*|SWEEP_*|WORKER_*|PARALLEL|GPU_RUNTIME|CUDA_PATH)
                export "$k=$v"
                ;;
        esac
    done < /proc/1/environ
fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] [run_with_error_trap] %s\n' "$(ts)" "$*" >&2; }

if [[ $# -lt 1 ]]; then
    log "usage: $0 <onstart-script> [args...]"
    exit 2
fi

ONSTART="$1"; shift

if [[ ! -x "$ONSTART" ]]; then
    log "ERROR: $ONSTART is not executable"
    exit 2
fi

# Capture all stderr to a log file in /tmp. Tee through to the real
# stderr so vast.ai's console view still sees output live.
WORK_DIR="${TMPDIR:-/tmp}/sweep-worker-$$"
mkdir -p "$WORK_DIR"
STDERR_LOG="$WORK_DIR/stderr.log"
exec 3>&2  # save original stderr
exec 2> >(tee -a "$STDERR_LOG" >&3)

log "starting $ONSTART (stderr -> $STDERR_LOG)"
log "instance ID: ${CONTAINER_ID:-<unset>}"
log "run ID:      ${SWEEP_RUN_ID:-<unset>}"

self_destroy_on_error() {
    local rc=$?
    if (( rc == 0 )); then
        log "onstart exited cleanly (rc=0) — no self-destroy"
        return
    fi

    log "onstart exited rc=$rc — running self-destroy"

    # Drain any in-flight tee output to the log.
    sync || true
    sleep 1

    # Refuse to self-destroy if we can't identify the instance. Better
    # to leave the box stuck (visible in fleet dashboard) than to
    # silently no-op the destroy.
    if [[ -z "${CONTAINER_ID:-}" ]]; then
        log "ERROR: CONTAINER_ID unset — cannot self-destroy. Box will keep running."
        return
    fi
    if [[ -z "${CONTAINER_API_KEY:-}" ]]; then
        log "ERROR: CONTAINER_API_KEY unset — cannot self-destroy. Box will keep running."
        return
    fi

    local run_id="${SWEEP_RUN_ID:-unknown-run}"
    local r2_prefix="s3://zentrain/${run_id}/errors/"

    # Annotate the log with exit context BEFORE upload so it's
    # self-contained.
    {
        echo "# === run_with_error_trap exit context ==="
        echo "# exit_code:    $rc"
        echo "# instance_id:  $CONTAINER_ID"
        echo "# run_id:       $run_id"
        echo "# host:         $(hostname)"
        echo "# timestamp:    $(ts)"
        echo "# onstart:      $ONSTART"
        echo "# === end context ==="
    } >> "$STDERR_LOG"

    /usr/local/bin/vastai-fleet self-destroy \
        --error-log "$STDERR_LOG" \
        --r2-prefix "$r2_prefix" 2>&1 || {
        log "WARN: vastai-fleet self-destroy failed (continuing — box may stay alive)"
    }
}

trap self_destroy_on_error EXIT

# Run the underlying onstart and propagate its exit code. We
# deliberately do NOT `exec` here — exec would replace this shell
# process and the EXIT trap would never fire.
"$ONSTART" "$@"
exit_rc=$?
log "onstart returned rc=$exit_rc"
exit "$exit_rc"
