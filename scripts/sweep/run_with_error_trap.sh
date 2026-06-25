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
#               "/usr/local/bin/onstart_unified.sh"]
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
#             then call `zenfleet-vastai self-destroy`.
#   - SIGTERM during shutdown: same as rc!=0 if the script has done
#             ANY work (avoid destroying an idle box that vast.ai is
#             rebooting for a maintenance reason).
#
# Idempotent + defensive: if any step in the trap fails, the others
# still attempt to run. The box ultimately destroys even if the log
# upload bombs.

set -uo pipefail

# Some vast.ai container init paths drop /sbin from PATH. dpkg needs
# /sbin/ldconfig + /sbin/start-stop-daemon at install time, and the
# onstart's `ldconfig -p | grep libnvrtc.so.12` probe needs ldconfig
# itself. Prepend the standard sbin dirs unconditionally.
export PATH="/usr/local/sbin:/usr/sbin:/sbin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

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

# ── GPU pre-flight ─────────────────────────────────────────────────
# We rent GPU boxes. CPU fallback at runtime silently turns a GPU-paid
# box into a slow CPU worker — wastes the user's budget. Refuse to
# start without a visible CUDA device, unless explicitly told otherwise
# via $ALLOW_CPU_FALLBACK=1.
if [[ "${ALLOW_CPU_FALLBACK:-0}" != "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "ERROR: nvidia-smi not found — this image expects a GPU host."
        log "       (set ALLOW_CPU_FALLBACK=1 to override)"
        exit 4
    fi
    if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        log "ERROR: nvidia-smi present but no GPU visible to this container."
        log "       (set ALLOW_CPU_FALLBACK=1 to override)"
        nvidia-smi 2>&1 | sed 's/^/  /' | head -10 >&2
        exit 4
    fi
    # Pin the gpu runtime to cuda so zenmetrics' auto-fallback chain
    # ([Cuda, Wgpu, Hip, Cpu]) can't silently land on CPU when cuda
    # init throws. Downstream scripts read $GPU_RUNTIME.
    if [[ -z "${GPU_RUNTIME:-}" || "$GPU_RUNTIME" == "auto" ]]; then
        export GPU_RUNTIME=cuda
        log "GPU_RUNTIME pinned to 'cuda' (no CPU fallback). Set ALLOW_CPU_FALLBACK=1 to relax."
    fi
    log "GPU pre-flight: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
    log "WARN: ALLOW_CPU_FALLBACK=1 set — runtime will use 'auto' GPU selection (may fall back to CPU)."
fi

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

    /usr/local/bin/zenfleet-vastai self-destroy \
        --error-log "$STDERR_LOG" \
        --r2-prefix "$r2_prefix" 2>&1 || {
        log "WARN: zenfleet-vastai self-destroy failed (continuing — box may stay alive)"
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
