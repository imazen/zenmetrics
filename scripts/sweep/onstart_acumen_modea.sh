#!/usr/bin/env bash
#
# onstart_acumen_modea.sh — vast.ai entrypoint for the Gate A
# castleCSF Mode A zensim-gpu feature-extraction fleet.
#
# Forks onstart_iwssim_backfill_v14.sh, swapping:
#   - METRIC: iwssim → zensim-gpu
#   - sidecar field: out_sidecar_iwssim → out_sidecar_zensim
#   - extra worker env: ACUMEN_MODE_A=1 + viewing-condition vars
#
# Paired with `Dockerfile.sweep.v26` (acumen-aware binary in the
# v26-acumen-* image tag). The v26 image bakes
# `/usr/local/bin/metric_chunk_worker.sh` which forwards the
# ACUMEN_* env vars to `zen-metrics score-pairs`.
#
# Tracking: imazen/zensim#40 Gate A.
#
# Required env vars (passed via vast.ai --env):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#   SWEEP_RUN_ID
# Optional:
#   WORKER_ID                  defaults to $(hostname)-$$
#   PARALLEL                   concurrent chunk workers per box (auto)
#   WORKDIR                    defaults to /workspace/sweep
#   SCRIPTS_R2_PREFIX          override; default s3://coefficient/jobs/<run>
#   GPU_RUNTIME                cuda / wgpu / cpu (default auto)
#   ACUMEN_PPD                 default 56 (lab reference)
#   ACUMEN_PEAK_NITS           default 100 (SDR sRGB)
#   ACUMEN_AMBIENT_NITS        default 5 (dim room)
#   METRIC                     default zensim-gpu (the only metric that
#                              honours --acumen-mode-a today)

set -uo pipefail

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

# Hydrate env from PID 1 (where vast.ai injects worker env vars).
if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL|WORKDIR|SCRIPTS_R2_PREFIX|GPU_RUNTIME|CUDA_PATH|CONTAINER_*|ACUMEN_*|METRIC)
                export "$k=$v"
                ;;
        esac
    done < /proc/1/environ
fi

ONSTART_LOG="${ONSTART_LOG:-/tmp/onstart_acumen.log}"
exec > >(tee -a "$ONSTART_LOG") 2> >(tee -a "$ONSTART_LOG" >&2)

# EXIT trap with self-destroy (matches iwssim v14 pattern; see that
# file for the why this is mandatory).
on_exit() {
    local rc=$?
    if [[ -n "${HEARTBEAT_PID:-}" ]]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    if (( rc == 0 )); then
        log "[on_exit] rc=0 — clean exit, no self-destroy"
        return
    fi
    log "[on_exit] rc=$rc — uploading log + self-destroying"
    sync || true
    sleep 1

    local run_id="${SWEEP_RUN_ID:-unknown-run}"
    local worker_id="${WORKER_ID:-$(hostname)-$$}"
    local container_id="${CONTAINER_ID:-unknown-container}"
    local r2_key="s3://coefficient/jobs/${run_id}/worker-logs/${worker_id}-failure.log"
    local upload_tmp=/tmp/on_exit_upload.log
    {
        echo "# === onstart_acumen exit context ==="
        echo "# exit_code:    $rc"
        echo "# worker_id:    $worker_id"
        echo "# container_id: $container_id"
        echo "# run_id:       $run_id"
        echo "# host:         $(hostname)"
        echo "# timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# === last 200 lines of $ONSTART_LOG ==="
        tail -n 200 "$ONSTART_LOG" 2>/dev/null || echo "(log unavailable)"
    } > "$upload_tmp"
    if [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" \
          && -n "${R2_SECRET_ACCESS_KEY:-}" ]] \
        && command -v s5cmd >/dev/null 2>&1; then
        s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
            --profile r2 cp "$upload_tmp" "$r2_key" >/dev/null 2>&1 \
            && log "[on_exit] log uploaded to $r2_key" \
            || log "[on_exit] WARN: log upload failed"
    fi
    if [[ -z "${CONTAINER_ID:-}" || -z "${CONTAINER_API_KEY:-}" ]]; then
        log "[on_exit] ERROR: CONTAINER_ID/CONTAINER_API_KEY unset — cannot self-destroy"
        return
    fi
    local url="https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/"
    log "[on_exit] DELETE $url"
    curl -fsSL --max-time 30 -X DELETE \
        -H "Authorization: Bearer ${CONTAINER_API_KEY}" \
        -H "Accept: application/json" \
        "$url" >/dev/null 2>&1 \
        && log "[on_exit] destroy DELETE accepted" \
        || log "[on_exit] WARN: destroy DELETE failed"
}
trap on_exit EXIT

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID missing}"

WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
PARALLEL="${PARALLEL:-0}"
[[ "$PARALLEL" == "auto" ]] && PARALLEL=0
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
WORKDIR="${WORKDIR:-/workspace/sweep}"
SCRIPTS_R2_PREFIX="${SCRIPTS_R2_PREFIX:-s3://coefficient/jobs/${SWEEP_RUN_ID}}"
METRIC="${METRIC:-zensim-gpu}"

# Acumen defaults: lab reference. Override via env.
export ACUMEN_MODE_A=1
export ACUMEN_PPD="${ACUMEN_PPD:-56}"
export ACUMEN_PEAK_NITS="${ACUMEN_PEAK_NITS:-100}"
export ACUMEN_AMBIENT_NITS="${ACUMEN_AMBIENT_NITS:-5}"
export METRIC GPU_RUNTIME

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Auto-detect PARALLEL from cgroup (same logic as iwssim v14).
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
    cgroup_ram=$(ram_gb_from_cgroup || echo 16)
    # zen-metrics scoring is GPU-bound; CPU parallelism mainly matters
    # for encoding which doesn't apply on a feature-extraction-only sweep
    # (chunks provide pre-encoded dist images). Cap at 4 — extra parallel
    # workers just contend for the GPU.
    PARALLEL=$(( cgroup_cpu < 4 ? cgroup_cpu : 4 ))
    (( PARALLEL < 1 )) && PARALLEL=1
fi
log "PARALLEL=$PARALLEL GPU_RUNTIME=$GPU_RUNTIME METRIC=$METRIC"
log "acumen viewing: ppd=$ACUMEN_PPD peak=$ACUMEN_PEAK_NITS ambient=$ACUMEN_AMBIENT_NITS"

# Verify the baked binary actually supports --acumen-mode-a. Belt-and-
# suspenders: the CI sanity check rejects images that don't have it,
# but verifying again at boot fails-loud-and-early if someone hand-
# patches the binary.
if ! /usr/local/bin/zen-metrics score-pairs --help 2>&1 | grep -q acumen-mode-a; then
    log "FATAL: baked zen-metrics binary does not support --acumen-mode-a; wrong image"
    exit 7
fi
log "baked zen-metrics --acumen-mode-a verified"

R2() {
    s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        --profile r2 "$@"
}

# Configure s5cmd profile from env so R2() works without ambient ~/.aws.
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS

heartbeat() {
    local phase="$1"
    local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    cat > /tmp/hb.json <<EOF
{"ts":"$ts","worker":"$WORKER_ID","phase":"$phase","gpu_runtime":"$GPU_RUNTIME","acumen":true}
EOF
    R2 cp /tmp/hb.json \
        "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/${WORKER_ID}.${phase}" \
        >/dev/null 2>&1 || true
}
heartbeat boot

log "pulling chunks.jsonl from $SCRIPTS_R2_PREFIX"
R2 cp "${SCRIPTS_R2_PREFIX}/chunks.jsonl" "$WORKDIR/chunks.jsonl" \
    || { log "FAIL pull chunks.jsonl"; exit 4; }
N_CHUNKS=$(wc -l < "$WORKDIR/chunks.jsonl")
log "loaded $N_CHUNKS chunks"

process_chunk() {
    local line="$1"
    local chunk_id out_sidecar
    chunk_id=$(printf '%s' "$line" | jq -r '.chunk_id')
    # Acumen chunks use out_sidecar_zensim_acumen to keep them in a
    # separate R2 prefix from the legacy zensim sidecars. Fall back to
    # out_sidecar_zensim if the generator hasn't been updated yet.
    out_sidecar=$(printf '%s' "$line" | jq -r '.out_sidecar_zensim_acumen // .out_sidecar_zensim')
    if [[ -z "$out_sidecar" || "$out_sidecar" == "null" ]]; then
        log "[skip] $chunk_id has no out_sidecar_zensim_acumen / out_sidecar_zensim field"
        return 0
    fi

    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${chunk_id}.claim"

    # Idempotent skip if sidecar already in R2.
    if R2 ls "$out_sidecar" 2>/dev/null | grep -q "${chunk_id}\.parquet"; then
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

    local LOG="/tmp/acumen-chunk-${chunk_id}.log"
    if printf '%s' "$line" \
        | /usr/local/bin/metric_chunk_worker.sh \
            --metric "$METRIC" \
            --gpu-runtime "$GPU_RUNTIME" \
            --out-sidecar-field "out_sidecar_zensim_acumen" \
            --work-dir "/tmp/acumen-${chunk_id}-$$" \
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

(
    while true; do
        sleep 60
        heartbeat alive 2>/dev/null || true
    done
) &
HEARTBEAT_PID=$!

# Fan out chunks across PARALLEL workers via xargs. Export the
# process_chunk function + env so each xargs-spawned subshell sees
# them.
export -f process_chunk log R2 heartbeat
export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
export SWEEP_RUN_ID WORKER_ID WORKDIR SCRIPTS_R2_PREFIX
export GPU_RUNTIME METRIC ACUMEN_MODE_A ACUMEN_PPD ACUMEN_PEAK_NITS ACUMEN_AMBIENT_NITS

xargs_rc=0
cat "$WORKDIR/chunks.jsonl" | xargs -I {} -P "$PARALLEL" bash -c '
    process_chunk "$@"
' _ {} || xargs_rc=$?

heartbeat done

log "all chunks processed; xargs_rc=$xargs_rc"
exit "$xargs_rc"
