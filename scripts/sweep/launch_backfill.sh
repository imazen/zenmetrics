#!/usr/bin/env bash
#
# launch_backfill.sh — unified vast.ai fleet launcher for metric backfill
# sweeps. Replaces the per-metric launch.sh / launch_imazen.sh files
# under iwssim_backfill/ / ssim2_backfill/ / cvvdp_backfill/.
#
# Drives the same per-instance create loop as the originals but folds
# all the parameter knobs behind one flag interface, and calls into
# zenfleet-vastai for the destroy half of the workflow (no more bash+python
# heredoc destroyers).
#
# Required tools on PATH:
#   - vastai (1.0.8 or newer)
#   - s5cmd
#   - gh (for ghcr.io token)
#   - python3 (only for parsing `vastai create instance --raw` output)
#   - zenfleet-vastai (operator CLI built from crates/zenfleet-vastai —
#     `cargo build --release -p zenfleet-vastai && cp
#     target/release/zenfleet-vastai ~/.local/bin/`)
#
# Required env vars (sourced from ~/.config/cloudflare/r2-credentials):
#   R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
#
# Flag-style invocation:
#
#   launch_backfill.sh \
#       --metric iwssim-gpu \
#       --run-id iwssim-backfill-2026-05-17 \
#       --chunks s3://coefficient/jobs/iwssim-backfill-2026-05-17/chunks.jsonl \
#       --max-dph 0.30 --n-boxes 30 --min-ram 8 --min-disk 20 \
#       --docker ghcr.io/imazen/zenmetrics-sweep:0.6.4-iwssim-fixed-6227c1a \
#       --onstart scripts/sweep/onstart_unified.sh
#
# Once the fleet is up the launcher prints the watch invocation that
# would auto-destroy at target — copy/paste to run as a detached
# background process (or invoke with --watch to run inline).
#
# All flags also accept env-var forms (METRIC, RUN_ID, CHUNKS, ...).
#
# Env-only knobs (additive, 2026-07-13 HDR-pairs fleet):
#   WORKER_PATH               chunk worker script to stage alongside onstart
#                             (default scripts/sweep/metric_backfill_chunk_worker.sh)
#   WORKER_R2_ACCESS_KEY_ID / WORKER_R2_SECRET_ACCESS_KEY / WORKER_R2_SESSION_TOKEN
#                             SCOPED temp creds to inject into the boxes instead
#                             of the sourced account creds (mint via the CF
#                             temp-access-credentials API; the session token is
#                             split into R2_SESSION_TOKEN_0..N to fit vast's
#                             256-char env value cap). Without these the
#                             launcher warns loudly and ships the sourced key.
#   SHARD_N                   opt-in static modulo sharding: inject
#                             -e SHARD_IDX=<k> -e SHARD_N=<n> per box, where k
#                             counts SUCCESSFUL launches (failed creates don't
#                             consume a shard). Orphaned shards (launch
#                             shortfall) are reported at the end.

set -euo pipefail
# shellcheck disable=SC1091
source ~/.config/cloudflare/r2-credentials

METRIC="${METRIC:-}"
RUN_ID="${RUN_ID:-}"
CHUNKS="${CHUNKS:-}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-${DOCKER:-}}"
ONSTART_PATH="${ONSTART_PATH:-${ONSTART:-}}"
N_BOXES="${N_BOXES:-30}"
MAX_DPH="${MAX_DPH:-0.30}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-8}"
MIN_DISK_GB="${MIN_DISK_GB:-20}"
# Minimum total GPU RAM (GB) per box. Default 0 = no filter. Set to
# 24 for v26+ sweeps to avoid the cubecl-cuda pool retention bug
# observed on 12 GB cards (RTX 3060/3080). Sweep workers running
# 372-feature zensim + 5 GPU metrics consistently brick those
# cards mid-run; 24 GB+ cards (A5000/3090/4090/A6000) have enough
# headroom that the bounded chunk cap suffices.
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-0}"
PARALLEL="${PARALLEL:-0}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
GHCR_USER="${GHCR_USER:-lilithriver}"
DRY_RUN="${DRY_RUN:-0}"
WATCH_INLINE="${WATCH_INLINE:-0}"
WATCH_MAX_WALL_MIN="${WATCH_MAX_WALL_MIN:-240}"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

[[ $# -gt 0 && ("$1" == "-h" || "$1" == "--help") ]] && usage 0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --metric) METRIC="$2"; shift 2;;
        --run-id) RUN_ID="$2"; shift 2;;
        --chunks) CHUNKS="$2"; shift 2;;
        --docker|--zenmetrics-image) ZEN_METRICS_IMAGE="$2"; shift 2;;
        --onstart) ONSTART_PATH="$2"; shift 2;;
        --n-boxes) N_BOXES="$2"; shift 2;;
        --max-dph) MAX_DPH="$2"; shift 2;;
        --min-cores) MIN_CORES="$2"; shift 2;;
        --min-ram) MIN_RAM_GB="$2"; shift 2;;
        --min-disk) MIN_DISK_GB="$2"; shift 2;;
        --min-gpu-ram) MIN_GPU_RAM_GB="$2"; shift 2;;
        --parallel) PARALLEL="$2"; shift 2;;
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --ghcr-user) GHCR_USER="$2"; shift 2;;
        --watch) WATCH_INLINE=1; shift;;
        --watch-max-wall-min) WATCH_MAX_WALL_MIN="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

[[ -z "$METRIC" ]]   && { echo "ERROR: --metric required" >&2; usage 1; }
[[ -z "$RUN_ID" ]]   && { echo "ERROR: --run-id required" >&2; usage 1; }
[[ -z "$CHUNKS" ]]   && { echo "ERROR: --chunks required" >&2; usage 1; }
[[ -z "$ZEN_METRICS_IMAGE" ]] && { echo "ERROR: --docker required" >&2; usage 1; }
[[ -z "$ONSTART_PATH" ]] && {
    # Try the conventional location.
    GUESS="scripts/sweep/onstart_${METRIC}_backfill.sh"
    if [[ -f "$GUESS" ]]; then
        ONSTART_PATH="$GUESS"
        echo "[launch_backfill] defaulting --onstart=$ONSTART_PATH" >&2
    else
        echo "ERROR: --onstart required (no $GUESS found)" >&2
        usage 1
    fi
}

[[ -f "$ONSTART_PATH" ]] || { echo "ERROR: --onstart $ONSTART_PATH does not exist" >&2; exit 1; }

BOOT_IMAGE="$ZEN_METRICS_IMAGE"
GHCR_TOKEN="$(gh auth token)"

echo "[launch_backfill] config"
echo "  METRIC:            $METRIC"
echo "  RUN_ID:            $RUN_ID"
echo "  CHUNKS:            $CHUNKS"
echo "  BOOT_IMAGE:        $BOOT_IMAGE"
echo "  ONSTART_PATH:      $ONSTART_PATH"
echo "  N_BOXES:           $N_BOXES"
echo "  MAX_DPH:           $MAX_DPH"
echo "  MIN_CORES:         $MIN_CORES"
echo "  MIN_RAM_GB:        $MIN_RAM_GB"
echo "  MIN_GPU_RAM_GB:    $MIN_GPU_RAM_GB"
echo "  MIN_DISK_GB:       $MIN_DISK_GB"
echo "  PARALLEL/box:      $PARALLEL"
echo "  GPU_RUNTIME:       $GPU_RUNTIME"
echo

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() { s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 "$@"; }

# Derive prefixes from the chunks path. CHUNKS like
# s3://coefficient/jobs/<run-id>/chunks.jsonl. SCRIPTS_R2_PREFIX is its
# parent.
SCRIPTS_R2_PREFIX="${CHUNKS%/chunks.jsonl}"
[[ "$SCRIPTS_R2_PREFIX" == "$CHUNKS" ]] && {
    echo "WARN: --chunks should end in /chunks.jsonl; using its parent as SCRIPTS_R2_PREFIX" >&2
    SCRIPTS_R2_PREFIX="${CHUNKS%/*}"
}

echo "[launch_backfill] verifying $CHUNKS is present"
if ! R2 ls "$CHUNKS" >/dev/null 2>&1; then
    echo "  ERROR: $CHUNKS missing in R2." >&2
    exit 1
fi
echo "  ok"

# Count chunks for the auto-derived watch target (n_chunks - 10 grace).
N_CHUNKS_RAW=$(R2 cat "$CHUNKS" 2>/dev/null | wc -l)
TARGET_SIDECARS=$(( N_CHUNKS_RAW - 10 ))
(( TARGET_SIDECARS < 1 )) && TARGET_SIDECARS=$N_CHUNKS_RAW
echo "[launch_backfill] $N_CHUNKS_RAW chunks → watch target $TARGET_SIDECARS (= chunks - 10 grace)"

# Upload onstart to the scripts prefix so workers can fetch it.
ONSTART_BASENAME="$(basename "$ONSTART_PATH")"
ONSTART_R2_KEY="${SCRIPTS_R2_PREFIX}/${ONSTART_BASENAME}"
echo "[launch_backfill] uploading $ONSTART_PATH → $ONSTART_R2_KEY"
R2 cp "$ONSTART_PATH" "$ONSTART_R2_KEY"

# Also upload the unified worker so onstart can fetch it (if not baked
# into the docker image). This is gated — only upload if the file
# exists on disk (which it should: this is part of feat/sweep-infra-
# unified). Override WORKER_PATH for onstarts with a different worker
# (e.g. hdr_pairs_chunk_worker.sh for persisted-pairs HDR fleets).
WORKER_PATH="${WORKER_PATH:-scripts/sweep/metric_backfill_chunk_worker.sh}"
if [[ -f "$WORKER_PATH" ]]; then
    WORKER_R2_KEY="${SCRIPTS_R2_PREFIX}/$(basename "$WORKER_PATH")"
    echo "[launch_backfill] uploading $WORKER_PATH → $WORKER_R2_KEY"
    R2 cp "$WORKER_PATH" "$WORKER_R2_KEY"
fi

# Driver filter rationale (2026-05-18, v19 image):
#
#   The v19 zenmetrics binary was built with CUDARC_CUDA_VERSION=12090,
#   which forces cudarc 0.19.4 to compile against the CUDA 12.9 binding
#   surface. None of the CUDA 13-only symbols
#   (cuCtxGetDevice_v2, cuCoredump{Register,Deregister}{Start,Complete}Callback)
#   are referenced by the resulting binary, so it loads cleanly on
#   drivers from 525.x through 580.x. We therefore relax the upper
#   ceiling that was needed for v14-v18 binaries.
#
#   Historical context (kept for future-self): v14-v18 was built with
#   cudarc auto-detecting CUDA 13.x from our local nvcc, dragging
#   cuCoredump* and cuCtxGetDevice_v2 dlsyms into the static load
#   path. Old drivers (<570) lacked v2; new drivers (>=570 with no
#   coredump callbacks) lacked Coredump*. The LD_PRELOAD stub at
#   /usr/local/lib/cuda_dlsym_stub.so papered over the latter but
#   not the former, so the v18 smoke still panicked on driver 555.
#
#   We now floor at driver 555 (CUDA 12.5+ ABI) — cudarc 0.19.4 emits
#   PTX with the CUDA 12.5+ minor version directive, and drivers older
#   than 555.42 reject the PTX with CUDA_ERROR_UNSUPPORTED_PTX_VERSION
#   at module load. The v21 smoke confirmed runtime-symbol panics were
#   eliminated; the surviving blocker on cheap-driver boxes is PTX-
#   version rejection. Bumping the floor from 525 -> 555 keeps
#   `cuda_max_good>=12.0` consistent with the actual driver ABI we
#   require. Historical note: 525 was the CUDA 12.0 first-release floor.
QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_max_good>=12.0 driver_version>=555.0.0 num_gpus=1"
# vast.ai's gpu_total_ram filter accepts GB units (the JSON field is
# MB but the query parser scales). gpu_total_ram>=24 = 24 GB.
if [[ "${MIN_GPU_RAM_GB}" -gt 0 ]]; then
    QUERY="${QUERY} gpu_total_ram>=${MIN_GPU_RAM_GB}"
fi
# GPU_FRAC_MIN (default 1.0 = dedicated GPU) — cheap "24 GB" offers
# on vast.ai are typically partial fractions (e.g. frac=0.2 = 4.8 GB
# usable). For sweeps with multi-MP source images, frac=1.0 is
# required or per-cell OOMs are observed. 2026-05-22 finding.
GPU_FRAC_MIN="${GPU_FRAC_MIN:-1.0}"
QUERY="${QUERY} gpu_frac>=${GPU_FRAC_MIN}"
echo "[launch_backfill] querying offers: $QUERY"
OFFERS_JSON=$(vastai search offers "$QUERY" --order 'dph_total' --raw)
OFFER_IDS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
seen = set()
out = []
for o in d:
    mid = o.get('machine_id')
    if mid in seen:
        continue
    seen.add(mid)
    out.append(o['id'])
    if len(out) >= ${N_BOXES}:
        break
print('\n'.join(str(x) for x in out))
")
n=$(echo "$OFFER_IDS" | wc -l)
echo "[launch_backfill] picked $n distinct offers (need $N_BOXES)"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "$OFFER_IDS" | head -10
    echo
    echo "DRY_RUN=1: not launching. Re-run without --dry-run to commit."
    exit 0
fi

[[ "$n" -lt 3 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

# Fix A (2026-05-18 EXP-LARGER-LARGE-V2): the prior heredoc-as-onstart-cmd
# pattern lost embedded `$` characters in vast.ai's API call (the box
# received an empty/truncated bootstrap, /var/log/onstart.log showed only
# `ERROR " ": command not found`). Replace with a base64-encoded payload
# so no quote escaping needs to survive the API hop. The payload writes
# the AWS credentials file from env vars (injected via --env) and execs
# the onstart pulled from R2.
ONSTART_BOOTSTRAP_RAW=$(cat <<'BOOT'
set -e
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = $R2_ACCESS_KEY_ID
aws_secret_access_key = $R2_SECRET_ACCESS_KEY
CREDS
# Scoped temp creds arrive with the session token split into
# R2_SESSION_TOKEN_0..N (vast's ~256-char env value cap). Reassemble
# BEFORE the first s5cmd call or the onstart fetch itself 403s.
ST="${R2_SESSION_TOKEN:-}"
if [ -z "$ST" ]; then
    for idx in 0 1 2 3 4 5 6 7 8 9; do
        eval part="\${R2_SESSION_TOKEN_${idx}:-}"
        [ -n "$part" ] && ST="${ST}${part}"
    done
fi
if [ -n "$ST" ]; then
    export AWS_SESSION_TOKEN="$ST" R2_SESSION_TOKEN="$ST"
    echo "aws_session_token = $ST" >> ~/.aws/credentials
fi
# Wait for s5cmd to be present (the v14 docker image bakes it; some
# upstream images install it at runtime — sleep briefly if absent).
for try in 1 2 3 4 5; do
    command -v s5cmd >/dev/null && break
    sleep 3
done
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    cp __ONSTART_R2_KEY__ \
       /usr/local/bin/onstart.sh
chmod +x /usr/local/bin/onstart.sh
# Route through the trap wrapper when the image bakes it (v15+). This
# gives the fleet self-destroy-on-crash semantics matching what the
# single-box smoke test (launch_backfill.sh --n-boxes 1) has: a panicked
# onstart uploads stderr to
# s3://zentrain/<run>/errors/<instance>.log and DELETEs its own
# vast.ai instance so a broken box doesn't keep burning $/hr.
if [[ -x /usr/local/bin/run_with_error_trap.sh ]]; then
    exec /usr/local/bin/run_with_error_trap.sh /usr/local/bin/onstart.sh
else
    exec /usr/local/bin/onstart.sh
fi
BOOT
)
# Substitute the R2 key into the placeholder.
ONSTART_BOOTSTRAP_RAW="${ONSTART_BOOTSTRAP_RAW//__ONSTART_R2_KEY__/$ONSTART_R2_KEY}"
# Base64-encode the entire payload so the bash -c arg is a fixed token.
ONSTART_BOOTSTRAP_B64=$(printf '%s' "$ONSTART_BOOTSTRAP_RAW" | base64 -w0)

INSTANCE_FILE="/tmp/${RUN_ID}/instances.txt"
mkdir -p "$(dirname "$INSTANCE_FILE")"
: > "$INSTANCE_FILE"

# Worker-side creds: prefer SCOPED temp creds (WORKER_R2_*) over the
# sourced account key. Session tokens exceed vast.ai's ~256-char env
# value cap, so split into R2_SESSION_TOKEN_0..N chunks — both
# onstart_unified.sh and onstart_hdr_pairs.sh reassemble them.
CRED_ENV_STR=""
if [[ -n "${WORKER_R2_ACCESS_KEY_ID:-}" ]]; then
    CRED_ENV_STR="-e R2_ACCESS_KEY_ID=${WORKER_R2_ACCESS_KEY_ID}"
    CRED_ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${WORKER_R2_SECRET_ACCESS_KEY:?WORKER_R2_SECRET_ACCESS_KEY required with WORKER_R2_ACCESS_KEY_ID}"
    if [[ -n "${WORKER_R2_SESSION_TOKEN:-}" ]]; then
        _tok="$WORKER_R2_SESSION_TOKEN"; _i=0
        while [[ -n "$_tok" ]]; do
            CRED_ENV_STR+=" -e R2_SESSION_TOKEN_${_i}=${_tok:0:240}"
            _tok="${_tok:240}"; _i=$((_i + 1))
        done
        echo "[launch_backfill] scoped worker creds: session token split into ${_i} parts"
    fi
else
    CRED_ENV_STR="-e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
    CRED_ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
    echo "[launch_backfill] WARNING: shipping the sourced R2 key to the fleet." >&2
    echo "  Mint scoped temp creds and pass WORKER_R2_ACCESS_KEY_ID/_SECRET/_SESSION_TOKEN" >&2
    echo "  (see ~/work/claudehints/topics/r2-credentials.md)." >&2
fi

i=0
launched=0
for offer_id in $OFFER_IDS; do
    i=$((i + 1))
    WORKER_ID="${RUN_ID}-w$i"
    LABEL="$WORKER_ID"

    ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID}"
    ENV_STR+=" ${CRED_ENV_STR}"
    ENV_STR+=" -e SWEEP_RUN_ID=${RUN_ID}"
    ENV_STR+=" -e WORKER_ID=${WORKER_ID}"
    ENV_STR+=" -e METRIC=${METRIC}"
    ENV_STR+=" -e PARALLEL=${PARALLEL}"
    ENV_STR+=" -e GPU_RUNTIME=${GPU_RUNTIME}"
    ENV_STR+=" -e SCRIPTS_R2_PREFIX=${SCRIPTS_R2_PREFIX}"
    [[ -n "${SWEEP_BIN_OVERRIDE:-}" ]] && \
        ENV_STR+=" -e SWEEP_BIN_OVERRIDE=${SWEEP_BIN_OVERRIDE}"
    [[ -n "${PARALLEL_CHUNKS:-}" ]] && \
        ENV_STR+=" -e PARALLEL_CHUNKS=${PARALLEL_CHUNKS}"
    [[ -n "${PARALLEL_CHUNKS_MAX:-}" ]] && \
        ENV_STR+=" -e PARALLEL_CHUNKS_MAX=${PARALLEL_CHUNKS_MAX}"
    [[ -n "${ADAPT_INTERVAL_SEC:-}" ]] && \
        ENV_STR+=" -e ADAPT_INTERVAL_SEC=${ADAPT_INTERVAL_SEC}"
    [[ -n "${ZENSIM_FEATURES_REGIME:-}" ]] && \
        ENV_STR+=" -e ZENSIM_FEATURES_REGIME=${ZENSIM_FEATURES_REGIME}"
    # Per-process chunk cap forwarded to the Rust worker. Default 20.
    [[ -n "${MAX_CHUNKS_PER_PROCESS:-}" ]] && \
        ENV_STR+=" -e MAX_CHUNKS_PER_PROCESS=${MAX_CHUNKS_PER_PROCESS}"
    [[ -n "${MAX_RESPAWNS:-}" ]] && \
        ENV_STR+=" -e MAX_RESPAWNS=${MAX_RESPAWNS}"
    # SWEEP_CLEANUP_BETWEEN_SOURCES (commit a21204f) — opt-in cubecl
    # pool flush. Safe only with PARALLEL_CHUNKS_MAX=1.
    [[ -n "${SWEEP_CLEANUP_BETWEEN_SOURCES:-}" ]] && \
        ENV_STR+=" -e SWEEP_CLEANUP_BETWEEN_SOURCES=${SWEEP_CLEANUP_BETWEEN_SOURCES}"
    [[ -n "${METRICS:-}" ]] && \
        ENV_STR+=" -e METRICS=${METRICS}"
    [[ -n "${JOBS:-}" ]] && \
        ENV_STR+=" -e JOBS=${JOBS}"
    # Explicit chunks URL so onstarts need not re-derive it by convention.
    ENV_STR+=" -e CHUNKS_R2=${CHUNKS}"
    # Opt-in: ship the vast API key so onstarts can self-destroy drained
    # boxes (INJECT_VAST_API_KEY=1). Without it a completed box IDLES at
    # full $/hr until an operator reaps it (2026-07-13 incident) — if not
    # injecting, run `zenfleet-vastai watch` or a log-tail reaper.
    if [[ "${INJECT_VAST_API_KEY:-0}" == "1" ]]; then
        VAST_KEY_VAL="${VAST_API_KEY:-$(cat ~/.config/vastai/vast_api_key 2>/dev/null || true)}"
        [[ -n "$VAST_KEY_VAL" ]] && ENV_STR+=" -e VAST_API_KEY=${VAST_KEY_VAL}"
    fi
    # Opt-in static modulo sharding (SHARD_N env): shard index counts
    # SUCCESSFUL launches so a failed create doesn't orphan its shard
    # mid-sequence. Any launch shortfall is reported after the loop.
    [[ -n "${SHARD_N:-}" ]] && \
        ENV_STR+=" -e SHARD_IDX=${launched} -e SHARD_N=${SHARD_N}"

    LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"

    # Use base64-decoded bootstrap to dodge vast.ai's API arg-mangling
    # of embedded `$` chars in the heredoc. Single quotes around the
    # base64 string keep the API-side parser from interpreting anything.
    ONSTART_CMD="bash -c 'echo ${ONSTART_BOOTSTRAP_B64} | base64 -d | bash'"
    OUT=$(vastai create instance "$offer_id" \
        --image "$BOOT_IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "$ONSTART_CMD" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        --raw 2>&1) || { echo "  $i fail: $(echo "$OUT" | head -c 200)"; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "  $i parse-fail: $(echo "$OUT" | head -c 200)"; continue; }
    # ssh-runtype instances are created in stopped state — explicit
    # start is required for the onstart-cmd to fire. (Same fix the
    # single-box smoke path (--n-boxes 1) applies; without this every box in
    # the fleet sits in actual_status=created indefinitely.)
    vastai start instance "$ID" >/dev/null 2>&1 || \
        echo "  $i WARN: start instance $ID failed (instance may still auto-start)"
    echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
    echo "  $i -> instance $ID ($WORKER_ID)"
    launched=$((launched + 1))
done

NLAUNCHED=$(wc -l < "$INSTANCE_FILE")
echo
echo "[launch_backfill] launched $NLAUNCHED instances (target $N_BOXES)"
echo "  manifest: $INSTANCE_FILE"
if [[ -n "${SHARD_N:-}" && "$NLAUNCHED" -lt "$SHARD_N" ]]; then
    echo "  WARNING: sharded launch shortfall — shards ${NLAUNCHED}..$((SHARD_N - 1)) are ORPHANED." >&2
    echo "  Remedy: once the fleet drains, launch ONE sweeper box with SHARD_N=1 SHARD_IDX=0 —" >&2
    echo "  per-chunk sidecar idempotency makes it skip everything already scored." >&2
fi
echo

# Suggest (or run) the watch command.
SIDECAR_R2_PREFIX="s3://zentrain/${RUN_ID}/"
WATCH_CMD=(
    zenfleet-vastai watch
    --label-prefix "$RUN_ID"
    --target-sidecars "$TARGET_SIDECARS"
    --r2-prefix "$SIDECAR_R2_PREFIX"
    --max-wall-min "$WATCH_MAX_WALL_MIN"
)

if [[ "$WATCH_INLINE" == "1" ]]; then
    echo "[launch_backfill] entering zenfleet-vastai watch (inline) — Ctrl+C to detach"
    exec "${WATCH_CMD[@]}"
else
    echo "[launch_backfill] to auto-destroy when complete:"
    printf '  '
    for w in "${WATCH_CMD[@]}"; do
        printf '%q ' "$w"
    done
    printf '\n'
    echo
    echo "[launch_backfill] or run inline by adding --watch to launch_backfill.sh"
fi
