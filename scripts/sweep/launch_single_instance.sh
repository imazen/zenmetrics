#!/usr/bin/env bash
#
# launch_single_instance.sh — single-box launcher for tuning + smoke
# testing the cvvdp / iwssim / ssim2 backfill flow before fanning
# out to a fleet via launch_backfill.sh.
#
# Differs from launch_backfill.sh in three ways:
#   - launches exactly ONE instance (no `n<3` quality protection)
#   - wraps the onstart in /usr/local/bin/run_with_error_trap.sh
#     when present in the image (v15+), so a crash uploads stderr
#     to R2 + self-destroys the box automatically
#   - prints the SSH connect + watch-util commands at the end so
#     the operator can monitor GPU util without external scripts
#
# Required env vars:
#   R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
#   (sourced from ~/.config/cloudflare/r2-credentials)
#
# Required CLI tools: vastai (≥1.0.8), s5cmd, gh, python3.
#
# Usage:
#   launch_single_instance.sh \
#       --metric cvvdp \
#       --run-id cvvdp-v15rc-2026-05-18 \
#       --chunks s3://coefficient/jobs/cvvdp-v15rc-2026-05-18/chunks.jsonl \
#       --docker ghcr.io/imazen/zen-metrics-sweep:v15 \
#       --onstart scripts/sweep/onstart_cvvdp_backfill_imazen.sh \
#       --max-dph 0.10

set -euo pipefail

# ── arg parsing ─────────────────────────────────────────────────────
METRIC=""
RUN_ID=""
CHUNKS=""
ZEN_METRICS_IMAGE="ghcr.io/imazen/zen-metrics-sweep:v15"
ONSTART_PATH=""
MAX_DPH="0.10"
MIN_CORES="${MIN_CORES:-4}"
MIN_RAM_GB="${MIN_RAM_GB:-8}"
MIN_DISK_GB="${MIN_DISK_GB:-20}"
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-10000}"
GHCR_USER="${GHCR_USER:-lilith}"
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"   # pin cuda — block CPU fallback by default
PARALLEL="${PARALLEL:-0}"

usage() {
    sed -n '2,30p' "$0" >&2
    exit "${1:-2}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --metric) METRIC="$2"; shift 2;;
        --run-id) RUN_ID="$2"; shift 2;;
        --chunks) CHUNKS="$2"; shift 2;;
        --docker) ZEN_METRICS_IMAGE="$2"; shift 2;;
        --onstart) ONSTART_PATH="$2"; shift 2;;
        --max-dph) MAX_DPH="$2"; shift 2;;
        --min-gpu-ram-mb) MIN_GPU_RAM_MB="$2"; shift 2;;
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --parallel) PARALLEL="$2"; shift 2;;
        -h|--help) usage 0;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

[[ -z "$METRIC" ]]   && { echo "ERROR: --metric required" >&2; usage 1; }
[[ -z "$RUN_ID" ]]   && { echo "ERROR: --run-id required" >&2; usage 1; }
[[ -z "$CHUNKS" ]]   && { echo "ERROR: --chunks required" >&2; usage 1; }
[[ -z "$ONSTART_PATH" ]] && { echo "ERROR: --onstart required" >&2; usage 1; }
[[ -f "$ONSTART_PATH" ]] || { echo "ERROR: $ONSTART_PATH missing" >&2; exit 1; }

# ── R2 creds ────────────────────────────────────────────────────────
if [[ -r ~/.config/cloudflare/r2-credentials ]]; then
    . ~/.config/cloudflare/r2-credentials
fi
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() { s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 "$@"; }

GHCR_TOKEN="$(gh auth token)"
[[ -n "$GHCR_TOKEN" ]] || { echo "ERROR: gh auth token returned empty" >&2; exit 1; }

# ── upload onstart + worker scripts to the run prefix ───────────────
SCRIPTS_R2_PREFIX="${CHUNKS%/chunks.jsonl}"
ONSTART_R2_KEY="${SCRIPTS_R2_PREFIX}/$(basename "$ONSTART_PATH")"

echo "[launch_single] config"
echo "  METRIC:            $METRIC"
echo "  RUN_ID:            $RUN_ID"
echo "  CHUNKS:            $CHUNKS"
echo "  BOOT_IMAGE:        $ZEN_METRICS_IMAGE"
echo "  ONSTART:           $ONSTART_PATH -> $ONSTART_R2_KEY"
echo "  MAX_DPH:           \$${MAX_DPH}/hr"
echo "  GPU_RUNTIME:       $GPU_RUNTIME"
echo "  PARALLEL:          $PARALLEL"
echo

echo "[launch_single] verifying $CHUNKS"
R2 ls "$CHUNKS" >/dev/null
echo "  ok"

echo "[launch_single] uploading $ONSTART_PATH"
R2 cp "$ONSTART_PATH" "$ONSTART_R2_KEY"

# Co-upload the metric-specific worker so the onstart can fetch it
# (skip if the image bakes it in; harmless duplicate otherwise).
WORKER_PATH="scripts/sweep/${METRIC}_backfill_chunk_worker.sh"
if [[ -f "$WORKER_PATH" ]]; then
    R2 cp "$WORKER_PATH" "${SCRIPTS_R2_PREFIX}/$(basename "$WORKER_PATH")"
fi
# Unified worker too (some onstarts use it instead).
UNIFIED_WORKER="scripts/sweep/metric_backfill_chunk_worker.sh"
if [[ -f "$UNIFIED_WORKER" ]]; then
    R2 cp "$UNIFIED_WORKER" "${SCRIPTS_R2_PREFIX}/$(basename "$UNIFIED_WORKER")"
fi

# ── pick the cheapest viable offer ──────────────────────────────────
# Driver filter rationale (2026-05-18, v19 image):
#
#   The v19 zen-metrics binary was built with CUDARC_CUDA_VERSION=12090,
#   which forces cudarc 0.19.4 to compile against the CUDA 12.9 binding
#   surface. None of the CUDA 13-only symbols
#   (cuCtxGetDevice_v2, cuCoredump{Register,Deregister}{Start,Complete}Callback)
#   are referenced by the resulting binary, so it loads cleanly on
#   drivers from 525.x through 580.x. We therefore relax the filter
#   compared to launch_backfill.sh's `driver_version<570.0.0` gate.
#
#   2026-05-19 update: floor bumped 525 -> 555 (CUDA 12.5+ ABI). cudarc
#   0.19.4 emits PTX with the CUDA 12.5+ minor version directive, and
#   drivers older than 555.42 reject the PTX at module load with
#   CUDA_ERROR_UNSUPPORTED_PTX_VERSION. v21 smoke eliminated runtime
#   symbol panics; PTX-version mismatch is the surviving blocker on
#   cheap-driver boxes. See launch_backfill.sh for full rationale.
#
#   If a NEW dlsym panic surfaces on a driver in this band, narrow the
#   floor further (driver_version>=570 is the next stable cut) rather
#   than re-imposing the upper ceiling.
QUERY="rentable=true reliability>0.99 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} gpu_total_ram>=$((MIN_GPU_RAM_MB / 1000)) cuda_max_good>=12.0 driver_version>=555.0.0 dlperf>=12 num_gpus=1"
echo "[launch_single] querying offers"
echo "  $QUERY"
OFFER_ID=$(vastai search offers "$QUERY" --order 'dph_total' --raw \
    | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
if isinstance(d, dict) and 'offers' in d: d = d['offers']
if not d: raise SystemExit('no offers match query')
print(d[0]['id'])
")
echo "  picked offer $OFFER_ID"

# ── boot-time bootstrap: download onstart from R2 + exec via the
#    image's run_with_error_trap wrapper if present (v15+). ──────────
ONSTART_BOOTSTRAP=$(cat <<BOOT
set -e
export AWS_ACCESS_KEY_ID="\$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="\$R2_SECRET_ACCESS_KEY"
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = \$R2_ACCESS_KEY_ID
aws_secret_access_key = \$R2_SECRET_ACCESS_KEY
CREDS
s5cmd --endpoint-url "https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" --profile r2 cp $ONSTART_R2_KEY /usr/local/bin/onstart.sh
chmod +x /usr/local/bin/onstart.sh
if [[ -x /usr/local/bin/run_with_error_trap.sh ]]; then
    exec /usr/local/bin/run_with_error_trap.sh /usr/local/bin/onstart.sh
else
    exec /usr/local/bin/onstart.sh
fi
BOOT
)

LABEL="${RUN_ID}-single"
ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID}"
ENV_STR+=" -e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
ENV_STR+=" -e SWEEP_RUN_ID=${RUN_ID}"
ENV_STR+=" -e WORKER_ID=${LABEL}"
ENV_STR+=" -e METRIC=${METRIC}"
ENV_STR+=" -e PARALLEL=${PARALLEL}"
ENV_STR+=" -e GPU_RUNTIME=${GPU_RUNTIME}"
ENV_STR+=" -e SCRIPTS_R2_PREFIX=${SCRIPTS_R2_PREFIX}"
# Forward optional run-time toggles when set in the launcher's env. The
# omni onstart reads SKIP_CLAIMS to bypass the R2 claim check (useful
# for single-instance smoke runs against a claim namespace already
# populated by a prior aborted run).
[[ -n "${SKIP_CLAIMS:-}" ]]      && ENV_STR+=" -e SKIP_CLAIMS=${SKIP_CLAIMS}"
[[ -n "${METRICS:-}" ]]          && ENV_STR+=" -e METRICS=${METRICS}"
[[ -n "${PARALLEL_CHUNKS:-}" ]]      && ENV_STR+=" -e PARALLEL_CHUNKS=${PARALLEL_CHUNKS}"
[[ -n "${PARALLEL_CHUNKS_MAX:-}" ]]  && ENV_STR+=" -e PARALLEL_CHUNKS_MAX=${PARALLEL_CHUNKS_MAX}"
[[ -n "${ADAPT_INTERVAL_SEC:-}" ]]   && ENV_STR+=" -e ADAPT_INTERVAL_SEC=${ADAPT_INTERVAL_SEC}"
[[ -n "${ZENSIM_FEATURES_REGIME:-}" ]] && ENV_STR+=" -e ZENSIM_FEATURES_REGIME=${ZENSIM_FEATURES_REGIME}"
# Pass the chunks URL through env. The unified Rust worker reads
# CHUNKS_R2; the bash onstart workers ignored it and derived from
# the SWEEP_RUN_ID. Forwarding here lets the smoke flow point at
# a different chunks namespace than the run ID without changing
# either codepath.
ENV_STR+=" -e CHUNKS_R2=${CHUNKS}"
LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"

echo "[launch_single] creating instance"
OUT=$(vastai create instance "$OFFER_ID" \
    --image "$ZEN_METRICS_IMAGE" --login "$LOGIN_STR" \
    --onstart-cmd "bash -c '$ONSTART_BOOTSTRAP'" \
    --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
    --raw 2>&1)
ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))")
[[ -z "$ID" ]] && { echo "ERROR: launch failed: $OUT" | head -c 500; exit 1; }

# vast.ai instances in ssh runtype are created in `stopped` state and
# need an explicit start to fire the onstart-cmd. (Without this, the
# instance sits at actual_status=created indefinitely. Observed
# 2026-05-18 — past sweeps may have started instances via a different
# vastai CLI version that auto-started.)
echo "[launch_single] starting instance $ID"
vastai start instance "$ID" 2>&1 | head -2

echo
echo "[launch_single] launched instance $ID (offer $OFFER_ID, label $LABEL)"
echo
echo "Monitor commands:"
echo "  fleet status:        vastai-fleet status --label-prefix '$RUN_ID'"
echo "  ssh in:              vastai ssh-url $ID  # copy that URL into ssh"
echo "  gpu util:            vastai execute $ID 'nvidia-smi dmon -c 30 -s u'"
echo "  follow logs:         vastai logs $ID --tail"
echo "  destroy now:         vastai destroy instance $ID"
echo "  sidecars produced:   s5cmd --profile r2 --endpoint-url $R2_ENDPOINT ls s3://zentrain/${RUN_ID}/${METRIC}_imazen/ | wc -l"
echo
echo "When this single instance proves throughput is good, fan out via:"
echo "  scripts/sweep/launch_backfill.sh --metric $METRIC --run-id $RUN_ID \\"
echo "    --chunks $CHUNKS --docker $ZEN_METRICS_IMAGE \\"
echo "    --onstart $ONSTART_PATH --n-boxes <N> --max-dph $MAX_DPH"
