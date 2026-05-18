#!/usr/bin/env bash
#
# launch_backfill.sh — unified vast.ai fleet launcher for metric backfill
# sweeps. Replaces the per-metric launch.sh / launch_imazen.sh files
# under iwssim_backfill/ / ssim2_backfill/ / cvvdp_backfill/.
#
# Drives the same per-instance create loop as the originals but folds
# all the parameter knobs behind one flag interface, and calls into
# vastai-fleet for the destroy half of the workflow (no more bash+python
# heredoc destroyers).
#
# Required tools on PATH:
#   - vastai (1.0.8 or newer)
#   - s5cmd
#   - gh (for ghcr.io token)
#   - python3 (only for parsing `vastai create instance --raw` output)
#   - vastai-fleet (built from crates/vastai-fleet — `cargo build
#     --release -p vastai-fleet && cp target/release/vastai-fleet ~/.local/bin/`)
#
# Required env vars (sourced from ~/.config/cloudflare/r2-credentials):
#   R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
#
# Flag-style invocation:
#
#   launch_backfill.sh \
#       --metric iwssim \
#       --run-id iwssim-backfill-2026-05-17 \
#       --chunks s3://coefficient/jobs/iwssim-backfill-2026-05-17/chunks.jsonl \
#       --max-dph 0.30 --n-boxes 30 --min-ram 8 --min-disk 20 \
#       --docker ghcr.io/imazen/zen-metrics-sweep:0.6.4-iwssim-fixed-6227c1a \
#       --onstart scripts/sweep/onstart_iwssim_backfill.sh
#
# Once the fleet is up the launcher prints the watch invocation that
# would auto-destroy at target — copy/paste to run as a detached
# background process (or invoke with --watch to run inline).
#
# All flags also accept env-var forms (METRIC, RUN_ID, CHUNKS, ...).

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
        --docker|--zen-metrics-image) ZEN_METRICS_IMAGE="$2"; shift 2;;
        --onstart) ONSTART_PATH="$2"; shift 2;;
        --n-boxes) N_BOXES="$2"; shift 2;;
        --max-dph) MAX_DPH="$2"; shift 2;;
        --min-cores) MIN_CORES="$2"; shift 2;;
        --min-ram) MIN_RAM_GB="$2"; shift 2;;
        --min-disk) MIN_DISK_GB="$2"; shift 2;;
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
# unified).
WORKER_PATH="scripts/sweep/metric_backfill_chunk_worker.sh"
if [[ -f "$WORKER_PATH" ]]; then
    WORKER_R2_KEY="${SCRIPTS_R2_PREFIX}/$(basename "$WORKER_PATH")"
    echo "[launch_backfill] uploading $WORKER_PATH → $WORKER_R2_KEY"
    R2 cp "$WORKER_PATH" "$WORKER_R2_KEY"
fi

QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_vers>=12.5 num_gpus=1"
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
s5cmd --endpoint-url "https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \\
    --profile r2 \\
    cp $ONSTART_R2_KEY \\
       /usr/local/bin/onstart.sh
chmod +x /usr/local/bin/onstart.sh
exec /usr/local/bin/onstart.sh
BOOT
)

INSTANCE_FILE="/tmp/${RUN_ID}/instances.txt"
mkdir -p "$(dirname "$INSTANCE_FILE")"
: > "$INSTANCE_FILE"

i=0
for offer_id in $OFFER_IDS; do
    i=$((i + 1))
    WORKER_ID="${RUN_ID}-w$i"
    LABEL="$WORKER_ID"

    ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID}"
    ENV_STR+=" -e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
    ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
    ENV_STR+=" -e SWEEP_RUN_ID=${RUN_ID}"
    ENV_STR+=" -e WORKER_ID=${WORKER_ID}"
    ENV_STR+=" -e METRIC=${METRIC}"
    ENV_STR+=" -e PARALLEL=${PARALLEL}"
    ENV_STR+=" -e GPU_RUNTIME=${GPU_RUNTIME}"
    ENV_STR+=" -e SCRIPTS_R2_PREFIX=${SCRIPTS_R2_PREFIX}"
    [[ -n "${SWEEP_BIN_OVERRIDE:-}" ]] && \
        ENV_STR+=" -e SWEEP_BIN_OVERRIDE=${SWEEP_BIN_OVERRIDE}"

    LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"

    OUT=$(vastai create instance "$offer_id" \
        --image "$BOOT_IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "bash -c '$ONSTART_BOOTSTRAP'" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        --raw 2>&1) || { echo "  $i fail: $(echo "$OUT" | head -c 200)"; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "  $i parse-fail: $(echo "$OUT" | head -c 200)"; continue; }
    echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
    echo "  $i -> instance $ID ($WORKER_ID)"
done

NLAUNCHED=$(wc -l < "$INSTANCE_FILE")
echo
echo "[launch_backfill] launched $NLAUNCHED instances (target $N_BOXES)"
echo "  manifest: $INSTANCE_FILE"
echo

# Suggest (or run) the watch command.
SIDECAR_R2_PREFIX="s3://zentrain/${RUN_ID}/"
WATCH_CMD=(
    vastai-fleet watch
    --label-prefix "$RUN_ID"
    --target-sidecars "$TARGET_SIDECARS"
    --r2-prefix "$SIDECAR_R2_PREFIX"
    --max-wall-min "$WATCH_MAX_WALL_MIN"
)

if [[ "$WATCH_INLINE" == "1" ]]; then
    echo "[launch_backfill] entering vastai-fleet watch (inline) — Ctrl+C to detach"
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
