#!/usr/bin/env bash
#
# DEPRECATED: use scripts/sweep/launch_backfill.sh --metric iwssim ...
# instead. This file is retained for in-flight runners and reference.
# New launches should use the unified launcher.
#
# iwssim_backfill/launch.sh — vast.ai fleet launcher for the IW-SSIM
# backfill (1.17M pairs from cvvdp_imazen_consolidated.parquet).
# Mirrors cvvdp_backfill/launch_imazen.sh — single-image flow, no
# Docker-in-Docker.
#
# Env vars:
#   SWEEP_RUN_ID            (default: iwssim-backfill-2026-05-17)
#   ZEN_METRICS_IMAGE       (default: ghcr.io/imazen/zenmetrics-sweep:0.6.4-iwssim-d9c0388)
#                           — also used as BOOT_IMAGE
#   N_BOXES                 (default: 30)
#   MAX_DPH                 (default: 0.30)
#   MIN_CORES               (default: 8)
#   MIN_RAM_GB              (default: 8 — IW-SSIM is lighter than CVVDP)
#   MIN_DISK_GB             (default: 20)
#   PARALLEL                (default: 0 = auto-detect)
#   GPU_RUNTIME             (default: auto)
#   DRY_RUN                 (default: 0)

set -euo pipefail
# shellcheck disable=SC1091
source ~/.config/cloudflare/r2-credentials

SWEEP_RUN_ID="${SWEEP_RUN_ID:-iwssim-backfill-2026-05-17}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-ghcr.io/imazen/zenmetrics-sweep:0.6.4-iwssim-d9c0388}"
BOOT_IMAGE="$ZEN_METRICS_IMAGE"

N_BOXES="${N_BOXES:-30}"
MAX_DPH="${MAX_DPH:-0.30}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-8}"
MIN_DISK_GB="${MIN_DISK_GB:-20}"
PARALLEL="${PARALLEL:-0}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DRY_RUN="${DRY_RUN:-0}"

GHCR_TOKEN="$(gh auth token)"
GHCR_USER="${GHCR_USER:-lilithriver}"

echo "[iwssim-backfill] launching fleet"
echo "  SWEEP_RUN_ID:      $SWEEP_RUN_ID"
echo "  BOOT_IMAGE:        $BOOT_IMAGE"
echo "  N_BOXES:           $N_BOXES"
echo "  MAX_DPH:           $MAX_DPH"
echo "  PARALLEL/box:      $PARALLEL"
echo "  GPU_RUNTIME:       $GPU_RUNTIME"
echo

SCRIPTS_R2_PREFIX="s3://coefficient/jobs/${SWEEP_RUN_ID}"
echo "[iwssim-backfill] verifying $SCRIPTS_R2_PREFIX has chunks.jsonl + iwssim_backfill_chunk_worker.sh"
if ! s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    ls "${SCRIPTS_R2_PREFIX}/" 2>/dev/null | grep -q chunks.jsonl; then
    echo "  ERROR: ${SCRIPTS_R2_PREFIX}/chunks.jsonl missing." >&2
    exit 1
fi
if ! s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    ls "${SCRIPTS_R2_PREFIX}/" 2>/dev/null | grep -q iwssim_backfill_chunk_worker.sh; then
    echo "  ERROR: ${SCRIPTS_R2_PREFIX}/iwssim_backfill_chunk_worker.sh missing." >&2
    exit 1
fi
echo "  ok"

QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_vers>=12.5 num_gpus=1"
echo "[iwssim-backfill] querying offers: $QUERY"
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
echo "[iwssim-backfill] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "$OFFER_IDS" | head -10
    echo
    echo "DRY_RUN=1: not launching. Re-run with DRY_RUN=0 to commit."
    exit 0
fi
[[ "$n" -lt 3 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

ONSTART_R2_KEY="${SCRIPTS_R2_PREFIX}/onstart_iwssim_backfill.sh"
echo "[iwssim-backfill] uploading onstart to $ONSTART_R2_KEY"
s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    cp scripts/sweep/onstart_iwssim_backfill.sh "$ONSTART_R2_KEY"

ONSTART_BOOTSTRAP=$(cat <<'BOOT'
set -e
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = $R2_ACCESS_KEY_ID
aws_secret_access_key = $R2_SECRET_ACCESS_KEY
CREDS
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    cp s3://coefficient/jobs/${SWEEP_RUN_ID}/onstart_iwssim_backfill.sh \
       /usr/local/bin/onstart.sh
chmod +x /usr/local/bin/onstart.sh
exec /usr/local/bin/onstart.sh
BOOT
)

INSTANCE_FILE="/tmp/iwssim-backfill-${SWEEP_RUN_ID}/instances.txt"
mkdir -p "$(dirname "$INSTANCE_FILE")"
: > "$INSTANCE_FILE"

i=0
for offer_id in $OFFER_IDS; do
    i=$((i + 1))
    WORKER_ID="${SWEEP_RUN_ID}-w$i"
    LABEL="$WORKER_ID"

    ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID}"
    ENV_STR+=" -e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
    ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
    ENV_STR+=" -e SWEEP_RUN_ID=${SWEEP_RUN_ID}"
    ENV_STR+=" -e WORKER_ID=${WORKER_ID}"
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
echo "[iwssim-backfill] launched $NLAUNCHED instances (target $N_BOXES)"
echo "  manifest: $INSTANCE_FILE"
