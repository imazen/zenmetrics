#!/bin/bash
# launch_cpu.sh — fleet launcher for IW-SSIM sweep chunks on vast.ai.
#
# Pairs with ghcr.io/imazen/zen-metrics-iwssim:<ver> (built from
# docker/Dockerfile.iwssim). Picks CPU-only offers — iwssim runs in
# milliseconds on CPU, so renting a GPU box would just burn money.
#
# Chunks consumed by this fleet MUST set `.metrics` to include
# "iwssim" (typically alongside zensim / ssim2 for triangulation).
#
# Env (all required):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
# Optional:
#   N_BOXES (default 20), MAX_DPH (default 0.05 — CPU is cheap),
#   MIN_CORES (default 8), MIN_RAM_GB (default 24),
#   MIN_DISK_GB (default 15),
#   IMAGE (default ghcr.io/imazen/zen-metrics-iwssim:0.6.0),
#   SWEEP_RUN_ID (default sweep-v17-iwssim-<today>), DRY_RUN
set -euo pipefail
source ~/.config/cloudflare/r2-credentials

DEFAULT_DATE=$(date -u +%Y-%m-%d)
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-v17-iwssim-${DEFAULT_DATE}}"
IMAGE="${IMAGE:-ghcr.io/imazen/zen-metrics-iwssim:0.6.0}"
N_BOXES="${N_BOXES:-20}"
MAX_DPH="${MAX_DPH:-0.05}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-24}"
MIN_DISK_GB="${MIN_DISK_GB:-15}"
DRY_RUN="${DRY_RUN:-0}"

GHCR_TOKEN="$(gh auth token)"
GHCR_USER="${GHCR_USER:-lilithriver}"

# `num_gpus=0` filters to CPU-only offers, which are an order of
# magnitude cheaper than the GPU pool.
QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} \
cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} \
num_gpus=0"

echo "[v17/iwssim] sweep_run_id=${SWEEP_RUN_ID}"
echo "[v17/iwssim] image=${IMAGE}"
echo "[v17/iwssim] querying: $QUERY"
OFFERS_JSON=$(vastai search offers "$QUERY" --order 'dph_total' --raw)
OFFER_IDS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
offers = d if isinstance(d, list) else d.get('offers', [])
seen, picked = set(), []
for o in offers:
    mid = o.get('machine_id')
    if mid in seen: continue
    seen.add(mid)
    picked.append(str(o['id']))
    if len(picked) >= int('$N_BOXES'): break
print('\n'.join(picked))
")
n=$(echo "$OFFER_IDS" | wc -w)
echo "[v17/iwssim] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then echo "$OFFER_IDS" | head -5; exit 0; fi
[[ "$n" -lt 3 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

INSTANCE_FILE="/tmp/${SWEEP_RUN_ID}_instances.txt"
> "$INSTANCE_FILE"
i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-w${i}"
    LABEL="zen-v17-iwssim-${i}"
    ENV_STR="-e R2_ACCOUNT_ID=$R2_ACCOUNT_ID \
-e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID \
-e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
-e SWEEP_RUN_ID=$SWEEP_RUN_ID \
-e WORKER_ID=$WORKER_ID \
-e SWEEP_GPU_RUNTIME=cpu"
    LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"
    OUT=$(vastai create instance "$offer_id" \
        --image "$IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "/usr/local/bin/zen-metrics-worker" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        --raw 2>&1) || { echo "  $i fail: $(echo "$OUT" | head -c 200)"; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "  $i parse-fail: $(echo "$OUT" | head -c 200)"; continue; }
    echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
    echo "  $i -> instance $ID"
done
echo
echo "[v17/iwssim] launched $(wc -l < "$INSTANCE_FILE") instances (target $N_BOXES)"
echo "[v17/iwssim] instance file: $INSTANCE_FILE"
