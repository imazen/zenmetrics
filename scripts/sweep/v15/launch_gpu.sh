#!/bin/bash
set -euo pipefail
source ~/.config/cloudflare/r2-credentials
SWEEP_RUN_ID="sweep-v15-2026-05-06"
IMAGE="ghcr.io/imazen/zen-metrics-sweep:0.6.3"
N_BOXES="${N_BOXES:-30}"
MAX_DPH="${MAX_DPH:-0.20}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-12}"
MIN_DISK_GB="${MIN_DISK_GB:-25}"
DRY_RUN="${DRY_RUN:-0}"
GHCR_TOKEN="$(gh auth token)"
GHCR_USER="lilithriver"

# Driver floor 555 = first NVIDIA release shipping CUDA 12.5 ABI. The
# v21 binary is built with CUDARC_CUDA_VERSION=12000 (no CUDA-13 dlsyms)
# but cudarc 0.19.4 emits PTX requiring the CUDA 12.5+ minor version
# directive — drivers <555.42 fail at module load with
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION. See launch_backfill.sh for full
# rationale.
QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_max_good>=12.0 driver_version>=555.0.0 num_gpus=1"
echo "[v15] querying: $QUERY"
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
echo "[v15] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then echo "$OFFER_IDS" | head -5; exit 0; fi
[[ "$n" -lt 5 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

INSTANCE_FILE="/tmp/v15-prep/v15_instances.txt"
> "$INSTANCE_FILE"
i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-w${i}"
    LABEL="zen-v15-${i}"
    ENV_STR="-e SWEEP_BIN_OVERRIDE=s3://coefficient/binaries/zen-metrics-0.6.7-linux-x86_64-gpu -e R2_ACCOUNT_ID=$R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY -e SWEEP_RUN_ID=$SWEEP_RUN_ID -e WORKER_ID=$WORKER_ID -e SWEEP_GPU_RUNTIME=cuda"
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
echo "[v15] launched $(wc -l < "$INSTANCE_FILE") instances (target $N_BOXES)"
