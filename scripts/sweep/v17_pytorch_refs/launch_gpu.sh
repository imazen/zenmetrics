#!/bin/bash
# launch_gpu.sh — fleet launcher for cvvdp sweep chunks on vast.ai.
#
# Pairs with ghcr.io/imazen/zen-metrics-cvvdp:<ver> (built from
# docker/Dockerfile.cvvdp). Picks single-GPU NVIDIA boxes with ≥12 GB
# VRAM at ≤$0.30/hr — cvvdp's display model needs the headroom on 4K
# frames but doesn't need a datacenter card.
#
# Chunks consumed by this fleet MUST set `.metrics` to ["cvvdp"] (or
# include it alongside others). The onstart loop is unchanged from v15.
#
# Env (all required):
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
# Optional:
#   N_BOXES (default 20), MAX_DPH (default 0.30), MIN_CORES (default 8),
#   MIN_RAM_GB (default 24), MIN_DISK_GB (default 25),
#   MIN_VRAM_GB (default 12), MIN_CUDA (default 12),
#   IMAGE (default ghcr.io/imazen/zen-metrics-cvvdp:0.6.0),
#   SWEEP_RUN_ID (default sweep-v17-cvvdp-<today>), DRY_RUN
set -euo pipefail
source ~/.config/cloudflare/r2-credentials

DEFAULT_DATE=$(date -u +%Y-%m-%d)
SWEEP_RUN_ID="${SWEEP_RUN_ID:-sweep-v17-cvvdp-${DEFAULT_DATE}}"
IMAGE="${IMAGE:-ghcr.io/imazen/zen-metrics-cvvdp:0.6.0}"
N_BOXES="${N_BOXES:-20}"
MAX_DPH="${MAX_DPH:-0.30}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-24}"
MIN_DISK_GB="${MIN_DISK_GB:-25}"
MIN_VRAM_GB="${MIN_VRAM_GB:-12}"
MIN_CUDA="${MIN_CUDA:-12}"
DRY_RUN="${DRY_RUN:-0}"

GHCR_TOKEN="$(gh auth token)"
GHCR_USER="${GHCR_USER:-lilithriver}"

QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} \
cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} \
cuda_max_good>=${MIN_CUDA} num_gpus=1 gpu_ram>=${MIN_VRAM_GB}"

echo "[v17/cvvdp] sweep_run_id=${SWEEP_RUN_ID}"
echo "[v17/cvvdp] image=${IMAGE}"
echo "[v17/cvvdp] querying: $QUERY"
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
echo "[v17/cvvdp] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then echo "$OFFER_IDS" | head -5; exit 0; fi
[[ "$n" -lt 3 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

INSTANCE_FILE="/tmp/${SWEEP_RUN_ID}_instances.txt"
> "$INSTANCE_FILE"
i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-w${i}"
    LABEL="zen-v17-cvvdp-${i}"
    # ZEN_METRICS_EXTERNAL_CVVDP is already baked into the image env.
    # We still pass SWEEP_GPU_RUNTIME=cpu because no in-process GPU
    # metrics run — the pycvvdp subprocess manages its own CUDA context.
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
echo "[v17/cvvdp] launched $(wc -l < "$INSTANCE_FILE") instances (target $N_BOXES)"
echo "[v17/cvvdp] instance file: $INSTANCE_FILE"
