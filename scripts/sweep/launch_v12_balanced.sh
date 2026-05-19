#!/usr/bin/env bash
# v12 sweep launcher — class-balanced 200-img unified sweep (jxl/avif/webp).
#
# Goal: test the cross-class generalization of the meta-picker. Uses 50 imgs
# per content class (gen-screen / gen-doc / gen-chart / gen-line) from the
# rebalanced corpus, sweeping a tight grid for ~30-min completion at 50+
# parallel workers.
#
# Prerequisites:
#   - GHCR credentials in `gh auth token` for the private docker image
#   - VAST_API_KEY in ~/.config/vastai/vast_api_key
#   - R2 credentials in ~/.config/cloudflare/r2-credentials
#   - 200 source PNGs at /tmp/v12-sweep-sources/ (or override with SOURCES_DIR)
#   - jobspec at /tmp/v12_chunks.jsonl (or override with CHUNKS_FILE)
#
# Usage:
#   scripts/sweep/launch_v12_balanced.sh [N_WORKERS]
#
# After completion: chunks land at s3://zentrain/sweep-v12-2026-05-06/<codec>/

set -euo pipefail
N_WORKERS=${1:-100}
SWEEP_RUN_ID="sweep-v12-2026-05-06"

set -a; source $HOME/.config/cloudflare/r2-credentials; set +a
export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key)
GH_TOKEN=$(gh auth token)

# Upload jobspec + sources
aws --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" s3 cp \
  /tmp/v12_chunks.jsonl "s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl" --quiet
aws --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" s3 cp \
  /tmp/v12-sweep-sources/ "s3://zentrain/${SWEEP_RUN_ID}/sources/" --recursive --quiet

# Find offers
OFFER_IDS=$(vastai search offers \
  'cpu_cores>=8 cpu_ram>16 dph_total<0.20 inet_down>50 reliability>0.97' \
  --order dph_total --raw 2>/dev/null | \
  python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
offers = d if isinstance(d, list) else d.get('offers', [])
seen = set(); picked = []
for o in offers:
    mid = o.get('machine_id')
    if mid in seen: continue
    seen.add(mid); picked.append(str(o['id']))
    if len(picked) >= ${N_WORKERS}: break
print(' '.join(picked))
")

INSTANCE_FILE="/tmp/${SWEEP_RUN_ID}_instances.txt"
> "$INSTANCE_FILE"

ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID} -e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID} -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY} -e SWEEP_RUN_ID=${SWEEP_RUN_ID} -e SWEEP_GPU_RUNTIME=cpu"

i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-w${i}"
    (
      OUT=$(vastai create instance "$offer_id" \
        --image "ghcr.io/imazen/zen-metrics-sweep:0.6.1" \
        --login "-u lilithriver -p ${GH_TOKEN} ghcr.io" \
        --disk 30 --label "$WORKER_ID" \
        --env "$ENV_STR -e WORKER_ID=${WORKER_ID}" \
        --onstart-cmd "/usr/local/bin/zen-metrics-worker" \
        --raw 2>&1)
      ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
      [[ -n "$ID" ]] && echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
    ) &
    if (( i % 20 == 0 )); then wait; fi
done
wait
echo "[v12] launched $(wc -l < $INSTANCE_FILE) workers"
