#!/usr/bin/env bash
# datagen_score_launch.sh <codec> — launch a vast.ai GPU box to score one codec's
# persisted variants (SPLIT GPU half). Mints scoped 6h R2 creds, picks a >=12GB
# fast-net reliable offer (>=12GB so the 12MP renditions don't OOM), launches the
# v29-split worker. Sidecars land at <prefix>/<codec>/sidecars/<metric>.parquet.
set -uo pipefail
CODEC="${1:?usage: datagen_score_launch.sh <codec>}"
PREFIX="${PREFIX:-picker-sweep-2026-06-22/datagen-2026-06-23}"
BUCKET="${BUCKET:-codec-corpus}"
METRICS="${METRICS:-butteraugli-gpu,cvvdp,ssim2-gpu,zensim-gpu}"
IMAGE="${IMAGE:-ghcr.io/imazen/zenmetrics-sweep:v29-split}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a

body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':21600,'prefixes':['picker-sweep-2026-06-22/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/dg_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/dg_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { echo "cred mint failed"; cat /tmp/dg_cred.json; exit 1; }

OFFER=$(vastai search offers "reliability>0.98 num_gpus=1 gpu_ram>=12 rentable=true inet_down>300 disk_space>50 cuda_vers>=12.0" --order dph_total --raw 2>/dev/null | python3 -c 'import json,sys
o=json.load(sys.stdin); o=o if isinstance(o,list) else o.get("offers",[])
print(o[0]["id"] if o else "")')
[ -n "${OFFER:-}" ] || { echo "no >=12GB offer found"; exit 1; }
echo "launch $CODEC score on offer $OFFER (metrics=$METRICS, ref=$PREFIX/ref)"
ENVS="-e R2_ACCOUNT_ID=$R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID=$AK -e R2_SECRET_ACCESS_KEY=$SK -e R2_SESSION_TOKEN=$ST -e ZEN_BUCKET=$BUCKET -e ZEN_RUN_PREFIX=$PREFIX/$CODEC -e ZEN_REF_PREFIX=$PREFIX/ref -e ZEN_METRICS=$METRICS"
vastai create instance "$OFFER" --image "$IMAGE" --disk 60 \
  --onstart-cmd "bash /usr/local/bin/split_score_worker.sh > /var/log/split.log 2>&1" \
  --env "$ENVS" --label "dgscore-$CODEC" --raw 2>&1 | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin); print("created id:",d.get("new_contract"),"ok:",d.get("success"))
except Exception: print("raw:",sys.stdin.read()[:200])'
