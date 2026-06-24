#!/usr/bin/env bash
# Launch an N-box GPU fleet to run a declared metric manifest on the zen job system.
# Based on gpu_e2e_proof.sh (the proven CUDA-ready onstart that avoids encoder_panic) but
# scaled to N boxes + a full manifest. The job system handles atomic claim + resume; rerun
# the same RUN to add boxes (they pick up the gap from the ledger).
#   usage: gpu_fleet_score.sh <codec> <manifest_local.json> <N_BOXES> [RUN]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
codec="$1"; MANIFEST_LOCAL="$2"; N="${3:-8}"
IMAGE="${ZEN_GPU_IMAGE:-ghcr.io/imazen/zenfleet-worker-exec-gpu:latest}"
BUCKET="codec-corpus"; DGP="picker-sweep-2026-06-22/datagen-2026-06-23"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
RUN="${4:-datagen-score-$codec-$(date -u +%Y%m%d-%H%M%S)}"
RUNP="jobs/$RUN"; CORPUS_PREFIX="$RUNP/corpus"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
echo "### GPU fleet: $codec  run=s3://$BUCKET/$RUNP/  image=$IMAGE  N=$N"
# scoped creds (run prefix only; corpus is copied under it so one prefix covers everything)
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':21600,'prefixes':['$RUNP/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/fleet_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/fleet_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { echo "cred mint failed"; cat /tmp/fleet_cred.json; exit 1; }
echo "minted scoped creds"
# copy renditions -> run corpus (server-side) so workers resolve cell.image_path
echo "copy 1482 renditions -> $CORPUS_PREFIX/ ..."
r2 cp "s3://$BUCKET/$DGP/ref/*" "s3://$BUCKET/$CORPUS_PREFIX/" >/dev/null 2>&1
r2 cp "$MANIFEST_LOCAL" "s3://$BUCKET/$RUNP/manifest.json" >/dev/null
MANIFEST="s3://$BUCKET/$RUNP/manifest.json"
NJOBS=$(python3 -c "import json;print(len(json.load(open('$MANIFEST_LOCAL'))))")
printf '{"paused":true}' > /tmp/fleet_ctl.json; CTLKEY="$RUNP/control.json"
r2 cp /tmp/fleet_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
echo "manifest=$NJOBS jobs, control paused"
# CUDA-ready onstart (vast runs --onstart-cmd, NOT the ENTRYPOINT; replicate the GPU env)
ONSTART='set +e
export PATH="/usr/local/sbin:/usr/sbin:/sbin:$PATH"
env | grep -E "^(AWS_|ZEN_)" >> /etc/environment
ldconfig 2>/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1
bash /usr/local/bin/fleet-entrypoint.sh 2>&1 | tee /var/log/zenfleet.log
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp /var/log/zenfleet.log "s3://$ZEN_BUCKET/$ZEN_RUN/worker-$ZEN_WORKER.log" 2>&1 | tail -1'
OFFERS=$(vastai search offers 'num_gpus=1 cuda_max_good>=12.6 gpu_ram>=8 disk_space>=30 rentable=true inet_down>200' -o 'dph+' --raw 2>/dev/null)
launched=0
for k in $(seq 1 "$N"); do
  OFFER=$(echo "$OFFERS" | python3 -c "import json,sys;o=json.load(sys.stdin);print(o[$((k-1))]['id'] if len(o)>$((k-1)) else '')")
  [ -z "$OFFER" ] && { echo "no offer for box $k"; continue; }
  ENVB="-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto -e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUNP -e ZEN_MANIFEST_URI=$MANIFEST -e ZEN_PROVIDER=vast-gpu -e ZEN_CORPUS_PREFIX=$CORPUS_PREFIX -e ZEN_CONTROL_KEY=$CTLKEY -e ZEN_IDLE_PASSES=10 -e ZEN_WORKER=vast-gpu-$k"
  vastai create instance "$OFFER" --image "$IMAGE" --label "group=$RUN" --disk 30 --env "$ENVB" --onstart-cmd "$ONSTART" 2>&1 | grep -iE 'new_contract|success' | head -1 && launched=$((launched+1))
done
echo "launched $launched/$N boxes on run $RUN; waiting 240s for boot+pull then RESUME"
sleep 240
printf '{"paused":false}' > /tmp/fleet_ctl.json
r2 cp /tmp/fleet_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
echo "### RESUMED run=$RUN  ledger=s3://$BUCKET/$RUNP/ledger/  (monitor: zenfleet-ctl catalog)"
echo "$RUN" > "/tmp/fleet_run_$codec.txt"
