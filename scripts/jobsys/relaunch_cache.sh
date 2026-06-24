#!/usr/bin/env bash
# Relaunch the avif scoring run on the cache image: destroy the 8 old (polluted/slow) boxes, launch N
# persistent+cache boxes. The job system resumes from the gap (done blobs kept). Streams progress to a
# log so the launch can't stall silently. Each vastai call is timeout-guarded so one hang can't block.
#   usage: relaunch_cache.sh [RUN] [N]
set -uo pipefail
RUN="${1:-datagen-score-zenavif-20260623-224512}"; N="${2:-24}"
RUNP="jobs/$RUN"; IMAGE=ghcr.io/imazen/zenfleet-worker-exec-gpu:latest; BUCKET=codec-corpus
LOG=/tmp/relaunch_cache.log; : > "$LOG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
log "destroying old boxes (exact label group=$RUN)"
oldids=$(timeout 40 vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);i=d if isinstance(d,list) else d.get('instances',[]);print(' '.join(str(x['id']) for x in i if x.get('label')=='group=$RUN'))")
# `vastai destroy` prompts [y/N]; pipe y (a bare stdin redirect defaults to N and silently no-ops).
for id in $oldids; do echo y | timeout 30 vastai destroy instance "$id" >/dev/null 2>&1 && log "destroyed $id"; done
log "minting scoped creds (8h)"
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':28800,'prefixes':['$RUNP/']}))")
timeout 30 curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/scale_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/scale_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { log "cred mint FAILED"; cat /tmp/scale_cred.json | tee -a "$LOG"; exit 1; }
MANIFEST="s3://$BUCKET/$RUNP/manifest.json"; CORPUS_PREFIX="$RUNP/corpus"; CTLKEY="$RUNP/control.json"
ONSTART='set +e
export PATH="/usr/local/sbin:/usr/sbin:/sbin:$PATH"
env | grep -E "^(AWS_|ZEN_)" >> /etc/environment
ldconfig 2>/dev/null
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1
bash /usr/local/bin/fleet-entrypoint.sh 2>&1 | tee /var/log/zenfleet.log
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp /var/log/zenfleet.log "s3://$ZEN_BUCKET/$ZEN_RUN/worker-$ZEN_WORKER.log" 2>&1 | tail -1'
log "searching offers"
OFFERS=$(timeout 40 vastai search offers 'num_gpus=1 cuda_max_good>=12.6 gpu_ram>=8 disk_space>=30 rentable=true inet_down>200' -o 'dph+' --raw 2>/dev/null)
NAVAIL=$(echo "$OFFERS" | python3 -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
log "offers available: $NAVAIL ; launching up to $N"
launched=0
for k in $(seq 1 "$N"); do
  OFFER=$(echo "$OFFERS" | python3 -c "import json,sys;o=json.load(sys.stdin);print(o[$((k-1))]['id'] if len(o)>$((k-1)) else '')")
  [ -z "$OFFER" ] && { log "offer pool exhausted at $((k-1))"; break; }
  ENVB="-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto -e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUNP -e ZEN_MANIFEST_URI=$MANIFEST -e ZEN_PROVIDER=vast-gpu -e ZEN_CORPUS_PREFIX=$CORPUS_PREFIX -e ZEN_CONTROL_KEY=$CTLKEY -e ZEN_IDLE_PASSES=10 -e ZEN_PERSISTENT_EXEC=1 -e ZEN_WORKER=vast-cache-$k"
  if timeout 40 vastai create instance "$OFFER" --image "$IMAGE" --label "group=$RUN-persist" --disk 30 --env "$ENVB" --onstart-cmd "$ONSTART" 2>&1 | grep -iqE 'new_contract|success'; then
    launched=$((launched+1)); log "launched cache box $launched/$N (offer $OFFER)"
  else
    log "create failed offer $OFFER"
  fi
done
log "DONE: launched $launched cache boxes on $RUN (control already paused:false; they claim the gap when ready)"
