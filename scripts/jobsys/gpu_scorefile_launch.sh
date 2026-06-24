#!/usr/bin/env bash
# Launch an N-box GPU fleet to run a ScoreFile manifest: per-chunk, NO-RE-ENCODE scoring of persisted
# variants. jobexec decodes the reference once, byte-range-fetches each pre-encoded variant out of the
# existing variants.tar (via the sha->offset index), and scores all 6 metrics + the 372-feature zensim
# sidecar. The manifest + variant_index.tsv are pre-built by build_scorefile_manifest.py (so this
# launcher does NOT re-upload anything). Streams progress to /tmp/scorefile_launch.log.
#   usage: gpu_scorefile_launch.sh <run_id> <codec_dir> <N_boxes>
set -uo pipefail
RUN="${1:?run id}"; CODEC="${2:-zenavif}"; N="${3:-8}"
SCALE="${ZEN_SCALE:-0}"   # ZEN_SCALE=1: add boxes to a LIVE run — skip the pause/240s-sleep/resume stall
DGP="${ZEN_DATAGEN_PREFIX:-picker-sweep-2026-06-22/datagen-2026-06-23}"
IMAGE="${ZEN_GPU_IMAGE:-ghcr.io/imazen/zenfleet-worker-exec-gpu:latest}"
BUCKET=codec-corpus; RUNP="jobs/$RUN"
LOG=/tmp/scorefile_launch.log; : > "$LOG"; log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
# scoped creds: rw on the run prefix + read on the datagen prefix (ref renditions + variants.tar)
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':28800,'prefixes':['$RUNP/','$DGP/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/sf_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/sf_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { log "cred mint failed"; cat /tmp/sf_cred.json | tee -a "$LOG"; exit 1; }
MANIFEST="s3://$BUCKET/$RUNP/manifest.json"; CTLKEY="$RUNP/control.json"
TAR="s3://$BUCKET/$DGP/$CODEC/variants.tar"; IDX="s3://$BUCKET/$RUNP/variant_index.tsv"
NJOBS=$(r2 cat "$MANIFEST" 2>/dev/null | python3 -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
if [ "$SCALE" = "0" ]; then
  printf '{"paused":true}' > /tmp/sf_ctl.json; r2 cp /tmp/sf_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
  log "run=$RUN codec=$CODEC jobs=$NJOBS image=$IMAGE; control paused; launching $N boxes"
else
  log "run=$RUN codec=$CODEC jobs=$NJOBS image=$IMAGE; SCALE mode (run already live, no pause); launching $N more boxes"
fi
ONSTART='set +e
export PATH="/usr/local/sbin:/usr/sbin:/sbin:$PATH"
env | grep -E "^(AWS_|ZEN_)" >> /etc/environment
ldconfig 2>/dev/null
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1
bash /usr/local/bin/fleet-entrypoint.sh 2>&1 | tee /var/log/zenfleet.log
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp /var/log/zenfleet.log "s3://$ZEN_BUCKET/$ZEN_RUN/worker-$ZEN_WORKER.log" 2>&1 | tail -1'
OFFERS=$(vastai search offers "reliability>0.98 num_gpus=1 gpu_ram>=12 rentable=true inet_down>300 disk_space>40 cuda_max_good>=12.6" --order dph_total --raw 2>/dev/null)
launched=0
for k in $(seq 1 "$N"); do
  OFFER=$(echo "$OFFERS" | python3 -c "import json,sys;o=json.load(sys.stdin);o=o if isinstance(o,list) else o.get('offers',[]);print(o[$((k-1))]['id'] if len(o)>$((k-1)) else '')")
  [ -z "$OFFER" ] && { log "offer pool exhausted at $((k-1))"; break; }
  ENVB="-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto -e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUNP -e ZEN_MANIFEST_URI=$MANIFEST -e ZEN_CONTROL_KEY=$CTLKEY -e ZEN_CORPUS_PREFIX=$DGP/ref -e ZEN_VARIANTS_TAR_URI=$TAR -e ZEN_VARIANT_INDEX_URI=$IDX -e ZEN_PERSISTENT_EXEC=1 -e ZEN_PROVIDER=vast-gpu -e ZEN_IDLE_PASSES=10 -e ZEN_WORKER=sf-$k"
  if timeout 40 vastai create instance "$OFFER" --image "$IMAGE" --label "group=$RUN" --disk 40 --env "$ENVB" --onstart-cmd "$ONSTART" 2>&1 | grep -iqE 'new_contract|success'; then
    launched=$((launched + 1)); log "launched box $launched/$N (offer $OFFER)"
  else log "create failed offer $OFFER"; fi
done
if [ "$SCALE" = "0" ]; then
  log "launched $launched; waiting 240s for boot+pull then RESUME"
  sleep 240
  printf '{"paused":false}' > /tmp/sf_ctl.json; r2 cp /tmp/sf_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
  log "### RESUMED run=$RUN (blobs: s3://$BUCKET/$RUNP/blobs/ — JSONL score+feature rows)"
else
  log "### SCALED +$launched boxes onto live run=$RUN (they join as they boot; no pause/stall)"
fi
