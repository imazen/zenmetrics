#!/usr/bin/env bash
# Launch Hetzner CX/CPX (cheap shared-vCPU) boxes to run a ScoreFile manifest — the CPU counterpart
# of gpu_scorefile_launch.sh for decode-bound fills (avif decode dominates; GPUs sat idle on vast).
# Same job-system contract: workers claim chunk jobs off jobs/<run>/manifest.json via R2-lease,
# byte-range-fetch variants out of the existing tar via variant_index.tsv, score CPU metrics, write
# blobs/. Boxes are cloud-init docker-ce; the worker is the baked CPU exec image (fleet.env).
#   usage: hetzner_scorefile_launch.sh <run_id> <N_boxes>
#   env:   ZEN_TAR_OVERRIDE (required — tar URI), ZEN_CORPUS_PREFIX_OVERRIDE (refs prefix on
#          codec-corpus), TYPES ("cx53 cx43 cpx62 cpx52"), LOCATIONS ("fsn1 nbg1 hel1"), RESUME=1 (skip
#          pause/resume dance — run already live)
set -uo pipefail
RUN="${1:?run id}"; N="${2:-2}"
. "$(dirname "$0")/fleet.env"
IMAGE="${ZEN_CPU_IMAGE:-$ZEN_FLEET_IMAGE_CPU}"
BUCKET=codec-corpus; RUNP="jobs/$RUN"
TAR="${ZEN_TAR_OVERRIDE:?set ZEN_TAR_OVERRIDE}"
CORPUS_PREFIX="${ZEN_CORPUS_PREFIX_OVERRIDE:?set ZEN_CORPUS_PREFIX_OVERRIDE}"
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
LOG=/tmp/hz_scorefile_launch.log; log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
# scoped RW creds: run prefix + datagen (tar) + corpus refs — 12h (CPU fills run long)
TARPFX=$(python3 -c "import sys;u='$TAR';print('/'.join(u.split('/')[3:-1])+'/')")
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':43200,'prefixes':['$RUNP/','$TARPFX','$CORPUS_PREFIX/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/hzsf_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/hzsf_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { log "cred mint failed"; cat /tmp/hzsf_cred.json | tee -a "$LOG"; exit 1; }
MANIFEST="s3://$BUCKET/$RUNP/manifest.json"; CTLKEY="$RUNP/control.json"
IDX="s3://$BUCKET/$RUNP/variant_index.tsv"
if [ "${RESUME:-0}" = "0" ]; then
  printf '{"paused":false}' > /tmp/hzsf_ctl.json; r2 cp /tmp/hzsf_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
fi
NJOBS=$(r2 cat "$MANIFEST" 2>/dev/null | python3 -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
log "run=$RUN jobs=$NJOBS image=$IMAGE boxes=$N types=${TYPES:-cx53 cx43 cpx62 cpx52}"
launch_box(){
  local wk="$1" name="hzsf-${RUN//[^a-zA-Z0-9-]/}-$1" ci typ loc ok=0 err
  ci=$(mktemp)
  cat > "$ci" <<EOF
#cloud-config
runcmd:
  - |
    docker run -d --name zensf --restart unless-stopped \
      -e AWS_ACCESS_KEY_ID='$AK' -e AWS_SECRET_ACCESS_KEY='$SK' -e AWS_SESSION_TOKEN='$ST' -e AWS_REGION=auto \
      -e ZEN_R2_ENDPOINT='$EP' -e ZEN_BUCKET='$BUCKET' -e ZEN_RUN='$RUNP' \
      -e ZEN_MANIFEST_URI='$MANIFEST' -e ZEN_CONTROL_KEY='$CTLKEY' \
      -e ZEN_CORPUS_PREFIX='$CORPUS_PREFIX' -e ZEN_VARIANTS_TAR_URI='$TAR' -e ZEN_VARIANT_INDEX_URI='$IDX' \
      -e ZEN_PERSISTENT_EXEC=1 -e ZEN_PROVIDER=hetzner-cpu -e ZEN_IDLE_PASSES=10 -e ZEN_WORKER='hz-$wk' \
      --entrypoint /usr/local/bin/fleet-entrypoint.sh '$IMAGE'
EOF
  for typ in ${TYPES:-cx53 cx43 cpx62 cpx52}; do
    for loc in ${LOCATIONS:-fsn1 nbg1 hel1}; do
      err=$(hcloud server create --name "$name" --type "$typ" --image docker-ce --location "$loc" \
        --ssh-key "$SSH_KEY" --label group="$RUN" --user-data-from-file "$ci" 2>&1) \
        && { log "$name launched ($typ/$loc)"; ok=1; break 2; } || true
    done
  done
  [ "$ok" = 1 ] || { log "$name FAILED all type/loc: $(echo "$err" | head -2)"; }
  rm -f "$ci"
}
for w in $(seq 1 "$N"); do launch_box "$w" & done
wait
log "### launched $N boxes onto run=$RUN (blobs: s3://$BUCKET/$RUNP/blobs/)"
log "teardown: hcloud server list -l group=$RUN -o noheader | awk '{print \$2}' | xargs -r -n1 hcloud server delete --poll-interval 5s"
