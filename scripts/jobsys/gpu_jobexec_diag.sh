#!/usr/bin/env bash
# Minimal GPU diagnostic: launch ONE vast GPU box that runs `zenmetrics jobexec` DIRECTLY (not via the
# worker, which swallows the executor's stderr) on a GPU metric, capturing the FULL stdout+stderr to R2.
# This pinpoints WHY metric jobs hit error_class=encoder_panic on a real GPU. Tears nothing down for
# you — print the destroy line at the end. ~$0.05/hr box, a few minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ZEN_GPU_IMAGE:-ghcr.io/imazen/zenfleet-worker-exec-gpu:latest}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
RUN="gpu-diag-$(date -u +%Y%m%d-%H%M%S)"
echo "### GPU jobexec diag run=$RUN image=$IMAGE"

# scoped creds
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':7200,'prefixes':['$RUN/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/diag_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/diag_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')

# upload one corpus image
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp "$ROOT/crates/zenmetrics-corpus/data/source.png" "s3://$BUCKET/$RUN/corpus/img.png" >/dev/null
echo "uploaded corpus img"

OFFER=$(vastai search offers 'num_gpus=1 cuda_max_good>=12.6 gpu_ram>=8 disk_space>=24 rentable=true verified=true inet_down>200' -o 'dph+' --raw 2>/dev/null \
  | python3 -c 'import json,sys;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')
[ -z "$OFFER" ] && { echo "no offer"; exit 1; }

ENVB="-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto \
-e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUN"

# The diagnostic onstart: set GPU-ready env, run jobexec directly on several runtimes, capture EVERYTHING.
ONSTART='set +e
export PATH="/usr/local/sbin:/usr/sbin:/sbin:$PATH"; ldconfig 2>/dev/null
SRC=/tmp/img.png
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "s3://$ZEN_BUCKET/$ZEN_RUN/corpus/img.png" "$SRC"
JOB="{\"kind\":{\"kind\":\"metric\",\"metric\":\"ssim2-gpu\"},\"inputs\":[\"aa\"],\"cell\":{\"image_path\":\"$SRC\",\"codec\":\"zenjpeg\",\"q\":80,\"knob_tuple_json\":\"{}\"}}"
LOG=/tmp/diag.log
{
  echo "=== uname / nvidia-smi ==="; uname -a; nvidia-smi 2>&1 | head -15
  echo "=== ldconfig libcuda ==="; ldconfig -p | grep -iE "libcuda\.so|libnvcuda" | head
  echo "=== ENV (cuda/ld/preload) ==="; env | grep -iE "LD_|CUDA|GPU" | sort
  echo; echo "=== jobexec ssim2-gpu (default Auto runtime) ==="
  echo "$JOB" | RUST_BACKTRACE=1 zenmetrics jobexec 2>&1; echo "[exit=$?]"
  echo; echo "=== zenmetrics score CLI ssim2-gpu --gpu-runtime cuda (encode first) ==="
  zenmetrics --version 2>&1
  echo; echo "=== list-metrics ==="; zenmetrics list-metrics 2>&1 | head -20
  echo; echo "=== CPU-metric jobexec (ssim2) to isolate encode-vs-GPU ==="
  echo "${JOB/ssim2-gpu/ssim2}" | RUST_BACKTRACE=1 zenmetrics jobexec 2>&1; echo "[exit=$?]"
} > "$LOG" 2>&1
cat "$LOG"
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "$LOG" "s3://$ZEN_BUCKET/$ZEN_RUN/diag.log"'

LOGIN_ARG=()
[ -n "${GHCR_TOKEN:-}" ] && LOGIN_ARG=(--login "-u ${GHCR_USER:-lilith} -p ${GHCR_TOKEN} ghcr.io")
vastai create instance "$OFFER" --image "$IMAGE" "${LOGIN_ARG[@]}" \
  --label "group=$RUN" --disk 24 --env "$ENVB" --onstart-cmd "$ONSTART" 2>&1 | tee /tmp/diag_create.json
echo "$RUN" > /tmp/diag_run.txt
echo "### created on offer $OFFER, run=$RUN. Diag log -> s3://$BUCKET/$RUN/diag.log"
echo "### read:  s5cmd --endpoint-url $EP cat s3://$BUCKET/$RUN/diag.log"
