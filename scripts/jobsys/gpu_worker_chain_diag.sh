#!/usr/bin/env bash
# Isolate WHY the worker chain (zenfleet-worker -> zenfleet-exec -> zenmetrics jobexec) hits
# encoder_panic on a real GPU while DIRECT `zenmetrics jobexec` scores fine (proven by gpu_jobexec_diag).
# Runs three escalating tests on ONE GPU box, capturing all stderr to R2:
#   A. direct jobexec               (expected: works)
#   B. via the zenfleet-exec shim   (adds the /bin/sh exec wrapper)
#   C. full zenfleet-worker on a 1-job manifest (adds the Command-spawn + piped stdio)
# Whichever first fails localizes the regression to that wrapper layer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ZEN_GPU_IMAGE:-ghcr.io/imazen/zenfleet-worker-exec-gpu:latest}"
BUCKET="${ZEN_FLEET_BUCKET:-coefficient}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
RUN="gpu-chain-$(date -u +%Y%m%d-%H%M%S)"
echo "### worker-chain diag run=$RUN"

body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':7200,'prefixes':['$RUN/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/chain_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/chain_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp "$ROOT/crates/zenmetrics-corpus/data/source.png" "s3://$BUCKET/$RUN/corpus/img.png" >/dev/null

OFFER=$(vastai search offers 'num_gpus=1 cuda_max_good>=12.6 gpu_ram>=8 disk_space>=24 rentable=true verified=true inet_down>200' -o 'dph+' --raw 2>/dev/null \
  | python3 -c 'import json,sys;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')
[ -z "$OFFER" ] && { echo "no offer"; exit 1; }
ENVB="-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto \
-e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUN"

ONSTART='set +e
export PATH="/usr/local/sbin:/usr/sbin:/sbin:$PATH"; ldconfig 2>/dev/null
SRC=/tmp/img.png
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "s3://$ZEN_BUCKET/$ZEN_RUN/corpus/img.png" "$SRC"
JOB="{\"kind\":{\"kind\":\"metric\",\"metric\":\"ssim2-gpu\"},\"inputs\":[\"aa\"],\"cell\":{\"image_path\":\"$SRC\",\"codec\":\"zenjpeg\",\"q\":80,\"knob_tuple_json\":\"{}\"}}"
# Build a 1-job manifest for the full worker test. The worker expects an array of DesiredJob with a job_id.
JID=$(printf "%s" "metric|ssim2-gpu|$SRC|zenjpeg|80" | sha256sum | cut -d" " -f1)
# Build a VALID 1-job manifest (local source) with python — no fragile string slicing.
JID="$JID" SRC="$SRC" python3 -c '
import json,os
print(json.dumps([{"job_id":os.environ["JID"],"kind":{"kind":"metric","metric":"ssim2-gpu"},
  "inputs":["aa"],"cell":{"image_path":os.environ["SRC"],"codec":"zenjpeg","q":80,"knob_tuple_json":"{}"}}]))' > /tmp/m.json
# Build a VALID 1-job manifest (R2 source via ZEN_CORPUS_PREFIX) — THIS mirrors the real e2e, where
# jobexec resolve_source() fetches the image from R2 with s5cmd using the inherited AWS_* env.
JID2=$(printf "%s" "metric|ssim2-gpu|corpus/img.png|zenjpeg|80" | sha256sum | cut -d" " -f1)
JID2="$JID2" python3 -c '
import json,os
print(json.dumps([{"job_id":os.environ["JID2"],"kind":{"kind":"metric","metric":"ssim2-gpu"},
  "inputs":["aa"],"cell":{"image_path":"corpus/img.png","codec":"zenjpeg","q":80,"knob_tuple_json":"{}"}}]))' > /tmp/m_r2.json
LOG=/tmp/chain.log
{
  echo "### A. DIRECT jobexec (local src)"; echo "$JOB" | RUST_BACKTRACE=1 zenmetrics jobexec 2>&1 | head -6; echo "[A exit=$?]"
  echo; echo "### B. via zenfleet-exec SHIM (local src, ZEN_EXEC=$ZEN_EXEC)"
  echo "$JOB" | RUST_BACKTRACE=1 /usr/local/bin/zenfleet-exec 2>&1 | head -25; echo "[B exit=$?]"
  echo; echo "### C. full zenfleet-worker (LOCAL src, the real fleet path)"
  RUST_BACKTRACE=1 zenfleet-worker --manifest /tmp/m.json \
    --ledger-out "s3://$ZEN_BUCKET/$ZEN_RUN/ledger/c.parquet" \
    --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "$ZEN_RUN/blobs" \
    --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "$ZEN_RUN/claims-c" \
    --r2-endpoint "$ZEN_R2_ENDPOINT" --exec /usr/local/bin/zenfleet-exec \
    --worker chaintest-local --provider diag 2>&1 | head -25; echo "[C exit=$?]"
  echo; echo "### D. full zenfleet-worker (R2 src via ZEN_CORPUS_PREFIX — EXACTLY the e2e path)"
  echo "    setting ZEN_CORPUS_PREFIX=$ZEN_RUN/corpus so jobexec fetches corpus/img.png from R2"
  ZEN_CORPUS_PREFIX="$ZEN_RUN/corpus" RUST_BACKTRACE=1 zenfleet-worker --manifest /tmp/m_r2.json \
    --ledger-out "s3://$ZEN_BUCKET/$ZEN_RUN/ledger/d.parquet" \
    --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "$ZEN_RUN/blobs" \
    --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "$ZEN_RUN/claims-d" \
    --r2-endpoint "$ZEN_R2_ENDPOINT" --exec /usr/local/bin/zenfleet-exec \
    --worker chaintest-r2 --provider diag 2>&1 | head -25; echo "[D exit=$?]"
} > "$LOG" 2>&1
cat "$LOG"
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "$LOG" "s3://$ZEN_BUCKET/$ZEN_RUN/chain.log"'

LOGIN_ARG=()
[ -n "${GHCR_TOKEN:-}" ] && LOGIN_ARG=(--login "-u ${GHCR_USER:-lilith} -p ${GHCR_TOKEN} ghcr.io")
vastai create instance "$OFFER" --image "$IMAGE" "${LOGIN_ARG[@]}" \
  --label "group=$RUN" --disk 24 --env "$ENVB" --onstart-cmd "$ONSTART" 2>&1 | tee /tmp/chain_create.json
echo "$RUN" > /tmp/chain_run.txt
echo "### run=$RUN -> chain.log at s3://$BUCKET/$RUN/chain.log"
