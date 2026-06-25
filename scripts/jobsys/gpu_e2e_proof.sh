#!/usr/bin/env bash
# End-to-end proof of the JOB-SYSTEM GPU-metric path on ONE vast.ai GPU box.
#
# This is the small, deliberate proof that closes the gap which forced ad-hoc SPLIT scoring:
# the standard exec image's base is non-CUDA, so GPU metrics fail. This drives the new
# ghcr.io/imazen/zenfleet-worker-exec-gpu image (CUDA + GPU zenmetrics jobexec + the worker
# loop) through the real job system: declare metric cells -> R2 lease queue -> ONE GPU worker
# claims + scores on the GPU -> DONE rows with real GPU scores land in the R2 ledger.
#
# It is NOT a fleet. One GPU box, ~4 cells, torn down by you after (TEARDOWN line printed).
#
# Env knobs:
#   ZEN_GPU_IMAGE   (default ghcr.io/imazen/zenfleet-worker-exec-gpu:latest)
#   GHCR_USER/GHCR_TOKEN — if the image is still `internal` on ghcr (not yet flipped public),
#                          pass a READ-ONLY ghcr token so vast can pull it via `--login`.
#                          Omit once the package is public (then no creds touch the box).
#   ZEN_FLEET_BUCKET (default zentrain)
#   ZEN_BOOT_WAIT_SECS (default 240)  — GPU boxes pull a ~950 MB image; give them time.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ZEN_GPU_IMAGE:-ghcr.io/imazen/zenfleet-worker-exec-gpu:latest}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"
CORPUS_DIR="${ZEN_E2E_CORPUS_DIR:-/tmp/gpu-e2e-corpus}"
SPEC="${ZEN_E2E_SPEC:-/tmp/gpu-e2e-spec.json}"
WAIT="${ZEN_BOOT_WAIT_SECS:-240}"

[ -f "$SPEC" ] || { echo "spec $SPEC missing"; exit 1; }
ls "$CORPUS_DIR"/*.png >/dev/null 2>&1 || { echo "no corpus pngs in $CORPUS_DIR"; exit 1; }

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
RUN="gpu-e2e-$(date -u +%Y%m%d-%H%M%S)"
CORPUS_PREFIX="$RUN/corpus"
echo "### GPU job-system e2e on s3://$BUCKET/$RUN/  image=$IMAGE"

# 1) scoped temp creds (object-read-write to THIS run only; never the root key on the box)
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':10800,'prefixes':['$RUN/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/gpu_e2e_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/gpu_e2e_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
echo "minted scoped creds (3h)"

# 2) upload the corpus under $RUN/corpus/ (root creds, workstation-only) so jobexec resolves
#    cell.image_path=corpus/img-XXXXX.png to s3://$BUCKET/$CORPUS_PREFIX/<image_path>
for f in "$CORPUS_DIR"/*.png; do
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
    s5cmd --endpoint-url "$EP" cp "$f" "s3://$BUCKET/$CORPUS_PREFIX/$(basename "$f")" >/dev/null
done
echo "uploaded $(ls "$CORPUS_DIR"/*.png | wc -l) corpus images under $CORPUS_PREFIX/"

# 3) declare the metric manifest and upload it
"$ROOT/target/release/zenfleet-ctl" declare --spec "$SPEC" --out /tmp/gpu_e2e_manifest.json
NJOBS=$(python3 -c 'import json;print(len(json.load(open("/tmp/gpu_e2e_manifest.json"))))')
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp /tmp/gpu_e2e_manifest.json "s3://$BUCKET/$RUN/manifest.json" >/dev/null
MANIFEST="s3://$BUCKET/$RUN/manifest.json"
echo "declared + uploaded $NJOBS-job manifest"

# 4) start the run PAUSED so the box idles on the control while it boots+pulls, then resume.
CTLKEY="$RUN/control.json"; printf '{"paused":true}' > /tmp/gpu_e2e_ctl.json
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp /tmp/gpu_e2e_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null

# 5) pick a GPU offer (cuda_max_good>=12.6 so the CUDA-12.6 runtime in the image matches) and create it.
OFFER=$(vastai search offers 'num_gpus=1 cuda_max_good>=12.6 gpu_ram>=8 disk_space>=24 rentable=true verified=true inet_down>200' -o 'dph+' --raw 2>/dev/null \
  | python3 -c 'import json,sys;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')
[ -z "$OFFER" ] && { echo "no GPU offer matched"; exit 1; }
echo "selected GPU offer $OFFER"

ENVBLOCK="-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto \
-e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUN -e ZEN_MANIFEST_URI=$MANIFEST \
-e ZEN_PROVIDER=vast-gpu -e ZEN_CORPUS_PREFIX=$CORPUS_PREFIX -e ZEN_CONTROL_KEY=$CTLKEY \
-e ZEN_IDLE_PASSES=8 -e ZEN_WORKER=vast-gpu-1"

# vast runs --onstart-cmd, NOT the image ENTRYPOINT. So we replicate the GPU-ready environment the
# image's own entrypoint (run_with_error_trap.sh) sets up before invoking fleet-entrypoint.sh:
#   - put /sbin on PATH and run `ldconfig` so the nvidia-runtime-injected driver libs (libcuda.so.1
#     etc.) are in the loader cache — WITHOUT this, cudarc's dlopen of libcuda fails and every metric
#     job dies with error_class=encoder_panic (observed 2026-06-23: the GPU is healthy at the vast
#     level but the driver isn't discoverable inside a container launched via a bare --onstart-cmd).
#   - a loud nvidia-smi + libcuda preflight whose output goes to vast's captured stdout (readable via
#     `vastai logs`) AND is uploaded to R2 (sidecars/gpu_preflight.txt) so it's diagnosable without SSH.
# Everything streams to /var/log/zenfleet.log AND stdout so both `vastai logs` and the R2 copy work.
ONSTART='set +e
export PATH="/usr/local/sbin:/usr/sbin:/sbin:$PATH"
env | grep -E "^(AWS_|ZEN_)" >> /etc/environment
ldconfig 2>/dev/null
{ echo "=== GPU preflight $(date -u) ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -2
  echo "--- libcuda in loader cache: ---"; ldconfig -p | grep -iE "libcuda\.so|libnvcuda" | head
  echo "--- libnvrtc/libcudart: ---"; ldconfig -p | grep -iE "libnvrtc\.so|libcudart\.so" | head
} > /tmp/gpu_preflight.txt 2>&1
cat /tmp/gpu_preflight.txt
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp /tmp/gpu_preflight.txt "s3://$ZEN_BUCKET/$ZEN_RUN/sidecars/gpu_preflight.txt" 2>&1 | tail -1
bash /usr/local/bin/fleet-entrypoint.sh 2>&1 | tee /var/log/zenfleet.log
s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp /var/log/zenfleet.log "s3://$ZEN_BUCKET/$ZEN_RUN/sidecars/zenfleet.log" 2>&1 | tail -1'

LOGIN_ARG=()
if [ -n "${GHCR_TOKEN:-}" ]; then
  LOGIN_ARG=(--login "-u ${GHCR_USER:-lilith} -p ${GHCR_TOKEN} ghcr.io")
  echo "passing ghcr --login (image still internal); flip the package public to drop this"
fi

vastai create instance "$OFFER" --image "$IMAGE" "${LOGIN_ARG[@]}" \
  --label "group=$RUN" --disk 24 --env "$ENVBLOCK" --onstart-cmd "$ONSTART" 2>&1 | tee /tmp/gpu_e2e_create.json
echo "created GPU instance on offer $OFFER (run=$RUN)"

# 6) wait for boot+pull, then RESUME.
echo "waiting ${WAIT}s for GPU box boot + ~950MB image pull, then RESUME…"
sleep "$WAIT"
printf '{"paused":false}' > /tmp/gpu_e2e_ctl.json
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp /tmp/gpu_e2e_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
echo "### RESUMED — RUN=$RUN"
echo "$RUN" > /tmp/gpu_e2e_run.txt
echo "creds: AK=$AK SK=$SK ST(len)=${#ST}  EP=$EP  BUCKET=$BUCKET" > /tmp/gpu_e2e_runinfo.txt
echo "TEARDOWN:  vastai destroy instance \$(vastai show instances --raw | python3 -c 'import json,sys;[print(i[\"id\"]) for i in json.load(sys.stdin) if i.get(\"label\")==\"group=$RUN\"]')"
