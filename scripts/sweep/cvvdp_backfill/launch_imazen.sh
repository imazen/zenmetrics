#!/usr/bin/env bash
#
# cvvdp_backfill/launch_imazen.sh — vast.ai fleet launcher for the
# IMAZEN-ONLY variant. Sibling to launch.sh; trades the pycvvdp
# parity column for a working single-image flow that standard
# vast.ai SSH instances can actually run (no Docker-in-Docker
# required).
#
# Architecture:
#   BOOT_IMAGE = ghcr.io/imazen/zen-metrics-sweep:<tag>
#               (has zen-metrics binary + s5cmd + jq baked in)
#   --onstart-cmd = bootstrap that pulls
#               onstart_cvvdp_backfill_imazen.sh from R2 and execs it
#
# The chunk_worker.sh runs in --skip-pycvvdp mode (no PYCVVDP_IMAGE)
# and uses the host zen-metrics binary (no --zen-metrics-image).
# Output is only the cvvdp_imazen_* sidecars; the cvvdp_pycvvdp_v054
# column is absent (finalize.sh tolerates parity=null).
#
# Env vars:
#   SWEEP_RUN_ID            (default: cvvdp-backfill-imazen-<YYYY-MM-DD>)
#   ZEN_METRICS_IMAGE       (default: ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-cuda124)
#                           — also used as BOOT_IMAGE
#   N_BOXES                 (default: 6)
#   MAX_DPH                 (default: 0.30)
#   MIN_CORES               (default: 8)
#   MIN_RAM_GB              (default: 8 — no pytorch load, much lighter)
#   MIN_DISK_GB             (default: 20 — only zen-metrics-sweep ~600 MB)
#   PARALLEL                (default: 2)
#   GPU_RUNTIME             (default: auto)
#   DRY_RUN                 (default: 0)

set -euo pipefail
# shellcheck disable=SC1091
source ~/.config/cloudflare/r2-credentials

SWEEP_RUN_ID="${SWEEP_RUN_ID:-cvvdp-backfill-imazen-$(date -u +%Y-%m-%d)}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-cuda124}"
BOOT_IMAGE="$ZEN_METRICS_IMAGE"

N_BOXES="${N_BOXES:-6}"
MAX_DPH="${MAX_DPH:-0.30}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-8}"
MIN_DISK_GB="${MIN_DISK_GB:-20}"
PARALLEL="${PARALLEL:-2}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DRY_RUN="${DRY_RUN:-0}"

GHCR_TOKEN="$(gh auth token)"
GHCR_USER="${GHCR_USER:-lilithriver}"

echo "[cvvdp-backfill-imazen] launching fleet (single-image mode)"
echo "  SWEEP_RUN_ID:      $SWEEP_RUN_ID"
echo "  BOOT_IMAGE:        $BOOT_IMAGE"
echo "  N_BOXES:           $N_BOXES"
echo "  MAX_DPH:           $MAX_DPH"
echo "  PARALLEL/box:      $PARALLEL"
echo "  GPU_RUNTIME:       $GPU_RUNTIME"
echo

# Pre-flight: chunks.jsonl + chunk_worker.sh + onstart on R2.
SCRIPTS_R2_PREFIX="s3://coefficient/jobs/${SWEEP_RUN_ID}"
echo "[cvvdp-backfill-imazen] verifying $SCRIPTS_R2_PREFIX has chunks.jsonl + chunk_worker.sh"
if ! s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    ls "${SCRIPTS_R2_PREFIX}/" 2>/dev/null | grep -q chunks.jsonl; then
    echo "  ERROR: ${SCRIPTS_R2_PREFIX}/chunks.jsonl missing." >&2
    exit 1
fi
echo "  ok"

# Offer search.
# cuda_vers>=12.5: cubecl-cuda via cudarc 0.19.4 needs the
# cuCoredumpDeregisterCompleteCallback symbol introduced in
# CUDA 12.5. Older filter (cuda_max_good>=12 / >=13) was on the
# GPU's max-supported version not the host driver's installed
# version — admitted GTX 1070/1080/2060 boxes with CUDA-12.0
# host drivers that panicked at score-pairs --metric cvvdp:
#   DlSym { source: "libcuda.so: undefined symbol:
#                    cuCoredumpDeregisterCompleteCallback" }
# cuda_vers is the right filter (machine's max-supported cuda
# based on installed driver version) per vastai docs.
QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_vers>=12.5 num_gpus=1"
echo "[cvvdp-backfill-imazen] querying offers: $QUERY"
OFFERS_JSON=$(vastai search offers "$QUERY" --order 'dph_total' --raw)
OFFER_IDS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
seen = set()
out = []
for o in d:
    mid = o.get('machine_id')
    if mid in seen:
        continue
    seen.add(mid)
    out.append(o['id'])
    if len(out) >= ${N_BOXES}:
        break
print('\n'.join(str(x) for x in out))
")
n=$(echo "$OFFER_IDS" | wc -l)
echo "[cvvdp-backfill-imazen] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "$OFFER_IDS" | head -10
    echo
    echo "DRY_RUN=1: not launching. Re-run with DRY_RUN=0 to commit."
    exit 0
fi
[[ "$n" -lt 3 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

# Upload onstart_cvvdp_backfill_imazen.sh.
ONSTART_R2_KEY="${SCRIPTS_R2_PREFIX}/onstart_cvvdp_backfill_imazen.sh"
echo "[cvvdp-backfill-imazen] uploading onstart to $ONSTART_R2_KEY"
s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    cp scripts/sweep/onstart_cvvdp_backfill_imazen.sh "$ONSTART_R2_KEY"

# Tiny bootstrap: install s5cmd + curl + jq if not present (image
# has them already), download onstart from R2, exec it.
ONSTART_BOOTSTRAP=$(cat <<'BOOT'
set -e
# zen-metrics-sweep image has s5cmd + jq + curl already.
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
mkdir -p ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = $R2_ACCESS_KEY_ID
aws_secret_access_key = $R2_SECRET_ACCESS_KEY
CREDS
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    cp s3://coefficient/jobs/${SWEEP_RUN_ID}/onstart_cvvdp_backfill_imazen.sh \
       /usr/local/bin/onstart.sh
chmod +x /usr/local/bin/onstart.sh
exec /usr/local/bin/onstart.sh
BOOT
)

INSTANCE_FILE="/tmp/cvvdp-backfill-imazen-${SWEEP_RUN_ID}/instances.txt"
mkdir -p "$(dirname "$INSTANCE_FILE")"
: > "$INSTANCE_FILE"

i=0
for offer_id in $OFFER_IDS; do
    i=$((i + 1))
    WORKER_ID="${SWEEP_RUN_ID}-w$i"
    LABEL="$WORKER_ID"

    ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID}"
    ENV_STR+=" -e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
    ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
    ENV_STR+=" -e SWEEP_RUN_ID=${SWEEP_RUN_ID}"
    ENV_STR+=" -e WORKER_ID=${WORKER_ID}"
    ENV_STR+=" -e PARALLEL=${PARALLEL}"
    ENV_STR+=" -e GPU_RUNTIME=${GPU_RUNTIME}"
    ENV_STR+=" -e SCRIPTS_R2_PREFIX=${SCRIPTS_R2_PREFIX}"
    # SWEEP_BIN_OVERRIDE: v15-style pattern. Onstart fetches this
    # URL (s3://… or https://…) at boot and replaces
    # /usr/local/bin/zen-metrics with it. Use when the docker
    # image's baked-in binary has the wrong cudarc feature set
    # (cuCoredumpDeregisterCompleteCallback gated on cuda-13020).
    [[ -n "${SWEEP_BIN_OVERRIDE:-}" ]] && \
        ENV_STR+=" -e SWEEP_BIN_OVERRIDE=${SWEEP_BIN_OVERRIDE}"

    LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"

    OUT=$(vastai create instance "$offer_id" \
        --image "$BOOT_IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "bash -c '$ONSTART_BOOTSTRAP'" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        --raw 2>&1) || { echo "  $i fail: $(echo "$OUT" | head -c 200)"; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "  $i parse-fail: $(echo "$OUT" | head -c 200)"; continue; }
    echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
    echo "  $i -> instance $ID ($WORKER_ID)"
done

NLAUNCHED=$(wc -l < "$INSTANCE_FILE")
echo
echo "[cvvdp-backfill-imazen] launched $NLAUNCHED instances (target $N_BOXES)"
echo "  manifest: $INSTANCE_FILE"
echo
echo "Monitor:"
echo "  SWEEP_RUN_ID=$SWEEP_RUN_ID bash scripts/sweep/cvvdp_backfill/status.sh"
echo "  vastai logs <instance_id>"
echo
echo "Tear down:"
echo "  bash scripts/sweep/destroy_all.sh"
