#!/bin/bash
#
# launch.sh — host-side vast.ai launcher for the cvvdp-backfill fleet
# (PINNED TASK).
#
# Modeled on scripts/sweep/v15/launch_gpu.sh but launches the
# cvvdp-backfill onstart (scripts/sweep/onstart_cvvdp_backfill.sh)
# instead of the v15 sweep onstart. Key differences:
#
#   - --image points at an `ubuntu:24.04` shell that docker-pulls the
#     two real images (zen-metrics-sweep + pycvvdp-scorer) at boot. We
#     don't ship a pre-built worker image because (a) the two scoring
#     images are 250 MB + 6.5 GB, baking them in would defeat
#     vast.ai's pull cache, and (b) the chunk-worker shell stitches
#     them together — there's no "single worker image" to bake.
#   - --env passes ZEN_METRICS_IMAGE and PYCVVDP_IMAGE alongside R2
#     creds; onstart_cvvdp_backfill.sh consumes both.
#   - SWEEP_RUN_ID + chunks.jsonf default to the
#     cvvdp-backfill-<date> naming convention.
#
# Required:
#   - ~/.config/cloudflare/r2-credentials sourcing R2_ACCOUNT_ID etc.
#   - vastai cli authenticated (`vastai login`)
#   - gh cli authenticated (used for the GHCR pull token)
#
# Usage:
#
#   bash scripts/sweep/cvvdp_backfill/launch.sh
#
# Environment overrides:
#   SWEEP_RUN_ID            (default: cvvdp-backfill-<YYYY-MM-DD>)
#   ZEN_METRICS_IMAGE       (default: ghcr.io/imazen/zen-metrics-sweep:0.6.4-aba984c)
#   PYCVVDP_IMAGE           (default: ghcr.io/imazen/pycvvdp-scorer:0.5.4)
#   N_BOXES                 (default: 6 — moderate fleet for the smoke pass)
#   MAX_DPH                 (default: 0.30 — pycvvdp wants more compute than v15)
#   MIN_CORES               (default: 8)
#   MIN_RAM_GB              (default: 16 — pytorch+cudart load is heavier)
#   MIN_DISK_GB             (default: 40 — pycvvdp image alone is 6.5 GB)
#   PARALLEL                (default: 2 — concurrent chunks per worker)
#   DRY_RUN                 (default: 0)

set -euo pipefail
source ~/.config/cloudflare/r2-credentials

SWEEP_RUN_ID="${SWEEP_RUN_ID:-cvvdp-backfill-$(date -u +%Y-%m-%d)}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-ghcr.io/imazen/zen-metrics-sweep:0.6.4-aba984c}"
PYCVVDP_IMAGE="${PYCVVDP_IMAGE:-ghcr.io/imazen/pycvvdp-scorer:0.5.4}"

# Boot image: a thin ubuntu:24.04 — the real scoring images get pulled
# inside onstart_cvvdp_backfill.sh. Don't reuse zen-metrics-sweep as
# the boot image: it would conflict with the dind-style docker pull
# of itself inside the container.
BOOT_IMAGE="${BOOT_IMAGE:-ubuntu:24.04}"

N_BOXES="${N_BOXES:-6}"
MAX_DPH="${MAX_DPH:-0.30}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-16}"
MIN_DISK_GB="${MIN_DISK_GB:-40}"
PARALLEL="${PARALLEL:-2}"
DRY_RUN="${DRY_RUN:-0}"

GHCR_TOKEN="$(gh auth token)"
GHCR_USER="${GHCR_USER:-lilithriver}"

echo "[cvvdp-backfill] launching fleet"
echo "  SWEEP_RUN_ID:      $SWEEP_RUN_ID"
echo "  ZEN_METRICS_IMAGE: $ZEN_METRICS_IMAGE"
echo "  PYCVVDP_IMAGE:     $PYCVVDP_IMAGE"
echo "  N_BOXES:           $N_BOXES"
echo "  MAX_DPH:           $MAX_DPH"
echo "  PARALLEL/box:      $PARALLEL"
echo

# Pre-flight: make sure chunks.jsonl + onstart_cvvdp_backfill.sh +
# cvvdp_backfill_chunk_worker.sh are already uploaded to R2. Workers
# pull from $SCRIPTS_R2_PREFIX in onstart.
SCRIPTS_R2_PREFIX="s3://coefficient/jobs/${SWEEP_RUN_ID}"
echo "[cvvdp-backfill] verifying $SCRIPTS_R2_PREFIX has chunks.jsonl + chunk_worker.sh"
if ! s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    ls "${SCRIPTS_R2_PREFIX}/" 2>/dev/null | grep -q chunks.jsonl; then
    echo "  ERROR: ${SCRIPTS_R2_PREFIX}/chunks.jsonl missing." >&2
    echo "  Generate + upload first:" >&2
    echo "    python3 scripts/sweep/generate_cvvdp_backfill_chunks.py --run-id $SWEEP_RUN_ID --unified-dir ... > /tmp/chunks.jsonl" >&2
    echo "    s5cmd cp /tmp/chunks.jsonl ${SCRIPTS_R2_PREFIX}/chunks.jsonl" >&2
    echo "    s5cmd cp scripts/sweep/cvvdp_backfill_chunk_worker.sh ${SCRIPTS_R2_PREFIX}/cvvdp_backfill_chunk_worker.sh" >&2
    exit 1
fi
echo "  ok"

# vast.ai offer search. Same filter shape as v15 with disk bumped for
# pycvvdp's image weight and ram bumped for pytorch loads.
QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_max_good>=12 num_gpus=1"
echo "[cvvdp-backfill] querying offers: $QUERY"
OFFERS_JSON=$(vastai search offers "$QUERY" --order 'dph_total' --raw)
OFFER_IDS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
offers = d if isinstance(d, list) else d.get('offers', [])
seen, picked = set(), []
for o in offers:
    mid = o.get('machine_id')
    if mid in seen: continue
    seen.add(mid)
    picked.append(str(o['id']))
    if len(picked) >= int('$N_BOXES'): break
print('\n'.join(picked))
")
n=$(echo "$OFFER_IDS" | wc -w)
echo "[cvvdp-backfill] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "$OFFER_IDS" | head -5
    echo
    echo "DRY_RUN=1: not launching. Re-run with DRY_RUN=0 to commit."
    exit 0
fi
[[ "$n" -lt 3 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

# Upload the launcher script's onstart variant to R2 so it survives
# inline-arg length limits. The actual --onstart-cmd is a tiny curl
# pulling onstart_cvvdp_backfill.sh from R2 and execing it.
ONSTART_R2_KEY="${SCRIPTS_R2_PREFIX}/onstart_cvvdp_backfill.sh"
echo "[cvvdp-backfill] uploading onstart to $ONSTART_R2_KEY"
s5cmd \
    --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 \
    cp scripts/sweep/onstart_cvvdp_backfill.sh "$ONSTART_R2_KEY"

# The inline onstart command is a tiny bootstrap that:
#   1. Installs s5cmd (needed before we can pull the real onstart).
#   2. s5cmd cp's onstart_cvvdp_backfill.sh from R2.
#   3. Execs it.
# The real onstart pulls chunks.jsonl, chunk_worker.sh, and the two
# docker images; runs the main worker loop.
ONSTART_BOOTSTRAP=$(cat <<EOF
set -e
apt-get update -q && apt-get install -yq curl ca-certificates >/dev/null
curl -fsSL "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz" \
    -o /tmp/s5cmd.tgz
tar xzf /tmp/s5cmd.tgz -C /usr/local/bin s5cmd
chmod +x /usr/local/bin/s5cmd
mkdir -p ~/.aws && cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = \${R2_ACCESS_KEY_ID}
aws_secret_access_key = \${R2_SECRET_ACCESS_KEY}
CREDS
s5cmd --endpoint-url "https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" --profile r2 \
    cp "${ONSTART_R2_KEY}" /usr/local/bin/onstart.sh
chmod +x /usr/local/bin/onstart.sh
exec /usr/local/bin/onstart.sh
EOF
)

INSTANCE_FILE="/tmp/cvvdp-backfill-${SWEEP_RUN_ID}/instances.txt"
mkdir -p "$(dirname "$INSTANCE_FILE")"
> "$INSTANCE_FILE"

i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-w${i}"
    LABEL="cvvdp-bf-${i}"

    ENV_STR="-e R2_ACCOUNT_ID=$R2_ACCOUNT_ID"
    ENV_STR+=" -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID"
    ENV_STR+=" -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY"
    ENV_STR+=" -e SWEEP_RUN_ID=$SWEEP_RUN_ID"
    ENV_STR+=" -e WORKER_ID=$WORKER_ID"
    ENV_STR+=" -e ZEN_METRICS_IMAGE=$ZEN_METRICS_IMAGE"
    ENV_STR+=" -e PYCVVDP_IMAGE=$PYCVVDP_IMAGE"
    ENV_STR+=" -e PARALLEL=$PARALLEL"
    ENV_STR+=" -e GHCR_TOKEN=$GHCR_TOKEN"
    ENV_STR+=" -e GHCR_USER=$GHCR_USER"
    ENV_STR+=" -e SCRIPTS_R2_PREFIX=$SCRIPTS_R2_PREFIX"

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

echo
echo "[cvvdp-backfill] launched $(wc -l < "$INSTANCE_FILE") instances (target $N_BOXES)"
echo "  manifest: $INSTANCE_FILE"
echo
echo "Monitor heartbeats:"
echo "  s5cmd --endpoint-url 'https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com' --profile r2 \\"
echo "      ls 's3://coefficient/heartbeats/${SWEEP_RUN_ID}/'"
echo
echo "Tail a worker log:"
echo "  vastai logs <instance_id>"
echo
echo "Tear down when complete:"
echo "  bash scripts/sweep/destroy_all.sh   # or per-instance: vastai destroy instance <id>"
