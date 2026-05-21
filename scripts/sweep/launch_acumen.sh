#!/usr/bin/env bash
# launch_acumen.sh — spin up N vast.ai instances for the Gate A
# castleCSF Mode A zensim-gpu feature-extraction sweep.
#
# Pairs with:
#   - Dockerfile.sweep.v26 → image tag v26-acumen-<sha>
#   - onstart_acumen_modea.sh (baked at /usr/local/bin/onstart_acumen.sh)
#   - metric_backfill_chunk_worker.sh (baked at /usr/local/bin/metric_chunk_worker.sh)
#   - chunks.jsonl from generate_acumen_chunks.py uploaded to
#     s3://coefficient/jobs/<run-id>/chunks.jsonl
#
# Tracking: imazen/zensim#40 Gate A.
#
# Required env / args:
#   SWEEP_RUN_ID                e.g. acumen-modea-gate-a-2026-05-21
#   IMAGE                       e.g. ghcr.io/imazen/zen-metrics-sweep:v26-acumen-<sha>
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
#                               R2 creds passed through to workers
#   CONTAINER_API_KEY           vast.ai API key (for self-destroy)
#
# Optional:
#   N_INSTANCES                 default 4
#   MIN_GPU_RAM_GB              default 8
#   MIN_RAM_GB                  default 24
#   MAX_BID_DPH                 default 0.20 ($/hr cap)
#   GPU_RUNTIME                 default cuda
#   ACUMEN_PPD                  default 56
#   ACUMEN_PEAK_NITS            default 100
#   ACUMEN_AMBIENT_NITS         default 5
#   DISK_GB                     default 24
set -euo pipefail

: "${SWEEP_RUN_ID:?SWEEP_RUN_ID missing}"
: "${IMAGE:?IMAGE missing (e.g. ghcr.io/imazen/zen-metrics-sweep:v26-acumen-<sha>)}"
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"
: "${CONTAINER_API_KEY:?CONTAINER_API_KEY missing (vast.ai API key)}"

N_INSTANCES="${N_INSTANCES:-4}"
MIN_GPU_RAM_GB="${MIN_GPU_RAM_GB:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-24}"
MAX_BID_DPH="${MAX_BID_DPH:-0.20}"
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"
ACUMEN_PPD="${ACUMEN_PPD:-56}"
ACUMEN_PEAK_NITS="${ACUMEN_PEAK_NITS:-100}"
ACUMEN_AMBIENT_NITS="${ACUMEN_AMBIENT_NITS:-5}"
DISK_GB="${DISK_GB:-24}"

# Per-sweep instance tracking file (CLAUDE.md "Per-sweep instance
# tracking — IMPORTANT" — DO NOT clobber another sweep's file).
INSTANCE_FILE="/tmp/${SWEEP_RUN_ID}_instances.txt"
> "$INSTANCE_FILE"

echo "[launch] run=$SWEEP_RUN_ID image=$IMAGE n=$N_INSTANCES" >&2
echo "[launch] viewing: ppd=$ACUMEN_PPD peak=$ACUMEN_PEAK_NITS ambient=$ACUMEN_AMBIENT_NITS" >&2

# vast.ai offer search — cheap consumer GPUs are fine for this work.
# `verified=true` excludes too many offers per CLAUDE.md.
OFFER_QUERY="gpu_ram>=${MIN_GPU_RAM_GB} cpu_ram>=${MIN_RAM_GB} cuda_max_good>=12.4 inet_down>=100 reliability>0.98 num_gpus=1"
echo "[launch] offer query: $OFFER_QUERY" >&2

offers=$(vastai search offers "$OFFER_QUERY" --order dph_total --raw \
    | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
items = d if isinstance(d, list) else d.get('offers', [])
items.sort(key=lambda x: x.get('dph_total', 9.99))
for x in items[:$N_INSTANCES * 3]:
    print(x['id'], x.get('dph_total', 0.0), x.get('gpu_name', 'unknown'))
")

if [[ -z "$offers" ]]; then
    echo "[launch] FATAL: no offers match query" >&2
    exit 3
fi

i=0
while read -r offer_id dph gpu_name; do
    if (( i >= N_INSTANCES )); then
        break
    fi
    if (( $(echo "$dph > $MAX_BID_DPH" | bc -l) )); then
        echo "[launch] offer $offer_id at \$$dph/hr exceeds MAX_BID_DPH=\$$MAX_BID_DPH; skip" >&2
        continue
    fi

    WORKER_ID="acumen-$(date +%s)-${i}"

    # Build single ENV_STR per vast.ai's preferred form (matches
    # launch_single_instance.sh) — multiple `--env "-e ..."` flags
    # cause "docker_build() error writing dockerfile" failures.
    ENV_STR="-e R2_ACCOUNT_ID=$R2_ACCOUNT_ID"
    ENV_STR+=" -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID"
    ENV_STR+=" -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY"
    ENV_STR+=" -e SWEEP_RUN_ID=$SWEEP_RUN_ID"
    ENV_STR+=" -e WORKER_ID=$WORKER_ID"
    ENV_STR+=" -e GPU_RUNTIME=$GPU_RUNTIME"
    ENV_STR+=" -e ACUMEN_PPD=$ACUMEN_PPD"
    ENV_STR+=" -e ACUMEN_PEAK_NITS=$ACUMEN_PEAK_NITS"
    ENV_STR+=" -e ACUMEN_AMBIENT_NITS=$ACUMEN_AMBIENT_NITS"
    ENV_STR+=" -e CONTAINER_API_KEY=$CONTAINER_API_KEY"

    LOGIN_ARG=""
    if [[ -n "${GHCR_USER:-}" && -n "${GHCR_TOKEN:-}" ]]; then
        LOGIN_ARG="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"
    fi

    echo "[launch] $WORKER_ID → vast.ai offer $offer_id (\$$dph/hr, $gpu_name)" >&2

    if [[ -n "$LOGIN_ARG" ]]; then
        create_out=$(vastai create instance "$offer_id" \
            --image "$IMAGE" \
            --login "$LOGIN_ARG" \
            --disk "$DISK_GB" \
            --label "$SWEEP_RUN_ID" \
            --raw \
            --env "$ENV_STR" \
            --onstart-cmd "/usr/local/bin/onstart_acumen.sh" \
            2>&1)
        rc=$?
    else
        create_out=$(vastai create instance "$offer_id" \
            --image "$IMAGE" \
            --disk "$DISK_GB" \
            --label "$SWEEP_RUN_ID" \
            --raw \
            --env "$ENV_STR" \
            --onstart-cmd "/usr/local/bin/onstart_acumen.sh" \
            2>&1)
        rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        echo "[launch] WARN: create failed for $offer_id: $create_out" >&2
        continue
    fi

    instance_id=$(echo "$create_out" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(d.get('new_contract', d.get('id', '')))
except Exception:
    pass
" || echo "")

    if [[ -n "$instance_id" ]]; then
        # vast.ai creates instances in `stopped` state — explicit
        # start is required to fire the onstart-cmd. Observed
        # 2026-05-18+; without this, instances sit at
        # actual_status=loading + cur_state=stopped + the
        # `docker_build() error writing dockerfile` placeholder
        # message indefinitely.
        echo "[launch] starting instance $instance_id" >&2
        vastai start instance "$instance_id" >&2 || true

        echo "$instance_id $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
        echo "[launch] OK $WORKER_ID instance=$instance_id" >&2
        i=$(( i + 1 ))
    else
        echo "[launch] WARN: instance_id missing from $create_out" >&2
    fi
done < <(echo "$offers")

echo "[launch] launched $i / $N_INSTANCES instances; tracked at $INSTANCE_FILE" >&2
echo "[launch] tail logs: tail -f /tmp/onstart_acumen.log on each instance" >&2
echo "[launch] heartbeats: s3://coefficient/heartbeats/${SWEEP_RUN_ID}/" >&2
echo "[launch] sidecars:   s3://zentrain/${SWEEP_RUN_ID}/zensim_acumen/" >&2
