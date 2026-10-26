#!/usr/bin/env bash
# Hydrate one immutable input archive without a manifest, claim or fit.
set -euo pipefail
host=$1 run=$2 image=$3 scratch=$4 pin=$5
[[ "$host" =~ ^[A-Za-z0-9_-]+$ && "$run" =~ ^[A-Za-z0-9_-]+$ ]]
[[ "$image" =~ ^ghcr.io/imazen/zenfleet-worker@sha256:[0-9a-f]{64}$ ]]
[[ "$scratch" =~ ^/[A-Za-z0-9_./+-]+$ && "/$scratch/" != *"/../"* ]]
[[ "$scratch" == /home/*/tmp/* || "$scratch" == /scratch/* ]]
[[ "$pin" =~ ^[0-9a-f]{64}$ ]]
ssh -o BatchMode=yes "$host" \
  FIT_RUN="$run" FIT_IMAGE="$image" FIT_SCRATCH="$scratch" FIT_PIN="$pin" 'bash -s' <<'REMOTE'
set -euo pipefail
export ZEN_STORE=lan
# shellcheck disable=SC1090
source "$HOME/.config/zen/s3env.sh"
mkdir -p "$FIT_SCRATCH"
name="zenfleet-prefetch-$FIT_RUN"
docker run --rm --name "$name" --cpus 1 --cpu-shares 256 --memory 2g --memory-swap 2g \
  -v "$FIT_SCRATCH:/scratch" -e TMPDIR=/scratch/tmp \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION -e AWS_REGION \
  -e ZEN_BUCKET=zentrain -e "ZEN_RUN=jobs/$FIT_RUN" -e "ZEN_R2_ENDPOINT=$EP" \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint nice "$FIT_IMAGE" -n 19 ionice -c 3 python3 -c '
import json,sys
from pathlib import Path
sys.path.insert(0,"/opt/fleet-fits/program")
from fit_cell_exec import fetch_data
root=fetch_data(sys.argv[1])
assert (root/".verified").read_text().strip()==sys.argv[1]
inventory=json.loads((root/"input_inventory.json").read_text())
print(json.dumps(dict(data_sha256=sys.argv[1],cache=str(root),verified_files=len(inventory["files"]),label=inventory["byc"]["label"],fit_invoked=False)))
' "$FIT_PIN"
REMOTE
