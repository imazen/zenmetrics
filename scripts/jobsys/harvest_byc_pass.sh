#!/usr/bin/env bash
# Incremental native harvest, immutable per-pass staging and retained-bake receipt.
set -euo pipefail
ctl=$1 run=$2 output=$3 cells=$4 pass=$5
[[ "$run" =~ ^[A-Za-z0-9_-]+$ && "$pass" =~ ^[A-Za-z0-9_-]+$ ]]
export TMPDIR="$HOME/tmp"
# shellcheck disable=SC1091
source scripts/lib/s3env.sh
stage=$(realpath -m "$cells/../fleet-harvest/$run-$pass")
cache=$(realpath -m "$cells/../fleet-harvest/$run/blobs")
mkdir "$stage"
mkdir "$stage/blobs"
mkdir -p "$cache"
python3 - "$cache" "$stage/blobs" <<'PYCACHE'
from pathlib import Path
import sys
cache, fresh = map(Path, sys.argv[1:])
for path in cache.iterdir():
    if path.is_file() and len(path.name) == 64:
        (fresh / path.name).hardlink_to(path)
PYCACHE
"$ctl" compact --run "$run" --endpoint "$EP" --out-dir "$output" --upload
python3 scripts/jobsys/harvest_fit_cells.py --manifest "$output/manifest.json" \
  --ids "$output/ids.tsv" --ledger "$output/snap_$run.parquet" \
  --blobs-prefix "s3://zentrain/jobs/$run/blobs" --endpoint "$EP" --scratch "$stage" \
  --byc-install-root "$cells" --done-only --install
python3 scripts/jobsys/byc_harvest_receipt.py --done-only --cells "$cells" \
  --manifest "$output/manifest.json" --ids "$output/ids.tsv" \
  --ledger "$output/snap_$run.parquet" --out "$output/${pass}_VERIFIED_HARVEST.json"
python3 - "$cache" "$stage/blobs" <<'PYCACHE'
from pathlib import Path
import sys
cache, fresh = map(Path, sys.argv[1:])
for path in fresh.iterdir():
    if path.is_file() and len(path.name) == 64 and not (cache / path.name).exists():
        (cache / path.name).hardlink_to(path)
PYCACHE
