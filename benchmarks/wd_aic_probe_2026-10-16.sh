#!/usr/bin/env bash
# Caller holds the shared heavy lock and run-heavy resource scope.
set -euo pipefail
models=$1
python=$2
reference=$3
site=$4
images=$5
config=$6
previous=$7
pairs=$8
out=$9
test ! -e "$out"
mkdir -p "$out"
budget_started=$(date +%s)
python3 - "$out/clock.json" "$budget_started" <<'PY'
import json,sys
from datetime import datetime,timezone
json.dump({'start_epoch':int(sys.argv[2]),'start_utc':datetime.now(timezone.utc).isoformat(),'budget_seconds':3600},open(sys.argv[1],'x'),indent=2)
PY
timeout --signal=TERM --kill-after=10 3600 just nn-wd-score "$models" "$pairs" "$out/native.tsv" ort-cuda
budget_remaining=$((3600 - $(date +%s) + budget_started))
if ((budget_remaining <= 0)); then exit 124; fi
timeout --signal=TERM --kill-after=10 "$budget_remaining" "$python" -I crates/zenmetrics-nn/models/wd/aic_probe.py --reference "$reference" --site "$site" --images "$images" --config "$config" --native-scores "$out/native.tsv" --previous-log "$previous" --out "$out/search" --budget-seconds "$budget_remaining"
