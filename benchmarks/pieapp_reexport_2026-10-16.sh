#!/usr/bin/env bash
# Historical probe: records the original dated environment, not a portable recipe.
# One bounded heavy step; caller holds shared flock/run-heavy.
set -euo pipefail
oldmodels=$1
models=$2
output=$3
root="$HOME/tmp/pieapp-oracle"
mkdir -p "$output"
"$root/venv/bin/python" crates/zenmetrics-nn/models/pieapp/review_reconstruction.py --checkpoint "$root/upstream/weights/PieAPPv0.1.pth" --receipt benchmarks/pieapp_review_2026-10-16/weight-identify.json --out "$output/reconstruction.json"
just nn-pieapp-export "$root/venv/bin/python" "$root/upstream/weights/PieAPPv0.1.pth" "$models"
cp "$oldmodels/pieapp_goldens.tsv" "$models/pieapp_goldens.tsv"
just nn-pieapp-review-shards "$models" "$output/shards"
