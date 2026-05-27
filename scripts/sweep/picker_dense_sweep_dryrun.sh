#!/usr/bin/env bash
# Dense q + size sweep for zenjpeg picker training data — LOCAL dry-run.
#
# Drives `zen-metrics sweep` (encode + score + persist) across a dense q
# grid and a small zenjpeg knob grid, once per source-size variant, then
# scores each (ref, encoded) pair with ssim2-gpu (the correct-monotone
# reach-ladder target — the shipped zensim metric has a known correctness
# defect on photo content, see the methodology doc). Encoded bytes,
# distorted-vs-source metric scores (ALL variants), and the 372-feature
# zensim sidecar are persisted per cell.
#
# This is the local, no-spend dry-run that de-risks the full-scale run.
# Parameterized by source dir + q grid + knob grid so the same script
# scales up (just point --sources at the K=20 clustered + log-spaced-size
# corpus and the budget grows; see the methodology doc for the full grid).
#
# Usage:
#   picker_dense_sweep_dryrun.sh <out_root> <zen_metrics_bin>
#
# Requires: zen-metrics built with `--features sweep,gpu,gpu-cuda` and a
# local CUDA GPU. NO fleet, NO cloud spend.
set -euo pipefail

OUT_ROOT="${1:-/mnt/v/zen/picker-dense-dryrun-2026-05-27}"
ZM="${2:-$HOME/work/zen/zenmetrics/target/release/zen-metrics}"

# Dense q grid: step 5 in 5..69, step 2 in 70..100 (29 levels). Matches the
# picker's ZQ_TARGETS density (CLAUDE.md "Dense sampling for trained models").
Q_GRID="5,10,15,20,25,30,35,40,45,50,55,60,65,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100"

# zenjpeg knob grid — exercises multiple categorical cells
# (subsampling | progressive | sharp_yuv | effort). 2x2x1x1 = 4 cells.
KNOB_GRID='{"subsampling":["444","420"],"progressive":[false,true],"sharp_yuv":[false],"effort":[1]}'

# Source-size variants. Each is a dir of PNGs at one size class. For the
# dry-run: native 512sq + a 256px Lanczos downscale. The full run adds
# 16-20 log-spaced sizes (see methodology doc).
declare -A SIZE_DIRS=(
  [sz512]="$OUT_ROOT/sources"
  [sz256]="$OUT_ROOT/sources_256"
)

LOGDIR="$OUT_ROOT/logs"
mkdir -p "$LOGDIR" "$OUT_ROOT/encoded" "$OUT_ROOT/features"

for sz in "${!SIZE_DIRS[@]}"; do
  SRCDIR="${SIZE_DIRS[$sz]}"
  if [[ ! -d "$SRCDIR" ]]; then
    echo "WARN: missing source dir $SRCDIR — skipping $sz" >&2
    continue
  fi
  echo "=== sweep size=$sz sources=$SRCDIR ===" | tee -a "$LOGDIR/driver.log"
  "$ZM" sweep \
    --codec zenjpeg \
    --sources "$SRCDIR" \
    --q-grid "$Q_GRID" \
    --knob-grid "$KNOB_GRID" \
    --metric zensim-gpu --metric ssim2-gpu \
    --gpu-runtime cuda \
    --output "$OUT_ROOT/pareto_${sz}.tsv" \
    --feature-output "$OUT_ROOT/features/feat_${sz}.parquet" \
    --encoded-out-dir "$OUT_ROOT/encoded/${sz}" \
    --pairs-tsv "$OUT_ROOT/pairs_${sz}.tsv" \
    2>&1 | tee -a "$LOGDIR/sweep_${sz}.log"
done

echo "=== sweep complete; building (ref,encoded) score pairs for the correct-monotone reach ladder ===" \
  | tee -a "$LOGDIR/driver.log"

# Re-score each (source, encoded) pair with ssim2-gpu via the batch path
# (the sweep's feature-extraction score column is raw/unmapped; batch gives
# the 0-100 mapped score). One pairs TSV across all sizes.
PAIRS="$OUT_ROOT/score_pairs_all.tsv"
{
  printf 'ref_path\tdist_path\timage_basename\tq\tknob_tuple_json\tsize_class\n'
  for sz in "${!SIZE_DIRS[@]}"; do
    SRCDIR="${SIZE_DIRS[$sz]}"
    ENCDIR="$OUT_ROOT/encoded/${sz}"
    [[ -d "$ENCDIR" ]] || continue
    for f in "$ENCDIR"/*.jpg; do
      [[ -e "$f" ]] || continue
      bn=$(basename "$f")
      src=$(echo "$bn" | sed -E 's/_[0-9a-f]{16}_zenjpeg.*//')
      q=$(echo "$bn" | grep -oE '_q[0-9]+_' | tr -d '_q')
      # knob hash is the trailing token; reconstruct knob_tuple_json from the
      # pareto TSV later in the parquet build (TSV-quoting JSON here is fiddly).
      printf '%s/%s.png\t%s\t%s\t%s\t\t%s\n' "$SRCDIR" "$src" "$f" "$src" "$q" "$sz"
    done
  done
} > "$PAIRS"
echo "score pairs: $(($(wc -l < "$PAIRS") - 1))" | tee -a "$LOGDIR/driver.log"

"$ZM" batch --metric ssim2-gpu --gpu-runtime cuda \
  --pairs "$PAIRS" \
  --output "$OUT_ROOT/score_pairs_ssim2_${RANDOM}.tsv" 2>&1 | tee -a "$LOGDIR/score_ssim2.log"
# Stable output name (batch is deterministic; rename last run)
mv "$OUT_ROOT"/score_pairs_ssim2_*.tsv "$OUT_ROOT/score_pairs_ssim2.tsv"

echo "=== done. pareto_*.tsv + features/feat_*.parquet + encoded/<sz>/*.jpg + score_pairs_ssim2.tsv ===" \
  | tee -a "$LOGDIR/driver.log"
