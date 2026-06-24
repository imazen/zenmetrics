#!/usr/bin/env bash
# Leave-one-out (LOO) feature ablation. Retrains the picker dropping ONE feature at
# a time and records val mean overhead. delta_f = overhead(drop f) - overhead(baseline):
#   delta_f > 0  -> feature f carries real marginal RD signal (removing it hurts) -> KEEP
#   delta_f <= 0 -> f is dead weight for the picker (removing it is neutral/helps) -> ABLATE
# Fixed --seed so the ONLY difference between runs is the dropped feature (clean signal).
# This is the gold standard Spearman correlation cleanup cannot give: Spearman sees only
# monotonic redundancy, never interaction terms or true marginal contribution.
#
# Too slow locally (N retrains); run on a dedicated big box with high --npar, kill when done.
#   usage: loo_ablation.sh <codec-config-module> <out-dir> [npar=8] [metric=ssim2] [seed=12345] [feat-subset-file]
set -uo pipefail
CONFIG="${1:?codec-config module name}"; OUTDIR="${2:?out dir}"
NPAR="${3:-8}"; METRIC="${4:-ssim2}"; SEED="${5:-12345}"; SUBSET="${6:-}"
TH="${TRAIN_HYBRID:-/home/lilith/work/zen/zenanalyze/zentrain/tools/train_hybrid.py}"
HIDDEN="${HIDDEN:-192,192,192}"
mkdir -p "$OUTDIR"; : > "$OUTDIR/results.tsv"; : > "$OUTDIR/progress.log"

# Feature list straight from the config's KEEP_FEATURES (or an explicit subset file for smoke runs).
if [ -n "$SUBSET" ] && [ -f "$SUBSET" ]; then
  mapfile -t FEATS < "$SUBSET"
else
  mapfile -t FEATS < <(python3 -c "import importlib; m=importlib.import_module('$CONFIG'); print('\n'.join(m.KEEP_FEATURES))")
fi
echo "LOO $CONFIG: ${#FEATS[@]} features, npar=$NPAR, seed=$SEED metric=$METRIC hidden=$HIDDEN" | tee -a "$OUTDIR/progress.log"

run_one(){
  local f="$1" tag="$1" drop=()
  [ "$f" != "__BASELINE__" ] && drop=(--drop-features "$f")
  CI=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 python3 "$TH" --codec-config "$CONFIG" \
    --objective size_optimal --metric-column "$METRIC" --metric-direction higher_better \
    --hidden "$HIDDEN" --activation leakyrelu --allow-unsafe --seed "$SEED" "${drop[@]}" \
    --out-json "$OUTDIR/m_$tag.json" --out-log "$OUTDIR/m_$tag.log" > "$OUTDIR/run_$tag.out" 2>&1
  local ov
  ov=$(grep -oE "Student: mean [0-9.]+%|Student metrics: argmin mean overhead [0-9.]+%" "$OUTDIR/run_$tag.out" | head -1 | grep -oE "[0-9.]+" | head -1)
  [ -z "$ov" ] && ov="NA"
  printf '%s\t%s\n' "$tag" "$ov" >> "$OUTDIR/results.tsv"
  printf '[%s] %s ov=%s (%s/%s)\n' "$(date -u +%H:%M:%S)" "$tag" "$ov" "$(wc -l < "$OUTDIR/results.tsv")" "$FEATS_N" >> "$OUTDIR/progress.log"
  rm -f "$OUTDIR/m_$tag.json" "$OUTDIR/m_$tag.json.manifest.json"
}
export -f run_one; export OUTDIR CONFIG METRIC SEED TH HIDDEN
export FEATS_N="$(( ${#FEATS[@]} + 1 ))"

run_one __BASELINE__                                  # baseline first (sequential)
printf '%s\n' "${FEATS[@]}" | xargs -P"$NPAR" -I{} bash -c 'run_one "$@"' _ {}
echo "LOO done: $(($(wc -l < "$OUTDIR/results.tsv"))) rows -> $OUTDIR/results.tsv" | tee -a "$OUTDIR/progress.log"
