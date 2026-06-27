#!/usr/bin/env bash
set -u
OUT=/home/lilith/picker-pp; REPO=/home/lilith/work/zen/zenmetrics; ZA=/home/lilith/work/zen/zenanalyze
F22=/mnt/v/output/imazen-26-features/imazen26_train_features_2026-06-22.tsv
cd "$REPO"
train() { local codec=$1 tgt=$2 omni=$3 col=$4
  [ -s "$omni" ] || { echo "SKIP $codec/$tgt (no omni)"; return; }
  echo "===== $codec / predict-$tgt ====="
  python3 scripts/picker/omni_to_pareto.py --omni "$omni" --features-tsv "$F22" --metric-col "$col" \
    --out-pareto "$OUT/train/${codec}.${tgt}.pareto.parquet" --out-features "$OUT/train/${codec}.features.tsv" 2>&1 | grep -E 'pareto:'
  # Mandatory-axis coverage gate (docs/MANDATORY_SWEEP_AXES.md): refuse to train a
  # picker on data missing a first-class mode (color / subsampling / sub-30s effort).
  if ! python3 scripts/picker/check_mandatory_coverage.py \
        --pareto "$OUT/train/${codec}.${tgt}.pareto.parquet" --codec "$codec"; then
    echo "ABORT $codec/$tgt: mandatory coverage failed — re-sweep with the axes pinned, do NOT train."
    return
  fi
  CUDA_VISIBLE_DEVICES="" PICKER_TARGET="$tgt" PYTHONPATH="scripts/picker:scripts/picker/configs:$ZA/zentrain/tools:$ZA/zentrain/examples" \
    python3 "$ZA/zentrain/tools/train_hybrid.py" --codec-config "${codec}_picker" --activation leakyrelu --hidden 192,192,192 2>&1 | grep -iE 'Student:|Wrote' | tail -2
  python3 "$ZA/tools/bake_picker.py" --model "$OUT/models/${codec}_predict_${tgt}_v0.1.json" --out "$OUT/models/${codec}_predict_${tgt}_v0.1.bin" --dtype f16 --bake-bin "$ZA/target/debug/zenpredict-bake" --allow-unsafe 2>&1 | grep -iE 'baked' | tail -1
}
for c in zenjpeg zenjxl zenwebp zenavif; do train $c zensim_a "$OUT/sweeps/${c}.zensim.tsv" score_zensim; done
train zenavif ssim2 "$OUT/sweeps/zenavif.ssim2.tsv" score_ssim2_gpu
echo "=== ALL PICKERS ==="; ls -la $OUT/models/*_predict_*.bin 2>/dev/null | awk '{print $NF, $5"B"}'
