#!/usr/bin/env bash
set -u
OUT=/home/lilith/picker-pp; REPO=/home/lilith/work/zen/zenmetrics; ZA=/home/lilith/work/zen/zenanalyze
# Features TSV MUST key on the same variant_name as the sweep omni. The
# clean-picker-corpus-2026-06-26 sweep emits `o_<id>.png.scale<W>x<H>` keys;
# clean_features_vn.tsv carries those. The older imazen26_train_features_*.tsv
# keys on a different convention (`<id>.scale<W>x<H>`) and joins to ZERO rows
# of this sweep — an empty pareto (the 2026-06-29 trap). Use the corpus's own
# variant-keyed features.
F22=/mnt/v/output/clean-picker-corpus-2026-06-26/clean_features_vn.tsv
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
  # NO --allow-unsafe: the zone-aware DATA_STARVED gate (unachievable high-zq
  # tail exempted as declared zones + single-config reference cells exempted)
  # lets a clean picker bake honestly. If bake REFUSES, the data has a genuine
  # gap (under-supply at low zq, or a real safety-tail) — fix the data, don't
  # force the bake.
  python3 "$ZA/tools/bake_picker.py" --model "$OUT/models/${codec}_predict_${tgt}_v0.1.json" --out "$OUT/models/${codec}_predict_${tgt}_v0.1.bin" --dtype f16 --bake-bin "$ZA/target/release/zenpredict-bake" 2>&1 | grep -iE 'baked|REFUSED' | tail -1
}
for c in zenjpeg zenjxl zenwebp zenavif; do train $c zensim_a "$OUT/sweeps/${c}.zensim.tsv" score_zensim; done
train zenavif ssim2 "$OUT/sweeps/zenavif.ssim2.tsv" score_ssim2_gpu
echo "=== ALL PICKERS ==="; ls -la $OUT/models/*_predict_*.bin 2>/dev/null | awk '{print $NF, $5"B"}'
