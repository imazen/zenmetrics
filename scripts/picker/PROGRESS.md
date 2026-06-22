# Picker pipeline — local encode→measure→train (started 2026-06-22)

Goal: encoded + measured + **trained ssim2-target pickers for each codec**
(zenjpeg/zenwebp/zenjxl/zenavif). Local-first on the 7950X + RTX 5070 (the
Hetzner/vast fleet is for *scale*; this proves the pipeline + produces the
first pickers). Survives compaction.

## Architecture (decided)
- **Encode + measure**: `zenmetrics sweep --plan rd_core --metric ssim2-gpu`
  (GPU binary built at master `0fed4441`, latest mains, zencodec 0.1.24 migration
  DONE on master so no patch). Omni TSV per codec.
- **Features**: zenanalyze **content** features (named feat_*), reused from the
  render-time TSV `imazen26_train_features_2026-06-14.tsv` (keyed `variant_name`
  = rendition basename w/o .png). NOT zensim features.
- **Adapt**: `scripts/picker/omni_to_pareto.py` — omni + features TSV →
  trainer's PARETO parquet (config_name=cell-id, zensim col = ssim2 score,
  size_class from dims, effective_max_zensim) + FEATURES TSV.
- **Train**: `zentrain/tools/train_hybrid.py --codec-config <codec>_ssim2`
  (configs in `scripts/picker/configs/`; opaque cell-id categorical, q = dial).
- **Bake**: `zenanalyze/tools/bake_picker.py` → ZNPR `.bin`.

## Paths
- corpus: `/mnt/v/output/picker-pipeline-2026-06-22/corpus/` (154 renditions,
  40/size-class, ≤4 MP cap — non-orchestrator GPU path OOMs on >~6 MP)
- sweeps: `.../sweeps/<codec>.tsv`
- train inputs: `.../train/<codec>.pareto.parquet` + `.../train/<codec>.features.tsv`
- models: `.../models/<codec>_ssim2_v0.1.{json,bin,log}`

## Commands
```
# corpus
python3 scripts/picker/select_corpus.py --src <renditions> --out <corpus> --per-class 40 --max-mp 4
# sweep (per codec; LD_LIBRARY_PATH for CUDA on WSL2)
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 nice -n19 ./target/release/zenmetrics sweep \
  --codec <codec> --sources <corpus> --q-grid 5,10,20,30,40,50,60,70,80,90,95 \
  --plan rd_core --metric ssim2-gpu --output <out>.tsv
# adapt
python3 scripts/picker/omni_to_pareto.py --omni <out>.tsv \
  --features-tsv /mnt/v/output/imazen-26-features/imazen26_train_features_2026-06-14.tsv \
  --out-pareto .../train/<codec>.pareto.parquet --out-features .../train/<codec>.features.tsv
# train
PYTHONPATH=scripts/picker/configs:<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
  python3 <zenanalyze>/zentrain/tools/train_hybrid.py --codec-config <codec>_ssim2 --activation leakyrelu
```

## Status
- [x] GPU sweep binary built (master 0fed4441 + sweep,png,jxl,gpu,gpu-cuda)
- [x] corpus selector + 4 MP cap; 154 renditions
- [x] adapter + 4 codec configs written
- [x] zenjpeg: swept (34188 cells, 0 fail) + adapted + trained (student 6.45% mean
      bytes-overhead val, argmin 44.2%, +2.06pp overfit) + baked (.bin 206725 B,
      roundtrip OK). Bake used --allow-unsafe: ALL quality gates pass; only
      DATA_STARVED_SIZE (≥50 rows/(size,zq)) fails — local corpus (~34-40/size after
      Pareto) < threshold; the fleet exists to scale this. Decomposed cell-id:
      strategy×sub (8 cells) + trellis_lambda scalar.
- [ ] zenjpeg adapt + train + bake (de-risk the train step)
- [/] avif/jxl/webp sweeps running (avif+jxl rd_core, webp modes_full b=60)
- [ ] commit pickers + parquets (pointer-file the big parquets)

## Gotchas hit
- `--feature-output` w/o `zensim-gpu` → CPU zensim features → STALL. Dropped it.
- >6 MP renditions OOM/thrash the non-orchestrator GPU path → `--max-mp 4`.
- `pkill -f <pat>` self-matches this shell; use `pkill -x zenmetrics`.
