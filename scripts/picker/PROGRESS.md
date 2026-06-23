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

## Fleet webp pickers — trained 2026-06-23 (NOT baked, see blocker)

Trained 2 zenwebp pickers on the **fleet** webp sweep (306,162 rows × 39 configs,
adapted to trainer format at `/mnt/v/output/zenmetrics/picker-fleet/train/`).
Size dist is small-heavy: small 245499 / tiny 35259 / medium 17082 / large 8322
rows (1398 unique renditions: 1121 small / 161 tiny / 78 medium / 38 large).

Config alignment: edited `configs/zenwebp_ssim2.py` (predict-ssim2 →
`webp_ssim2.pareto.parquet`) and `configs/zenwebp_picker.py` (predict-zensim-a →
`webp_zensim.pareto.parquet`) to point PARETO/FEATURES at the fleet files
(absolute paths) + OUT_* at `picker-fleet/models/`. Both use the same recipe
(modes_full 11-cell decomposition: format×tuning×syuv categorical + method scalar;
51 KEEP_FEATURES; 15 log1p FEATURE_TRANSFORMS) so they differ ONLY in target.

Results (held-out val n=5124, MLP 112→128→128→33, 35233 params, 68.8 KB f16):
- predict-ssim2:   student mean overhead **3.32%**, argmin 47.7%, overfit +0.50pp
- predict-zensim-a: student mean overhead **3.91%**, argmin 30.6%, overfit +0.45pp
  (both BEAT the jpeg fleet baselines of 5.13% / 6.32% on mean overhead)

**BAKE BLOCKED — non-DATA_STARVED gates fail (NOT bypassed per task rule):**
- predict-ssim2: DATA_STARVED_SIZE (ok-to-bypass) + **PER_SIZE_TAIL** (medium p99
  174% > 80%; medium has only 78 renditions, one image `9354.scale512x512`
  dominates the tail).
- predict-zensim-a: DATA_STARVED_SIZE + **PER_ZQ_TAIL ×3** (zq78/82/84 p99 85-96%)
  + **WORST_ROW** (large/zq82 `9920.scale2048x2048` 201.3% > 200%).
  These are the small-heavy size imbalance (78 medium / 38 large) biting the
  under-sampled p99 tails — the *bulk* fits well (means 3.3/3.9%). bake gate is
  all-or-nothing; `--allow-unsafe` would bypass these too, which the task forbids.
  Models written as JSON only; awaiting user decision on the tail-risk bypass.

## Gotcha: loky/HistGB DEADLOCKS under run-heavy (fixed via env)
`train_hybrid.py`'s HistGB teacher phase uses joblib `n_jobs=-1`. Under
`run-heavy` (systemd-user cgroup scope) with its default `OMP_NUM_THREADS=24`,
loky spawns ~24 process workers that EACH open 24 OpenMP threads → ~576-thread
oversubscription → all park in `futex_wait`, 0% CPU, hangs forever (reproduced
in isolation: bare = 1.5s, under run-heavy = ∞). FIX (env on the launch, no
zenanalyze edit): `env OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=11 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 run-heavy --jobs 11 -- python3 train_hybrid.py ...` → one HistGB
per worker, no inner threads, full run in ~105s.

## Gotchas hit
- `--feature-output` w/o `zensim-gpu` → CPU zensim features → STALL. Dropped it.
- >6 MP renditions OOM/thrash the non-orchestrator GPU path → `--max-mp 4`.
- `pkill -f <pat>` self-matches this shell; use `pkill -x zenmetrics`.
