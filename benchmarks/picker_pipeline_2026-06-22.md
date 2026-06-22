# SSIM2-target codec pickers — local pipeline results (2026-06-22)

Encoded → measured → trained pickers for each codec, end-to-end on the
water-cooled 7950X + RTX 5070 (native GPU metrics). Target = SSIMULACRA2
(the "ssim2-approximating, not zensim Profile-A" goal). Local-first to prove
the pipeline + produce first pickers; the Hetzner/vast fleet (image
`zen-metrics-sweep:v28`) scales it.

## Provenance
- Binary: master `0fed4441` + path-patched codec mains, `--features sweep,png,jxl,gpu,gpu-cuda`.
- Corpus: `train_renditions_2026-06-14` size-stratified subset (≤4 MP cap — the
  non-orchestrator GPU path OOMs >~6 MP). jpeg/jxl 154 imgs (40/size); webp/avif
  64 imgs (16/size, avif/webp encode is slower).
- Sweep: `zenmetrics sweep --plan <rd_core|modes_full> --metric ssim2-gpu`,
  q-grid 5,10,20,30,40,50,60,70,80,90,95.
- Features: zenanalyze content features (`imazen26_train_features_2026-06-14.tsv`,
  110 named feat_*, joined on rendition `variant_name`).
- Adapter: `scripts/picker/omni_to_pareto.py` (omni + features → trainer pareto+features;
  cell-id → categorical+scalar decomposition; ssim2 → the trainer's `zensim` target col).
- Train: `zentrain/tools/train_hybrid.py --activation leakyrelu --hidden 192,192,192`.
- Bake: `zenanalyze/tools/bake_picker.py` → ZNPR `.bin` (roundtrip-verified).
- Models: `/mnt/v/output/picker-pipeline-2026-06-22/models/<codec>_ssim2_v0.1.{json,bin}`.

## Results (held-out validation)
| codec | imgs | cells×scalar | configs | student bytes-overhead | argmin acc | scalar RMSE | .bin bytes |
|---|---|---|---|---|---|---|---|
| zenjpeg | 154 | 8 (strategy×sub) × trellis_lambda | 24 | **6.45%** | 44.2% | λ 0.088 | 206725 |
| zenavif | 64* | 8 (sub×bd×qm) × speed | 24 | **8.86%*** | 33.9% | speed 1.64 | 206725 |
| zenjxl | 154 | 3 (mode×variant) × effort | 9 | 29.64% | 59.5% | effort 1.31 | 200344 |
| zenwebp | 64 | 2 (format×tuning×syuv) × method | 12 | 50.27% | 87.8% | method 1.31 | 199069 |

\* avif row is the first-cut on partial sweep data (~21 imgs); full 64-img retrain in progress.

## Reading the numbers
- **Bytes-overhead** = how much bigger than the per-row Pareto-optimal config the
  picker's choice is, on held-out images (lower = better). The student MLP often
  beats the per-row teacher because it generalizes.
- **jpeg/avif are strong** (6–9%): rd_core gives them a rich 24-config space that
  decomposes into 8 categorical cells + a continuous scalar — the picker has real
  choices to interpolate.
- **jxl/webp are coarse first-cuts** (30–50%): rd_core yields only 9 (jxl) / 12
  (webp) configs collapsing to 3 / 2 categorical cells — too few to pick well.
  **Fix: modes_full with a larger `--plan-budget`** (jxl/webp have rich modes_full
  axes) → more cells → lower overhead. webp also needs lossy-focused cells (the
  modes_full lossless/multipass cells are slow + low-value for an ssim2 picker).

## Known limitations (→ fleet scales these)
- **Bakes used `--allow-unsafe`**: all QUALITY gates pass (overhead tails,
  size-invariance, overfit) — the only failures are `DATA_STARVED_SIZE`
  (≥50 train rows/(size,zq)): the local corpus (34–40/size after Pareto) is under
  the threshold, and tiny images can't reach low ssim2 (structurally empty cells).
  The fleet's full corpus closes this.
- **ZQ_TARGETS trimmed to 30–94** (tiny floor ~54; >95 rarely reachable). Per-size
  target grids would let medium/large target lower — future work.
- Corpus capped at 4 MP (GPU memory on the non-orchestrator path; the v28 fleet
  image's orchestrator handles larger via the OOM→strip→CPU ladder).
