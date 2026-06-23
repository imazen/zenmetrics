# Codec pickers — predict-ssim2 + predict-zensim-a, local pipeline (2026-06-22)

Encoded → measured → trained pickers for each codec, two target families each,
end-to-end on the water-cooled 7950X + RTX 5070 (native GPU/CPU metrics).
Local-first to prove the pipeline + produce first pickers; the Hetzner/vast
fleet scales it.

## Two target families per codec (8 models)

- **predict-ssim2** — target = SSIMULACRA2 (`ssim2-gpu` score). The
  "ssim2-approximating" goal.
- **predict-zensim-a** — target = zensim Profile-A (`zensim` CPU score).

| codec | corpus (imgs / sizes) | predict-ssim2 overhead | predict-zensim-a overhead |
|---|---|---|---|
| zenjpeg | 154 / tiny+small+medium+large | **7.18%** | **9.36%** |
| zenjxl  | 154 / tiny+small+medium+large | 32.52% | **10.73%** |
| zenwebp | 64  / tiny+small+medium+large | 50.27% | 0.00%* |
| zenavif | 40  / tiny+small only          | **8.21%** | **11.73%** |

Overhead = held-out mean bytes over the per-row Pareto-optimal config (lower
better). `.bin` ≈ 200–207 KB each (ZNPR f16, roundtrip-verified).

\* webp predict-zensim-a is degenerate (only 6 categorical cells collapse to a
near-trivial pick) — first-cut, same coarseness as webp ssim2 (50%). Fix is
`modes_full` with a larger budget for richer lossy cells.

## Provenance

- **Binary**: master `8c373d54`+ (the deadlock fix, below), built
  `--features sweep,png,jxl,gpu,gpu-cuda`.
- **Features**: zenanalyze **latest** content features
  `imazen26_train_features_2026-06-22.tsv` (110 named `feat_*`, joined on
  rendition `variant_name`). Supersedes the 06-14 set — zenanalyze commit
  `4654359` changed ~7 features (edge_slope_stdev, variance/covariances,
  spectral_slope_y, chroma); renditions are byte-identical so the encodes stayed
  valid and only re-adapt + retrain was needed.
- **Corpus**: `train_renditions_2026-06-14` size-stratified (≤4 MP cap). jpeg/jxl
  154 (40/size), webp 64; **avif on the ≤256² small corpus (40 imgs)** — see
  limitations.
- **Sweep**: `zenmetrics sweep --plan rd_core|modes_full`, q-grid
  5,10,20,30,40,50,60,70,80,90,95.
  - predict-ssim2 jpeg/jxl/webp: reused the existing `ssim2-gpu` omni sweeps.
  - predict-zensim-a jpeg/jxl/webp: fresh `--metric zensim` (CPU) sweeps,
    **`--encoded-out-dir` persisted** (3.9 GB encodes: jpeg 34188 / jxl 8778 /
    webp 2688 files).
  - avif: **one combo sweep** `--metric ssim2-gpu --metric zensim` (both targets,
    one encode, persisted) — 10560 cells in 43 s.
- **Adapt**: `scripts/picker/omni_to_pareto.py` (omni + features → trainer
  pareto+features; cell-id → categorical+scalar; `--metric-col` selects the
  target → trainer's `zensim` column).
- **Train**: `zentrain/tools/train_hybrid.py --activation leakyrelu --hidden
  192,192,192`, `PICKER_TARGET={ssim2|zensim_a}` (configs in
  `scripts/picker/configs/`).
- **Bake**: `zenpredict-bake` → ZNPR `.bin`.
- **Artifacts**: `/home/lilith/picker-pp/{models,train,enc}/` (pointer-filed —
  >30 KB, not committed).

## The deadlock that gated this run (fixed: 8c373d54)

The zensim sweeps with `--encoded-out-dir` hung at a variable cell count
(423–5833). Root cause was **not** a memory leak: CPU metrics
(zensim/ssim2/butteraugli/dssim) parallelize internally with rayon, and the
sweep's per-cell dispatch held the global `MetricCache` GPU-cache mutex
(`cache.rs:265`) **while running them inside the outer rayon `par_iter`**. The
lock holder descended into zensim's nested rayon `join` and waited for worker
threads — but all 28 other workers were parked on that same mutex
(`futex_do_wait`). `--encoded-out-dir` just perturbed timing enough to trigger
it every run. Fix: CPU metrics (`!metric.requires_gpu()`) take the uncached
`run_metric` path with no global lock (the documented contract); they now
parallelize instead of serialize. Verified: corpus_sm zensim+persist went from
deadlock → 8880 cells/2 s; the ssim2-gpu+zensim *combo* (the original stall) now
runs 10560 avif cells in 43 s.

Diagnosed without root (ptrace_scope=1): `/proc/<pid>/task/*/wchan` (all-futex ⇒
lock, not I/O) → gdb-as-child + timer-interrupt backtrace → `cache.rs:265` +
`zensim::…convert_source_to_xyb_into`.

## Known limitations (→ fleet scales these)

- **avif corpus is small (40 imgs, tiny+small only)**: avif CPU encode is the
  bottleneck (re-encode per cell), and ssim2-gpu on >~2 MP risks OOM with the
  leaked GPU memory on the 12 GB card. The avif models only see two size classes
  — first-cut. The fleet (bigger boxes, more GPU VRAM) covers the full size
  range.
- **Bakes used `--allow-unsafe`**: all QUALITY gates pass; only
  `DATA_STARVED_SIZE` (≥50 train rows/(size,zq)) fails — local corpus is under
  threshold per cell. The fleet's full corpus closes this.
- **webp/jxl ssim2 are coarse** (30–50%): `rd_core` yields too few cells;
  `modes_full` with a larger budget is the fix.
- **GPU memory leak (~4 GB) on this box** from repeated SIGKILLs during the
  deadlock hunt — unrelated to the pickers; clears on a clean GPU.
