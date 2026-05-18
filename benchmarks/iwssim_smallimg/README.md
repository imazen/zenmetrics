# Adaptive IW-SSIM small-image validation — 2026-05-17

`run_validation.py` is the harness; `report_2026-05-17.txt` is the
result of one local run on RTX 5070.

## Question

When IW-SSIM is forced to score sub-176-px inputs via the
`IwssimConfig::allow_small=true` reflect-pad path (added in
4e01232c), does the score still rank distortion-severity in agreement
with ssim2 on the same pairs?

Three host-side preprocessing strategies are compared:

- **`iwssim_reflect`** — reflect-pad short axis to 176 then stock
  iwssim (current default; the only one implemented in
  `iwssim-gpu` today).
- **`iwssim_upscale`** — Lanczos upscale to ≥176 then stock iwssim.
- **`iwssim_tile`** — repeat content (`tile_pad`) until both axes
  ≥176 then stock iwssim.

All three are tested against `ssim2_gpu` as the anchor (ssim2 handles
small inputs natively at any dim).

## Method

- **Corpus**: 49 CID22 validation-set originals (512×512).
- **Native dims**: {64, 96, 128, 176} via Lanczos downsample.
- **Quality steps**: q ∈ {5, 20, 50, 80, 95} via `zen-metrics
  sweep --codec zenjpeg`.
- **Total pairs**: 49 × 4 × 5 = 980.
- **GPU**: RTX 5070 (CUDA 12.6, cubecl ce de2f985 fork).
- **Per-metric runtime**: ~10 s for 980 pairs (each).

For each (src, native_dim) we compute pooled Spearman ρ and pairwise
rank-flip rate of each strategy against `ssim2_gpu` and against the
`q` ordering. Pooled means across q's at fixed (src, dim) and also
across all (src, q) at fixed dim — the latter is the headline.

## Results (980 pairs, see `report_2026-05-17.txt`)

**Absolute thresholds** (ρ ≥ 0.85, flip ≤ 10%): every strategy
FAILS at every dim including the stock 176-px baseline. The
iwssim-vs-ssim2 disagreement floor on this CID22-JPEG corpus is
**~11.4% flip rate** even with no preprocessing. Setting an absolute
10% threshold without measuring the baseline was wrong.

**Relative to dim=176 stock baseline** (Δρ ≥ -0.02, Δflip ≤ +0.02):
every strategy at every sub-176 dim PASSES. Adaptive iwssim
introduces ≤0.01 drift in Spearman and ≤0.01 drift in flip rate vs
running iwssim natively on a 176-px image.

Best strategy per dim (higher ρ vs ssim2 = better):

| native_dim | best          | runner-up        | worst           | spread |
|------------|---------------|------------------|-----------------|--------|
| 64         | tile (0.9374) | upscale (0.9351) | reflect (0.9277) | 0.010  |
| 96         | tile (0.9421) | reflect (0.9384) | upscale (0.9373) | 0.005  |
| 128        | tile (0.9395) | upscale (0.9350) | reflect (0.9328) | 0.007  |
| 176        | (all equal: 0.9341 — no preprocessing applied at this dim)        |

**Tile beats reflect-pad and upscale at every sub-176 dim**, by a
small but consistent margin.

## Recommendation

1. **Adaptive iwssim is usable** — no strategy degrades the metric
   meaningfully vs the stock 176-px baseline. Sub-176 inputs can be
   scored with adaptive iwssim for codec sweeps / picker work
   without worrying that the metric breaks down at small sizes.

2. **Switch default strategy from reflect-pad to tile.** Implementation
   complexity is identical (host-side preprocessing before upload);
   tile is marginally better at all sub-176 dims.

3. **Expose strategy as an enum** rather than the current
   `allow_small: bool`. Future strategies (e.g. blurred padding,
   content-aware extension) can be added without breaking the API.

4. **Do NOT claim iwssim "agrees with ssim2"** in any documentation —
   the 11% baseline flip rate is intrinsic to the two metrics
   responding differently to JPEG artifacts.

## Limitations + caveats

- **q as ground truth is weak.** q is the JPEG encoder's quality
  dial, not a perceptual rating. Strong correlation with metric
  scores is expected by construction. A proper validation would
  use KonJND-1k or CID22 with MOS, but no native small-image MOS
  data was available.
- **Single codec (zenjpeg).** Other codecs (webp, avif, jxl) may
  show different per-strategy drift, particularly avif/jxl which
  have qualitatively different artifact patterns. A second sweep
  with all 5 codecs would close this gap.
- **49 sources, not 100 as planned.** CID22 validation set has 49
  refs total; sweep used all of them. Statistical power per (dim,
  strategy) cell is 245 pairs, which is enough to distinguish
  strategy gaps of ≥0.005 but not gaps below ~0.002.
- **Spearman over a 5-point q grid is coarse.** A denser q grid
  (10-15 steps) would surface finer-grained ranking errors. The
  per-source Spearman is 1.0 trivially across all strategies; the
  signal lives in cross-source pooling.

## Reproducing

```sh
cd /home/lilith/iwssim-val   # or wherever the harness lives
python3 run_validation.py --num-sources 100
cat report.txt
```

Phases: `--phase {prep,score,analyze,all}`. Prep is CPU-bound (~6s
per 5 refs); score is GPU-bound (~10s per 980-pair metric run).
Analyze is < 1s.
