# Picker pre-processing & scalar-head smoothness (2026-06-22)

Follow-up to `picker_pipeline_2026-06-22.md`: making the zenanalyze feature
pre-processing and the scalar heads "optimal and smooth." Two principled config
changes landed; two structural limits were measured and are deferred to the
re-sweep + fleet.

## What changed (config-only, `scripts/picker/configs/`)

1. **`FEATURE_TRANSFORMS = log1p`** on the 15 heavy-tailed, strictly-positive
   features (measured on `imazen26_train_features_2026-06-22`: pixel_count tail
   9352×, laplacian_variance 365×, luma_kurtosis 367×, the chroma horiz/vert/peak
   sharpness family 80–180×, high_freq_energy_ratio, dct_compressibility_*). Fed
   raw to the trainer's StandardScaler, their outliers dominate and the bulk
   collapses; `log1p` (0 params, applied before StandardScaler, baked into the
   model JSON so inference matches) compresses the tail to a near-Gaussian,
   smooth input. Bounded-[0,1] and low/left-skew features stay raw. Wired via
   `picker_config_common.feature_transforms()` (env `PICKER_NO_TRANSFORMS=1`
   disables, for ablation).

2. **Tightened `SCALAR_DISPLAY_RANGES`** to the actual swept ranges so the scalar
   head normalizes onto the full [0,1] instead of a sub-interval:
   jpeg λ (0,25)→(0,15); jxl effort (1,9)→(5,9); avif speed (0,10)→(2,6).
   (webp method (0,6) already matched.)

## What the measurements actually say (be honest)

**The overhead metric is noise-limited at this corpus size.** 3-seed A/B of
log1p ON vs OFF, jpeg (154 imgs, ~15-image held-out val):

| target | log1p ON (3 seeds) | log1p OFF (3 seeds) |
|---|---|---|
| ssim2    | 7.27 / 8.71 / 9.16 | 7.80 / 5.57 / 9.27 |
| zensim_a | 9.26 / 8.30 / 8.05 | 10.37 / 6.75 / 8.12 |

Seed-to-seed variance is **±2–3 pp for the *same* config** — larger than the
ON/OFF effect (means ~tied). So:
- The single-seed overhead numbers in `picker_pipeline_2026-06-22.md` are point
  estimates with ±2–3 pp noise — read them as "~7±2.5%", not "7.2%".
- `log1p` is kept because it's principled, smoother conditioning that doesn't
  hurt — **not** because it measurably lowers overhead (it can't be shown to, at
  this corpus size). Honest "smooth," not a fabricated win.

**The scalar heads are limited by degenerate sampling, not normalization.**
Every scalar axis takes only 3–4 distinct values in `rd_core`:
jpeg trellis_lambda ∈ {0, 14.5, 14.75} (effectively **binary** on/off),
jxl effort ∈ {5,7,9}, avif speed ∈ {2,4,6}, webp method ∈ {0,2,4,6}. A scalar
head can't be "smooth" predicting a 3-value target — the range tweak helps the
normalization but the signal is coarse by construction. **The real fix is denser
scalar sampling in the sweep plan** (e.g. trellis_lambda 2,4,…,20; effort 1–9;
speed 0–10), which lands with the re-sweep.

## Next (deferred — GPU busy with the zensim-gpu repair agent)

- **jxl/webp `modes_full` re-sweep** to fix the coarse 32%/50% ssim2 overheads
  (too few categorical cells in `rd_core`), with **denser scalar sampling** so
  the scalar heads have real continuous signal.
- **Re-validate on the bigger corpus** — the only way to get below the ±2–3 pp
  noise floor and make "optimal" a measured claim rather than a principled one.
  This is the fleet's job.
