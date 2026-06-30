# Image-aware lossy-codec ordering projection — 2026-06-30

A per-family **linear projection of zenanalyze features** that orders lossy codecs by predicted
bytes-at-target. Fit on **confound-corrected** data (no re-sweep). This is the interpretable,
one-shot order that the picker uses for step 3 (see `docs/HOW_THE_PICKER_DECIDES.md`).

## Files
- [`picker_projection_2026-06-30.json`](picker_projection_2026-06-30.json) — the model. `cols`
  (102 inputs: 101 qualified `name@hex8` zenanalyze features + `target_zq`), `families`
  (`[jpeg, webp, jxl, avif]`), `weights[fam] = {w: [102], b}` — a **folded affine on RAW inputs**
  (StandardScaler folded into the Ridge weights; verified equal to the sklearn pipeline, atol 1e-4),
  plus a `test_vector` + `test_scores` round-trip pair. Score = `w·x + b` = predicted `ln(bytes)`;
  **lower = fewer bytes = better**; order = ascending; feed straight into
  `zenpicker::RouteDecision::resolve` (masked argmin).
- Regenerate: `scripts/picker/export_projection.py` (re-fits + folds + round-trip-checks + writes).

## What it does
Held-out test (8,888 cells), one-shot, extra bytes vs the perfect oracle (corrected data):

| order | mean | p90 |
|---|---|---|
| image-blind always-JXL | 29.97% | 82.39% |
| image-blind always-AVIF | 21.62% | 57.76% |
| **this linear projection** | **3.85%** | **13.96%** |
| full MLP (for reference) | ~4.0% | ~11% |

Dropping the raw `log_pixels` input costs nothing (3.85% either way — size is already carried by
the qualified features), so the runtime needs only the Offer's features + the quality target.

## Interpretability
The **raw** folded weights are 1/σ-scaled (low-variance features get large raw weights), so don't
read importance off them directly. The **standardized** coefficients rank the differentiators:
every family's bytes scale with size + `target_zq`; the codec-discriminating signals are
`info_weight` (texture/entropy), `uniformity`, and `max_dim`. See
`scripts/picker/linear_projection_order.py` for the per-family standardized drivers.

## Provenance
- Data: `s3://zentrain/canonical/2026-06-27/<codec>/{train,test}.parquet` (local mirror
  `/mnt/v/output/canonical-picker-2026-06-27/`), origin even/odd split (no rendition leakage).
- Features sidecar: `/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet`
  (101 qualified source-only features, 0 NaN).
- Corrected oracle: `bytes_to_reach(target)` (cheapest encode reaching ≥ target zq) over each
  codec's BEST swept speed (AVIF best of s2/s4/s6/s8). Paired support (≥2 codecs reach the target).
- Confounds corrected (no re-sweep): AVIF speed (`avif_speed_correct.py`), coverage/MNAR via paired
  comparison, corpus skew via size×quality reweighting (`corrected_ranking.py`).
- Corrected pairwise ranking: **AVIF 2.07 ≈ JXL 2.05 ≫ WebP 1.22 > JPEG 0.65.**

## Next
Bake this projection as the lossy `ROUTER_LOSSY` ZNPR (1 linear layer, per-family `ln(bytes)`
outputs, argmin) via the existing `zenpredict` bake path, replacing the MLP lossy router. `route()`
already consumes a per-family-score router through `RouteDecision::resolve`, so no route.rs code
change — only the baked model.
