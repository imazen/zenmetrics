# Metric–codec bias: ssim2 vs zensim rank codecs differently (2026-06-30)

**Finding.** Targeting **ssim2** vs **zensim** picks a *different lossy format* 33–48% of the time
**even on clean coverage** (all 4 codecs reach the target on both metrics) — dominated by **jxl→avif**
(and jxl→webp) flips. This is a real metric-character difference, not a coverage artifact.

**Mechanism (per-codec ssim2 bias = ssim2 achieved at a given zensim, minus the cross-codec mean;
`scripts/picker/metric_rd_part2.py`, graph `metric_bias.png`):**
- **JXL under-scores ssim2 everywhere** (−0.3…−1.4σ-ish, worst mid-range). Good on zensim, worse on ssim2.
- **WebP over-scores ssim2 the most** (+0.6…+1.3, uniform).
- **AVIF flips**: under-scores at low-q (−1.6 @ z47) → over-scores at high-q (+1.0 @ z90+).
- **JPEG flips** the other way (+1.1 low-q → −0.7 high-q).

So "ssim2 favors AVIF" is really **"ssim2 disfavors JXL."** Hypothesis (not proven): zensim (XYB,
MLP-perceptual) is *perceptually aligned with JXL* (VarDCT in XYB optimizes for what zensim rewards);
SSIMULACRA2's structural model rewards WebP/AVIF smoothness. **The metric encodes a codec preference.**

**Tails (watch these):**
1. Bias is **quality-dependent** — AVIF/JPEG curves cross zero; WebP/JXL consistent. Disagreement
   concentrates where curves diverge (mid-high q for jxl→avif).
2. **High-q coverage artifact**: JXL has **39 rows ≥ z95** (11.5k ≥ z92) → above ~z95 JXL drops out
   and AVIF wins by default. Secondary; the known `jxl-lossy-swept-only-to-q90` gap. Re-sweep JXL to q95.
3. **Per-image variance is huge** (p10–p90 ssim2 band ≈ 40 pts at mid-zensim); the *median* bias is
   only ~1pt. Most of the pick disagreement is this spread, not the median shift.

**Implications.**
- The **shipped zensim router is mildly JXL-biased** (shares JXL's perceptual home-court). Picking on
  zensim structurally over-picks JXL vs a structural metric.
- **ssim2 targeting needs its own trained router** (the canonical data carries `score_ssim2`; same
  pairwise-discriminant pipeline). The earlier "ship zensim now / bake ssim2 next" scope stands.

Data: `/mnt/v/output/canonical-picker-2026-06-27` (2.75M lossy rows, both `score_zensim` + `score_ssim2`).
Scripts: `scripts/picker/metric_{agree,calib,rd_investigation,rd_part2}.py`. Graphs:
`/mnt/v/output/picker-metric-investigation/{rd_curves,metric_transfer,metric_bias}.png`.
