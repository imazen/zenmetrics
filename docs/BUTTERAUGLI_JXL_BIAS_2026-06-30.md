# Butteraugli on JXL: does a 3rd metric side with ssim2 or zensim? (2026-06-30)

**Context.** The companion investigation (`METRIC_CODEC_BIAS_2026-06-30.md`) found that
targeting **ssim2** vs **zensim** picks a different lossy format 33–48% of the time, dominated
by **jxl→avif** flips: zensim FAVORS jxl (its XYB/perceptual home-court), ssim2 DISFAVORS jxl.
The open question: is that a real perceptual signal or a zensim idiosyncrasy? A **third,
independent** perceptual metric — **butteraugli** (max-norm + 3-norm) — breaks the tie.

**Gap filled.** The canonical picker corpus has zensim+ssim2 but NO butteraugli. The 2026-06-24
6-metric data has butteraugli + zensim for webp/avif/jpeg but NO jxl. This run scored
**butteraugli-gpu on JXL** to complete the matrix.

## Method

- **JXL butteraugli**: scored `butteraugli-gpu` (→ `butteraugli_max_gpu` + `butteraugli_pnorm3_gpu`)
  over **box-21** of the existing sweep run `jxl-lossy-vardct-1782609551` (REUSED its persisted
  variants + omni per-cell mapping — no re-encode). **59,220 jxl-lossy cells**, 188 renditions
  (18 origins o_95xx), q5→q90, via the canonical heterogeneous-split scorer
  (`scripts/sweep/split_score_worker.sh`, image `ghcr.io/imazen/zenmetrics-sweep:v29-split`) on
  ONE vast.ai RTX 3060. Sidecar written to the run prefix:
  `s3://zentrain/jxl-lossy/runs/jxl-lossy-vardct-1782609551/butter-score/box-21/sidecars/butteraugli-gpu.parquet`.
  Joined to canonical jxl `score_zensim` on `(image_path, codec, q, knob_tuple_json)`.
- **webp/avif/jpeg butteraugli + zensim**: the 2026-06-24 6-metric data
  (`/mnt/v/zen/zensim-training/2026-06-24/unified/<codec>/`).
- **Bias** = median butteraugli achieved at a given zensim, minus the cross-codec mean at that
  zensim. **Butteraugli is a DISTORTION metric (lower = better)**, so the sign convention is the
  OPPOSITE of `metric_bias.png` (which used ssim2, higher=better):
  - bias < 0 ⇒ LOWER butteraugli at matched zensim ⇒ BETTER ⇒ butteraugli **FAVORS** that codec (sides with zensim)
  - bias > 0 ⇒ HIGHER butteraugli at matched zensim ⇒ WORSE ⇒ butteraugli **DISFAVORS** that codec (sides with ssim2)

**CAVEAT (honest):** this is a **cross-corpus** comparison — jxl renditions (clean-picker o_95xx)
share **0** renditions with the 2026-06-24 webp/avif/jpeg (imazen-26). Matched-zensim normalization
controls for quality, but the cross-codec mean mixes content. The **sign** of the jxl bias is
robust when it is large and consistent across the zensim band; the absolute magnitude is less
precise than a within-image comparison would be.

## Result

<!-- FILLED FROM THE REAL RUN -->
**59,220 jxl-lossy cells scored, 0 failures**, all 59,220 joined to canonical `score_zensim`
(train+validate+test). Sample sizes: jxl 59,220 / webp 128,699 / avif 29,889 / jpeg 16,830.

**JXL delivers WORSE (higher) butteraugli than every other codec at matched zensim — across the
ENTIRE quality band (zensim 47→89).** jxl ranks **4th of 4 (worst) in every single bin**, both
norms. Bias (achieved butteraugli − cross-codec mean; **>0 = worse = disfavored**):

- **max-norm: jxl mean bias +1.357** (raw rank 4.00/4). Peaks mid-range (+1.8 @ z71), never < +0.6.
- **3-norm:  jxl mean bias +0.364** (raw rank 4.00/4). Same sign everywhere, smaller magnitude.

Raw medians make the gap concrete — at zensim ≈ 71, butteraugli-max: **jxl 5.07** vs avif 2.45 /
webp 2.78 / jpeg 2.77. jxl carries roughly **2× the max-norm distortion** to reach the same zensim.

**Max-norm vs 3-norm DIFFER in magnitude (not sign):** jxl's disadvantage is ~3.7× larger in
**max-norm** (worst-pixel error, +1.357) than in **3-norm** (aggregate error, +0.364). Butteraugli's
*worst-region* norm penalizes jxl hardest — consistent with VarDCT's localized ringing/blocking being
the thing zensim (XYB, MLP-perceptual) forgives and butteraugli's max-norm punishes.

**Internal consistency (not a metric bug):** within jxl, zensim and butteraugli agree on jxl's own
quality *ordering* — spearman(zensim, butter_max) = **−0.912**, (zensim, butter_3norm) = **−0.96**
(higher zensim ⇒ lower butteraugli, as expected). They disagree only on jxl's *level relative to other
codecs* at a matched zensim. That is exactly a codec-preference difference, not noise.

## Verdict

**Butteraugli SIDES WITH ssim2 — it DISFAVORS JXL.** The 2-vs-1 that zensim implied is broken the
other way: **two independent perceptual metrics (ssim2 structural + butteraugli, incl. its
worst-region max-norm) both say jxl looks worse at a matched quality level than webp/avif/jpeg;
only zensim favors jxl.** This strongly supports the companion investigation's hypothesis that the
zensim router is **JXL-biased** because zensim shares jxl's XYB/perceptual home-court — it is NOT
that ssim2 is uniquely anti-jxl. A picker trained to target zensim will over-pick jxl relative to
what two other perceptual metrics reward.

Both butteraugli norms agree on the sign; max-norm shows it far more strongly than 3-norm.

**Caveat (repeated):** cross-corpus (jxl box-21 clean-picker vs webp/avif/jpeg 2026-06-24 imazen-26,
0 shared renditions). The matched-zensim normalization controls for quality but the cross-codec mean
mixes content; the sign is robust because it is large (rank 4/4 in 22/22 bins) and monotone across the
whole band. A within-image confirmation (encode webp/avif/jpeg on the same box-21 sources + butteraugli)
would tighten the magnitude but is very unlikely to flip a rank-4/4-everywhere result.

Data: jxl sidecar `s3://zentrain/jxl-lossy/runs/jxl-lossy-vardct-1782609551/butter-score/box-21/sidecars/butteraugli-gpu.parquet`
(59,220 rows, `butteraugli_max_gpu` + `butteraugli_pnorm3_gpu`). Script: `scripts/picker/butter_jxl_bias.py`.
Graphs: `/mnt/v/output/picker-metric-investigation/butter_jxl_{bias,rd}.png`.
