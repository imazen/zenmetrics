# Integrated-PU21 ssim2 HDR validation — UPIQ HDR subset (2026-06-09)

Measured SROCC vs JOD ground truth on the UPIQ HDR subset (n = 380
(ref, dist) pairs, absolute-luminance EXRs). One row per feeding /
implementation; higher is better. Full motivation + run logs:
imazen/zenmetrics#25.

| feeding / implementation                                   | SROCC (UPIQ HDR, JOD) |
|------------------------------------------------------------|----------------------:|
| PU-SSIM literature bar (Aydin-style PU + SSIM, UPIQ paper)  |                 0.740 |
| **CPU integrated PU21** — fast-ssim2 git main `35f198af`, feature `hdr-pu` | **0.7044** |
| **GPU integrated PU21** — ssim2-gpu `compute_linear_nits` (`XybFlavor::Pu21`), commit `de2ced69` | **0.7040** |
| zensim PU front-end (zensim PR #44 prototype)               |                 0.694 |
| input-feed PU shell (PU-encode fed as *input* to the unchanged cube-root pipeline) | ~0.61 |

Method one-liner: PU21 (`banding_glare`) replaces the cube-root at the
perceptual-encoding layer of the SSIMULACRA2 XYB transform, taking
absolute-luminance linear RGB (cd/m²) directly — versus the shell
variants that PU-encode the *input* and leave the pipeline untouched.
The integrated form recovers ~0.09 SROCC over the input-feed shell and
lands within 0.04 of the literature PU-SSIM bar; GPU matches the CPU
prototype within 0.0004.

Provenance:

- GPU scorer: `crates/ssim2-gpu/examples/upiq_pu_score_cuda.rs` at
  zenmetrics commit `de2ced69` (CUDA, RTX 5070), pipelines cached per
  image dimension, whole-image mode.
- CPU prototype: fast-ssim2 git main `35f198af`, feature `hdr-pu`
  (NOT in the crates.io 0.8.1 release — the CPU umbrella path stays on
  the `PuRescale` u8 shell until a release ships the feature).
- Truth: UPIQ JOD scores (HDR subset), pairs TSV as in issue #25.
- SROCC deltas of ±0.001 across runs are expected (f32 reduction
  nondeterminism does not reorder ranks materially at n = 380).

Routed into production by zenmetrics commit `112a4517`
(`HdrFeeding::IntegratedPuNits`: GPU-class ssim2 only; CPU ssim2 /
dssim / iwssim stay on the u8 PU shell — see `hdr::hdr_feeding` docs
for the per-metric rationale).

## Addendum 2026-06-09 (late): PU-IW-SSIM measured — u8 shell is itself lossy

Same harness (UPIQ HDR subset, n=380, JOD truth; artifacts regenerated under
`/mnt/v/output/zenmetrics/upiq-pu/` after a /tmp wipe). `iwssim-gpu` (CUDA,
BT.601 gray on code values — no linearization, so NOT the ssim2 double-transform
bug), scored over PuRescale-u8 PNGs identical to the production shell.

| variant | SROCC |
|---|--:|
| published PU-SSIM (UPIQ baseline, float PU) | 0.740 |
| plain SSIM (skimage) on our gray PU-u8 PNGs | 0.682 |
| PU(bt709-luma) gray u8 → iwssim-gpu | 0.648 |
| per-channel PU u8, peak=10000 (no clip) → iwssim-gpu | 0.631 |
| **production shell** (per-channel PU u8, peak=1000) → iwssim-gpu | **0.628** |

Decomposition of the 0.74→0.628 gap: **~0.06 = the u8 shell itself**
(quantization + 255/PU(peak) rescale vs float PU values — proven by the plain-SSIM
control landing at 0.682, not 0.74); **~0.02 = per-channel PU vs PU-of-luminance**;
**~0.03 = IW information-weighting on PU-HDR statistics**; highlight clip ≈ 0.

Implications: (1) the earlier "iwssim shell is correct, not a gap" claim is
REVISED — the layer is structurally right but the u8 round-trip costs real
correlation even for bare metrics; (2) open fix: feed PU(luma) as float planes
into the iwssim core (it consumes f32 gray internally) — expected recovery to
~0.68–0.70; the residual IW drag vs plain PU-SSIM is a metric property, not a
feeding bug. Method + scripts: `/mnt/v/output/zenmetrics/upiq-pu/pu_encode_upiq.py`,
`pu_encode_variants.py`.

## Addendum 2 (2026-06-10): float PU(luma) feeding — iwssim jumps to 0.81

`crates/iwssim/examples/upiq_pu_float.rs`: `PU21(bt709-luma(nits)) · 255/PU21(1000)`
fed as **float** gray planes into `Iwssim::score_gray` (the canonical f32 entry the
u8 path itself funnels into) — identical scale to the production shell, zero
quantization. n=380, same UPIQ HDR harness:

| variant | SROCC |
|---|--:|
| **PU-MS-SSIM float** (`iw_flag=false`) | **0.8123** |
| **PU-IW-SSIM float** (`iw_flag=true`) | **0.8076** |
| HDR-VDP-2 (UPIQ baseline) | 0.812 |
| PU-SSIM (published, single-scale) | 0.740 |
| ssim2 integrated PU (GPU) | 0.704 |
| production u8 shell -> iwssim | 0.628 |

The u8 round-trip cost iwssim ~0.18 SROCC (0.628 -> 0.808), far beyond the ~0.06
the single-scale skimage control suggested — multi-scale structure amplifies the
quantization damage. Float PU-MS-SSIM ties HDR-VDP-2 as our best non-learned HDR
score; only learned PU-PieAPP (0.875) is above. Caveat: published PU-SSIM is
single-scale SSIM; ours is multi-scale (a stronger base metric) — the comparison
row to quote against UPIQ baselines is HDR-VDP-2. IW-vs-MS delta is -0.005 (IW
weighting still mildly negative on PU-HDR statistics).

Routing implication: `hdr_feeding(iwssim)` should feed float PU(luma) gray
(`score_gray`), NOT `SdrU8(PuRescale)`. CPU seam exists today; GPU iwssim needs a
gray-f32 ingress (analog of ssim2-gpu's `compute_linear_nits`).

## Addendum 3 (2026-06-10): full Mohammadi panel, apples-to-apples on the identical 380 HDR pairs

SROCC-only rows above are now superseded by the full panel (canonical
`zensim-validate` `panel` binary, built from zensim PR #44 branch c3caf273;
baselines re-paneled from UPIQ's published per-condition objective scores on
the SAME 380-pair HDR subset — baseline SROCCs reproduce the published values
exactly, validating the harness). Per-config TSVs:
`/mnt/v/output/zenmetrics/upiq-pu/panel_*.tsv`.

| metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--:|--:|--:|--:|--:|--:|
| PU-PieAPP (learned) | 0.8748 | 0.8751 | 0.6889 | 0.0000 | 0.9740 | 0.4839 |
| **float PU-MS-SSIM (ours, iw_flag=false)** | **0.8123** | **0.8107** | **0.6206** | 0.0000 | **0.9596** | **0.5855** |
| HDR-VDP-2 | 0.8117 | 0.8011 | 0.6086 | 0.0000 | 0.9558 | 0.5985 |
| **float PU-IW-SSIM (ours, routed)** | 0.8076 | 0.8033 | 0.6155 | 0.0000 | 0.9583 | 0.5956 |
| PU-SSIM (published) | 0.7395 | 0.7369 | 0.5490 | 0.0000 | 0.9335 | 0.6760 |
| PU-FSIM | 0.7185 | 0.7017 | 0.5253 | 0.0000 | 0.9316 | 0.7125 |
| zensim-PU profile A (MLP, X=4) | 0.6935 | 0.6866 | 0.5039 | 0.0053 | 0.9146 | 0.7271 |
| zensim-PU PreviewV0_2 (linear, X=4) | 0.6869 | 0.6920 | 0.5062 | 0.0079 | 0.9073 | 0.7219 |
| zensim-PU PreviewV0_1 | 0.6525 | 0.6677 | 0.4799 | 0.0079 | 0.8874 | 0.7445 |
| PU-PSNR | 0.5485 | 0.6041 | 0.3984 | 0.0079 | 0.8310 | 0.7969 |

Findings the panel adds beyond SROCC:
1. **float PU-MS-SSIM beats HDR-VDP-2 on EVERY stat** (PLCC +0.010, KROCC
   +0.012, PWRC +0.004, Z-RMSE −0.013) — not a tie; a clean full-panel win.
   PU-IW-SSIM also beats HDR-VDP-2 on PLCC/KROCC/PWRC/Z-RMSE. Both second
   only to learned PU-PieAPP.
2. **zensim-PU shows no hidden pathology**: stats agree (~0.69 across rank
   stats, OR ≈ 0, Z-RMSE in family with PU-FSIM). The MLP (A) wins the rank
   stats while linear V0_2 wins PLCC/Z-RMSE — rank-better vs scale-better;
   any future zensim HDR ship should re-fit the output calibration on PU
   scores rather than inherit the SDR spline.
3. zensim-PU sits ~0.11 SROCC below the float-fed SSIM-family — consistent
   with the chroma/X-scale frontier (PU-XYB opponent weighting) rather than
   the luminance encoding; the float-feeding lesson (no u8, no clip)
   is already native to its planar f32 path.

## Addendum 4 (2026-06-10): our cvvdp-gpu + butteraugli rows — display-peak sensitivity

Scored through the production `HdrScorer` route (new example
`crates/zenmetrics-api/examples/upiq_hdr_score.rs`; LinearPlanes + display
model, CUDA, same 380 pairs; butter negated for panel orientation):

| metric (ours) | peak | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|--|--:|--:|--:|--:|--:|--:|
| **cvvdp-gpu** | **10000** | **0.8309** | **0.8340** | **0.6519** | 0.0000 | **0.9634** | **0.5518** |
| cvvdp-gpu | 1000 (HDR_PEAK_NITS default) | 0.7580 | 0.7614 | 0.5884 | 0.0105 | 0.9338 | 0.6483 |
| butteraugli pnorm3 | 10000 | 0.6404 | 0.6974 | 0.5040 | 0.0184 | 0.8640 | 0.7167 |
| butteraugli pnorm3 | 1000 | 0.6281 | 0.6882 | 0.4884 | 0.0184 | 0.8608 | 0.7255 |
| butteraugli max | 1000 | 0.6303 | 0.6679 | 0.4753 | 0.0184 | 0.8698 | 0.7443 |

Findings:
1. **cvvdp-gpu @ display-peak 10000 is our best metric on UPIQ HDR** — beats
   HDR-VDP-2 (0.8117) and float PU-MS-SSIM (0.8123) on every panel stat;
   second only to learned PU-PieAPP (0.8748).
2. **The default 1000-nit display peak costs cvvdp 0.073 SROCC** on this
   content (max 12 528 nits): `LinearPlanes` clamps at the display peak by
   design ("a real peak-nit display can't show more"), but UPIQ's subjective
   data was collected on a ~6000-nit SIM2 display — when correlating against
   subjective data, the display model must match the *experiment's* display,
   not the default. The knob exists (`HdrScorer::new` peak_nits); pick it
   from content/metadata or the dataset's display spec.
3. butteraugli is peak-insensitive (+0.012) and weak on HDR overall (0.63-0.64)
   — its psychovisual model is SDR-referred; intensity_target scales the range
   but not the model. For HDR work prefer cvvdp / float PU-MS-SSIM.

## Addendum 5 (2026-06-10): cvvdp display-peak sweep + butteraugli bug-hunt control

**cvvdp-gpu display-peak sweep** (same 380 HDR pairs, content max 12 528 nits;
UPIQ's narwaria/korshunov subjective data was collected on SIM2 displays
~4000–6000 nits):

| display peak | SROCC | PLCC | KROCC | PWRC | Z-RMSE |
|--:|--:|--:|--:|--:|--:|
| 1000 (HDR_PEAK_NITS default) | 0.7580 | 0.7614 | 0.5884 | 0.9338 | 0.6483 |
| 4000 | 0.8153 | 0.8231 | 0.6367 | 0.9595 | 0.5679 |
| 6000 (≈ experiment display) | 0.8245 | 0.8292 | 0.6454 | 0.9620 | 0.5589 |
| **10000 (PQ max, no clip)** | **0.8309** | **0.8340** | **0.6519** | **0.9634** | **0.5518** |

Monotone PAST the physical display: "match the experiment's display" recovers
most of the loss (0.758→0.825) but unclamped is better still — the
LinearPlanes clamp always destroys signal the metric could use. Guidance: for
metric-correlation use, set the cvvdp display peak to content-max / PQ-max
(no clipping); reserve display-matched peaks for display-referred QA
questions.

**butteraugli implementation control** (is 0.63 on HDR a bug?): re-ran today's
butteraugli-gpu over TID2013 (n=3000, local corpus, MOS truth):
`butter_max 0.6683 / pnorm3 0.6622` vs our committed 2026-05-01 baseline
`0.6696` — Δ0.001, exact reproduction. The SDR→HDR drop is only ~0.03–0.04,
consistent with an SDR-referred psychovisual model out of domain, NOT an
implementation defect. (No published butteraugli-on-UPIQ exists to compare
against; the self-consistency control is the instrument.) Verdict: our
butteraugli is healthy; it is simply not an HDR metric — route HDR to
cvvdp(no-clip) / float PU-MS-SSIM.

## Addendum 6 (2026-06-10): paired-significance correction + source-paper alignment

Read Mantiuk & Azimi 2021 (PU21, PCS) and Hanji et al. 2022 (SIGGRAPH "caveats
of quality assessment") + supplement. Their core discipline — small metric
deltas cannot rank methods; even a careful protocol could not order its top-8
metrics at 95% confidence — applied to OUR table via paired bootstrap
(B=2000, same 380 pairs):

| comparison | ΔSROCC | 95% CI | verdict |
|---|--:|--|--|
| float PU-MS-SSIM − HDR-VDP-2 | +0.0005 | [−0.030, +0.029] | **TIE** (retracts addendum-3's "beats on every stat") |
| float PU-IW-SSIM − HDR-VDP-2 | −0.004 | [−0.034, +0.022] | tie |
| cvvdp@10k − HDR-VDP-2 | +0.019 | [−0.017, +0.052] | suggestive, NOT proven (P=0.87) |
| cvvdp@10k − float PU-MS-SSIM | +0.019 | [−0.006, +0.045] | not proven (P=0.93) |
| **cvvdp@10k − cvvdp@6000** | **+0.0065** | **[+0.0015, +0.0128]** | **SIGNIFICANT** — no-clip > display-matched survives rigor |

CORRECTED conclusions: the top cluster {cvvdp@10k, float PU-MS-SSIM, float
PU-IW-SSIM, HDR-VDP-2} is statistically indistinguishable at n=380; ordering
within it is noise. What IS established (large effects, far beyond CI width):
float feeding >> u8 shell (+0.18); every top-cluster member >> PU-SSIM 0.740;
unclamped display peak > clamped for cvvdp (paired same-metric test).

Source-paper takeaways recorded for this work:
1. PU21 paper validates our exact choices on the same UPIQ HDR pairs:
   banding_glare is their measured-best variant; PU values are >= 0 by design
   (MS-SSIM-safe); derivation is luminance-based (per-channel RGB is a
   pragmatic extension — consistent with our PU-of-luma > per-channel
   measurement); their UPIQ protocol feeds absolute units, no tone mapping.
2. Caveats paper: their 1000-nit clamp was experiment-display-matched (PQ1000
   monitor) — consistent with our display-matching finding, and our
   unclamped-wins result reflects UPIQ's brighter SIM2 display.
3. Caveats: SDR metrics on LINEAR HDR values are significantly worse than any
   PU/mu-law adaptation (never feed linear to SDR metrics); tone-map-then-
   metric also degrades several metrics; PU21 vs mu-law: no significant
   difference in their task (PU21 kept for perceptual basis).
4. For future reconstruction-style HDR eval (ultrahdr gain-map work):
   global tone/color error dominates FR metrics — apply their CRF/polynomial
   correction (3rd-degree poly on PQ-luma + u'v' chroma, Tikhonov toward
   identity) before scoring, or rankings reflect CRF inversion error, not
   artifact quality. Minimum-measurable-increment idea: report the metric
   delta needed for alpha=0.05 alongside any method comparison.
5. PU21-VSI and PU21-PIQE (no-reference) were their best performers — VSI
   (saliency-weighted) is a candidate addition to our metric set.

## Addendum 7 (2026-06-10): PU-SSIM from-pixels replication attempt — NOT reproduced; provenance reframe

We replicate UPIQ's *correlation pipeline* exactly (their released per-condition
scores → our JOD join reproduces 0.8748/0.8117/0.7395/0.7185/0.5485 to 4
decimals). We do NOT reproduce their PU_SSIM=0.7395 when computing from pixels:

| our from-pixels attempt (skimage SSIM, gaussian 11×11 σ1.5) | SROCC |
|---|--:|
| float PU21(luma), L = full PU range | 0.6619 |
| float PU21(luma)·255/PU(1000), L=255 | 0.6855 |
| u8-quantized (earlier control) | 0.682 |
| Wang-style 4× downsample variant | 0.6205 (falsified) |
| +0.6 nit display black-lift variant | 0.6653 (falsified) |

Float ≈ u8 here ⇒ quantization was NOT the single-scale-SSIM gap (it was
decisive only for multi-scale: iwssim 0.628→0.808). Remaining explanation:
**provenance** — UPIQ's objective scores (Mikhailiuk et al. 2020) PREDATE PU21
(2021); their released `PU_SSIM` is the PU08 encoding under their MATLAB
protocol. PU08 differs from PU21 most at low luminance (negative values below
0.5 cd/m², different shape) — plausibly worth the ~0.05–0.08 on dark narwaria
content. Full replication would require running gfxdisp's `pu21`/UPIQ
benchmark wrappers on these pairs (open, low priority).

**Consequence for the ssim2-vs-ssim question**: like-for-like inside ONE
pipeline, SSIMULACRA2 IS better than single-scale SSIM on PU-HDR — integrated
PU-ssim2 0.704 vs our-pixels PU21-SSIM 0.66–0.69. It only looked worse against
the published 0.740 because that number is a different encoding + protocol.
The real internal anomaly is MS-SSIM (0.812) > ssim2 (0.704): ssim2's TRAINED
weights/error-norms are calibrated to cube-root-XYB SDR statistics (domain
shift under PU-HDR) and its chroma channels add PU-domain noise (every strong
HDR performer here is luminance-only), while MS-SSIM is untrained, luma-only,
and robust. Retuning ssim2's weights on PU-HDR data is the obvious (data-
blocked) lever — same blocker as zensim-PU's 0.694.
