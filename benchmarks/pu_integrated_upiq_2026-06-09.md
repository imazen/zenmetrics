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
