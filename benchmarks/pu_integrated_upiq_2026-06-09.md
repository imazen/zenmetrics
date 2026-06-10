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
