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
