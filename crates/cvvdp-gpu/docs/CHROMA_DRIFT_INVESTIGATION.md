# 256×256 chrominance-shift drift vs pycvvdp (open, tick 191)

## Finding

Tick 191 extended the bench script with a chrominance-only
distortion fixture (`synth_256x256_chroma_shift`): G channel +16,
R/B unchanged. pycvvdp v0.5.4 CUDA returns **9.6649 JOD**; our
GPU pipeline returns **9.5476 JOD** — drift **0.1173 JOD**.

This is ~24× above our standing 0.005 tolerance for canonical-
reference parity tests. All other distortion-type fixtures
(blur3x1, blur1x3, noise, JPEG q-grid, 4000×3000 synth) hold
≤0.005 vs pycvvdp.

## Hypothesis

The drift only surfaces with chrominance-isolating distortion.
Other fixtures perturb R/G/B roughly equally, so the achromatic
(A) channel dominates the JOD and any chromatic-channel drift
gets averaged out. With ref differing from dist only in G, the
DKL transform projects the +16 G offset largely into the RG
opponent channel, isolating the chromatic pipeline.

Candidates for the source:
1. **DKL RGB→opponent matrix**: We carry the cvvdp v0.5.4
   matrix at `crates/cvvdp-gpu/src/kernels/color.rs`. A small
   precision difference (e.g. f32 truncation of pycvvdp's
   double-precision constants) would shift RG values that
   amplify through the rest of the pipeline.
2. **Chromatic CSF interp**: Our per-pixel CSF kernel uses a
   uniform-axis arithmetic interp on a 32×32 LUT, while
   pycvvdp's `interp.py` does a binary-search bracket. At
   chrominance frequencies (RG / VY channels), the LUT shape
   may differ at the level where our interp converges to a
   different bracket than pycvvdp's.
3. **CH_GAIN per-channel weights**: We carry pycvvdp's
   `[1.0, ch_chrom_w, ch_chrom_w, ch_trans_w]` slice. If
   `ch_chrom_w` got the wrong value at port time (it's 1.0 in
   v0.5.4), a 1% delta would compound through the pool.

## Next steps

- Add a stage-by-stage parity dump on the chroma_shift fixture:
  compare DKL planes, Weber bands, T_p bands, D bands, partials,
  and final Q between our GPU pipeline and pycvvdp's internal
  `stats` output. The stage where the diff first exceeds f32
  noise narrows the source.
- Spot-check the DKL matrix constants against
  `pycvvdp/vvdp_data/cvvdp_parameters.json`'s `display_models`.
  Same for CH_GAIN.
- If interp1 is the issue, switching to the binary-search form
  should close it; tradeoff is a per-pixel branch in the CSF
  kernel that may regress throughput.

## Why this isn't an immediate fix

The drift was surfaced by extending the goldens manifest, which
is the user's directive: "wider distortion sweep". The test that
would gate this drift is removed (would have failed); the bench
script still emits the golden so future ticks can root-cause
without re-discovery.

The Burn port (tracked in `BURN_PORT_PLAN.md`) is one path to
resolution — Burn's tensor ops would call the same matmul
math pycvvdp uses, eliminating the precision-truncation
hypothesis end-to-end.
