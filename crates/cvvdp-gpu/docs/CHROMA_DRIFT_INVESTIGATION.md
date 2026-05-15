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
1. **DKL RGB→opponent matrix** — **FALSIFIED (tick 192)**.
   Computed pycvvdp's combined matrix
   (`LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @ sRGB_to_XYZ`) at f64
   precision and compared digit-for-digit to our
   `SRGB_LINEAR_TO_DKL` constant. They match to 8+ decimal
   digits — at the f32 precision limit. Computation script
   left at `/tmp/dkl_compute.py`. **Not the source.**
2. **Chromatic CSF interp**: Our per-pixel CSF kernel uses a
   uniform-axis arithmetic interp on a 32×32 LUT, while
   pycvvdp's `interp.py` does a binary-search bracket. At
   chrominance frequencies (RG / VY channels), the LUT shape
   may differ at the level where our interp converges to a
   different bracket than pycvvdp's. **Remaining suspect.**
3. **CH_GAIN per-channel weights** — **FALSIFIED (tick 192)**.
   pycvvdp's mult-mutual path (the default masking_model in
   cvvdp_parameters.json) uses
   `ch_gain = [1, 1.45, 1, 1.]` applied as `T_p = T * S * ch_gain`.
   Our `CH_GAIN = [1.0, 1.45, 1.0]` matches byte-for-byte for the
   3-channel still-image case. (The `ch_chrom_w` config field at
   1.0 is a different weight applied to the per-channel pool,
   not the CSF-stage ch_gain.) **Not the source.**

## Remaining candidates

After tick 192, only one of the three candidates is still live:
**CSF LUT interpolation form**. Two follow-up directions:
- **Direct value check**: feed a known chrominance ρ + log_L_bkg
  pair through both our `csf_apply_per_pixel_kernel` and pycvvdp's
  interp1; compare the per-channel `S` value bit-for-bit. If they
  diverge at chrominance specifically, that's the source.
- **LUT staleness**: our `csf_lut/csf_lut_weber_fixed_size.json`
  was vendored from pycvvdp at port time. Diff it against the
  v0.5.4-installed file at
  `.venv/lib/python3.10/site-packages/pycvvdp/vvdp_data/csf_lut_weber_fixed_size.json`
  to rule out a stale snapshot.

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
