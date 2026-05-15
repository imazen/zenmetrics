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
2. **Chromatic CSF interp / LUT data** — **FALSIFIED (tick 193)**.
   Compared our vendored `csf_lut/v0_5_4.rs` arrays against
   pycvvdp's installed `csf_lut_weber_fixed_size.json`:
   axes (LOG_L_BKG_AXIS, LOG_RHO_AXIS) and all three channel
   matrices (o0_c1/c2/c3) agree to **5e-11 precision** —
   well below the f32 noise floor.
   Compared interp methods: pycvvdp uses **`interp1q`**
   (uniform-axis rescale) on the L_bkg axis and
   **`batch_interp1d`** (torch.searchsorted, binary) on the rho
   axis. Our host_scalar `interp1_clamped` uses binary search
   on both. For a uniformly-sampled axis (LOG_L_BKG_AXIS is
   uniform in log space, stride ~0.2032), both methods produce
   the same `t` value and bracket — mathematically equivalent.
   Not the source.
3. **CH_GAIN per-channel weights** — **FALSIFIED (tick 192)**.
   pycvvdp's mult-mutual path (the default masking_model in
   cvvdp_parameters.json) uses
   `ch_gain = [1, 1.45, 1, 1.]` applied as `T_p = T * S * ch_gain`.
   Our `CH_GAIN = [1.0, 1.45, 1.0]` matches byte-for-byte for the
   3-channel still-image case. (The `ch_chrom_w` config field at
   1.0 is a different weight applied to the per-channel pool,
   not the CSF-stage ch_gain.) **Not the source.**

## Remaining candidates

All three original candidates are now falsified. The drift must
come from a stage I haven't yet enumerated. New candidates to
investigate (tick 194+):

- **Display model luminance computation**: cvvdp's
  `apply_display_model` maps sRGB byte → linear → cd/m² via
  `Y_peak * (R + R_refl)` style math. A small constant difference
  in `Y_peak`, `Y_black`, or `Y_refl` between our
  `DisplayModel::STANDARD_4K` and pycvvdp's display config
  could shift the chromatic channels disproportionately because
  the chromatic CSF response is steeper at low luminance.
- **Band-multiplier (`band_mul`) rule**: We apply `band_mul=2.0`
  on non-edge non-baseband levels (cvvdp's `lpyr.get_band`
  doubles non-edge bands). Edge cases at the smallest /
  largest levels — what if pycvvdp's rule differs slightly?
- **Pool weights**: `BASEBAND_W` (3-channel still-image
  `baseband_weight`) and `PER_CH_W` ([1.0, 1.0, 1.0] for 3ch)
  multiply per-band contributions before the L_p fold.
  Worth diffing against `cvvdp_parameters.json`'s
  `baseband_weight` field.

## Direct next-step idea

The cleanest diagnostic: spool up `dump_channels.py` from
pycvvdp on the chroma_shift fixture, then have our host_scalar
dump the same intermediate stages. Compare stage-by-stage to
find where the first divergence appears.

The Burn port (`BURN_PORT_PLAN.md`) remains the most robust
resolution path: by calling pycvvdp's exact matmul / conv ops
through Burn, the drift sources at every untested constant or
clamp would be eliminated wholesale.

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
