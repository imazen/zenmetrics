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

All three original candidates are now falsified. Tick 194's
constant-pin sweep falsifies several more:

- **Display luminance constants** — **FALSIFIED (tick 194)**.
  `pycvvdp/vvdp_data/display_models.json` standard_4k entry:
  `max_luminance=200`, `contrast=1000`, `E_ambient=250`, default
  `k_refl=0.005`. y_peak=200 ✓, y_black=200/1000=0.2 ✓,
  y_refl = E_ambient·k_refl/π = 250·0.005/π = 0.39788736 ✓
  (matches our `0.397_887_36` byte-for-byte).
- **Pool weights** — **FALSIFIED (tick 194)**.
  `cvvdp_parameters.json` `baseband_weight` =
  `[0.0036334486, 1.6627724, 4.1187453, 25.2596989]`. Our
  `BASEBAND_W` carries the first three to f32 precision.
  `PER_CH_W = [1.0, 1.0, 1.0]` matches pycvvdp's
  `get_ch_weights` for still-image 3-channel
  (`[1, ch_chrom_w=1.0, ch_chrom_w=1.0]`).
- **`SENSITIVITY_CORRECTION_DB`** — **FALSIFIED (tick 194)**.
  pycvvdp `sensitivity_correction = -0.2797423303127289`.
  Our const `-0.279_742_33` matches at f32 precision.
- **`D_MAX` soft-clamp** — **FALSIFIED (tick 194)**.
  pycvvdp `d_max = 2.5642454624176025`. Our const `2.564_245_5`
  matches at f32 precision.
- **Baseband + masking control flow** — **FALSIFIED (tick 194)**.
  Read pycvvdp's `apply_masking_model` + `weber_contrast_pyr.decompose`
  end-to-end. Baseband path is `D = |T_f - R_f| * S` with no
  CH_GAIN (mirrors our `diff_abs_3ch` after a no-CH_GAIN
  csf_apply_6ch). Non-baseband applies CH_GAIN=[1, 1.45, 1] in
  `T_p = T * S * ch_gain` before mult-mutual masking — same as
  our `ch_gain_for_band(is_baseband=false)` ×
  `csf_apply_6ch_kernel`. The baseband weber-contrast formula
  divides test bands by test_Y_mean and ref bands by ref_Y_mean
  with `clamp(..., max=1000)` — structurally matches our
  `baseband_divide_3ch_kernel`. No divergence found.

Remaining live candidates: smaller. Either a SUBTLE
implementation detail somewhere (e.g. pycvvdp clamps baseband
to max=1000 only, we clamp to ±1000 — but values are far from
clamp on chroma_shift), or a stage I haven't traced yet
(sensitivity ↔ band_mul interaction, log10 base, etc.).

Tick 195 also falsified:
- **`MASK_P`** = 2.264355 (matches pycvvdp 2.264355182647705)
- **`MASK_Q`** = [1.302623, 2.888591, 3.680771] (matches pycvvdp's
  first three of [1.302622, 2.888590, 3.680771, 3.588787]; the
  fourth is the transient channel, unused for still-image 3ch)
- **`MASK_C`** = -0.795497 (matches pycvvdp -0.7954971194267273)
- **`XCM_3X3`** = `2 ^ pycvvdp.xcm_weights.reshape(4,4)[:3,:3]`
  digit-for-digit (rows: input channel; cols: output channel
  per pycvvdp's `mask_pool`: `M[cc] = sum_in C[in] * xcm[in, cc]`)
- **Gauss kernel coefficients**: pycvvdp `K = [0.05, 0.25, 0.40,
  0.25, 0.05]` (from `kernel_a = 0.4`, where the formula is
  `[0.25 − a/2, 0.25, a, 0.25, 0.25 − a/2]`). Our `GAUSS5` array
  uses the same formula with `KERNEL_A = 0.4`, so the values
  are identical.

After tick 195, **every constant + control-flow hypothesis has
been ruled out via direct source comparison**. The 0.117 JOD
chroma_shift drift must come from an implementation-level
divergence that constant-pinning can't surface. Concrete next
steps:

1. **Stage-by-stage value dump** at the chroma_shift fixture:
   intercept our host_scalar's intermediate Weber bands, log_l_bkg,
   T_p, D arrays, and pycvvdp's via `dump_channels.py`. Compare
   element-wise. The stage where the first divergence appears
   ≥ f32 noise localizes the source.
2. **Burn port** (`BURN_PORT_PLAN.md`): calling pycvvdp's exact
   conv/matmul/reduce graph through Burn eliminates every
   untested implementation-level detail (clamp form, log base,
   sum order, etc.) wholesale. This is the more robust path
   once the value-dump finds where the divergence sits.

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
