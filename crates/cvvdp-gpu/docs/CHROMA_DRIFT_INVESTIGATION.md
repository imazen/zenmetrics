# 256×256 chrominance-shift drift vs pycvvdp (RESOLVED, tick 204)

> **Note on file references**: Below, `examples/chroma_shift_drift_probe.rs`
> refers to the probe file as it existed during the investigation
> (ticks 191-208). The file was renamed to
> `examples/manifest_parity_probe.rs` in tick 229 to reflect that
> it now walks all 6 manifest fixtures rather than just chroma_shift.
> Historical references are preserved as-written so the timeline
> stays accurate.

## Resolution — pycvvdp overrides baseband CSF rho to 0.1 cy/deg

**Root cause**: `cvvdp_metric.py:628` does `rho_band[bb] = 0.1`
for the baseband. Our pipeline used the geometric
`band_frequencies(ppd, w, h)` value (0.190 at 256²) instead.
For chromatic channels (RG, VY) the low-rho CSF gives high S
values; using the wrong rho lowered our baseband D → lowered
Q_sc[RG/VY] → lowered Q_tc → lowered JOD.

**Fix** (tick 204):
- New `pub const CSF_BASEBAND_RHO: f32 = 0.1` in
  `kernels/csf.rs`.
- `host_scalar::predict_jod_still_3ch` uses
  `CSF_BASEBAND_RHO` for the baseband CSF lookup.
- `Cvvdp::new` builds `logs_row[last]` from
  `precompute_logs_row(CSF_BASEBAND_RHO, ...)` — the GPU CSF
  apply kernels consume this row at baseband.

**Verification**: `examples/chroma_shift_drift_probe.rs`:
```
cvvdp-gpu (current):  9.664865
cvvdp-gpu host_scalar: 9.664865
pycvvdp golden:       9.664865
GPU - pycvvdp:        +0.000000
host - pycvvdp:       +0.000000
```

Drift CLOSED to f32 precision. Re-enabled the
`compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift`
test at the standard 0.005 JOD tolerance.

## Follow-up: 73×91 odd-dim residual — CLOSED (tick 206)

**Root cause**: pycvvdp's `gausspyr_reduce` has a parity-check bug.
Looking at the horizontal-pass right-column patch in
`pycvvdp/lpyr_dec.py:204-209`:

```python
if (x.shape[-2] % 2)==1: # odd number of columns   ← comment says columns
    y[...,:,-1] += y_a[...,:,-1]*K_horiz[...,0,3] + y_a[...,:,-2]*K_horiz[...,0,4]
else: # even number of columns
    y[...,:,-1] += y_a[...,:,-1]*K_horiz[...,0,4]
```

The check uses `x.shape[-2]` which is the **input ROW count**, not
column count. The comments say "odd/even number of columns" but the
implementation tests rows. For mixed-parity inputs (e.g. 6×5 → 3×3
at level 4→5 of the 73×91 pyramid; or 46×37 → 23×19 at level 1→2),
pycvvdp applies the wrong patch.

The reduce at 73×91's right-column baseband:
- Pycvvdp (with bug): right-col gauss[3×3] = [27.18, 43.55, 45.90]
- Our reflect (correct math): [36.78, 59.66, 62.82]

This propagates through Weber/CSF/D/pool to yield the ~0.006 JOD
residual.

**Fix** (tick 206):
- `host_scalar` `gausspyr_reduce_scalar`: rewritten from pure
  reflection to zero-pad + explicit pycvvdp-bug-compatible patches
  (vertical patches by sh parity, horizontal first-col always,
  horizontal last-col by sh parity = the bug).
- GPU `downscale_kernel`: keeps the reflect-based main path (it
  matches pycvvdp for all same-parity-sw/sh inputs, which covers
  the 4K + 256² corpus) plus a delta correction at the right
  column when `sw` and `sh` parities differ:
  - `sw odd, sh even`: subtract 0.05·vscratch[sw-2] + 0.20·vscratch[sw-1]
  - `sw even, sh odd`: add 0.05·vscratch[sw-2] + 0.20·vscratch[sw-1]

**Verification**:
- `compute_dkl_jod_matches_pycvvdp_at_73x91_odd` (new): GPU JOD
  9.3904 = pycvvdp golden 9.3904 at f32 precision (diff = 0.0000).
- All 23 pipeline_color tests pass.
- chroma_shift, 12 MP synth, 256² blur1x3 / blur3x1 / noise all
  still pass at 0.005 JOD tolerance (unchanged; even+even inputs
  hit no parity mismatch).

## Earlier (resolved) follow-up: 73×91 odd-dim residual (tick 205, was open)

After ticks 196-204 closed the chroma_shift drift to f32 precision,
the canonical 256² and 4K synth fixtures all pass pycvvdp parity
at the standard 0.005 JOD tolerance. The 73×91 odd-dim fixture is
the one outlier:

| Call (ref, dist) | Ours    | pycvvdp | diff   |
|------------------|---------|---------|--------|
| (ref, dist)      | 9.38409 | 9.39608 | 0.012  |
| (dist, ref)      | 9.38988 | 9.39037 | 0.0005 |

Note both implementations are asymmetric here (different JOD when
test/ref roles swap) because the 73×91 synth_pair_odd_dim fixture's
dist has direction-asymmetric perturbations per channel
(R−8, G−4, B+12) — `predict(ref, dist) ≠ predict(dist, ref)` is
expected from the metric. The 0.006-0.012 JOD residual sits in our
chain somewhere distinct from chroma_shift's baseband CSF rho.

Candidates for the source (small-image specific):
- Gauss pyramid boundary handling at very small bands (5×6, 3×3).
- log_l_bkg statistic at the 3×3 baseband (only 9 pixels — pycvvdp
  uses spatial-mean for `weber_g1` baseband; our host_scalar uses
  the per-pixel L_bkg of the achromatic plane, which at 3×3 is
  effectively a few-pixel average too).
- PU blur skip threshold at bands ≤ 6 pixels per axis (our
  threshold is `w > 6 && h > 6` — strictly greater, matching
  pycvvdp's `pu_padsize = 6`).
- Edge effects in the 5-tap separable Gaussian at very small
  bands (5×6 has 5 wide — kernel radius 2 means every pixel
  touches the boundary).

Verified other fixtures still pass at 0.005:
- 4000×3000 synth (12 MP)        ≤ 0.005
- 256×256 blur3x1                ≤ 0.005
- 256×256 blur1x3                ≤ 0.005
- 256×256 noise                  ≤ 0.005
- 256×256 chroma_shift           = 0.000000 (tick 204)

This is a smaller drift than chroma_shift was (0.012 vs 0.117), and
only triggers on odd dimensions ≤ ~100. Tracked for a future
investigation tick. No code change needed to ship the chroma_shift
fix — the existing odd-dim test `compute_dkl_jod_matches_host_scalar_on_odd_dims`
still gates GPU vs host_scalar at 0.005 (passes; our two paths
agree even though they jointly diverge from pycvvdp).

## Earlier history

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

## Tick 196 — found a real LUT bug, narrowed the gap

`compute_dkl_planes_matches_pycvvdp_dkl_at_chroma_shift_sentinels`
(new test) compared our DKL planes to pycvvdp's at 10 sentinel
pixels of the chroma_shift fixture. Low-byte sentinels matched
exactly; high-byte sentinels diverged by up to **0.198 cd/m²**
(at byte 232).

Root-caused to **`SRGB8_TO_LINEAR_LUT` having wrong values at
high bytes**:
- byte 217: ours **0.6941793** vs correct **0.6938717365** (+3.1e-4)
- byte 230: ours **0.7919172** vs correct **0.7912979126** (+6.2e-4)
- byte 232: ours **0.8076336** vs correct **0.8069522381** (+6.8e-4)

The constants were copied from
`zensim-gpu::kernels::color::SRGB8_TO_LINEARF32_LUT` at port time.
The doc comment named the right formula
(`((p + 0.055) / 1.055)^2.4` for p > 0.04045) but the literal
numbers diverged from the formula's outputs. Regenerated from
the canonical sRGB EOTF at f64 → f32, replaced our LUT.

After the fix:
- DKL planes now bit-identical with pycvvdp at all 10 sentinels
  (max diff 3.8e-5, pure f32 noise). Locked by the new
  parity test in CI.
- All 74 existing tests still pass — the LUT-value differences
  are well inside the f32-precision margins of the other
  fixtures (12 MP synth, blur3x1, blur1x3, noise, JPEG q-grid).
- **The chroma_shift JOD still drifts by 0.117** (9.5474 vs
  9.6649). With DKL bit-identical, the divergence is now
  **downstream of color transform**.

Next-tick target: instrument the Weber-contrast pyramid output
on the chroma_shift fixture. If our weber bands match pycvvdp's,
drift is further downstream (CSF / masking / pool).

## Tick 197-198 — weber bit-identical, drift downstream

Tick 197 added the band-0 weber-stage parity probe. Result:
max diff 2.3e-7 — bit-identical with pycvvdp.

Tick 198 extended to all 8 bands:
- band 0: max diff 2.3e-7
- band 1: max diff 2.7e-7
- band 2: max diff 2.1e-7
- band 3: max diff 1.1e-7
- band 4: max diff 2.2e-7
- band 5: max diff 2.0e-7
- band 6: max diff 2.2e-7
- band 7 (baseband): max diff 2.4e-7

All bands bit-identical with pycvvdp across all 10 sentinel
pixels. **The 0.117 JOD chroma_shift drift is now provably
downstream of the entire pyramid stage.**

The complete sequence:
- DKL (sRGB → DKL): ✓ bit-identical (tick 196, after LUT fix)
- Gauss pyramid reduce: ✓ implicit (Weber consumes its output)
- Gauss pyramid expand: ✓ implicit
- Weber-contrast subtract + L_bkg divide: ✓ bit-identical
- log_l_bkg: ✓ implicit (same input as weber)

So the drift is in one (or more) of:
- CSF apply (T_p = weber · S · ch_gain · band_mul)
- mult-mutual masking + PU blur
- Spatial pool (L_p norm with safe_pow)
- 3-stage Minkowski fold
- met2jod piecewise

Most likely candidate: CSF apply, since at chroma_shift the
chromatic channels (RG, VY) carry most of the signal and the
CSF interp at chrominance frequencies + log_L_bkg is the
densest part of the downstream pipeline.

## Tick 199 — CSF apply has ~0.9% rel divergence

Tick 199 added `compute_dkl_t_p_bands_ref_matches_pycvvdp_at_chroma_shift_all_bands`:
dumps pycvvdp's T_p (= weber · S · ch_gain) on the REF side
across all 8 bands. (We compare REF only because pycvvdp uses
REF's log_l_bkg for the S lookup in both T_test_p and T_ref_p
computation, while our compute_dkl_t_p_bands uses each-call's
input. The JOD path is fine — it uses REF's log_l_bkg for both
sides via the bands_dis split.)

Result on chroma_shift:
- band 0 REF: max abs 3.2e-3, rel 1.0e-3
- band 1 REF: max abs 4.1e-2, rel 6.3e-4
- band 2 REF: max abs 8.8e-2, rel 3.6e-4
- band 3 REF: max abs 7.1e-2, rel 3.6e-4
- **band 4 REF: max abs 7.8e-3, rel 8.8e-3** (worst rel)
- band 5 REF: max abs 1.3e-3, rel 8.9e-3 (also worst rel)
- band 6 REF: max abs 2.8e-4, rel 6.4e-4
- band 7 REF: max abs 3.1e-3, rel 3.4e-4

Max relative drift: **0.89%** at band 4-5.

This is small but real. Pure f32 noise would be ~1e-7 relative;
0.9% is well above that. The chain  rho → log10(rho) →
LUT interp → 10^ → × sens_corr_factor has ~5 f32 operations,
each at f32-precision noise (1e-7). 0.9% relative means there's
a real algorithmic divergence, not f32 noise.

Most likely source: subtle f32 ordering in our `interp1_clamped`
vs pycvvdp's `interp1q` (uniform-axis rescale). For LOG_L_BKG_AXIS
which IS uniform in log space, the two methods are
mathematically equivalent — but f32 storage may make the stored
axis values slightly non-uniform (e.g. -2.301029 vs -2.301030 at
ULP boundaries). Binary search on slightly-non-uniform values
can pick a different bracket than uniform-rescale.

Why this becomes 0.117 JOD: a 0.9% relative T_p drift propagates
through pool (lp_norm with p=2, preserves relative error) and
the 3-stage Minkowski fold (also p=2-4, near-preserves). Then
met2jod maps Q → JOD with a STEEP local slope at chroma_shift's
Q-value (around the kink at Q=0.1, jod_a/p_e regime). A 0.9%
relative input shift at that slope produces a 0.1+ JOD output
shift.

The fix path: switch our `interp1_clamped` to a uniform-axis
form (compute `ind = (x_q - x[0]) / (x[-1] - x[0]) * (N-1)` and
`imin = floor(ind)`, matching pycvvdp's interp1q exactly). Or
use the same form pycvvdp uses to remove the implementation
divergence wholesale.

## Tick 200 — host_scalar interp1_uniform on L_bkg

Implemented the uniform-rescale form for `LOG_L_BKG_AXIS`
(`interp1_uniform` in `kernels/csf.rs`); the rho axis stays on
binary search (its first interval has ratio 0.3228 vs the
regular 0.5 — not uniformly log-spaced). All 78 parity tests
still pass after the swap.

## Tick 201 — drift did NOT close

Re-measured on the chroma_shift fixture via
`examples/chroma_shift_drift_probe.rs`:

```
cvvdp-gpu (current):  9.547440
pycvvdp golden:       9.664865
abs diff:             0.117425
```

**Bit-identical to the pre-tick-200 number** (0.1174). So the
L_bkg interp form was not the source — the T_p REF-side 0.89%
relative drift comes from somewhere else in the CSF apply step,
or the drift surfaces further downstream.

Remaining hypotheses (in order of suspicion now):
1. **CSF apply step** still has a non-interp divergence
   (sensitivity_correction order? f32 cast of LUT-returned
   value? `10^(sens_corr/20)` vs `pow10` form?).
2. **Masking model (`mult-mutual`)** — cross-channel pooling
   via `XCM_3X3` and the `mask_pool` step. T_p divergence may
   amplify through the |T_test| − |T_ref| difference path.
3. **f32 accumulation order in pool** — our `Atomic<f32>::fetch_add`
   in `pool_band_kernel` has non-deterministic reduce order.
   pycvvdp uses torch's deterministic sum. For chroma_shift
   where most-bands' contributions are similar magnitude,
   accumulation order could matter for the 4th decimal.

## Tick 201 update — D bands DIVERGE at band 4 (7% rel)

Stage-4 parity probe shipped:
- `scripts/cvvdp_goldens/dump_d_chroma.py` produces
  `pycvvdp_d_chroma_shift.json` (post-masking, post-PU-blur,
  pre-pool D values at 10 sentinel pixels per band).
- `compute_dkl_d_bands_matches_pycvvdp_at_chroma_shift_all_bands`
  parity test verifies against the golden.

Per-band rel diff (cvvdp-gpu vs pycvvdp at chroma_shift):
- band 0 D: rel **2.3e-3**
- band 1 D: rel 8.2e-4
- band 2 D: rel 6.3e-4
- band 3 D: rel 7.6e-4
- **band 4 D: rel 7.0e-2** (worst — 8× amplification vs T_p's 0.9%)
- band 5 D: rel 2.1e-2
- band 6 D: rel 1.4e-2
- band 7 D (baseband): rel 8.0e-4

**Verdict**: The masking model **amplifies** the T_p drift by
roughly 8× where T_p's rel error peaks (band 4-5). Given
`D_u = safe_pow(|T_p-R_p|, mask_p) / (1 + M)` with
`mask_p = 2.264`, the amplification path is:

1. T_p_test - T_p_ref: at chroma_shift the diff is small (RG
   shifted by 16/255 ≈ 6%) — a 0.9% rel error in either T_p
   becomes a much larger rel error in the residual when
   `|T_p_test - T_p_ref|` is much smaller than `T_p`.
2. `^mask_p` (= 2.26): doubles the rel residual error.
3. (1 + M) denominator: M can be O(1), so it dampens but
   doesn't restore.

So the **root cause is still in the CSF apply step** (the
0.9% T_p drift), but it surfaces dramatically in D at the
band where T_p's rel error peaks. Tick 200's L_bkg
interp1_uniform didn't help — the rho-axis CSF lookup or
the sensitivity_correction application is the next suspect.

## Tick 202 update — host_scalar S matches pycvvdp at f32 noise

Stage-5 raw-S parity probe shipped:
- `scripts/cvvdp_goldens/dump_s_chroma.py` produces
  `pycvvdp_s_chroma_shift.json` with per-band raw S values
  (pre-sens_corr) at 10 sentinel pixels, plus pycvvdp's
  per-pixel `log_l_bkg_ref` so the host comparison runs on
  the same inputs.
- `sensitivity_scalar_matches_pycvvdp_raw_csf_at_chroma_shift_all_bands`
  parity test feeds the SAME `log_l_bkg_ref` into our
  `sensitivity_scalar(rho, log_l_bkg, cc)` and compares.

Per-band rel diff (our host_scalar S vs pycvvdp raw S):
- band 0: rel **8.1e-7**     band 4: rel **5.0e-7**
- band 1: rel 1.1e-6          band 5: rel 6.2e-7
- band 2: rel 6.1e-7          band 6: rel 1.0e-6
- band 3: rel 5.9e-7          band 7: rel 5.4e-7

**Verdict**: host_scalar's CSF lookup is **bit-identical to
pycvvdp at the f32 noise floor (1e-6 rel)**. The CSF table +
interp implementation is not the source.

So where does the 0.9% T_p drift come from? Combining
established findings:

- Weber bands: bit-identical (tick 198, 2.7e-7 abs)
- ch_gain: constant array, bit-identical
- host_scalar's CSF S: bit-identical to pycvvdp (tick 202, 1e-6 rel)
- BUT: GPU pipeline T_p (`compute_dkl_t_p_bands`) diverges
  0.9% rel from pycvvdp (tick 199)

Therefore: **the GPU `csf_apply_*` kernel diverges from the
host scalar by ≥0.9% rel** at chroma_shift. The host scalar
chain is correct; the GPU port has an arithmetic discrepancy.

### Suspect: `exp(log_s_corr * LN_10)` vs `10^log_s * 10^sens_corr`

The GPU kernel folds sens_corr into the log-space sum then
applies a single `exp(x * LN_10)` for the final S, whereas
the host scalar applies `10^log_s` (one transcendental) and
multiplies by a constant `10^(sens_corr/20)` (precomputed
literal). f32 transcendental routines differ:

- `f32::exp(x * LN_10)` on cubecl-cuda → typically `__expf`
  (fast-math, ~3.5 ULP)
- Host `10f32.powf(log_s)` → libm or rustc's powf, more
  accurate (~1 ULP)

3.5 ULP at S ≈ 300 (band 4) is ≈ 8.4e-5 abs / 304 = 2.8e-7
rel — well under 0.9%. So fast-math precision alone doesn't
account for the drift.

## Tick 203 — `inv_step` constant fixed; absolute T_p drift down 800×, headline JOD drift unchanged

Stage-6 GPU-vs-host T_p parity probe shipped:
- `compute_dkl_t_p_bands_matches_host_scalar_per_pixel_at_chroma_shift`
  compares GPU T_p (`compute_dkl_t_p_bands`) to host_scalar
  T_p (Weber × `sensitivity_corrected_scalar` × ch_gain) pixel-
  by-pixel with per-pixel rel tolerance. The existing
  `_matches_host_scalar` test uses band-max-normalized rel
  tolerance and missed the per-pixel rel drift.

**Pre-fix** GPU vs host T_p divergence at chroma_shift:
  band 0 abs **7.6e-3** rel 7.8e-3
  band 1 abs 6.2e-2     rel 1.8e-2
  band 4 abs 7.7e-3     rel 3.1e-2
  band 5 abs 1.3e-3     rel 20% (small-T_p pixels)

The 7.6e-3 abs at band 0 implies a real arithmetic discrepancy
between GPU and host. Reading the GPU `csf_apply_per_pixel_kernel`
revealed:

```
let inv_step = f32::new(4.920_640_4); // 31 / (4.0 - (-2.30103))
```

Computing 31 / 6.3010299957 = 4.91983057 (correct).
Computing 31 / 6.3 = 4.92063492 (truncated denominator → matches literal).

**The literal was computed as `31/6.3` (dropping the .0103
suffix of the axis range) — a 1.6e-4 relative error baked
into the bracket-index arithmetic.** This survived ticks
196-202 because the constant is documented with a misleading
comment (`31 / (4.0 - (-2.30103))`) that yields the wrong
value only if you literally compute the comment text instead
of the underlying math.

### Fix

Updated `inv_step` to `4.919_830_6` at all three GPU CSF kernels
(`csf_apply_per_pixel_kernel`, `csf_apply_3ch_kernel`,
`csf_apply_6ch_kernel`).

**Post-fix** GPU vs host T_p divergence at chroma_shift:
  band 0 abs **9.5e-6** rel 7.4e-3   ← 800× tighter abs
  band 1 abs 1.4e-4    rel 1.8e-2
  band 4 abs 1.5e-4    rel 3.1e-2
  band 5 abs 4.8e-5    rel 20% (small-T_p pixels)

Absolute T_p divergence collapsed by ~800× at band 0 and
~400× at band 1. Remaining per-pixel rel is concentrated at
tiny-T_p pixels (where the denominator amplifies f32 noise);
absolute contributions there are negligible.

### Why the headline JOD didn't move much

Re-measured via `examples/chroma_shift_drift_probe.rs`:
  cvvdp-gpu = 9.547566 (was 9.547440)
  pycvvdp golden = 9.664865
  abs diff = **0.117298** (was 0.117425)

JOD shift of 1.3e-4 — the bands' lp_norm pool integrates
over the whole spatial extent; large-magnitude pixels
(where the inv_step error contributed <1% rel error) dominate
the pool. The fix is correct and necessary (a 800× absolute
tightening at the kernel output is a real correctness win),
but the chroma_shift JOD divergence has another source
upstream or in pooling.

All 20 pipeline_color tests pass after the fix — the canonical
parity gates (12 MP synth at 0.005 JOD; blur3x1/blur1x3/noise
at 0.005 JOD) all still pass.

### Next probe (queued for next tick)

The remaining 0.117 JOD drift on chroma_shift can't be from
CSF apply anymore (now within f32 noise at large magnitudes).
Likely sources:
1. **Pool reduction order**: our `pool_band_kernel` uses
   `Atomic<f32>::fetch_add` (non-deterministic reduce order).
   pycvvdp uses torch's deterministic sum. For chroma_shift
   the RG channel is the dominant contributor; f32 sum order
   can shift accumulated sums by ~1e-5 abs over O(10⁴) pixels
   — potentially significant when the final Q sits near the
   met2jod kink at Q=0.1.
2. **`band_mul` placement**: pycvvdp applies `band_mul`
   inside `lpyr.get_band` before CSF; we apply it via
   `ch_gain_for_band` after. Math is identical at f64 but
   f32 ordering may differ.
3. **Per-pixel ch_gain at baseband**: baseband bypasses
   ch_gain in pycvvdp (`is_baseband` branch); our kernel
   path needs verification.

See `examples/manifest_parity_probe.rs` for the canonical
end-to-end measurement (was `chroma_shift_drift_probe.rs` until
tick 229; see note at top of this doc).

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
