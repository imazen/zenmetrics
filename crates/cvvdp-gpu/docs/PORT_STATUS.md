# cvvdp-gpu port status

Tracking faithful-port progress against the Python reference
(`gfxdisp/ColorVideoVDP`). One row per pipeline stage.

| Stage              | Module                 | Status                                   | Parity check                              |
|--------------------|------------------------|------------------------------------------|-------------------------------------------|
| sRGB → linear      | `kernels/color`        | host scalar + cubecl kernel body         | host 2e-3 vs pycvvdp; GPU 3e-5 vs scalar  |
| Display model      | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| RGB → DKL          | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| Laplacian pyramid  | `kernels/pyramid`      | host scalar + all 3 cubecl kernels       | pycvvdp 3 bands + 3 cuda kernels parity   |
| CSF weighting      | `kernels/csf`          | scalar (log10 L_bkg) + kernel + table     | 60 pts vs pycvvdp + GPU scale parity      |
| Contrast masking   | `kernels/masking`      | scalar mult-mutual + PU σ=3 blur + CH_GAIN| 4×4×3 small-band <1e-3 rel; whole-image    |
| Per-band pooling   | `kernels/pool`         | host scalar lp_norm + 3-stage pool       | 3 fixtures vs pycvvdp <1e-3 abs           |
| Host fold / JOD    | `kernels/pool`         | host scalar met2jod (smooth piecewise)   | 3 fixtures + kink continuity              |
| Composed pipeline  | `host_scalar`          | end-to-end sRGB → JOD on corpus          | bounded, broadly monotone (gap vs pycvvdp)|

## Reference version pin

`gfxdisp/ColorVideoVDP` **v0.5.4** (latest tag as of 2026-05-14).
Driver script in `scripts/cvvdp_goldens/` runs `pycvvdp==0.5.4` to
produce parity goldens. When bumping: also bump the R2 prefix
(`v1` → `v2`), the `GOLDEN_VERSION` const in `tests/common/mod.rs`,
and the version assertion in `tests/parity.rs`.

The cvvdp parameter JSON gets vendored into
`crates/cvvdp-gpu/data/cvvdp_v0.5.4.json` once the script lands (small
~5 KB file, safe to commit) and loaded through `params::CvvdpParams`.

## Out of scope (v0)

- Video / temporal channels (sustained + transient).
- Foveation / gaze maps.
- HDR display models — sRGB-std only for the initial parity pass.

## Open questions

- **(Resolved tick 21)** Phase-uncertainty Gaussian blur in
  masking. cvvdp's σ=3 separable Gaussian for bands > 6×6 is now
  applied via `mult_mutual_band` + `phase_uncertainty_band`.
  Closed by replicating torchvision's `GaussianBlur(13, 3.0)`
  kernel + reflect padding. Whole-image parity gate via `shadow_jod`
  closed ~0.5-1.5 JOD of the gap.

- **cvvdp v0.5.4 uses `weber_contrast_pyr`, NOT vanilla Laplacian**
  for the `contrast = "weber_g1"` config. `weber_contrast_pyr` (from
  `pycvvdp.lpyr_dec`) builds bands as Weber-contrast ratios
  (`band = (gauss[k] - expand(gauss[k+1])) / gauss[k+1]`-flavoured
  forms) on the linear DKL planes, and the call site implicitly
  log10s the gauss output before passing it to CSF. The Rust port
  currently uses vanilla Laplacian + explicit log10 in host_scalar.
  This is the dominant cause of the remaining ~1.4 JOD gap to
  pycvvdp at high q (shadow q90 = 8.60 vs pycvvdp 9.99). Porting
  weber_contrast_pyr is the biggest single chunk left.

- **`lpyr.get_band` multiplies non-edge Laplacian bands by 2.0**
  (cvvdp's `lpyr_dec.get_band` does `band_mul = 2.0` for
  `0 < bb < n-1`, else 1.0). Our port doesn't replicate this gain
  on the band readout. Affects masking input magnitudes by 2× on
  bands 1..n-2.

- **cvvdp's baseband uses `|T_f - R_f| * S` (no masking model)**
  with `rho_band[last] = 0.1` clamp before the CSF lookup. Trivial
  to wire once weber_contrast_pyr brings the baseband magnitudes
  into the same units cvvdp's Q_per_ch[last] reflects (~6.88 for
  VY in the v1 manifest; vanilla Laplacian + |T-R|*S gives ~712,
  the 100× discrepancy that defeated tick 23's attempt).

- **cvvdp bug: column-parity check in `gausspyr_reduce`.** Line 206
  of cvvdp v0.5.4's `lpyr_dec.gausspyr_reduce` checks
  `x.shape[-2] % 2` (row count) when deciding the right-column edge
  fix-up — the variable being patched is `y[...,:,-1]`, the
  rightmost column, so the parity check should clearly use
  `x.shape[-1] % 2` (column count). Doesn't affect the
  zenmetrics-corpus (all 2^k square inputs through the pyramid),
  but will cause a divergence on non-square inputs at odd-height-
  but-even-width levels. To preserve bit-stable parity our port
  reproduces the bug verbatim; document it here and re-evaluate when
  the cvvdp pin moves.

  Status: pure-symmetric-reflection happens to be equivalent to
  cvvdp's `zero-pad + explicit edge patches` for even-input dims, so
  `gausspyr_reduce_scalar` matches cvvdp exactly on the corpus's
  pyramid levels. `gausspyr_expand_scalar` now uses cvvdp's explicit
  edge-replication scheme (`interleave_zeros_and_pad`) so the
  constant-signal test passes across the whole buffer.
- **Per-band CSF weight precomputation**: should the host upload one
  flat `f32` array (`n_levels × N_CHANNELS`) or one tensor per band?
  Single flat upload is simpler; keep unless a per-band variant becomes
  necessary.
- **Atomics for pooling**: cubecl-cpu doesn't yet support
  `Atomic<f32>::fetch_add` (per zensim-gpu's lib.rs). Use per-band
  per-block partials with a tree reduction, same shape as zensim-gpu's
  fused features kernel, so the CPU runtime works.
