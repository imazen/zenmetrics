# cvvdp-gpu port status

Tracking faithful-port progress against the Python reference
(`gfxdisp/ColorVideoVDP`). One row per pipeline stage.

| Stage              | Module                 | Status                                   | Parity check                              |
|--------------------|------------------------|------------------------------------------|-------------------------------------------|
| sRGB → linear      | `kernels/color`        | host scalar + cubecl kernel body         | host 2e-3 vs pycvvdp; GPU 3e-5 vs scalar  |
| Display model      | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| RGB → DKL          | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| Laplacian pyramid  | `kernels/pyramid`      | host scalar + all 3 cubecl kernels       | pycvvdp 3 bands + 3 cuda kernels parity   |
| Weber-contrast pyr | `kernels/pyramid`      | host scalar + weber_contrast_compute_kernel | scalar via shadow_jod; kernel 14-pt parity |
| CSF weighting      | `kernels/csf`          | scalar + csf_apply_per_pixel_kernel + table | scalar parity + per-pixel kernel <1e-3 rel|
| Contrast masking   | `kernels/masking`      | scalar + no-blur kernel + PU blur kernels | scalar + no-blur + PU σ=3 blur all parity |
| Per-band pooling   | `kernels/pool`         | host scalar + pool_band_kernel (atomic)  | 3 host fixtures + GPU vs lp_norm_mean     |
| Host fold / JOD    | `kernels/pool`         | host scalar met2jod (smooth piecewise)   | 3 fixtures + kink continuity              |
| Composed pipeline  | `host_scalar`          | end-to-end sRGB → JOD on corpus          | <0.01 JOD vs pycvvdp v1 manifest          |

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

- **(Resolved tick 24)** cvvdp v0.5.4 uses `weber_contrast_pyr` for
  the `contrast = "weber_g1"` config. Ported as
  `kernels::pyramid::weber_contrast_pyr_dec_scalar`; the shadow JOD
  on the corpus now matches pycvvdp within 0–0.7 JOD across q1–q90
  (was 1.4–1.7 before this tick). The shadow now slightly
  *overshoots* pycvvdp at low q — see `band_mul = 2.0` below.

- **(Resolved tick 25)** `lpyr.get_band` multiplies non-edge
  Laplacian bands by 2.0. Applied at the host_scalar consumption
  site as a `band_mul` scaling — keeps the Weber-pyramid storage
  canonical, mirrors cvvdp's readout pattern.

- **(Resolved tick 25)** Baseband bypass formula
  (`|T_f - R_f| * S`, no masking, no CH_GAIN). Wired in
  host_scalar; the Weber-pyramid magnitudes work cleanly with this
  formula (no 100× blow-up the tick-23 vanilla-Laplacian attempt
  hit).

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
