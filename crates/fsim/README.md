# fsim

Pure-Rust CPU implementation of **FSIM** (feature similarity index)
and **FSIMc** (chromatic variant), ported for behavioral fidelity to
the authors' reference `FR_FSIMc.m` (Zhang, Zhang, Mou & Zhang, IEEE
TIP 20(8), 2011) and validated against it under GNU Octave.

- `fsim_plane_f32` / `fsim_luma8` — grayscale FSIM.
- `fsim_rgb8` — returns `Scores { fsim, fsimc }`: FSIM on the luma
  plane plus the FSIMc chroma term (YIQ, same matrix as the reference).

Pipeline, matching the reference: optional 2× box-filter subsampling
when min dimension > 256 (factor F = max(1, floor(min/256)), applied
recursively), Scharr gradients, `phasecong2`-style log-Gabor
phase-congruency maps (4 scales × 6 orientations, 2D-FFT filterbank),
PC + gradient similarity maps weighted by `max(PC1, PC2)`, FSIMc chroma
term `|ISim·QSim|^0.03`. Constant inputs return `NaN`, as the
reference's phase-congruency maps degenerate to `0/0`.

The 2D FFT is a self-contained radix-2 + Bluestein implementation
vendored from this repository's `crates/hdrvdp/src/fft.rs` (same
provenance; HDR-specific helpers omitted).

## Validation

`src/tests.rs` checks 18 Octave goldens (grayscale, RGB/YIQ, luma;
even/odd dims; subsampled sizes; identical and degenerate inputs) —
see `validation/` for the generator and provenance. Scalar and SIMD
tiers are bit-identical by construction (fixed-order f32 elementwise
math, fixed-order f64 pooling).

## Features

- `std` (default) — `std` + `alloc`; without it the crate is `no_std`
  + `alloc`.
- `avx512` — enable the x86-64 AVX-512 tier at runtime dispatch.
- `parallel` — rayon row-parallelism.
- `_dev` — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2-FMA / AVX-512 / NEON / wasm128) via
`archmage`/`magetypes`, dispatched on first call.
