# vsi

Pure-Rust CPU implementation of **VSI** (visual saliency-induced
index), ported for behavioral fidelity to the authors' reference
`VSI.m` (Zhang, Shen & Li, IEEE TIP 23(10), 2014) and validated
against it under GNU Octave.

- `vsi_rgb8` — VSI of two packed sRGB RGB8 pairs (the reference is
  RGB-only; there is no grayscale variant).

Pipeline, matching the reference: **SDSP** saliency maps — each RGB
channel resampled to 256×256 by the antialiased bilinear `imresize`
(triangle kernel broadened on shrink, symmetric whole-point padding,
full-kernel-sum normalisation), D50 CIE-Lab conversion, single-scale
log-Gabor bandpass via 2D FFT, centre and warm-colour priors, resize
back, `mat2gray`. Opponent channels `L/M/N`, box-mean `conv2 'same'`
decimation by `F = max(1, round(min(w,h)/256))` then `1:F:end`
subsample, Scharr gradients, and the quality map
`gradSim^0.4 · VSSim · real((ISim·QSim)^0.02)` pooled by
`max(SM1, SM2)` (NaN-propagating, MATLAB `max` semantics). Flat inputs
return `NaN`, as the reference's saliency normalisation degenerates.

The 2D FFT is a self-contained radix-2 + Bluestein implementation
vendored from this repository's `crates/hdrvdp/src/fft.rs` (same
provenance; HDR-specific helpers omitted). The `imresize` port is
verified bit-identical to Octave's `imresize(A,[m n],'bilinear')` at
17 scale combinations — see `validation/imresize_check.m`.

## Validation

`src/tests.rs` checks 14 Octave goldens (RGB pairs at even/odd dims,
`F=1` and `F=2` decimation regimes, identical and degenerate inputs,
replicated gray) — see `validation/` for the generator and
provenance. Scalar and SIMD tiers are bit-identical by construction
(fixed-order f32 elementwise math, fixed-order f64 pooling).

## Features

- `std` (default) — `std` + `alloc`; without it the crate is `no_std`
  + `alloc`.
- `avx512` — enable the x86-64 AVX-512 tier at runtime dispatch.
- `parallel` — rayon across the reference/distorted work pairs
  (bit-identical at any thread count).
- `_dev` — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2-FMA / AVX-512 / NEON / wasm128) via
`archmage`/`magetypes`, dispatched on first call.
