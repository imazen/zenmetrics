# msssim

Pure-Rust CPU implementation of **MS-SSIM** (multi-scale structural
similarity), ported for behavioral fidelity to the authors' reference
`msssim.m` + `ssim_index_new.m` (Wang, Simoncelli & Bovik, IEEE
Asilomar 2003) and validated against it under GNU Octave.

- `msssim_plane_f32` — MS-SSIM of two 0–255-scale planes.
- `msssim_rgb8` — on the unrounded MATLAB `rgb2gray` luma plane
  (`0.2989/0.5870/0.1140`).

Pipeline, matching the reference: per level, 11×11 separable
Gaussian-weighted `filter2 'valid'` statistics (`mu`, `sigma²`,
`sigma12`), SSIM and contrast–structure maps with `C1 = (0.01·255)²`,
`C2 = (0.03·255)²`, both map means; between levels a forward 2×2 box
`imfilter 'symmetric'/'same'` + `1:2:end` decimation; final score
`prod(mcs[1..L−1]^w[1..L−1]) · mssim[L]^w[L]` on the canonical
`[0.0448 0.2856 0.3001 0.2363 0.1333]` weights.

Level count: the reference hard-errors below `min(w,h) < 11·2^(L−1)`;
this port auto-selects `level = min(5, floor(log2(min/11))+1)` and
uses the first `level` weights — identical to calling the reference
with `(level=L, weight=w(1:L))` (the goldens pin this at L = 1, 2, 3
and 5). `min < 11` returns `Error::TooSmall`.

Negative means make the reference score complex (`x.^w → |x|^w·e^{iπw}`);
this port returns its real part — `mag·cos(π·Σw_neg)` — matching what
the reference's `prod` yields after `real()`. Identical inputs score
exactly `1.0`, including constant ones (no NaN — `C1, C2 > 0`).

## Validation

`src/tests.rs` checks 15 Octave goldens (even/odd dims, levels 1–5,
the 176 boundary, identical/constant/zero, negative-mean, RGB) — see
`validation/` for the generator and provenance. Worst observed delta
~8.8e-6 (f32 planes vs the reference's f64). Scalar and SIMD tiers
are bit-identical by construction (fixed-order f32 elementwise math,
fixed-order f64 pooling).

## Features

- `std` (default) — `std` + `alloc`; without it the crate is `no_std`
  + `alloc`.
- `avx512` — enable the x86-64 AVX-512 tier at runtime dispatch.
- `parallel` — rayon across the ref/dis filter pairs (bit-identical
  at any thread count).
- `_dev` — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2-FMA / AVX-512 / NEON / wasm128) via
`archmage`/`magetypes`, dispatched on first call.
