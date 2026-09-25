# vif

Pure-Rust CPU implementation of **VIFp** — pixel-domain Visual
Information Fidelity (Sheikh & Bovik, *Image Information and Visual
Quality*, IEEE TIP 15(2), 2006), ported for behavioral fidelity to
the authors' multiscale scalar-GSM release `vifp_mscale.m` and
validated against it under GNU Octave.

- `vif_plane_f32` — VIFp of two 0–255-scale planes.
- `vif_rgb8` — on the unrounded MATLAB `rgb2gray` luma plane
  (`0.2989/0.5870/0.1140`); the reference is single-channel, so this
  is the house convention for RGB input.

Pipeline, matching the reference: four scales; at each scale an
N-tap Gaussian (`N = 2^(5−scale)+1` → 17, 9, 5, 3, `σ = N/5`,
rank-1 separable) produces `filter2 'valid'` mean/variance/
covariance statistics; a scalar GSM regression yields `g` and
`sv_sq` through the reference's exact masking order; the information
sums accumulate `Σ log10(1 + g²·σ1²/(sv²+2))` and
`Σ log10(1 + σ1²/2)`. Between scales the images are
`filter2 'valid'` + `1:2:end` decimated by that scale's kernel.
`vifp = num/den`.

Degenerate behavior is reproduced, not guarded: a scale whose
`'valid'` map is empty contributes zero, a flat reference makes
`sigma1_sq ≡ 0` so `den = 0` and the score is `NaN`, and
`min(w,h) < 17` leaves every scale empty → `NaN`. Identical
textured inputs score `~1.0` (`0.999999999992` — the GSM gain stays
just under one), identical constant inputs score `NaN`. VIFp can
exceed 1 on sharpened inputs — the reference's own behavior.

## Validation

`src/tests.rs` checks 17 Octave goldens (even/odd dims, empty-scale
boundaries at 37×41/24×24/16×16, identical/constant/zero degenerates,
constant-distorted → `0.0`, RGB luma) — see `validation/` for the
generator and provenance. Worst observed delta ~`5e-13`.

**f64 pipeline:** the reference's `1e-10` masks sit ~4 orders above
f64's noise floor on a 0–255 signal; an `f32` statistics plane
leaves ~`1e-3` residual variance on flat input (`E[x²]−μ²`
cancellation) and would defeat every mask. The port therefore casts
`f32`/`u8` inputs to `f64` on gather and runs `f64` end-to-end —
`f64x4` lanes (`f64x8` under `avx512`) via
`archmage`/`magetypes`, bit-identical across tiers and thread
counts.

## Features

- `std` (default) — `std` + `alloc`; without it the crate is `no_std`
  + `alloc`.
- `avx512` — enable the x86-64 AVX-512 tier at runtime dispatch.
- `parallel` — rayon across the ref/dis filter pairs (bit-identical
  at any thread count).
- `_dev` — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2 / AVX-512 / NEON / wasm128), dispatched on
first call.
