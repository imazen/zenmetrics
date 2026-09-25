# haarpsi

Pure-Rust CPU port of **HaarPSI** — the Haar wavelet perceptual
similarity index of R. Reisenhofer, S. Bosse, G. Kutyniok, T. Wiegand,
*"A Haar Wavelet-Based Perceptual Similarity Measure for Images"*, Signal
Processing: Image Communication 61 (2018).

The port follows the authors' MIT-licensed reference implementation
`HaarPSI.m` (<https://github.com/rgcda/haarpsi>, redistributed under
`validation/` with its license) and is verified against it under GNU
Octave (see [`validation/`](validation/)).

## Algorithm

1. Optional preprocessing (`preprocessWithSubsampling`, the reference's
   third argument, default on): `conv2(X, ones(2,2)/4, 'same')` then
   decimation at the odd indices — a forward-looking 2×2 box mean with
   zero-padded borders, matching MATLAB's even-kernel `same` anchor.
2. sRGB → **YIQ** for color input (`Y = 0.299R + 0.587G + 0.114B`, the
   reference's I and Q rows) — no quantization.
3. Three-scale Haar decomposition in horizontal and vertical
   orientations; each filter is a rank-1 box-sum × strip-difference, so
   the whole pass is separable stencil work on padded planes.
4. Luma similarity `(2|a||b| + C)/(|a|² + |b|² + C)` on scales 1–2
   (C = 30), averaged over the two scales and both orientations; weights
   `max(|a|, |b|)` from scale 3.
5. Color: the same similarity on the 2×2-smoothed `|I|` and `|Q|`
   planes, pooled with the luma orientation weights.
6. Logistic pooling `Σ sigmoid(sim·α)·W / ΣW` (α = 4.2), inverse
   logistic, squared. An all-zero pair has zero total weight and yields
   `NaN` — the reference's behavior.

## API

```rust
let s = haarpsi::haarpsi_rgb8(&ref_rgb, &dist_rgb, w, h, w * 3)?;
// s ≈ 1.0 for identical images (the logistic inverse is exact at sim = 1)
```

- `haarpsi_plane_f32(r, d, w, h, stride)` — strided f32 planes on the
  0..255 scale, reference default preprocessing.
- `haarpsi_plane_f32_opts(.., subsample)` — explicit preprocessing flag
  (the reference's third argument).
- `haarpsi_rgb8(r, d, w, h, stride)` — interleaved sRGB via the YIQ
  color path.
- `haarpsi_luma8(r, d, w, h, stride)` — sRGB → unrounded BT.601 luma
  through the grayscale path (no color channel).

## Determinism

Every map value is a fixed-order f32 expression and the final pooling
accumulates f64 in a fixed lane-grouped order, so every SIMD tier
(`scalar`, `v3` AVX2, `v4` AVX-512 behind `avx512`, `neon`, `wasm128`)
and every `parallel` thread count produces **bit-identical** output.
`libm` covers `log`/`exp` so cross-platform agreement holds.

Features: `std` (default), `parallel` (rayon row parallelism),
`avx512`, `_dev` (per-tier entry points for parity tests/disassembly).

## Verification

`src/tests.rs` pins 16 golden rows produced by `HaarPSI.m` under GNU
Octave on deterministic integer patterns: grayscale with and without
subsampled preprocessing, odd sizes, identical inputs, constant and
zero-weight degenerate inputs, the RGB/YIQ path, and the luma path.
All pass within 5e-5 — the residual is the port's f32 plane pipeline vs
the reference's f64.
