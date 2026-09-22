# gmsd

Pure-Rust CPU port of **GMSD** — Gradient Magnitude Similarity Deviation, the
full-reference image quality index by Wufeng Xue, Lei Zhang, Xuanqin Mou and
Alan C. Bovik ("Gradient Magnitude Similarity Deviation: A Highly Efficient
Perceptual Image Quality Index", *IEEE Transactions on Image Processing*
23(2), 2014).

The port follows **libgmsd** by Tom Clunie
(<https://github.com/clunietp/libgmsd>, commit `de646c9a`), a portable C
implementation of the authors' `GMSD.m`. libgmsd is MIT-licensed; its
copyright and licence notice are kept in [`LICENSE-libgmsd`](LICENSE-libgmsd)
and at the head of [`src/kernel.rs`](src/kernel.rs), which carries the ported
arithmetic.

## Algorithm

1. 2×2 average, keep every second sample (half resolution).
2. Prewitt gradient magnitude on both images (`[1 1 1; 0 0 0; −1 −1 −1]/3`
   and its transpose, zero-padded `conv2 'same'`).
3. Per-sample similarity `GMS = (2·m_r·m_d + c) / (m_r² + m_d² + c)`.
4. **Deviation pooling**: the score is the sample standard deviation of the
   GMS map. 0 = identical; larger = worse.

There are no trained parameters. The only constant is `c`.

### Which `c`

`c = 170` on the 0..255 intensity scale. This is what the authors' `GMSD.m`
and libgmsd use. The paper states `c = 0.0026` for intensities normalised to
0..1; `0.0026 · 255² = 169.065`, so the two are the same constant to 0.55 %.
The port follows the reference implementations (170) so that its output is
comparable with published GMSD numbers and bit-comparable with libgmsd.

## API

```rust
let r = gmsd::GrayImage::new(&ref_plane, width, height, stride)?; // f32, 0..255
let d = gmsd::GrayImage::new(&dist_plane, width, height, stride)?;
let score = gmsd::gmsd(r, d)?;                 // .gmsd (deviation), .mean_gms
let mut map = vec![0.0f32; (width / 2) * (height / 2)];
let score = gmsd::gmsd_with_map(r, d, &mut map)?; // + the GMS map
```

- Input is any strided f32 gray plane. `gmsd_rgb8` / `rgb8_to_gray` convert
  sRGB8 the way libgmsd's command-line tool does:
  `round(0.299·R + 0.587·G + 0.114·B)`.
- Feature `pixels`: `gmsd_pixels` scores any zenpixels `PixelSlice`, converted
  row by row to sRGB8 by zenpixels-convert and then to the same 8-bit luma.
- The GMS map is a spatial map at half resolution (`map_dims(w, h)`).
- `GmsdScore::mean_gms` is the map mean (the paper's GMSM), free alongside.
- Feature `parallel`: rayon over 64-row bands. The score is bit-identical at
  every thread count (per-row sums are combined in row order).

## Parity with libgmsd

On even-sized input the GMS map is **bit-identical** to libgmsd's: every
expression reproduces libgmsd's float evaluation order with no FMA, and the
unit tests pin the fast path to a straight-line transcription of it. The score
differs only by f64 rounding (one-pass shifted accumulation instead of
libgmsd's two-pass mean/variance). The measured record over real image pairs
is `benchmarks/gmsd_parity_2026-09-22.md`.

Deliberate difference: on **odd** width or height the half-resolution grid is
`⌊w/2⌋ × ⌊h/2⌋` — the size libgmsd documents — and the trailing row/column is
dropped. (libgmsd's `downsample_2x2` writes one element past that allocation
on odd input; `GMSD.m` instead keeps an extra half-zero-padded sample.)

## SIMD

One `#[arcane]` entry per tier (`v3` AVX2, `neon`, `wasm128`, `scalar`) through
archmage; the per-row helpers (2×2 decimation; gradient + GMS + pooling sums)
are magetypes `#[rite]` variants of the same tier, inlined into it. Every tier
is bit-identical to the scalar tier (tested).

## License

AGPL-3.0-only OR the Imazen commercial license, like the rest of zenmetrics.
The ported libgmsd arithmetic remains available under its MIT licence
(`LICENSE-libgmsd`).
