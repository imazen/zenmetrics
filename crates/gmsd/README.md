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

One `#[arcane]` entry per tier (`v4` AVX-512 behind crate feature `avx512`,
`v3` AVX2, `neon`, `wasm128`, `scalar`) through archmage; the per-row helpers
(sRGB8 → gray + 2×2 decimation; gradient + GMS + pooling sums) are magetypes
`#[rite]` variants of the same tier, inlined into it. Every tier is
bit-identical to the scalar tier (tested).

The sRGB8 conversion is integer SIMD: for `S = 299·R + 587·G + 114·B`,
`(S + 499) / 1000` equals `round(0.299·R + 0.587·G + 0.114·B)` everywhere
except `S ≡ 500 (mod 1000)`, where the f64 sum lands either side of the
half-integer — those rare lanes are recomputed in f64, so the result is
bit-identical to libgmsd (proved exhaustively over all 2^24 triplets). The
2×2 decimation then sums the four integer luma values and multiplies by 0.25
once — bit-identical to libgmsd's ordered f32 chain because every partial sum
is a multiple of 0.25 ≤ 255.

`gmsd_rgb8` never materialises full-size gray planes: each band converts and
decimates only the input rows it reads, in one fused pass, in its own tier
and thread. Published speed numbers come from zensim's `ssim2_speed_bar`
owner, not from `examples/split_timing.rs` diagnostics.

## MDSI (quarantined development addition)

`mdsi_rgb8(reference, distorted, width, height, stride_bytes)` returns the
default summation MDSI distance as `Result<f64>`. Inputs are interleaved
gamma-encoded RGB8 with a byte stride; the scorer performs no colour
management, orientation or alpha processing. The caller supplies RGB in
the intended common colour space. The CLI exposes `--metric mdsi` under
the existing `cpu-gmsd` feature for `score` and `score-pairs`.

MDSI is implemented from the paper (Nafchi, Shahkolaei, Hedjam and Cheriet,
"Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image
Quality Evaluator", *IEEE Access* 4, 2016) and validated against scores computed
by the authors' reference software, which is not included here or used at run
time. Where the paper is silent, [`docs/MDSI_CHOICES.md`](docs/MDSI_CHOICES.md)
records the reading taken and why. On 116 image pairs (odd sizes, sub-64 px,
1 px, and averaging factors 1 to 4) the score agrees with the authors' scores
to a relative difference of 4.84e-10 at most, inside the 1e-9 gate; a wrong
constant fails 109 of the 116 (the other 7 are exact-zero pairs no constant
moves). The model uses unrounded f64 luminance and two opponent channels,
size-dependent box filtering with zero-padded odd edges, a fused-luminance
gradient term, chromaticity similarity with C3=550, and principal complex
fourth roots for negative combined similarities in the pooling.

The box average is the only full-resolution work (the averaged planes are about
256 samples on the short side), and it runs in exact integer arithmetic. The
score is bit-identical to the straight-line evaluation of the paper's equations
kept in `src/mdsi.rs` (`reference`) at every archmage tier and every thread
count. The 116-pair gate is an ordinary test that the caller selects with
`GMSD_MDSI_GATE=require|skip` and `GMSD_MDSI_TARGETS=<table>` (see the
workspace `justfile`).

`ms_gmsd_rgb8` and `ms_gmsdc_rgb8` are quarantined paper-derived variants
of Zhang et al.'s four-scale masked GMSD and colour extension. They use
unrounded f64 YIQ, zero-padded Prewitt with c=170, replicated odd-edge 2x2
averaging, population deviation, and scale-3 I/Q RMSE. These conventions
are explicit choices where the paper is underspecified. Independent
NumPy qualification is pending; no author-software parity is claimed.

## License

AGPL-3.0-only OR the Imazen commercial license, like the rest of zenmetrics.
The ported libgmsd arithmetic remains available under its MIT licence
(`LICENSE-libgmsd`).
MDSI is written from the paper; see the attribution above.
