# psnrhvs

Pure-Rust CPU port of **PSNR-HVS** and **PSNR-HVS-M** —

- **PSNR-HVS** (K. Egiazarian, J. Astola, N. Ponomarenko, V. Lukin,
  F. Battisti, M. Carli, *"New full-reference quality metrics based on
  HVS"*, VPQM-06): PSNR on 8×8 DCT coefficients weighted by a contrast
  sensitivity function.
- **PSNR-HVS-M** (N. Ponomarenko, F. Silvestri, K. Egiazarian, M. Carli,
  J. Astola, V. Lukin, *"On between-coefficient contrast masking of DCT
  basis functions"*, VPQM-07): additionally subtracts a per-block
  masking threshold before weighting — the metric codec-evaluation
  reports (Daala/AV1-style `psnr_hvs`) usually mean.

The port follows the authors' reference implementation `psnrhvsm.m`
(Nikolay Ponomarenko, <http://ponomarenko.info>). That implementation is
licensed for educational/research use and is not redistributed here; the
port implements the published algorithm and constants, verified against
the reference under GNU Octave (see [`validation/`](validation/)).

## Algorithm

1. Tile into 8×8 blocks (default step 8, the reference's `wstep`;
   partial edge strips are dropped, as the reference does).
2. Orthonormal 2D DCT-II of both blocks (MATLAB `dct2` convention).
3. **Masking threshold** (`maskeff`, HVS-M only): non-DC coefficient
   energy `Σ z_dct²·MaskCof` scaled by the ratio of the four 4×4-quadrant
   variances to the whole-block variance, `sqrt(e·pop)/32`; block mask =
   `max(mask_ref, mask_dist)`.
4. Per coefficient `u = |A_dct − B_dct|`: **PSNR-HVS** accumulates
   `(u·CSFCof)²`; **PSNR-HVS-M** uses `u' = max(u − mask/MaskCof, 0)` on
   non-DC coefficients (DC is never masked).
5. `10·log10(255²/mean_energy)`, or `100000` when the energy is zero
   (identical / visually indistinguishable).

## API

```rust
let s = psnrhvs::psnrhvs_luma8(&ref_rgb, &dist_rgb, w, h, w * 3)?;
// s.psnr_hvs      — PSNR-HVS   (dB, higher = better)
// s.psnr_hvs_m    — PSNR-HVS-M (dB, >= psnr_hvs)
```

- `psnrhvs_plane_f32(r, d, w, h, stride)` — strided f32 planes on the
  0..255 scale (`psnrhvs_plane_f32_step` exposes `wstep`).
- `psnrhvs_rgb8(r, d, w, h, stride)` — interleaved sRGB; each channel is
  scored as a plane and the scores are averaged (the multi-plane
  convention codec harnesses report; `RgbScore` also carries the
  per-channel `PlaneScore`s).
- `psnrhvs_luma8(r, d, w, h, stride)` — BT.601 luma
  (`round(0.299R + 0.587G + 0.114B)`, same as `gmsd`'s), the
  PSNR-HVS-Y codec-benchmark shape.

Images need at least one complete 8×8 block (`Error::TooSmall` below).

## Determinism

Per-block energies accumulate in f32 in fixed coefficient order; the
global sum is f64 in raster order, so every SIMD tier (`scalar`, `v3`
AVX2, `v4` AVX-512 behind `avx512`, `neon`, `wasm128`) and every
`parallel` thread count produces **bit-identical** output. Final scoring
uses `libm::log10` for cross-platform bit agreement.

Features: `std` (default), `parallel` (rayon band parallelism),
`avx512`, `_dev` (per-tier entry points for parity tests/disassembly).

## Verification

`src/tests.rs` pins 15 golden rows produced by `psnrhvsm.m` under
Octave (plane patterns covering step≠8, sub-8×8-multiple sizes,
constant regions, DC shifts, clamping; plus per-channel and luma rows
for the color entry points). All pass within 1e-3 dB — the residual is
the port's f32 block pipeline vs the reference's f64; the observed
delta is ≤ ~4e-4 dB.
