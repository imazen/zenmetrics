# iwssim

Pure-Rust IW-SSIM (Wang & Li 2011) — and, with information weighting switched
off, a true MS-SSIM (Wang, Simoncelli & Bovik 2003). CPU implementation; the
`iwssim-gpu` sibling crate runs the same algorithm on CubeCL.

## Two metrics in one pipeline

The pipeline is the Wang & Li formulation: 5-scale dyadic pyramid, 11×11
Gaussian SSIM statistics per scale, cross-scale combination
`Π |wmcs_j|^{β_j}` with the standard exponents
`β = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`.

- **`IwssimParams { iw_flag: true }` (default) — IW-SSIM**: each scale's
  contrast-structure map is pooled with information-content weights from the
  GSM model (parent-band conditioned, per-pixel mutual information).
- **`IwssimParams { iw_flag: false }` — MS-SSIM**: each scale pools by plain
  mean. With weighting off, the Wang & Li construction reduces exactly to
  canonical MS-SSIM.

Both modes are locked to the reference implementation
([Python-IW-SSIM](https://github.com/Jack-guo-xy/Python-IW-SSIM), commit
`f9de37c`) by committed goldens: `goldens/python_iwssim_2026-05-27.json`
(IW) and `goldens/python_msssim_2026-06-10.json` (MS-SSIM), enforced by
`tests/parity_python.rs` on deterministic fixtures (identical pairs ≤ 1e-5,
distorted ≤ 5e-3).

## Input

`score(ref_rgb, dist_rgb)` takes sRGB8 and converts to gray via BT.601 on
code values (no linearization), matching the reference. `score_gray` is the
canonical entry — float gray planes in 0..255 scale, `width × height`
samples. Minimum dimension 176 (5-level pyramid + 11×11 valid blur);
`allow_small` tiles smaller inputs up to the floor.

The float gray entry matters for HDR: feeding `PU21(bt709-luma)·255/PU21(peak)`
as **floats** (no u8 round-trip) scored SROCC 0.808 (IW) / 0.812 (MS-SSIM)
on the UPIQ HDR subset vs 0.628 through a u8 shell — see
`benchmarks/pu_integrated_upiq_2026-06-09.md` and the umbrella's
`HdrFeeding::PuLumaGrayF32` routing.

## Batch and streaming

- `warm_reference` / `score_with_warm_ref{,_gray}` reuse the reference-side
  pyramid + per-scale eigendecomposition across many distorted candidates.
- `score_strip{,_gray}` processes bounded-height strips for large images;
  strip and whole-image paths agree to ≤ 1e-6 (`tests/strip_parity.rs`).
