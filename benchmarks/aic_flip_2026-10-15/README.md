# LDR-FLIP and CW-SSIM: JPEG AIC-4 checks

FLIP is a Rust port of the LDR algorithm in [NVlabs/flip](https://github.com/NVlabs/flip/blob/d3ab02a4425eb167d8a4021b382a2bafc6e6c838/src/cpp/FLIP.h), pinned to `d3ab02a4425eb167d8a4021b382a2bafc6e6c838`. The CLI metric `flip` emits `flip` (lower is better), using the reference viewing conditions: distance 0.7 m, width 3840 pixels, physical width 0.7 m. The BSD-3-Clause notice is retained in `crates/zenmetrics-cli/LICENSE-flip`.

The oracle is `flip-evaluator==1.7`; its source distribution SHA-256 is `88204ec5000fa90b26e00f6f21e8d73f18e5afac7e8a23cb0803b117b3da081b`. It reproduces all 61 published AIC `FLIP` values exactly. Version 1.7 sums its map in row order in f32. The CLI retains that convention; the pinned newer reference's f64 sum would change the published column. Pixel arithmetic, CSF filtering, Hunt-adjusted colour difference, edge/point detection and error exponent follow the C++ reference. Images use native resolution with nearest-sample border extension.

The 61-pair published gate has median/max scalar error **0 / 0**. The 21 ladder maps (three images at 64, 128, 256, 512, 1024, 2048 and 4096 square pixels) have maximum per-pixel error **2.9802322387695312e-8** and maximum scalar error **0**, below the requested 1e-6 gates. Packed and strided RGB ingress must agree bit for bit.

`flip_probe.rs` exports the production Rust map and mean without involving a decoder. `verify_flip.py` decodes the same RGB8 PNGs for both implementations and compares every raw cell; no colormap or screenshot is compared. Paths to the external AIC and ladder fixtures are explicit in the script. Install the oracle in a separate virtual environment; it is not a runtime dependency of the Rust CLI.

## CW-SSIM

The exact AIC pyramid has not been established. `cw-ssim` emits `cw_ssim` (higher is better); the selected implementation has median/max published error **2.9390669198381403e-7 / 1.106269287983741e-4**, with the largest residual on `S21_JXLM_09.png`. This is not an exact reproduction claim.

The implementation follows [Wang and Simoncelli (2005)](https://www.cns.nyu.edu/pub/lcv/wang05a.pdf) and [Sampat et al. (2009)](https://doi.org/10.1109/TIP.2009.2025923): rounded BT.601 luma, four complex steerable-pyramid levels, eight orientations (order seven), the fourth oriented level, valid 7×7 windows, K=0, and orientation-averaged Gaussian spatial pooling with σ = uncropped band height/4. The radial transition spans one octave. Two exactly zero-energy windows are defined to have similarity 1. Inputs smaller than 64×64 are refused. RGB rows can be strided.

The independent reference uses [pyrtools 1.0.11](https://github.com/LabForComputationalVision/pyrtools/blob/1ff70157232b9efc9f3ab043230fe6854690e516/src/pyrtools/pyramids/SteerablePyramidFreq.py), under the MIT notice retained in `LICENSE-pyrtools`. Its source file SHA-256 is `f234aa9442927e83d3add8791e9e31d5e489a9041938c2878174d1433b4b156c`. Rust adapts the existing in-tree scalar radix-2/Bluestein FFT to f64; no dependencies were added. Unused fine orientation bands are not materialized.

The selected variant uses actual centred DFT-bin coordinates at odd sizes and interpolated radial/angular masks. Selection minimized the worst absolute published error over the 61-pair panel. Measurements are retained in `cw_pyramid_search_summary.tsv`, `cw_grid_search_summary.tsv`, `cw_diagnostic_grid_search.json` and `cw_aic.json`:

| Convention | Median absolute error | Maximum absolute error |
|---|---:|---:|
| Original pyrtools grid, BLAS luma, level 4, valid, Gaussian | 4.5422022842966925e-7 | 8.901036304693744e-4 |
| True DFT grid, f64 explicit luma, interpolated angular mask | 2.939066942042601e-7 | 1.1062692880392522e-4 |
| True DFT grid, f64 explicit luma, analytic angular mask | 3.010417111148911e-7 | 1.1099389968893814e-4 |
| True DFT grid, f32 explicit luma, interpolated angular mask | 2.8370234161911867e-7 | 1.1086544580862778e-4 |
| True DFT grid, Gaussian σ = cropped height/4 | 1.3208148066390368e-4 | 5.677828847750144e-3 |

Earlier levels, uniform pooling, and same-size zero-padded windows had larger residuals in the full panel. An inclusive-endpoint frequency grid was also worse on the four diagnostic pairs. No pyiqa source or package was read.

The final Rust/reference check holds the scalar ingress expression fixed: `round_ties_even(0.299R + 0.587G + 0.114B)` in f64. NumPy/BLAS matrix multiplication can reassociate those terms and change a small number of half ties; its distinct measurements remain in the search receipts. With explicit ingress on both sides, all 61 Rust scores agree with the independent Python pyramid within **1.1114442699522442e-12**, retaining the 1e-9 oracle gate. The initial matrix-multiply comparison and its failed gate are retained in the full logs rather than treated as pyramid errors.

`verify_cw_explicit.py` runs that final gate; `verify_cw.py` retains the initial matrix-multiply comparison. `input_hashes.tsv` identifies every AIC and ladder PNG used. `check_ci.sh` runs the CLI library tests, the full workspace format check with CI's package selection, CI's exact workspace Clippy flags, and the inventory test. Generated full logs and search matrices are referenced by `validation.meta`.

Implementation commits: FLIP `4f0dac38006cada675c7f2b607d5d1ac875cec57`; CW-SSIM `636f6b2c84dccd04f94637295f87935d66d5f02b`. Gate recipes and diagnostics are in `5778ea61ecf02eba9bf1e37ae6f28958f02e6316`. Oracle output files are created exclusively; choose a fresh output directory when rerunning.

## Coordinator review notes (2026-10-15)

- **FLIP pooling.** The AIC-4 `FLIP` column is flip-evaluator 1.7's row-major f32 running sum, now the
  CLI metric `flip-v17`. The f32 sum loses precision as the map grows: on a uniform 0.075 map it reads
  low by 8.7e-6 (512²), 9.4e-5 (1024²), 1.0e-3 (2048²) and 1.7e-2 (4096²) relative to the true mean
  (sequential f32 vs exact, measured by the reviewer). `flip` therefore sums in f64, as NVlabs FLIP.h
  does at the pinned revision. Both are pinned against flip-evaluator 1.7 in
  `flip_and_cw_ssim_match_pinned_reference_values` (zenmetrics-cli).
- **CW-SSIM selection is in-sample.** The grid, sigma and mask variant were chosen on the same 61 pairs
  the residual is reported on. No pair reproduces exactly (best 6.45e-9; 20 of 61 above 1e-6). Only
  the DFT-grid and sigma choices are clearly separated by the data; luma and angular-mask options move
  the worst case by under 4e-7.
- Parity measured on Linux/glibc only (`powf`/`exp` from the platform maths library).
