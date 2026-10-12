# CPU optimization measurements (2026-10-12)

Host r7900x, baseline 82eb7e9d, release (thin LTO), runtime AVX2,
`parallel` feature. Synthetic RGB pair from `examples/wall.rs`, 4096×2048,
4 calls, allocations and conversion included. No target-cpu=native.
All commands ran under run-heavy (16G cap) and perf stat -e cycles.
Recipes: `just -f crates/msssim/justfile wall <binary> <threads>`.

| Wang path | Baseline ms/call | Row-ring ms/call | Baseline cycles (4 calls + setup) | Row-ring cycles |
|---|---:|---:|---:|---:|
| 1 thread | 405.3 | 156.4 | 8,875,473,216 | 3,482,765,233 |
| 16 threads | 162.9 | 75.0 | 18,097,010,462 | 5,984,502,855 |

run-heavy peak-RSS: baseline 1t 0.49 GiB / 16t 0.91 GiB;
row-ring 1t 0.19 GiB / 16t 0.21 GiB. These are measured process peaks,
not allocation estimates. Loaded shared host; wall times are observations,
not performance thresholds.

Baseline profile (cycles, including kernel): horizontal filter 25.13%,
vertical filter 19.59%, RGB entry/conversion 6.33%, unresolved kernel
symbol 4.68%, memset 3.49%. Kernel symbols were restricted by perf settings.

The row-ring implementation forms squares/cross-products inside the
horizontal stencil, keeps eleven horizontal rows per moment, and combines
vertical moments directly into the two score maps. It borrows the original
planes instead of cloning them. The raster-order f64 pool, f32 tap order,
decimation, level selection, and complex-product convention are unchanged.
Band halos are recomputed per worker. Maps remain materialized to preserve
exact raster pooling; the public API and thresholds are unchanged.

Validation: default tests, parallel/_dev/avx512 tests (Octave goldens,
scalar/AVX2 parity and libvmaf FFI parity), plus band/thread equality at
11×11, 17×23, 176×177, 257×193, 513×257 with 1/2/3/16 threads.

## libvmaf-compatible path

`examples/wall_libvmaf.rs` measures both float_ssim and float_ms_ssim
on its deterministic u16 pair. Runtime f64x4 AVX2 dispatch preserves
multiply-then-add order (no FMA), f32 output storage, and reflected borders.
The 9×9 filter remains a full 81-tap convolution: factoring its rounded
2-D coefficients would change the filter. Fixed 15-element input windows
cover four stride-two outputs and let the compiler prove tap bounds.

| Path | Threads | Baseline ms/call | SIMD ms/call |
|---|---:|---:|---:|
| float_ssim | 1 | 58.1 | 47.2 |
| float_ms_ssim | 1 | 972.8 | 454.2 |
| float_ssim | 16 | 57.8 | 43.8 |
| float_ms_ssim | 16 | 498.8 | 122.9 |

Combined cycles: baseline 1t **11,310,273,709 for 2 calls of each metric**;
SIMD 1t **10,639,316,407 for 4 calls of each metric**. Baseline 16t
39,814,417,881 vs SIMD 23,301,384,403 (both 4 calls of each).
Peak-RSS: baseline 1t 0.43 GiB, 16t 0.95 GiB; SIMD 1t 0.43 GiB,
16t 0.91 GiB. run-heavy reports these process peaks; the two examples
exercise different input/score conventions and should not be compared
as equivalent algorithms.

Baseline top five user-cycle symbols: filter_pixel 56.76%, vertical Gaussian
24.66%, horizontal Gaussian 6.14%, score fold 6.03%, plane conversion 2.36%.
After initial SIMD (before the fixed-array 9×9 specialization): decimation
35.16%, score fold 16.74%, vertical Gaussian 16.19%, horizontal Gaussian
14.40%, plane conversion 6.42%.

Tests also compare every convolution/decimation output bit with an ordered
scalar oracle across odd/even sizes, factors 2/3/4, and 2/3/4/9-tap kernels.
Default and parallel tests, including real libvmaf frame/scale/LCS parity,
pass without any threshold change. Clippy adds no warnings; existing FFI
safety-comment warnings remain. API and no_std behavior are retained.

Final user-cycle profiles after fixed-array decimation: Wang fused band 59.30%,
RGB entry/conversion 23.06%, score/pool entry 5.93%, box decimation 5.71%,
fixture/main 1.28%. libvmaf-compatible decimation 27.43%, score fold 19.50%,
vertical Gaussian 17.85%, horizontal Gaussian 16.18%, plane conversion 7.48%.
These profiles used eight calls at 4096×2048; measured profile walls were
157.1 ms/call (Wang), 43.3/367.8 ms/call (float_ssim/float_ms_ssim).
They are separate observations from the perf-stat A/B table.

## Fixed-block RGB ingress and final original-to-optimized A/B

RGB conversion now handles eight pixels through fixed `[u8; 24]` blocks,
with SIMD channel conversion and the original ordered f32 coefficient
expression. Rows and scalar tails honor stride; the new test compares every
luma sample bit with the original scalar formula over widths 1/7/8/9/17/40/257
and padded rows. Plane-input entry points do not use this conversion.

Incremental eight-call A/B against the row-ring-only version: 1t
152.8→134.1 ms/call, 6,643,481,468→5,818,799,747 cycles; 16t
76.7→61.7 ms/call, 11,353,832,801→10,637,125,164 cycles. Peak-RSS
0.19 GiB (1t) and 0.21 GiB (16t) on both sides.

Final eight-call A/B against the original 82eb7e9d binary:

| Threads | Original ms/call | Final ms/call | Original cycles | Final cycles |
|---|---:|---:|---:|---:|
| 1 | 399.2 | 134.1 | 16,944,650,157 | 5,780,082,632 |
| 16 | 175.2 | 68.6 | 35,931,397,485 | 10,781,038,221 |

run-heavy peak-RSS: original 0.49/1.03 GiB (1t/16t), final 0.19/0.21 GiB.
Final Wang top five user-cycle symbols: fused band 71.93%, score/pool entry
7.22%, decimation 7.18%, RGB conversion 7.03%, fixture/main 1.42%.
The profile's eight-call wall was 136.4 ms/call. Pool order and box
arithmetic remain unchanged; intermediate map storage preserves exact
raster folding. Neither variant's filters are declared optimal by this
synthetic benchmark.

After RGB changes, default and parallel/_dev/avx512 suites pass, including
FFI parity. Scoped fmt and clippy pass without new warnings; msssim
`cargo check --no-default-features` also passes.
