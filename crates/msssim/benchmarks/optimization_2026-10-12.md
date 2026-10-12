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
