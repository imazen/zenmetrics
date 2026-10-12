# CPU optimization measurements (2026-10-12)

Host r7900x, baseline 405b2157 (IW sources unchanged from 82eb7e9d).
Release thin LTO, runtime AVX2 dispatch, parallel feature, no target-cpu=native.
Synthetic gray pair from `examples/wall.rs`, 4096×2048, four calls including
scoring allocations. Scorer construction and fixture generation precede the
wall timer, but perf cycles include both. The example uses 0..1 gray values;
these timings do not establish throughput on a natural-image corpus.
All heavy commands use run-heavy, 16G cap. Recipes are in `../justfile`.

| Threads | Baseline ms/call | Packed Gram ms/call | Baseline cycles | Packed cycles |
|---|---:|---:|---:|---:|
| 1 | 852.5 | 581.9 | 18,585,204,663 | 12,972,895,904 |
| 16 | 396.4 | 368.9 | 25,424,175,743 | 20,363,344,903 |

run-heavy peak-RSS was 0.66 GiB for all four runs. Shared host load changes
wall measurements; the subsequent profile run measured 922.0 ms/call and is
not the A/B timing. Cycle sampling still identified the remaining hot paths.

Baseline top five user-cycle symbols: Gram accumulation 47.61%, quadratic
form 8.64%, vertical Gaussian 7.26%, information-weight map 6.31%, horizontal
Gaussian 3.71%. After: packed Gram 16.16%, quadratic form 12.26%, vertical
Gaussian 11.59%, information-weight map 8.93%, horizontal Gaussian 5.80%.

The default nine-tap neighborhood (with/without parent) uses fixed triangular
SIMD packing: each lane accumulates one Gram coefficient in the original
pixel order. Accumulators remain live across the band instead of loading and
storing matrix rows for every pixel. Tap offsets are hoisted out of the pixel
loop. Multiplication and addition remain separate f64 operations. Custom
neighborhood sizes keep the generic path; the eigensolver, quadratic form,
information weights, pyramid, and pooling are unchanged.

Validation: Python IW/MS goldens, strip and warm-reference parity,
materialized covariance equality, plus exact comparison to the previous
generic Gram kernel with nonzero initial accumulators and split row calls.
No threshold or public API changes. Clippy introduces no warnings.
