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

A follow-up experiment specialized quadratic-form loops to literal 9/10
bounds. It passed parity but was rejected: in an eight-call 1t A/B the packed
Gram baseline measured 664.8 ms/call and 28,069,531,466 cycles; the specialization
measured 923.4 ms/call and 34,279,682,415 cycles. At 16t (four calls),
552.4→543.0 ms/call and 20,995,591,653→20,321,476,217 cycles did not offset the
single-thread regression. The existing quadratic-form kernel is retained.
run-heavy peak-RSS: 0.67 GiB for both eight-call 1t runs, 0.66 GiB for both
16t runs. These were separate, more heavily loaded runs than the main table.

Disassembly of the retained Gram AVX2 function contains vmulpd/vaddpd and
no fused multiply-add: the two-rounding accumulation remains explicit.
The repository advisory quality kit was unavailable (`just quality --quick`
reported `kit not found`); scoped fmt and clippy were run directly instead.

CI lint on e2dc4f74 failed at fmt-check on an inherited formatting hunk in
`crates/mad-iqa/src/fft.rs:834`, outside this lane's permitted crates.
Scoped `cargo fmt --check -p msssim -p iwssim` passes. Run 38254913417,
job 114830527851 contains the formatting diff. Local default and parallel
IW-SSIM suites pass; remote full-matrix green is not established.
