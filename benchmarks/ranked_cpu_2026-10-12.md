# Ranked CPU metric optimization, 2026-10-12

Measured on `r7900x` (Ryzen 9 7900X, 24 logical CPUs), not on Zen 5.
Release builds use runtime dispatch, `parallel,avx512`, and no
`target-cpu=native`. Source and dependencies come from the CI-pinned snapshot.
The procedural RGB pair is defined by each crate's `examples/wall.rs`.
Each observation scores 3355×2516 five times. Single-thread runs pin CPU 2
and collect `perf stat -e instructions,cycles`; sixteen-thread runs use the
host's available CPUs. These wall observations are exploratory, without
statistical intervals; the earlier suite treats changes below 15% as noise.

## VSI

Baseline source: `81dcb599`; baseline receipts and recipe: `c1fe73b0`.
Candidate: the accompanying retained-column decimation change.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 642.3 | 410.0 |
| One-thread instructions, five calls | 52,080,700,950 | 28,109,445,107 |
| One-thread cycles, five calls | 17,585,307,564 | 11,102,197,534 |
| Sixteen-thread ms/call | 314.5 | 271.3 |
| One-thread run-heavy peak RSS, GiB | 0.53 | 0.50 |
| Sixteen-thread run-heavy peak RSS, GiB | 0.57 | 0.55 |

The horizontal scratch holds only columns consumed by the vertical pass.
Both sums retain their previous accumulation order and normalization.
An independent ordered box-filter test compares output bits for factors
1–12, narrow images, odd dimensions, and partial windows. The existing
Octave goldens, stride checks, and SIMD-tier checks pass with unchanged
expectations. Default tests and all-feature tests pass; all-feature clippy
with `-D warnings` and scoped rustfmt pass.

The no-default-feature test command exposes an existing unqualified
`eprintln!` in `vsi/src/tests.rs`; it cannot compile that test configuration.
Default tests cover the implementation without `parallel`. No test was
ignored or given a wider tolerance.

Receipts:
[baseline 1T](vsi_baseline_1t_2026-10-12.log),
[baseline 16T](vsi_baseline_16t_2026-10-12.log),
[candidate 1T](vsi_candidate_1t_2026-10-12.log),
[candidate 16T](vsi_candidate_16t_2026-10-12.log).

Reproduce with `just snap-cargo @ build --release -p vsi
--features parallel,avx512 --example wall`, preserve the resulting binary,
then `just ranked-wall <binary> <1-or-16> <receipt-path>`.
