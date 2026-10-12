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

## FSIM

Baseline source: `81dcb599`. Candidate: the accompanying retained-column
decimation change. The RGB/YIQ conversion is unchanged in this comparison.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 876.1 | 725.0 |
| One-thread instructions, five calls | 107,894,319,179 | 90,361,554,508 |
| One-thread cycles, five calls | 24,234,072,459 | 19,841,131,868 |
| Sixteen-thread ms/call | 449.0 | 262.1 |
| One-thread run-heavy peak RSS, GiB | 0.12 | 0.09 |
| Sixteen-thread run-heavy peak RSS, GiB | 0.12 | 0.11 |

The decimation test covers the same ordered full-filter oracle as VSI.
Default and all-feature tests pass, including the existing grayscale,
RGB/YIQ, luma, stride and tier goldens with unchanged assertions.
All-feature clippy with `-D warnings` and scoped rustfmt pass; an
unreachable internal type alias was restricted to crate visibility.

Receipts:
[baseline 1T](fsim_baseline_1t_2026-10-12.log),
[baseline 16T](fsim_baseline_16t_2026-10-12.log),
[candidate 1T](fsim_candidate_1t_2026-10-12.log),
[candidate 16T](fsim_candidate_16t_2026-10-12.log).

Use the VSI reproduction commands with `-p fsim`. The workspace's
advisory `just quality` cannot start on this checkout because the quality
kit is absent; scoped fmt, tests and clippy were run explicitly.

## HaarPSI

Baseline production source: `4b92dc7e`, with the accompanying wall example.
Candidate: vectorized libm-equivalent logistic pooling. The bounded kernel
uses libm 0.2.16's reduction cutoffs, constants, polynomial and ordered f32
operations; other inputs use scalar libm. The f64 pooling order is unchanged.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 173.1 | 153.5 |
| One-thread instructions, five calls | 16,038,749,636 | 14,215,812,899 |
| One-thread cycles, five calls | 4,851,167,440 | 4,376,666,673 |
| Sixteen-thread ms/call | 36.2 | 33.3 |
| Run-heavy peak RSS, GiB, both thread counts | 0.18 | 0.18 |

The wall deltas are below the suite's 15% noise caution; the retained
evidence is the reduction in retired instructions, not a qualified wall
speedup. The 8.4 MP score bits match exactly: `3fe818588024d55a`.
Default and all-feature tests pass with unchanged golden/tier assertions.
A dense bounded-domain grid, reduction-boundary neighbors, subnormals,
signed zero and out-of-domain values compare the logistic results to libm
bit for bit (NaNs remain NaNs), on scalar and dispatched kernels.
All-feature clippy with `-D warnings` and scoped rustfmt pass. Disassembly
of `__attune_pool_band_inner_v4x` contains packed multiply/add/divide
instructions in the polynomial; there is no FMA in that formula.

Receipts:
[baseline 1T](haarpsi_baseline_1t_2026-10-12.log),
[baseline 16T](haarpsi_baseline_16t_2026-10-12.log),
[candidate 1T](haarpsi_vector_candidate_1t_2026-10-12.log),
[candidate 16T](haarpsi_vector_candidate_16t_2026-10-12.log).
Use the VSI reproduction commands with `-p haarpsi`.

Additional HaarPSI checks retain exact score bits at 64² (100 calls,
`3fea26b2015e7bf8`) and 1024² (ten calls, `3fe840abcb2299c2`).
At 64², instructions changed from 393,150,402 to 375,765,570;
both wall observations rounded to 0.6 ms/call. At 1024² they changed
from 3,908,961,616 to 3,456,720,450; wall changed from 21.1 to 18.4
ms/call. Peak RSS stayed at 0.02 GiB and 0.03 GiB respectively.
These wall deltas also remain below the noise caution.
[64² baseline](haarpsi_baseline_small_2026-10-12.log),
[64² candidate](haarpsi_candidate_small_2026-10-12.log),
[1024² baseline](haarpsi_baseline_1024_2026-10-12.log),
[1024² candidate](haarpsi_candidate_1024_2026-10-12.log).

The first prototype kept scalar argument-reduction selection; it retired
14,657,449,477 instructions for the large-image leg and returned the same
score bits. The landed version also vectorizes that selection.
[prototype 1T](haarpsi_candidate_1t_2026-10-12.log),
[prototype 16T](haarpsi_candidate_16t_2026-10-12.log).

## RGB SSIM

Baseline production source: `16ef88a9`, with the accompanying
`msssim/examples/classical_wall.rs` harness. It compiles the CLI scoring
source directly; native CLI tests exercise the production input conversion.
The candidate retains f64, reflected borders, all coefficients, tap order
and the final score fold. Interior convolutions accumulate sixteen outputs
locally before storing, instead of reading and writing the full output row
for every tap. Disassembly contains packed f64 multiply/add operations.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 1621.1 | 1000.9 |
| One-thread instructions, five calls | 65,983,425,931 | 60,424,291,115 |
| One-thread cycles, five calls | 44,582,556,016 | 27,525,912,117 |
| Sixteen-thread ms/call | 463.1 | 453.8 |
| Run-heavy peak RSS, GiB, both thread counts | 0.55 | 0.55 |

The sixteen-thread wall observations are within the noise caution. Both
thread counts return exactly `3febf5b219da7a49`. An independent reflected
scalar filter compares both intermediate and output bits, including square
and product ingress, narrow shapes, odd dimensions and block tails.
All four classical tests pass through the native CLI. Its full CPU-metrics
library configuration passes 33 tests, with two pre-existing ignored tests.
The lean example passes clippy with `-D warnings` and scoped rustfmt.
Full CLI clippy remains blocked by existing unused/unreachable code and
doc-format warnings in GPU dependencies. No expectations were changed.

Receipts:
[baseline 1T](ssim_baseline_1t_2026-10-12.log),
[baseline 16T](ssim_baseline_16t_2026-10-12.log),
[candidate 1T](ssim_candidate_1t_2026-10-12.log),
[candidate 16T](ssim_candidate_16t_2026-10-12.log).
Build with `just snap-cargo @ build --release -p msssim
--features parallel,avx512 --example classical_wall`, then use `ranked-wall`.

## MS-SSIM scout audit

The proposed exact separable replacement for libvmaf's 9×9 decimator
cannot preserve its stored coefficients. Their leading 2×2 minor is
exactly `-32317628725/73786976294838206464`, so the f32 table is not rank
one. A 255-valued impulse distinguishes the original coefficient from
the proposed center-row factorization after f32 rounding: 0.18207000195980072
versus 0.18215307593345642. This audits coefficient/output equality; it
does not measure an approximate implementation's final score error.

Run `just lpf-rank-check`; the [receipt](msssim_lpf_rank_2026-10-12.log)
records the source hash and an exactly separable negative control.
The existing `dispatched_filters_match_ordered_scalar` test requires
exact decimated output bits. All-feature tests pass, including six real
libvmaf feature/frame comparisons, with unchanged expectations.

The next native MS-SSIM scout suggestion is already implemented in the
current `kernel.rs`: products form inside the horizontal filter and five
eleven-row moment rings feed the vertical filter and statistic combine.
Rayon bands own their halos, and the final fold stays in raster order.
There are no materialized full-image product planes to remove from that
path. These two scout findings are source checks, not new speedup claims.

## Subsequent target measurements

The next passes use the existing `cpu-profile/new-metrics-wall` production
CLI driver with its pinned zenbench git revision `ac675d3` (resolved
version 0.1.10). `just ranked-zenbench` retains the binary hash, bounded
run settings, full console report, resource receipt and per-metric TSV.
Latency cells score one pair; `par1/par4/par8` cells instead measure
concurrent independent pairs on dedicated pools. They answer distinct
questions and are labelled explicitly. Inputs here use the driver's
procedural RGB pair, which differs from the earlier crate wall examples.

The initial NLPD observations use 30 interleaved rounds at 3355×2516,
one Rayon thread, with source `bdc458e0`:
[zenbench report](nlpd_baseline_zenbench_1t_2026-10-12.log),
[TSV](nlpd_baseline_zenbench_1t_2026-10-12.tsv).
They establish the next candidate's baseline, not an optimization result.
