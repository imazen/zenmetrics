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


## NLPD

Baseline production source: `bdc458e0`. The candidate computes vertical
blur and Laplacian subtraction into three rotating rows per Rayon band,
then normalizes those rows directly. It removes one full-image scratch
plane. The IQA-compatible path reuses its existing spare upsample plane;
its filter and normalization formulas remain unchanged. The vector path
retains the five blur FMAs, scalar tail, ordered normalization additions
and exact division.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 470.54 | 421.94 |
| One-thread instructions, five calls | 28,760,631,590 | 27,222,626,924 |
| One-thread cycles, five calls | 13,070,938,636 | 11,710,515,648 |
| Sixteen-thread ms/call | 251.68 | 223.24 |
| Run-heavy peak RSS, GiB, both thread counts | 0.57 | 0.51 |

The wall changes remain below the earlier suite's 15% noise caution.
The retained result is the measured memory reduction and instruction
reduction. All observations return score bits `401be9088775ba00`.
A separate four-block zenbench comparison reports exploratory latency
change -11.416297557380386%, with an A/A change of -1.1171817940483302%.
Every candidate and control passed an exact score-bit checker. This is
exploratory evidence, without a confirmation interval or promotion verdict.
The hard system cpuset covered CPU 2 and descendants; an earlier attempt
refused the requested set because the user scope did not enforce it.
[Selected paired results](nlpd_pair_2026-10-12.csv) and
[provenance/raw-evidence pointer](nlpd_pair_2026-10-12.meta).

The independent materialized-band test compares every normalized pixel
bit for all six levels, odd dimensions, vector tails and parallel band
boundaries. Default and all-feature tests pass; the CI-pinned snapshot's
all-feature tests and clippy with `-D warnings` pass. The no-default
wasm32-unknown-unknown library build passes. Existing assertions are unchanged.
Disassembly inspection found redundant bounds checks in the first trial's
vertical row loads; partitioning each source row before the vector loop
removed the observed regression. That trial measured 632.32 ms and 0.51 GiB,
and is retained as a rejected timing result.

Receipts: [baseline 1T](nlpd_baseline_1t_2026-10-12.log),
[candidate 1T](nlpd_partition_candidate_1t_2026-10-12.log),
[baseline 16T](nlpd_baseline_16t_2026-10-12.log),
[candidate 16T](nlpd_candidate_16t_2026-10-12.log),
[first trial](nlpd_candidate_1t_2026-10-12.log).
The [queued zenbench trial](nlpd_candidate_zenbench_1t_2026-10-12.log)
was cancelled while waiting for another experiment; it produced no metric TSV.

## IW-SSIM

Baseline production source: `ad6b49b9`, with the accompanying wall output
that exposes score bits. The candidate forms each square/cross-product row,
then immediately runs the horizontal Gaussian on that row. It removes the
full-image product scratch from `compute_cs`. Horizontal vector FMAs,
scalar tails, vertical filtering and the final reduction remain ordered
as before. Test and development builds retain the materialized kernels
as independent comparison paths.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 555.47 | 516.72 |
| One-thread instructions, five calls | 39,948,799,791 | 37,866,480,168 |
| One-thread cycles, five calls | 15,542,770,749 | 14,646,625,910 |
| Sixteen-thread ms/call | 358.27 | 330.04 |
| One-thread run-heavy peak RSS, GiB | 0.67 | 0.69 |
| Sixteen-thread run-heavy peak RSS, GiB | 0.67 | 0.68 |

Peak RSS did not fall in this workload; no process-memory saving is
claimed. Retired instructions fell. Wall deltas remain below the suite's
15% noise caution. The four-block interleaved zenbench comparison reports
an exploratory -6.061191841489869% latency change, with an A/A change of
-0.7082316070614776%. Its hard system scope confines descendants to CPU 2
and 16 GiB. Every arm passes the exact score checker; all large-image
observations return `3fefff0b71a660d0`. There is no confirmation interval
or promotion verdict.

The ordered materialized-product test compares every horizontal output
bit for squares and cross-products, including odd shapes, narrow valid
regions, SIMD tails and parallel band boundaries. Default and all-feature
tests pass, including Python goldens and strip parity, with unchanged
assertions. All-target all-feature clippy and production-feature library
clippy pass with `-D warnings`; the no-default wasm32 library check passes.
Disassembly contains packed product multiplies followed by the eleven
ordered Gaussian FMAs. Scope builds use the existing pinned snapshot.

Receipts: [baseline 1T](iwssim_baseline_1t_2026-10-12.log),
[candidate 1T](iwssim_candidate_1t_2026-10-12.log),
[baseline 16T](iwssim_baseline_16t_2026-10-12.log),
[candidate 16T](iwssim_candidate_16t_2026-10-12.log),
[paired export](iwssim_pair_2026-10-12.csv),
[raw-evidence pointer and resource receipt](iwssim_pair_2026-10-12.meta).
Reproduce with the VSI build/wall recipes using `-p iwssim`, and the
`ranked-pair-plan` / `ranked-pair-scope` recipes recorded in the metadata.

## PSNR-HVS

Baseline source: `62b5a420`. Its native profile places 38.91% of sampled
cycles in the AVX-512 block band; four indexed gathers carry substantial
local-period samples. The candidate loads eight contiguous samples for
each block and transposes eight blocks with magetypes. A sixteen-lane
batch joins two transposed groups. Arithmetic, DCT tap order, masking and
the raster fold remain unchanged. The candidate band's sampled share is
33.45%; its annotated disassembly contains no indexed gathers. Sample
shares are profile observations, not absolute speedups.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 146.03 | 132.56 |
| One-thread instructions, five calls | 6,400,082,339 | 5,839,578,228 |
| One-thread cycles, five calls | 4,209,813,128 | 3,861,499,873 |
| Sixteen-thread ms/call | 30.99 | 30.41 |
| Run-heavy peak RSS, GiB, both thread counts | 0.24 | 0.24 |

Both score bits stay `40310dd0a5ede6f9/403415c9efc235c4`. Default and
all-feature goldens/tier assertions pass unchanged. The new indexed-load
oracle checks every transposed bit for padded rows, block steps 1/4/8/9/17,
both eight-lane groups, every short-tail length, signed zero and a NaN
payload. All-target all-feature clippy with `-D warnings`, scoped fmt
and the no-default wasm32 library check pass.

The four-block paired observation gives exploratory latency change
-6.35602267886079%, with A/A change +1.028594259878394%. Both HVS and
HVS-M are checked exactly in every arm. Wall changes remain within the
earlier suite's 15% noise caution; there is no confirmation verdict. The
retained result is fewer instructions and removal of indexed gathers,
with unchanged measured RSS.

Receipts: [baseline 1T](psnrhvs_baseline_1t_2026-10-12.log),
[candidate 1T](psnrhvs_candidate_1t_2026-10-12.log),
[baseline 16T](psnrhvs_baseline_16t_2026-10-12.log),
[candidate 16T](psnrhvs_candidate_16t_2026-10-12.log),
[baseline profile](psnrhvs_profile_2026-10-12.txt),
[candidate profile](psnrhvs_candidate_profile_2026-10-12.txt),
[paired results](psnrhvs_pair_2026-10-12.csv),
[paired provenance](psnrhvs_pair_2026-10-12.meta). Profile metadata beside
the reports records raw sample paths, binary hashes and resource receipts.
Use `ranked-wall` with `-p psnrhvs`, `ranked-profile`, and the paired
recipes.

## HDR-VDP-3

Baseline source: `62b5a420`, using the extended existing `perf_probe`
example. Both generations consume the same absolute-luminance field;
v3 explicitly uses Quality, 30 pixels/degree, age 24, no surround,
luminance-default emission and reference options. The v3 candidate
reuses one f64 plan per axis, including twiddles, Bluestein chirps and
filter transforms. One scratch buffer serves successive transforms.
Convolution shares its plans between forward and inverse transforms.
Butterfly arithmetic and axis order remain unchanged.

| One-thread observation, three calls | Baseline | Candidate |
|---|---:|---:|
| 250² seconds per call | 0.648 / 0.643 / 0.643 | 0.423 / 0.420 / 0.418 |
| 250² retired instructions | 46,881,022,790 | 30,653,210,397 |
| 250² cycles | 10,695,895,767 | 7,002,355,294 |
| 256² seconds per call | 0.380 / 0.355 / 0.335 | 0.338 / 0.315 / 0.297 |
| 256² retired instructions | 19,112,998,532 | 17,000,579,747 |
| 512² seconds per call | 1.401 / 1.385 / 1.387 | 1.300 / 1.232 / 1.231 |
| 512² retired instructions | 77,369,534,518 | 68,983,043,340 |
| 250² / 256² run-heavy peak RSS, GiB | 0.02 | 0.02 |
| 512² run-heavy peak RSS, GiB | 0.09 | 0.09 |

The non-power-of-two case benefits from cached Bluestein setup. Its
four-block paired latency change is exploratory -34.17826938080178%,
with A/A change +0.21689984320927078%. Every invocation's three output
digests match. There is no confirmation interval or promotion verdict.
The raw point observations at 256² and 512² remain within the earlier
15% noise caution. Memory peaks did not fall.

With sixteen Rayon threads and affinity to CPUs 0–23, the 250² baseline
is 0.650/0.644/0.642 seconds; candidate 0.425/0.419/0.419 seconds.
This particular v3 workload does not improve with that thread setting.
Both thread settings return digest `d4052bae6722d1b6`; the 256² and
512² digests are `5b3aad535338dd14` and `bf6301622110de34`. The untouched
v2 control returns `8598d191fde2f435` with 0.040/0.034/0.034 seconds
in both builds at 256². Digests include complete maps and scalar outputs.

Default/all-feature unit tests pass. The full all-feature suite retains
its official/reference goldens and map assertions, including the 21-case
v3 corpus, unchanged. New tests compare every real/imaginary output bit
against the previous per-transform implementation for repeated zero and
nonzero inputs, radix-2 and arbitrary lengths, and mixed 2-D shapes.
All-target all-feature clippy passes with `-D warnings`. The no-default
wasm32 library check and clippy pass; the v2 feature-conditional local
variable no longer triggers an unused-mut warning there.

Receipts: [250² baseline](hdrvdp3_baseline_250_2026-10-12.log),
[250² candidate](hdrvdp3_candidate_250_2026-10-12.log),
[256² baseline](hdrvdp3_baseline_256_2026-10-12.log),
[256² candidate](hdrvdp3_candidate_256_2026-10-12.log),
[512² baseline](hdrvdp3_baseline_512_2026-10-12.log),
[512² candidate](hdrvdp3_candidate_512_2026-10-12.log),
[16T baseline](hdrvdp3_baseline_250_16t_2026-10-12.log),
[16T candidate](hdrvdp3_candidate_250_16t_2026-10-12.log),
[v2 baseline](hdrvdp2_baseline_256_2026-10-12.log),
[v2 candidate](hdrvdp2_candidate_256_2026-10-12.log),
[paired export](hdrvdp3_pair_2026-10-12.csv),
[paired provenance and resource receipt](hdrvdp3_pair_2026-10-12.meta).
Build `-p hdrvdp --features parallel,avx512 --example perf_probe` in
the pinned snapshot, then use `ranked-hdr-probe` or
`ranked-hdr-pair-plan` / `ranked-pair-scope`.

## FSIM fused ingress

Baseline source: `68651fc6`, retaining the earlier decimated-column win.
The candidate converts RGB values to Y/I/Q inside each consumed
horizontal box window, avoiding full-resolution color planes. Conversion
values, horizontal/vertical sum order and final normalization are
unchanged. Independent compact rows run through Rayon when large enough;
the luma-only path shares the same implementation with one channel.
Factor-one inputs retain direct plane conversion and avoid its previous
extra copy.

| Measurement | Baseline | Candidate |
|---|---:|---:|
| One-thread ms/call | 669.44 | 594.10 |
| One-thread instructions, five calls | 90,358,060,491 | 83,985,712,591 |
| One-thread cycles, five calls | 18,551,116,239 | 16,167,906,900 |
| Sixteen-thread ms/call | 262.75 | 136.64 |
| One-thread run-heavy peak RSS, GiB | 0.09 | 0.07 |
| Sixteen-thread run-heavy peak RSS, GiB | 0.11 | 0.13 |

The single-thread RSS falls; the sixteen-thread observation uses more
RSS. The timing change there is exploratory, without an interval. The
four-block single-thread paired comparison reports -13.544669810273469%
latency change and A/A +0.13874499293387374%, within the earlier suite's
15% noise caution. No confirmation or promotion verdict is claimed.
Every arm returns both exact score bits
`3fefd56321153b56/3fefb97ee068a939`.

The independent materialized conversion/filter test compares every
Y/I/Q and luma output bit at factors 1–12, odd/narrow shapes, partial
windows, padded rows and truncated final padding. Its large case also
exercises the parallel compact-row path. Default/all-feature goldens,
stride and tier tests pass unchanged; all-target all-feature clippy with
`-D warnings`, scoped fmt and the no-default wasm32 library check pass.

Receipts: [baseline 1T](fsim_fusion_baseline_1t_2026-10-12.log),
[candidate 1T](fsim_fusion_candidate_1t_2026-10-12.log),
[baseline 16T](fsim_fusion_baseline_16t_2026-10-12.log),
[candidate 16T](fsim_fusion_candidate_16t_2026-10-12.log),
[paired export](fsim_fusion_pair_2026-10-12.csv),
[paired provenance and resource receipt](fsim_fusion_pair_2026-10-12.meta).
Use the preceding FSIM build command and the `ranked-wall` /
`ranked-pair-plan` / `ranked-pair-scope` recipes.


## Daala PSNR-HVS baseline and profile

The production CLI driver built from `9b8de0dd` measured 466.4643 ms per
call over 30 exploratory rounds at 3355×2516, one Rayon thread, CPU 2.
The sentinel score was 34.765447061926444. This uses the driver's
procedural pair and the Daala studio-601 YUV444 route, separately from
the Ponomarenko path above. It establishes a baseline, not a speedup.
[Report](psnrhvs_daala_baseline_zenbench_1t_2026-10-12.log) and
[TSV](psnrhvs_daala_baseline_zenbench_1t_2026-10-12.tsv).

A separate sampled run recorded 16,628 samples with zero lost. The
self-cycle shares were 23.23% in `calc_psnrhvs`, 22.19% in
`od_bin_fdct8`, 5.58% in `od_bin_fdct8x8`, 18.01% in
`studio601_yuv444` and 7.55% in `roundf`.
[Profile](psnrhvs_daala_profile_2026-10-12.txt).
The production conversion preserves ordered f32 arithmetic and
rounds each Y/Cb/Cr component separately. A conversion optimization
must compare all 256³ RGB triples against that rounding contract.
The Daala block raster and coefficient fold must retain their ordered
f32 accumulation; the integer DCT must keep its column-first rounding.

Baseline resource receipt: `run-heavy: done rc=0 17s | peak-RSS 0.14GiB |
min-avail 24058MiB | peak-load 0.50`. Profile receipt:
`run-heavy: done rc=0 19s | peak-RSS 0.17GiB | min-avail 24031MiB |
peak-load 0.54`. Sampled timings are kept separate from the baseline.
Reproduce with `just ranked-zenbench BINARY 1 PREFIX psnrhvs-daala
3355x2516` and `just ranked-zenbench-profile BINARY PREFIX
psnrhvs-daala 3355x2516`.
