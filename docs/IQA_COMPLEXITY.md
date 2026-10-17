# IQA method complexity: operations, halo, access and parallel structure

Status: **provisional, in progress (2026-10-17).** Counts below are tallies of
source code at a stated commit, not measurements. Timings used for comparison
are existing provisional benchmark rows; final timing numbers are pending
cleaner controlled runs.

## Why

Wall-clock rankings mostly measure how much optimization effort a metric has
received. This analysis describes the work a method's definition asks for, so
that metrics can be compared independently of how far each implementation has
been tuned.

The approach follows M. Pedzisz (Nokia), *Complexity Assessment of
Full-Reference Image Quality Metrics*, a JPEG AIC-4 contribution circulated
as a preliminary draft dated 2026-10-07 (cited with the owner's permission;
later revisions may differ). That draft counts primitive scalar operations
along each metric's source-grounded path from input pair to score, before
SIMD, FMA fusion, threading and compiler transforms, weights them by
declared relative costs (add/sub/mul 1, comparison/abs/negation 1, sqrt 4,
division 6, exp/log/trig 20, softplus 25, general power 40, element copies
0), and normalizes per input pixel at 840×944, the AIC-4 test resolution. It
reports from about 3 weighted operations per pixel (PSNR-Y) to about 4,265
(Butteraugli, RGB), and recommends weighted counts tied to identified
implementations over runtime for comparing complexity. Its declared weights
are available here as `scripts/complexity/weights_pedzisz_2026_draft.json`.

This repository extends that approach with halo radius and access pattern
(the owner's request), per-precision classes, a work/span view of parallel
structure, operation weights measured on a named CPU, and a comparison with
measured CPU and wall time. Counts are of zenmetrics' implementations, not
of the reference implementations the draft used, so per-metric values are
not expected to match it.

It does not produce a hardware-independent runtime. Operation weights belong
to one microarchitecture, ISA tier and precision. Measured time also depends on
vectorization, cache behaviour and threading, which the counts do not model.

## Unit and scope

- **Per original input pixel**: totals are divided by `W·H` of one input image.
  A full-reference method that processes both images counts both
  (`images: 2`); stages that combine the pair count once.
- **Intrinsic stages only**: colour conversion, filtering, pyramids,
  transforms, pooling and the final formula. Decode, encode, file I/O,
  allocation, model loading, weight upload and thread-pool start are excluded
  from the counts. Where they are measured they are reported separately.
- **Codecs are excluded** from all tallies, including the optimization-effort
  tally.
- **Shapes**: counts are formulas in `W`, `H` and method parameters, evaluated
  at 256², 1024², 1920×1080 and 3840×2160. Pyramid depth, crops, patch grids and
  resize schedules make per-pixel cost shape-dependent; the evaluated shape is
  always stated.

## Operation classes

Logical scalar operations, per element, by precision (`f16 f32 f64 i8 u8 i16
u16 i32 u32 i64`):

| class | counts |
|---|---|
| `add` | add, subtract, accumulate |
| `mul` | multiply |
| `fma` | one fused multiply-add where the source requests it (`mul_add`, FMA intrinsics, magetypes `mul_add`). Rust never contracts `a*b+c` on its own, so written `a*b+c` is `mul` + `add`. Flops count an `fma` as 2. |
| `div`, `rcp` | division; reciprocal (including approximate reciprocal with refinement, refinement ops counted separately) |
| `sqrt`, `rsqrt` | square root; reciprocal square root |
| `exp`, `log`, `pow`, `trig` | transcendental calls (`log10`, `log2`, `ln` are `log`; `cbrt` is `pow`; `atan2`, `cos` are `trig`) counted as one call each. Their internal cost is a weight, not an op count. |
| `cmp`, `abs` | compare, min/max, clamp, select; absolute value |
| `cvt` | numeric conversion (u8→f32, f32→f64, rounding to integer) |
| `int` | integer arithmetic and index computation that is part of the method (quantizers, histogram bins), not loop bookkeeping |
| `load`, `store` | logical element reads and writes of image-sized buffers |

Bytes read and written per element are recorded with an access class:
`pointwise`, `sep_stencil` (separable filter, radius `r` per pass),
`stencil2d`, `block`, `downsample`, `upsample`, `resample`, `transform`
(DCT, FFT, wavelet), `reduction`, `gather`, `histogram`, `sort`,
`data_dependent`, and, for neural networks, `conv` and `matmul`.

## Halo, span and parallel structure

Each stage records its stencil radius in its own sample grid and the grid's
scale in original pixels. The evaluator sums `radius × scale` along the
dependency chain to give the **cumulative halo**: how far, in original pixels,
an output depends on its input. A strip or tile implementation must carry at
least this many rows of overlap.

The **span** of a stage is the length of its dependency chain in element
steps under the method's mathematical definition: 1 for pointwise and stencil
work, `log2(n)` for a sum or mean (reductions are associative in the method
even when an implementation sums sequentially for bit stability), the row
length for a recursive (IIR) filter, the iteration count for a data-dependent
loop. Pyramid levels that depend on each other add their spans (`chain`).

From these the evaluator reports **work/span**, an upper bound on ideal
parallel speedup with unlimited processors; the **minimum stage width**, the
fewest independent elements at any stage (coarse pyramid levels and final
formulas); the number of **global barriers** (stages that need a complete
reduction before the next starts); and **serial stages**. SIMD, threads and
batching are separate questions: SIMD needs regular stride-1 access within a
stage; threads need width and few barriers; batching needs repeated
independent calls. Arithmetic intensity (flops per logical byte) places a
stage against a memory roofline; it ignores cache reuse between stages.
Measured thread scaling answers a different question again and is reported
beside these bounds, not substituted for them.

## Counting conventions (v2, after the pilot)

The pilot (`ssim`, `gmsd`, `ssim2`, counted independently by two lanes) showed
where unwritten conventions changed totals. These rules settle them:

1. **Inline approximations are counted as written.** A transcendental the
   source computes inline (fast-ssim2's cube root: bit manipulation plus
   Newton steps) is its arithmetic; a library call (`f64::exp`, `powf`) is one
   call of its class.
2. **Implementation-only work is a separate class.** Strip-overlap
   recomputation, padding or format copies and materialized intermediates the
   method does not need are stages with `"class": "implementation"`. Totals
   report intrinsic and implementation flops separately.
3. **Bytes are compulsory traffic.** `bytes_read`/`bytes_written` per element
   are the stage's input planes read once and output written once. Per-tap
   reads of a stencil appear only in the `load` count.
4. **Halo is per axis.** Each stage has `axis` (`x`, `y`, `xy`, `none`) and a
   one-sided reach `radius` in its own grid; `scale` converts to original
   pixels. A 2×2 box downsample reaches 1 sample. Only stages whose level `l`
   consumes level `l−1` (pyramid downsampling) set `chain`; a blur applied at
   each level does not chain across levels.
5. **Recursive filters** record the effective support radius the
   implementation uses for its warm-up as `radius`, are `serial`, and have a
   span equal to the scanned row or column length.
6. **Default path.** Counts follow the code path the CLI runs by default;
   where a SIMD path fuses or blocks work differently from the scalar fallback
   (fast-ssim2's horizontal blur is 4-way blocked: 3 mul + 10.5 fma + 3 add per
   output; its vertical pass is the plain recurrence), the default path is
   counted and the difference noted. These are therefore counts of an
   implementation, not of the method in the abstract.

Pilot reconciliation is recorded in
`benchmarks/iqa_complexity_2026-10-17/reconcile/pilot.md`.

## Neural networks

Neural methods are counted at the logical layer level from the exported ONNX
graph: multiply-accumulates per convolution or matrix multiply (counted as
`fma`), plus normalization, activation, pooling and resize element work, at
the network's actual input shape after the method's own resize or crop
schedule. Runtime kernels (im2col, Winograd, fused epilogues) execute different
operation mixes; their counts are not the method's.

## Weights and measured comparison

A weighted cost multiplies each class by a per-instruction cost on a named
microarchitecture and ISA tier: reciprocal throughput for a throughput
estimate and latency for dependency-chain estimates. Summing latencies is not
a runtime prediction. Weight sources are recorded per entry: LLVM's
`llvm-mca` scheduling model for the CPU (a model, not a measurement) and
measured probes with hardware counters. No weight is entered from memory.

Predicted cost is compared with measured CPU time and wall time separately,
within one host, device, thread count, mode, ingress, shape and revision. Fits
follow the repository's convention: intercept from small sizes, per-pixel slope
from large sizes, never one line across the whole range. Poor correlations
and large residuals are reported, not removed.

## Measured runs (controlled, provisional)

`benchmarks/iqa_complexity_2026-10-17/measured/` holds one controlled series
per file: one host (i265, Core Ultra 7 265K), one binary, one pinned set of
P-cores, one shape (1024×1024), one content pair (zenmetrics-corpus photo vs
its q20 JPEG), warm calls in one process. `scripts/complexity/perf_deltas.py`
runs `complexity-reps` under `perf stat` at 1 and 1+K calls and reports the
difference per call, so decode, setup and process start are excluded; three
repeats per metric are kept. CPU time (`task-clock`) and wall time are
separate columns. These are still single-host, single-shape numbers; they are
not a ranking across hosts.

**Measured thread scaling is a property of the implementation, not of the
method.** From one build with each metric crate's own threading enabled
(`i265_par_1024.tsv`), the 8-thread speedup over 1 thread on P-cores was:

| speedup at 8 threads | metrics |
|---|---:|
| below 1.2× (not threaded on this path) | 47 |
| 1.2–2× | 12 |
| 2–4× | 18 |
| 4–6× | 4 |
| 6–9× | 2 (gmsd, ssim) |

Median 1.03× over 83 metrics. The PSNR family, VMAF, CAMBI, FLIP, CIEDE2000
and the pyiqa-convention SSIM/MS-SSIM variants do not thread in this build, so
their 1.0× says nothing about what the method allows. Intrinsic parallelism
is instead bounded by the tallies' work/span and barrier counts (section
"Halo, span and parallel structure"); the two answer different questions and
are reported side by side, never substituted.

### First predicted-vs-measured comparison (n = 15, provisional)

Tallies landed so far (one counter each, swap recounts pending) against the
single-thread 1024×1024 run, `analysis/correlation_1t_1024_*.tsv`:

| weights | Spearman (95 % bootstrap) | Pearson of logs | residual spread | outside 3× of the fit |
|---|---|---|---:|---|
| flop count | 0.93 (0.74–0.99) | 0.91 | 17.6× | ciede2000, mdctpsnr, nlpd-iqa, ssimulacra1 |
| i265 P-core scalar reciprocal throughput | 0.94 (0.73–0.99) | 0.93 | 26.4× | mdctpsnr, ssimulacra1 |
| i265 P-core scalar latency | 0.93 (0.75–1.00) | 0.92 | 20.4× | ciede2000, ssimulacra1 |
| Pedzisz 2026 draft declared weights | 0.91 (0.69–0.98) | 0.93 | 24.0× | mdctpsnr, ssimulacra1 |

CPU time and wall time agree at one thread, as they should. The high rank
correlation comes mostly from the three-decade spread in cost between PSNR
and mDCT-PSNR; within a decade, measured time departs from the tally by up to
a factor of 26. Those residuals are where implementation choices dominate:
mDCT-PSNR runs about 3.5× faster than its count predicts (FMA-cloned,
vectorized kernels), SSIMULACRA1 about 7.5× slower (its count and its code
path are both on the recount list). Weighting by measured instruction costs
does not change the picture. The comparison will be redone when all tallies
are reconciled; n = 15 of 115 CLI names.

## Optimization effort

Question: how many optimization iterations, and how much agent time, went
into each metric, so that speed can be read against effort. Codecs are
excluded throughout.

**Definitions.** An *optimization commit* changes a metric family's source and
states a performance purpose or a measured speed change. An *iteration* is an
optimization commit that also cites verification evidence (a benchmark file,
or a measurement plus a parity or bit-identity check). Commit counts are not
labor time. *Active agent time* is the sum of gaps of at most 300 s between
consecutive events of one agent session; *elapsed* is last minus first event.
Both describe agent activity, not human effort, and neither is estimated
where records are missing.

**Agent-session evidence (measured 2026-10-17,
`scripts/complexity/agent_effort.py`).** Each host aggregates its own Claude
Code, Codex and Devin records and emits counts only; no transcript content is
published. Family source paths: `scripts/complexity/effort_families.json`.

| host | sessions in a zenmetrics checkout | sessions touching a family | union active hours | sessions touching exactly one family | earliest touching session |
|---|---:|---:|---:|---:|---|
| i265 | 18 | 11 | 63.5 | 0 | 2026-10-11 |
| dev | 12 | 9 | 29.8 | 3 (all CVVDP) | 2026-08-05 |
| r7900x | 9 | 9 | 10.2 | 1 (VIF) | 2026-09-25 |
| mac | 3 | 3 | 2.1 | 1 (VSI) | 2026-10-14 |

Findings:

- **Per-metric agent time is not recoverable from these records.** 27 of 32
  touching sessions edited several metric families at once (coordinators,
  sweeps, shared harnesses), so any split of their time between metrics would
  be an estimate. Per-family numbers in the JSON files are non-additive.
- **Most of the history is missing.** Records start in August on dev and in
  October elsewhere, while most metric crates landed between 2026-05-03 and
  2026-09-25 (`scripts/dashboard/favorites.json`); local Claude Code
  transcripts are pruned over time. One further agent host had no in-scope
  sessions; r5900xt refused the key and was not read.
- Git-based iteration counts per family are tallied separately
  (`benchmarks/iqa_complexity_2026-10-17/effort/`), with the same exclusions.

## Files

- `scripts/complexity/evaluate.py`: schema validation and evaluation
  (`--check`, `--shape`, `--json`).
- `benchmarks/iqa_complexity_2026-10-17/tallies/<metric>.<counter>.json`: one
  tally per metric and independent counter.
- `benchmarks/iqa_complexity_2026-10-17/reconcile/`: disagreements between
  counters and how source evidence settled them.
- `benchmarks/iqa_complexity_2026-10-17/README.md`: coverage, results and
  missing cases.
