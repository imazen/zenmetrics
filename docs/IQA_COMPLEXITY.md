# IQA method complexity: operations, halo, access and parallel structure

Status: **provisional, in progress (2026-10-17).** Counts below are tallies of
source code at a stated commit, not measurements. Timings used for comparison
are existing provisional benchmark rows; final timing numbers are pending
cleaner controlled runs.

## Why

Wall-clock rankings mostly measure how much optimization effort a metric has
received. This analysis describes the work a method's definition asks for, so
that metrics can be compared independently of how far each implementation has
been tuned. The owner described the approach (attributed by the owner to
Maciej; no written source for it was found in our repositories, so the
formulation here is ours): weighted operations per pixel, plus halo radius and
memory access pattern.

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

## Files

- `scripts/complexity/evaluate.py`: schema validation and evaluation
  (`--check`, `--shape`, `--json`).
- `benchmarks/iqa_complexity_2026-10-17/tallies/<metric>.<counter>.json`: one
  tally per metric and independent counter.
- `benchmarks/iqa_complexity_2026-10-17/reconcile/`: disagreements between
  counters and how source evidence settled them.
- `benchmarks/iqa_complexity_2026-10-17/README.md`: coverage, results and
  missing cases.
