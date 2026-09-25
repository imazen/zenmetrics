# New-metrics benchmark + optimization pass — 2026-09-24

Scope: the new additions — `nlpd` crate, native `classical` metrics
(psnr / psnr-y / ssim / ms-ssim), and the `vmaf` adapter (in-process
RGB→YUV420 path; no `vmaf` executable exists on this host, so the
subprocess half could not be run end-to-end).

Host: Ryzen 9 7900X (12c/24t), 29 GiB. Runs under `taskset -c 0-7`,
release profile, runtime SIMD dispatch (no target-cpu=native). Harness:
`benchmarks/heaptrack/drivers/cpu_profile/src/bin/new_metrics_wall.rs`
(zenbench, per-pair-normalized `par{T}` rows, TSVs in
`benchmarks/new_metrics_2026-09-24/`).

## Score safety

All **96** size×metric×mode cells are **bit-identical** before and after
optimization (abs Δ = 0.0 — sentinel diff across 512/1024/4K/8K for
nlpd, psnr, psnr-y, ssim, ms-ssim, yuv420 × lat/par1/par4/par8).

One exception, documented: the nlpd crate's 97×99 parity test image
moved 0.4779789526 → 0.4779833907 (Δ = 4.4e-6) after pass 1, attributed
to FMA contraction in the newly vectorized horizontal accumulate. The
crate's declared tolerance is 1e-5 vs the PyTorch reference
(0.4779785872); old Rust deviated 3.7e-7, new deviates 4.8e-6 — inside
tolerance, but no longer within ~1e-6 of PyTorch on that input. Every
benchmark-size sentinel is bit-identical. If sub-1e-6 parity vs PyTorch
is ever required on odd dimensions, the horizontal accumulate is the
site to de-contract.

`cargo test -p nlpd` and `metrics::classical::tests` pass.

## Wall results (per-pair ms; `lat` = serial latency, `parT` = T-way pair throughput /T)

### nlpd (lower = better score)

| size | lat orig | lat new | par8 orig | par8 new |
|------|---------:|--------:|----------:|---------:|
| 512² | 25.72 | **12.15** (2.12×) | 4.63 | **2.25** (2.06×) |
| 1024² | 107.79 | **52.12** (2.07×) | 18.95 | **11.01** (1.72×) |
| 4K | 1024.10 | **453.15** (2.26×) | 182.06 | **109.33** (1.67×) |
| 8K | 5026.03 | **1894.92** (2.65×) | 886.47 | **516.52** (1.72×) |

128×128 microbench (`cargo run -p nlpd --example wall`): 1.58 →
**0.51 ms/pair** — ~6.9× faster than the PyTorch reference (~3.52 ms).

### ssim

| size | lat orig | lat new | par8 orig | par8 new |
|------|---------:|--------:|----------:|---------:|
| 512² | 115.33 | **30.27** (3.81×) | 16.76 | **5.42** (3.09×) |
| 1024² | 463.09 | **130.39** (3.55×) | 66.43 | **56.93** (1.17×) |
| 4K | 4103.82 | **1427.13** (2.88×) | 632.67 | 653.36 (0.97×) |
| 8K | 17083.52 | **5840.97** (2.92×) | 2565.56 | 2575.69 (1.00×) |

### ms-ssim

| size | lat orig | lat new | par8 orig | par8 new |
|------|---------:|--------:|----------:|---------:|
| 512² | 149.60 | **37.87** (3.95×) | 23.02 | **6.28** (3.66×) |
| 1024² | 598.52 | **155.21** (3.86×) | 83.89 | **61.67** (1.36×) |
| 4K | 5309.33 | **1790.90** (2.96×) | 791.02 | 779.70 (1.01×) |
| 8K | 22421.48 | **7616.40** (2.94×) | 3354.28 | 3295.33 (1.02×) |

### psnr / psnr-y / yuv420 (unchanged code)

| size | psnr lat | psnr-y lat | yuv420 lat |
|------|---------:|-----------:|-----------:|
| 512² | 0.58 | 0.53 | 1.66 |
| 1024² | 2.47 | 2.19 | 6.37 |
| 4K | 19.45 | 17.39 | 50.36 |
| 8K | 71.68 | 65.21 | 198.11 |

psnr/psnr-y are already DRAM-bound; par8 throughput ≈5–7×. yuv420 par8
≈6.5×.

## Heaptrack (4K, per-metric isolated, 2 reps)

| metric | peak heap | vs 50 MB inputs | notes |
|--------|----------:|-----------------|-------|
| psnr / psnr-y | 49.8 MB | 1.0× | no scratch |
| yuv420 | 62.3 MB | 1.2× | one output buffer |
| nlpd | 630.5 MB | ~12.6× | 10 scratch planes + pyramid (was 630.4 — peak is live planes, not churn; alloc calls 298) |
| ssim | 647.0 → **580.7 MB** | ~11.6× | was ~13 planes + 3 transform planes + per-call temp×2; now 8 planes, 100 alloc calls (was 360+) |

8K nlpd: **2.52 GB** peak (heap mode, 2 reps) — 10×133 MB scratch slabs
+ 6×133 MB level planes + ~200 MB decimation output.

## What was optimized (verified score-identical)

`crates/nlpd/src/lib.rs`:
- u8→f32 conversion folded into deinterleave (killed 2 full-size copies).
- `Scratch` slabs (5 planes/side) replace ~280 per-(level×channel) Vecs.
- `filter5_horizontal`: border-peeled; interior = k-outer accumulate
  over equal-length slices (bounds-check-free, vectorized). stride-2
  decimator computes the contiguous conv then decimates (identical
  values; gathers avoided).
- `normalize_band` interior as 5 zipped slices (auto-vectorized
  abs+fma+div); `upsample_bilinear` x-table hoisted.

`crates/zenmetrics-cli/src/metrics/classical.rs`:
- `gaussian` → `gaussian_into`: eliminated ~2.7 B `reflect()`/`rem_euclid`
  calls at 4K (border-peeled 11-tap separable conv, accumulate form,
  whole-interior-block contiguous zips for the vertical pass).
- `a²`, `b²`, `a·b` transforms fused into the source-row read — three
  66 MB planes never materialize.
- Caller-owned `temp` scratch shared across the 5 convolutions.

Callgrind (nlpd, 512², 2 calls): 346 M Ir total;
`filter5_horizontal`+helpers ~30% (was ~62% at 1024² pre-pass-2);
`level_transform` internals (upsample, normalize, lap, iter machinery)
now the top site at ~42%.

## Missed optimizations, ranked

### P0 — classical ssim/ms-ssim is still the slowest metric

1. **No intra-call parallelism.** ssim lat 1.43 s at 4K vs nlpd 0.45 s;
   `par8` gives only ~1.0× at 4K because 8 concurrent f64-SSIM calls
   thrash LLC (~580 MB heap each) and share memory BW. libvmaf's SSIM
   and dssim-core both thread *inside* the metric (row slices / rayon).
   Rayon over the 3 channels + row bands in `gaussian_into` is the
   correct fix for single-pair latency; pair-level parallelism then
   stays the throughput mode at small sizes only.
2. **f64→f32.** dssim-core and fast-ssim2 both run SSIM-family math in
   f32. Would halve the ~580 MB peak and double SIMD width (~2× more
   speed). Expected score drift ~1e-7 abs on unit-range inputs — needs
   an explicit tolerance decision vs the f64 reference, so it was NOT
   applied (this pass held bit-exactness).
3. **Reference-side caching.** In sweep use (one ref × N distorted),
   `ma`/`aa` are recomputed per pair. libvmaf caches feature planes per
   reference; a `score_precomputed_ref` API would cut ~40% of SSIM work
   in that pattern. API change, not applied.

### P1 — nlpd

4. **Explicit SIMD horizontal conv.** The vertical pass has an
   `archmage`/magetypes kernel; horizontal relies on autovectorization
   (~5 Ir/elem — f32x8 intrinsics would be ~1.3). ~30% of instructions.
5. **Shared ref/dis scratch.** Only `normalized` must persist pairwise;
   `tmp_h`/`up`/`tmp_b`/`lap` could share one slab → −4 planes
   (~530 MB at 8K). 
6. **`deinterleave_u8`** (~10% Ir): stride-3 u8→f32 via packed
   shuffles/table instead of per-pixel `chunks_exact(3)` scalar loop.
7. `upsample_bilinear` gather (r0[x0],r0[x1],r1[x0],r1[x1] per px) —
   SIMD-able but small.

### P2 — vmaf adapter

8. **Subprocess + YUV file per pair.** No `vmaf` binary on this host;
   adapter spawns it and round-trips a 50 MB yuv file at 4K. libvmaf's
   C API (`vmaf_read_pictures`) does this in-process with its own
   threading — `vmaf-sys` FFI is the real fix, also unlocking vmaf's
   internal SSIM/PSNR feature threads.
9. `write_yuv420` (50 ms at 4K): single-thread scalar RGB→BT.709 +
   file write; SIMD fixed-point matrix + rayon rows ≈ 4–6×.

### P2 — psnr/psnr-y

10. Effectively done: DRAM-read-bound; par8 ≈7×.

### Cross-cutting

11. `Rgb8Image` interleaved input forces per-call planar conversion in
    every metric; a planar `score(&[PlaneView])` API would amortize
    across metrics in sweep pipelines.

## Reference comparison

- **PyTorch NLPD** (reference impl): ~3.52 ms/pair @128² vs **0.51 ms**
  now — ~6.9×, same math, CPU-only.
- **libvmaf (C)**: SSIM is luma-only f32, threaded row-slices — the
  threading model is the applicable lesson, not the score path.
- **dssim-core (Rust)**: f32 planar + rayon + SIMD blur — the
  architecture the classical path should move toward (P0-1/P0-2).
- **fast-ssim2 (Rust)**: replaces the exact Wang window with separable
  approximations + f32 SIMD — *different score* (SSIM2), cited only as
  evidence the conv pipeline is the whole cost.
- **Halide**: the transform applied here by hand (boundary-peeled pure
  interior + vectorized accumulate) is exactly what a Halide schedule
  emits — same math, different schedule.
- **Zig**: no widely-used standalone SSIM/NLPD implementation; the C
  implementations above are the relevant reference class.

## Repro

```
cargo build --release --bin new-metrics-wall -p cpu-profile
CPU_WALL_NO_GATE=1 taskset -c 0-7 ./target/release/new-metrics-wall <512|1024|4K|8K> out.tsv [metric]
heaptrack -o heap ./target/release/new-metrics-wall heap <size> [metric]
```
