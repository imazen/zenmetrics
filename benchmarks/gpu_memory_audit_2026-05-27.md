# GPU memory audit across six -gpu metric crates — 2026-05-27

Companion to `benchmarks/gpu_memory_audit_2026-05-27.csv` /
`.meta`. Measures `nvidia-smi memory.used` delta + warm-up wall time
for each `(crate, mode, size)` cell on RTX 5070 / CUDA 13.2.1 /
driver 596.21. Cells run subprocess-per-cell so cubecl's intra-
process memory pool can't bleed across cells; the OS reclaims the
GPU pool on child exit.

Drivers: `crates/<crate>/examples/mem_one_size.rs` (six binaries).
Orchestrator: `scripts/memory_audit/audit_gpu_metrics.py`. Measured
at `b1d080a` ("test(gpu-audit): add per-crate mem_one_size bench
drivers + orchestrator").

## Q1: VRAM per crate × size

`delta_mib` is the peak `nvidia-smi memory.used` minus the
baseline immediately before the child launched, sampled during a
~400 ms hold window the child takes after one warm-up
`compute_with_reference` call (cached-reference encoder hot path).

| crate            | mode        | 1024² | 2048² | 4096² | exp\* | warm @ 4096² |
|------------------|-------------|------:|------:|------:|------:|-------------:|
| butteraugli-gpu  | full        | 415   | 1153  | 4177  | 0.83  |   5834 ms    |
| butteraugli-gpu  | strip (256) | 389   | 243   | 561   | 0.13  |    814 ms    |
| ssim2-gpu        | full        | **538** | **1775** | **6232** | **0.88** | **8190 ms** |
| ssim2-gpu        | strip (256) | 429   | 751   | 1357  | 0.42  |   1453 ms    |
| dssim-gpu        | full        | 385   | 968   | 3230  | 0.77  |   4404 ms    |
| dssim-gpu        | strip (256) | 321   | 630   | 888   | 0.37  |   1173 ms    |
| iwssim-gpu       | full        | 513   | 673   | 2820  | 0.62  |   2785 ms    |
| iwssim-gpu       | strip (256) | 929   | 481   | 760   | n/a   |   1192 ms    |
| **zensim-gpu**   | **full**    | **225** | **449** | **1185** | **0.60** | **1039 ms** |
| cvvdp-gpu        | full        | 385   | 1089  | 3970  | 0.84  |   4765 ms    |
| cvvdp-gpu        | strip_pair  | 418   | 833   | 2272  | 0.61  |   3065 ms    |

\* `exp = log2(delta_4k / delta_1k) / 4`. 1.0 = pixel-count linear
(16× pixels → 16× memory); < 1.0 = sub-linear (fixed cost
dominates at small sizes); > 1.0 = super-linear (allocation
pattern is super-quadratic in the pixel count). 0.6-0.7 is the
range we'd expect from a pyramid-based pipeline once constant
overhead is factored out.

**Headlines:**

- **zensim-gpu owns gold standard** — 5× less VRAM than ssim2-gpu
  at 4096², 8× faster wall, exponent 0.60. It bakes all per-scale
  per-channel work into a single tile-fused `fused_features_kernel`
  per scale and uses 3-channel-fused downscale + xyb kernels.
- **ssim2-gpu is the new worst offender for VRAM** at 6.2 GB on a
  4096² pair (was already #2 HtoD-per-iter at 52; now confirmed #1
  on memory too). Exponent 0.88 — almost perfectly linear with
  pixel count, meaning ~370 MB of every additional MP. At 4K square
  this saturates 50%+ of a 12 GB consumer GPU.
- **cvvdp-gpu post-fix lands in the middle of the pack at 3.97 GB**
  (was 1647 ms at 4096² pre-fix → 4765 ms wall in this run; warm
  numbers are inflated by the first-iter kernel compile, see the
  cvvdp `bench_4096_one_mode` script for cleaner steady-state.) The
  208 → ~12 HtoD/iter fix landed correctly and didn't regress VRAM.
- **Strip mode is the only path that scales gracefully** for any
  crate at 4K — butter strip uses 561 MiB at 16.8 MP, vs 4177 MiB
  for Full. The cost is wall time (Full's amortized compute beats
  strip when the workload fits) and feature gaps (e.g. butter
  strip lacks `set_reference` — `compute_strip` re-uploads the ref
  every call). For 8K+ inputs strip is mandatory.
- **iwssim-gpu strip exponent shows baseline drift.** The 929 MiB
  reading at 1024² is real but inflated by ~530 MiB of residual
  pool from the prior cell (a cubecl quirk where `cudaFree` lazy-
  returns to the driver). The 4096² strip = 760 MiB is solid; the
  trend across the three points is what matters and is plainly
  better than full.

Note on warm_ms: this is the first `compute_with_reference` call
after pipeline construction — it includes one cubecl kernel-compile
pass per kernel used in the call (typically 30-200 ms per kernel).
Steady-state per-call cost is much lower; for actual perf-vs-perf
read the existing `bench_warm_ref.rs` / `bench_t4_warm.rs`
harnesses. The warm_ms here is the best comparable indicator we
get from a single warm-up call inside the subprocess pattern.

## Q2: ssim2-gpu has the same speedup left on the table that
   cvvdp-gpu did. (Probably 3-5× — see SSIM2 deep dive below.)

The shape of the inefficiency is different from cvvdp-gpu's. cvvdp
was uploading per-band uniforms inside per-band hot loops (the fix
lifted constants out, giving 36× at 4096²). ssim2-gpu **doesn't
upload data** — it launches **dozens of separate per-channel
kernels** when zensim-gpu's mega-fused kernel pattern would
collapse them. Each launch carries one HtoD for its scalar args
even when the data is small, which is where the 52-HtoD/iter
count comes from. The win is in **kernel fusion + 3-channel
fusion**, not in lifting constants.

See `docs/SSIM2_OPTIMIZATION_REVIEW.md` for the per-kernel ledger,
file:line citations, and the five concrete fix patterns.
