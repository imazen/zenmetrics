# Metric dispatch consolidation — ONE way to score, sdr/hdr × cpu/gpu × all 6

**Status: in progress (started 2026-06-26). DONE: C1, C1b, C2. NEXT: C3, C4, C5, C6.**
- **C1** (3a7dda0d): cpu-metrics → cpu-cvvdp/cpu-iwssim; run_metric GPU→native-CPU failover
  for cvvdp/iwssim; CI guards (`*_scores_on_native_cpu_in_cpu_build`). Verified cvvdp=9.598 CPU.
- **C1b** (14e2eba9): score-pairs's own GPU-only guard fixed → falls through to run_metric's native
  CPU. Verified CPU-sweep `score-pairs --metric cvvdp`=9.268. CPU fleet unblocked for cvvdp.
- **C2**: cubecl-cpu is NEVER dispatched. `GpuRuntime::Cpu → Backend::Cpu` (native, was CubeclCpu);
  cpu-metrics → `zenmetrics-api/cpu-metrics` (all six native, so the umbrella CPU failover works for
  ssim2/dssim/butter/zensim too); `cpu_fallback_backend()` returns `Backend::Cpu` even in a no-cpu
  build (fail loud, never silently run cubecl-cpu); CI guard `no_gpu_runtime_maps_to_cubecl_cpu`.

Goal (user directive): make the
"a metric looks GPU-only / its CPU path is unwired / there are several
inconsistent ways to score" class of bug **structurally impossible** — one
dispatch entry that every caller funnels through, automatic native GPU→CPU
failover for all six metrics, CPU compiled by default, sdr+hdr uniform. Delete
parallel paths; change defaults as needed.

## Root cause (why cvvdp "looked GPU-only", source-traced 2026-06-26)

Five overlapping scoring paths with inconsistent feature/cuda gating:

1. `score-pairs` typed `CvvdpBatchScorer` bypass (`main.rs:1224/1993/2134`,
   `#[cfg(gpu-cvvdp)]`) — short-circuits `run_metric()` for per-pair instance reuse.
2. `run_metric()` (`metrics/mod.rs`) — the umbrella single-shot path.
3. `MetricCache` (`metrics/cache.rs`) — cached umbrella path (sweep/batch); routes
   cvvdp via `compute_umbrella` only under `#[cfg(gpu-cvvdp)]`, else `_ => run_metric`.
4. `zenmetrics-orchestrator` scoring (`orchestrator_runner::orchestrator_score_one`,
   `sweep/run.rs::score_via_orchestrator`) — **cuda-gated**; the `not(orchestrator-cuda)`
   variant is a stub `Err("requires orchestrator-cuda")`, and `score_via_orchestrator`
   is **bit-rotted** (uses removed `zenmetrics_orchestrator::{Task,TaskData}` / `run_single`,
   which moved to `::executor::` and are themselves cuda-gated) → `sweep,orchestrator`
   won't compile.
5. `zenmetrics-api::cpu_dispatch` (`Backend::Cpu`, native SIMD `cvvdp::Cvvdp`/`iwssim`)
   — the *correct* CPU path, but gated behind `cpu-cvvdp`/`cpu-iwssim` which the CLI
   only pulls via `orchestrator-cpu-cvvdp` (→ pulls the broken orchestrator).

Three concrete defects:
- **D1**: CLI `MetricKind::{Cvvdp,Iwssim}` are classified `"GPU"` (`metrics/mod.rs:165-166`)
  — the enum itself encodes "GPU-only". No `Cvvdp`/`CvvdpGpu` CPU/GPU split like ssim2 has.
- **D2**: the auto-backend ladder is `[Cuda,Wgpu,Hip,CubeclCpu]` (`metrics/mod.rs::auto_order`,
  `capability.rs:130 cpu_fallback_backend()==CubeclCpu`). `CubeclCpu` = cubecl kernels on CPU,
  which **panics on `atomic<f32>` for cvvdp**. The real CPU path `Backend::Cpu`→`cpu_dispatch`
  is **absent from the ladder**, so "failover" lands on a broken backend, never the SIMD port.
- **D3**: CLI `cpu-metrics` = `[butteraugli,zensim,fast-ssim2,dssim-core,rgb,imgref]` only —
  does NOT forward to `zenmetrics-api/cpu-metrics` (which has clean standalone `cpu-cvvdp`=`[dep:cvvdp,cvvdp/std]`,
  `cpu-iwssim`). So native CPU cvvdp/iwssim aren't compiled in default/gpu/sweep builds.

## Target architecture — ONE switch

`Metric::new(kind, Backend::Cpu, …)` already routes to native `cpu_dispatch` for all six
(`metric.rs:691/829`) when the `cpu-<m>` feature is on. So:

- **One function** `score_unified(kind, ref, dist, params, prefer)` (CLI side, or promoted into
  the umbrella): try native GPU backends (`Cuda→Wgpu→Hip`, only if a `gpu-<m>` feature is built
  AND the runtime inits) → **fall back to `Backend::Cpu` (native `cpu_dispatch`)** → else a loud
  "no backend compiled for <m>: enable `gpu-<m>` or `cpu-<m>`" error. `CubeclCpu` drops out of the
  default ladder (opt-in only; it's a dev/debug path, not a fallback).
- **The ONE way IS the orchestrator's memory-aware scheduler — NOT a delete (corrected 2026-06-26
  per user).** Scoring N variants × M GPU metrics naively (variant-major) holds M pipelines in VRAM
  at once or reloads a pipeline per variant (thrash) + re-uploads the ref every time. The orchestrator's
  `Orchestrator::run_all` (pool.rs:2333) ALREADY sorts tasks by `(metric, w, h, ref_hash, task_id)` →
  **metric-major**: each (metric, dims) group dispatches together, pipeline built once, reference cached,
  swept across all variants, then the next metric. That + the warm-pool + OOM-ladder is exactly the
  intelligence we want as the single path. Mirrors the SPLIT fleet's `for m in METRICS: score-pairs -m`.
- **The blocker is purely cuda-gating, not architecture.** The executor (run_all/run_single, executor.rs)
  is `#![cfg(all(feature="bench", feature="cuda"))]` — so the metric-major scheduler only runs on CUDA;
  CPU + non-cuda GPU fall to the "requires orchestrator-cuda" stub. But the executor uses the umbrella
  `Metric` + a `CpuAdapter` (Backend::Cpu already wired, "CpuNotYetWired shim removed") with NO direct
  cubecl-cuda calls. So **un-cuda-gate the executor** (`cfg(all(bench,cuda))` → `cfg(bench)`) → the
  scheduler becomes backend-agnostic (CPU/wgpu/hip/cuda via the umbrella).
- **Then funnel everything through it**: `score-pairs`/`sweep`/`batch` build the (variant×metric) task
  list and hand it to `run_all`; `MetricCache`/`CvvdpBatchScorer` become (or are subsumed by) the
  orchestrator's per-(metric,dims) warm slots — not separate ad-hoc caches. The orchestrator is the one
  intelligent scoring path, CPU and GPU alike.
- **CPU compiled by default**: CLI `cpu-metrics` forwards to `zenmetrics-api/cpu-metrics` (all six). ✅ C2.
- **sdr/hdr**: the scheduler takes the hdr feeding per metric (cvvdp/butter native linear planes; others
  PU21) — unify the `main.rs` hdr branch into the task build, not a parallel loop.
- **Backend enum**: `Backend::Cpu` (native) is the fallback everywhere; `cpu_fallback_backend()` returns
  `Backend::Cpu`, never `CubeclCpu`. ✅ C2. (cubecl-cpu is never dispatched.)

## Chunks (land each compiling + tested)

- **C1 — CPU-by-default + native-CPU failover in `run_metric` (category-killer).**
  CLI `cpu-metrics` → `+ zenmetrics-api/cpu-metrics` (+ deps). `run_metric` (and the one switch)
  try GPU then `Backend::Cpu`. cvvdp/iwssim no longer hardcoded GPU at dispatch. Regression test:
  a **no-GPU build scores all six on CPU** (`score-pairs --metric cvvdp` → real JOD). CI gate so it
  can't regress.
- **C2 — collapse the ladder + `cpu_fallback_backend()=Backend::Cpu`**; drop `CubeclCpu` from the
  default ladder (opt-in flag only).
- **C5 (was "strip", CORRECTED to "un-gate + make canonical") — un-cuda-gate the orchestrator
  executor** so its metric-major `run_all` scheduler (pool.rs:2333; pipeline-once-per-metric, cached
  ref, swept across all variants, OOM ladder) runs on CPU + any GPU backend, not just CUDA. The executor
  already uses the umbrella `Metric` + `CpuAdapter` with no direct cubecl-cuda → change
  `#![cfg(all(bench,cuda))]` → `#![cfg(bench)]` and walk the compile fallout. This is the memory-aware
  scoring intelligence we must KEEP (user, 2026-06-26: "lots of intelligence needed to keep mem use
  manageable … sweep a metric at a time across all variants, so you don't hold in memory or thrash gpu").
  C5a (done, d6bc8b94) already de-bit-rotted `score_via_orchestrator` by delegating to the maintained entry.
- **C3 — funnel `score-pairs`/`sweep`/`batch` through the orchestrator's `run_all`** (the one
  memory-aware path); fold `CvvdpBatchScorer` + `MetricCache` into the orchestrator's per-(metric,dims)
  warm slots rather than keeping them as separate ad-hoc caches. (Depends on C5.)
- **C4 — unify hdr** into the orchestrator task build (per-metric feeding), not a parallel loop.
- **C6 — docs + semver**: README/CLAUDE.md reflect "one way = the orchestrator scheduler"; `cargo semver-checks`.

## Invariants / tests that make the bug impossible

- `tests/` (CI, no-GPU): every `MetricKind` produces a finite score on a tiny pair via the one
  switch. Fails the build if any metric has no CPU fallback compiled.
- A guard test asserting there is exactly one public scoring entry the subcommands call (no
  `CvvdpBatchScorer`/`orchestrator_score_one` references at call sites).
