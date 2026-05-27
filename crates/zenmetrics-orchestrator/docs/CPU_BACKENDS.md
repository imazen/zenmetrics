# Phase 6 — CPU backend wiring

This document is the canonical mapping between each `MetricKind` and the
CPU reference implementation the orchestrator's OOM-fallback ladder
calls when no GPU backend fits. Update it whenever a CPU backend is
added, swapped, or marked unavailable.

## Per-metric mapping

| MetricKind | CPU crate         | Feature flag    | Cached-ref API     | Notes |
|------------|-------------------|-----------------|--------------------|-------|
| Cvvdp      | `cvvdp` (in-tree) | `cpu-cvvdp` | yes (`warm_reference`) | Pure-Rust CPU port; matches pycvvdp v0.5.4 within ≤1e-3 JOD. |
| Ssim2      | `ssimulacra2`     | `cpu-ssim2`     | no (recompute)     | Builds Xyb per-call; no warm-state API. |
| Dssim      | `dssim-core`      | `cpu-dssim`     | yes (`DssimImage` cache) | Multi-scale LAB; reuses prepared image. |
| Butter     | `butteraugli`     | `cpu-butter`    | no (recompute)     | `butteraugli()` is one-shot. |
| Zensim     | `zensim`          | `cpu-zensim`    | no (recompute)     | `Zensim::compute()` is one-shot. |
| Iwssim     | *(none)*          | —               | —                  | No clean CPU reference upstream. Chooser routes around. |

## Feature flags

- `cpu-cvvdp`, `cpu-ssim2`, `cpu-dssim`, `cpu-butter`, `cpu-zensim` are
  additive. Each pulls only its own dependency.
- `cpu-all` is the convenience flag bundling all five.
- Default features intentionally do NOT include any `cpu-*`. Callers
  that only care about capability detection pay no dep cost. Sweep
  workers that want the fallback ladder must enable explicitly:
  `--features cuda,cpu-all`.

## Iwssim honest-stop rationale

There is no clean CPU reference crate for IW-SSIM in the wider Rust
ecosystem. The original IW-SSIM paper has a MATLAB implementation
(Wang et al., 2011) and `iwssim-gpu` ports the algorithm directly. We
deliberately did NOT spend the implementation budget porting it to CPU
in Phase 6 because:

1. The orchestrator's CPU backend is a fallback for VRAM-constrained
   environments. Iwssim is a parity-style metric with relatively low
   production volume; sites that run out of VRAM on Iwssim can either
   pick a different metric or move to a higher-VRAM machine.
2. Writing a from-scratch CPU port without a published reference
   implementation to validate against risks shipping a metric whose
   scores diverge from the GPU path — exactly the failure mode the
   CLAUDE.md "zero tolerance for precision loss" rule flags.

What happens at runtime when a caller asks for `MetricKind::Iwssim`
with `Backend::Cpu`:

- The chooser surfaces `RejectReason::CpuMetricUnavailable` for the
  Cpu candidate — visible in `BackendChoice::considered`.
- The OOM-fallback ladder advances past Cpu to the next available
  backend (which, for Iwssim, is none beyond GpuStrip).
- If both GPU candidates also fail, `run_single` returns
  `OrchestratorError::FullyExhausted`. The caller can:
  - Try a smaller image size
  - Use a different metric
  - Move to a higher-VRAM GPU

If upstream eventually publishes a clean Rust CPU implementation
(e.g., `iwssim-core` on crates.io) we will wire it behind a new
`cpu-iwssim` feature flag in a follow-up release.

## Cached-reference semantics

CPU adapters expose `set_reference` / `compute_with_cached_reference`
to mirror the umbrella `Metric` shape. The pool's worker dispatches
through the cached path when both:

1. The auto-detect window saw the same `(metric, w, h, ref_hash)`
   tuple recently.
2. The adapter reports `supports_cached_ref() == true`.

Per-crate cached-ref behavior:

- **cvvdp** (`supports_cached_ref = true`): true warm path —
  `warm_reference` precomputes the reference's Weber pyramid + DKL
  planes; subsequent `score_with_warm_ref` calls skip ~50 % of the
  pipeline. Worth promoting whenever a reference is reused.
- **dssim-core** (`supports_cached_ref = true`): caches the prepared
  `DssimImage<f32>` (multi-scale LAB representation); `compare` reuses
  it. Speedup ~2× on reference-reuse workloads.
- **ssimulacra2**, **butteraugli**, **zensim**
  (`supports_cached_ref = false`): no upstream cached-ref API. The
  adapter caches the *bytes* of the reference so the cached-ref call
  shape still works, but the implementation recomputes the per-call
  XYB / pixel transform. Correctness preserved, no speedup.

## RAM characteristics

CPU references use system RAM, not VRAM. The chooser models this as
`vram_mib = 0` (CPU is always feasible from a VRAM perspective);
real RAM ceilings are deferred to Phase 7's ResourceBudget work.

Approximate per-pixel scratch (measured 2026-05-27 on 7950X / 128 GB
RAM workstation, peak resident-set during compute):

| Metric  | bytes/pixel | 1024² (3 MP) | 4096² (48 MP) |
|---------|-------------|--------------|---------------|
| cvvdp   | ~5-7        | ~20 MiB      | ~120 MiB      |
| zensim  | ~10-15      | ~40 MiB      | ~250 MiB      |
| butter  | ~30-40      | ~100 MiB     | ~600 MiB      |
| dssim   | ~40         | ~120 MiB     | ~700 MiB      |
| ssim2   | ~50         | ~150 MiB     | ~850 MiB      |

`butteraugli` and `ssimulacra2` at 4096² are within tolerance for a
128 GB workstation but **not** for a 16 GB GitHub Action runner. If
the orchestrator is targeting a low-RAM environment, callers should
either:

- Disable the expensive CPU backends (`--no-default-features --features
  cuda,cpu-cvvdp,cpu-zensim` keeps the cheap ones).
- Use the GPU pathway and accept FullyExhausted on tight environments.

A future Phase 7 ResourceBudget revision will measure CPU RAM during
the warm() bench and reject CPU candidates whose predicted RAM exceeds
free system memory by the safety margin.

## Acceptance gates

- All CPU backends construct + compute successfully on a 256² synthetic
  pair: `tests/cpu_backend.rs::all_backends_construct_and_compute`.
- cvvdp parity vs cvvdp-gpu: |diff| < 1e-3 JOD at 256² and 1024²
  (covered in cvvdp's own parity tests, referenced here for the
  ladder).
- OOM-forced fallback returns a CPU Score with `backend_used = Cpu`:
  `tests/cpu_backend.rs::oom_fallback_routes_to_cpu`.
- chooser picks Cpu when GPU is mocked-unavailable:
  `tests/cpu_backend.rs::chooser_picks_cpu_when_gpu_oom`.

## Build matrix

The crate's CI exercises three feature combinations:

```bash
# Capability detection only — no CPU deps pulled.
cargo check -p zenmetrics-orchestrator

# Full GPU build with no CPU fallback.
cargo check -p zenmetrics-orchestrator --no-default-features --features cuda

# Production sweep worker config.
cargo test  -p zenmetrics-orchestrator --no-default-features \
            --features cuda,cpu-all
```

Per-CPU-backend selective builds (one feature at a time) are smoke-
tested in `tests/cpu_backend.rs` to make sure no implicit
cross-feature dependency creeps in.
