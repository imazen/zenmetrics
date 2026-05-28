# Phase 6 — CPU backend wiring

This document is the canonical mapping between each `MetricKind` and the
CPU reference implementation the orchestrator's OOM-fallback ladder
calls when no GPU backend fits. Update it whenever a CPU backend is
added, swapped, or marked unavailable.

## Per-metric mapping

| MetricKind | CPU crate         | Feature flag    | Cached-ref API     | Notes |
|------------|-------------------|-----------------|--------------------|-------|
| Cvvdp      | `cvvdp` (in-tree) | `cpu-cvvdp` | yes (`warm_reference`) | Pure-Rust CPU port; matches pycvvdp v0.5.4 within ≤1e-3 JOD. |
| Ssim2      | `fast-ssim2` (Imazen) [^ssim2-8h] | `cpu-ssim2`     | yes (`Ssimulacra2Reference`) | SIMD-accelerated SSIMULACRA2 (archmage AVX2/AVX-512/NEON/WASM128). Accepts `ImgRef<[u8; 3]>` directly; precompute path skips ~50 % of pipeline. |
| Dssim      | `dssim-core`      | `cpu-dssim`     | yes (`DssimImage` cache) | Multi-scale LAB; reuses prepared image. |
| Butter     | `butteraugli`     | `cpu-butter`    | no (recompute)     | `butteraugli()` is one-shot. |
| Zensim     | `zensim`          | `cpu-zensim`    | no (recompute)     | `Zensim::compute()` is one-shot. |
| Iwssim     | *(none)*          | —               | —                  | No clean CPU reference upstream. Chooser routes around. |

[^ssim2-8h]: Phase 8h (2026-05-27) replaced the original `ssimulacra2`
    0.5 wiring (from Phase 6, commit `0fc139a3`) with Imazen's
    SIMD-accelerated `fast-ssim2` 0.8 per the global crate index. The
    swap unlocked three improvements: (1) `ImgRef<[u8; 3]>` input
    skips the manual `Xyb::try_from(Rgb::new(...))` transcode, (2)
    SIMD path via `archmage` gives 2-3× speedup on AVX2/AVX-512/NEON
    hosts, (3) `Ssimulacra2Reference` precompute API replaces the
    "stash bytes for shape parity" fallback with a true warm path.
    `ssim2-gpu`'s parity dev-dependency on upstream `ssimulacra2` is
    a separate concern and was not touched (the ssim2-gpu crate tests
    GPU↔upstream-CPU agreement; this row tests orchestrator CPU
    fallback behaviour).

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
- **fast-ssim2** (`supports_cached_ref = true`, Phase 8h): caches a
  `Ssimulacra2Reference` (precomputed reference XYB + sub-bands +
  blur scratch). Subsequent `compare` calls skip ~50 % of the
  pipeline. Speedup ~2× on reference-reuse workloads. Replaces the
  Phase 6 byte-stash fallback that the upstream `ssimulacra2 0.5`
  wiring required.
- **butteraugli**, **zensim**
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

fast-ssim2 documents ~24 image-sized f32 planes plus a downscale
pyramid (`fast_ssim2::MAX_IMAGE_PIXELS` = 16384²) and caps inputs to
16384² to bound the working set; the per-pixel estimate matches the
prior `ssimulacra2 0.5` row. `butteraugli` and `fast-ssim2` at 4096²
are within tolerance for a 128 GB workstation but **not** for a
16 GB GitHub Action runner. If
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

## Other CPU adapter choices to audit (added Phase 8h, 2026-05-27)

Phase 8h swapped `ssim2` from upstream `ssimulacra2 0.5` to
Imazen's SIMD-accelerated `fast-ssim2`. The other four Phase 6
wirings were not changed in 8h, but they MAY benefit from similar
swaps in a future tick. This section documents the audit so the
next session doesn't have to re-discover the picture.

| Metric  | Current CPU crate | Imazen SIMD alternative? | Action |
|---------|-------------------|--------------------------|--------|
| Butter  | `butteraugli` 0.9.2 | **Already Imazen.** `imazen/butteraugli` is the canonical workspace; uses `archmage` AVX-512/AVX2/NEON dispatch. The `0.9.2` workspace-pin and the `butteraugli` crate name happen to overlap with the historical Google name but this IS the Imazen pure-Rust port. | **No swap needed.** Already correct. |
| Dssim   | `dssim-core` 3.5 (kornelski/dssim) | **No Imazen alternative known** as of 2026-05-27. `dssim-core` upstream is well-maintained, uses Rayon for parallelism, and is the de-facto Rust SSIM library. We do not have an in-house SSIM crate — `fast-ssim2` is SSIMULACRA2 (different algorithm), and `zensim` is a separate metric family. | **No swap planned.** Keep `dssim-core` unless an Imazen alternative ships. |
| Zensim  | `zensim` (workspace) | N/A — already Imazen. | **No swap needed.** |
| Cvvdp   | `cvvdp` (workspace, in-tree) | N/A — already Imazen, in-tree. The Phase 8c renamed this from the historical name; the implementation is the pure-Rust CPU port that matches pycvvdp v0.5.4 within ≤1e-3 JOD. | **No swap needed.** |
| Ssim2   | `fast-ssim2` 0.8 (Imazen, **Phase 8h**) | Now correct. | Already swapped. |

### Honesty note on the `butteraugli` crate name

The `butteraugli` crate name on crates.io was originally claimed by
Imazen for the pure-Rust port; the global crate index (`~/.claude/CLAUDE.md`)
lists `butteraugli` as "Butteraugli perceptual image difference metric"
under the Metrics section without a "Pure Rust" prefix, but the
`imazen/butteraugli` repo is the source-of-truth (BSD-3-Clause,
archmage SIMD, no C FFI). Production callers consuming `butteraugli =
"0.9"` from crates.io get the Imazen implementation. The audit row
above is correct: this is already an Imazen SIMD path.

### When to revisit

If any of the following change, re-run this audit:

- A new Imazen SIMD SSIM-family crate ships on crates.io (we'd swap
  the dssim row).
- The upstream `dssim-core` regresses on a SIMD path that we'd want
  to fix in a fork (we'd vendor or fork).
- A Phase X migration moves `cvvdp` to its own crates.io publication
  (the row stays correct; the build-from-path adjustment is
  Cargo-only).
- `butteraugli 0.9` releases a major version bump with API changes
  (the adapter's `ButteraugliParams::new()` call site needs review).
