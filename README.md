# zenmetrics

Multi-vendor GPU implementations of the perceptual image quality
metrics Imazen runs in production, plus a unified CLI.

Built on CubeCL via the [`zenforks-cubecl`](https://crates.io/crates/zenforks-cubecl)
publication of [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)
(0.10.x — carries pinned-upload + PTX-cache-widening + Metal
`Atomic<f32>` capability patches for our use case). A single
`#[cube]`-annotated Rust kernel source dispatches across CUDA (NVIDIA),
WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a
build-time CPU fallback.

## Metric crates

The six GPU metric crates plus the two CPU reference crates that the
orchestrator's CPU ladder dispatches to:

| Crate | Metric | Range / shape | Parity reference |
|---|---|---|---|
| [`butteraugli-gpu`](crates/butteraugli-gpu/) | Butteraugli | distance, max-norm (default) + libjxl 3-norm | [`butteraugli`](https://crates.io/crates/butteraugli) 0.9.4 |
| [`ssim2-gpu`](crates/ssim2-gpu/) | SSIMULACRA2 | 0–100, higher better | [`ssimulacra2`](https://crates.io/crates/ssimulacra2) 0.5 |
| [`dssim-gpu`](crates/dssim-gpu/) | DSSIM | distance, 0 = identical | [`dssim-core`](https://crates.io/crates/dssim-core) 3.4 |
| [`iwssim-gpu`](crates/iwssim-gpu/) | IW-SSIM (Wang & Li 2011) | `[0, 1]`, 1.0 = identical | [`iwssim`](crates/iwssim/) (in-tree CPU port) |
| [`zensim-gpu`](crates/zensim-gpu/) | zensim feature extractor | 228-feature vector + scalar score 0–100 | [`zensim`](https://github.com/imazen/zensim) 0.3.0 |
| [`cvvdp-gpu`](crates/cvvdp-gpu/) | ColorVideoVDP (still-image, GPU) | JOD ~3–10, higher better | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) 0.5.4 |
| [`iwssim`](crates/iwssim/) | IW-SSIM (CPU reference + SIMD) | `[0, 1]`, 1.0 = identical | self (pure-Rust port) |
| [`cvvdp`](crates/cvvdp/) | ColorVideoVDP (still-image, CPU) | JOD ~3–10 + per-pixel diffmap | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) 0.5.4 |

The CPU side of each metric is supplied by an external reference crate
([`fast-ssim2`](https://crates.io/crates/fast-ssim2) 0.8.1,
[`dssim-core`](https://crates.io/crates/dssim-core) 3.4,
[`butteraugli`](https://crates.io/crates/butteraugli) 0.9.4,
[`zensim`](https://github.com/imazen/zensim) 0.3.0) or an in-tree crate
([`cvvdp`](crates/cvvdp/), [`iwssim`](crates/iwssim/)). All six metrics
expose a CPU backend (the IW-SSIM CPU port landed in 2026-05; see the
[Modes × metrics support matrix](#modes--metrics-support-matrix)).

### Supporting crates

| Crate | Role |
|---|---|
| [`zenmetrics-api`](crates/zenmetrics-api/) | Umbrella: one `MetricKind` enum + one `Metric` type dispatching to all six per-crate opaque scorers |
| [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/) | Capability-aware backend chooser + persistent benchmark cache + OOM fallback ladder + warm worker pool |
| [`zen-metrics-cli`](crates/zen-metrics-cli/) | `zen-metrics` CLI front-end (score / batch / compare / sweep) |
| [`zenmetrics-corpus`](crates/zenmetrics-corpus/) | Shared test-image corpus (test infra) |
| [`iwssim-filter-codegen`](crates/iwssim-filter-codegen/) | Build-time generator for the IW-SSIM separable blur filters |
| [`cvvdp-conformance`](crates/cvvdp-conformance/) | pycvvdp conformance fixtures + parity harness for the cvvdp crates |

The workspace also contains the vast.ai / Hetzner / RunPod / Salad
sweep-fleet crates (`zen-cloud-*`, `zencloud-hetzner`,
`zenfleet-orchestrator`, `zen-job-*`, `zen-ledger`, `zen-jobdash`,
`zen-sweep-worker`, `zenstats`) that drive the backfill pipeline; they
are infrastructure, not part of the metric API.

## Recommended entry point: `zenmetrics-orchestrator`

For any caller that scores **more than one (ref, dist) pair** —
sweeps, picker training, RD curves, batch comparison, anything with
multiple tasks — use [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/).
It adds three things every previous in-tree caller had to hand-roll:

1. **Backend selection.** Persistent per-machine benchmark cache picks
   the fastest backend that fits available VRAM for each task. Knows
   which `(metric, size)` combinations OOM on this machine and avoids
   them on subsequent runs.
2. **OOM-safe fallback ladder.** `GpuFull → GpuStrip → (Cvvdp:
   GpuStripPair) → Cpu`. Each downgrade is recorded in the cache so the
   same machine never tries the failing combination twice.
3. **Cached-reference auto-detect.** xxhash3 hashes ref bytes per task,
   promotes consecutive same-ref tasks to the `set_reference` +
   `compute_with_cached_reference` API for the 1.5–3× speedup that
   sweeps benefit from.

**Quick decision table:**

| Caller shape | Use |
| --- | --- |
| One `(ref, dist)` per process, no fallback needed | `zenmetrics-api` directly |
| Batch / sweep / picker training / RD curve | **`zenmetrics-orchestrator`** |
| Streaming workload | **`zenmetrics-orchestrator`** |
| OOM-tolerant scoring | **`zenmetrics-orchestrator`** |
| One-ref / many-dist workloads | **`zenmetrics-orchestrator`** |

See [`crates/zenmetrics-orchestrator/README.md`](crates/zenmetrics-orchestrator/README.md)
for quickstart, the streaming + batch APIs, OOM handling details,
cached-ref semantics, CPU backend selection, capability cache lifecycle,
and the full configuration surface. Migration code samples in
[`crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md`](crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md).

The `zen-metrics` CLI routes scoring through the orchestrator by
default (since Phase 7.7.1, 2026-05-27). The legacy direct-dispatch
path remains available via `zen-metrics --use-legacy-scheduler …` (or
`ZENMETRICS_USE_LEGACY_SCHEDULER=1`) — useful when an archived parquet
sidecar needs bit-identical regeneration, or when comparing the two
paths for parity. The orchestrator path itself was validated as
bit-identical to legacy across all 54 cells (6 metrics × 3 sizes × 3
qs) on RTX 5070 + 7950X — see
[`benchmarks/orchestrator_parity_2026-05-27_phase771_run3.csv`](benchmarks/orchestrator_parity_2026-05-27_phase771_run3.csv)
for the per-cell data. The `--use-orchestrator` flag and
`ZENMETRICS_USE_ORCHESTRATOR` env var are accepted for
backwards-compat with pre-Phase-7.7.1 scripts / Docker images but
emit a deprecation warning.

The new sweep image
[`Dockerfile.sweep.v27`](Dockerfile.sweep.v27) bakes the orchestrator
features in and ships
[`scripts/sweep/onstart_orchestrator.sh`](scripts/sweep/onstart_orchestrator.sh)
as an entrypoint that drives the per-cell scoring through the
orchestrator's worker pool.

One per-metric carve-out remains: butteraugli stays on the legacy
direct-dispatch path because `ButteraugliOpaque::new_with_memory_mode`
resolves Auto to strip-mode (butter is strip-preferred), which drops
to single-resolution scoring and diverges from the legacy CLI's
always-multires output by ~14-30 %. The orchestrator transparently
falls back to legacy for butter; sweeps emit the same column shape
in both paths.

## SRCC sanity table

Spearman rank correlation coefficient against published still-image
MOS datasets, sign-normalized so higher = better. These figures are
**illustrative, sourced externally** (the published
[Cloudinary SSIMULACRA2 benchmark](https://github.com/cloudinary/ssimulacra2_rs)
table for the reference metrics) — they are not regenerated by any
harness in this repo, so treat them as an order-of-magnitude sanity
check on metric discrimination, not a committed measurement. The
metric each crate computes is bit-comparable to the cited reference,
so the reference's published SRCC transfers.

| Metric | TID2013 | KADID-10k | CID22 |
|---|---|---|---|
| `dssim-gpu` (= DSSIM) | 0.871 | 0.856 | 0.872 |
| `ssim2-gpu` (= SSIMULACRA2) | 0.819 | 0.785 | 0.885 |
| `butteraugli-gpu` (3-norm) | 0.664 | 0.543 | 0.794 |
| `iwssim-gpu` (= IW-SSIM) | (not benchmarked here) | | |
| `zensim-gpu` (= zensim) | (Imazen-internal benchmark) | | |
| `cvvdp-gpu` (= ColorVideoVDP) | (pending — reference is pycvvdp 0.5.4) | | |

## Memory modes

Every GPU metric crate exposes a `MemoryMode` enum + a
`new_with_memory_mode` constructor so callers choose how the GPU
working set is laid out. The umbrella ([`zenmetrics-api`](crates/zenmetrics-api/))
re-exports a single user-facing enum and converts to each crate's own
`MemoryMode` at the call boundary:

```rust
// zenmetrics_api::MemoryMode — the portable subset every metric accepts.
pub enum MemoryMode {
    /// Per-crate `resolve_auto` picks the variant that fits the cap. Default.
    Auto,
    /// Whole-image working set on device.
    Full,
    /// Vertical strips of `h_body` body rows + the crate's halo per
    /// side. `h_body == None` lets the resolver pick the largest body
    /// that fits the cap.
    Strip { h_body: Option<u32> },
    /// Reserved — every per-crate `From` maps `Tile` to `Auto` today.
    Tile { h: u32, w: u32 },
}
```

cvvdp-gpu additionally exposes two cvvdp-specific variants on its
**typed** enum (`cvvdp_gpu::MemoryMode`) that the umbrella's portable
subset does not carry, because they change the one-shot/cached-ref
shape or the JOD value:

- `StripPair { h_body }` — Mode B: ref and dist both walk in strips
  together (no full-ref cache). Best for one-shot CLI callers; the
  orchestrator surfaces it as `Backend::GpuStripPair`.
- `CappedPyramid { levels }` — JOD-shifting safety net that truncates
  pyramid depth to shrink the deepest-band blur halo. **Not
  bit-identical to Full** — opt-in only; `Auto` never picks it.

The full per-metric breakdown — which modes each crate exposes on CPU
and GPU, and the exact constructor to invoke each — is in the
[Modes × metrics support matrix](#modes--metrics-support-matrix) and
[API surface](#api-surface-invoking-each-mode) sections below.

### Auto policy and the orchestrator's crossover

`MemoryMode::Auto` resolves per crate by:

1. Reading the VRAM cap: `ZENMETRICS_VRAM_CAP_BYTES` (decimal usize)
   when set, else a live free-VRAM probe (cubecl / `nvidia-smi`), else
   an 8 GB default.
2. Estimating the whole-image working-set bytes via the per-crate
   `estimate_gpu_memory_bytes` helper (zensim-gpu additionally reserves
   `CUBECL_OVERHEAD_BYTES` ≈ 193 MiB for the runtime pool).
3. Picking Full when it fits and the crate is not strip-preferred;
   else picking Strip with an auto-sized `h_body` that fits the cap.
4. Returning `Error::TooBigForFull { needed, cap }` when neither fits.

Only **butteraugli-gpu** is strip-preferred — its `resolve_auto` tries
Strip *first* and picks it even when Full would fit, because the strip
walker is the faster path on that crate
([`crates/butteraugli-gpu/src/memory_mode.rs`](crates/butteraugli-gpu/src/memory_mode.rs)).
dssim-gpu, ssim2-gpu, iwssim-gpu, zensim-gpu, and cvvdp-gpu are
Full-preferred — `Auto` only drops to Strip when Full exceeds the cap.

When the [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/)
drives scoring it does **not** rely on per-crate `Auto` alone — it runs
a cost-model-aware backend chooser over its persistent benchmark cache.
A `ChooserConfig::vram_safety_margin` (default 0.15) is held back, and
the chooser picks the fastest backend that fits. For a single cold call
(`ExecContext::OneShot`, task #146) it additionally consults the
measured one-shot CPU/GPU crossover
([`benchmarks/cpu_gpu_crossover_2026-05-29.tsv`](benchmarks/cpu_gpu_crossover_2026-05-29.tsv))
and routes small images to CPU rather than paying the GPU
context-init floor; the warm pool / sweep path stays `Batch` and ranks
on warm steady-state cost. See
[API surface](#api-surface-invoking-each-mode).

### Backwards compatibility

The historical `Metric::new(backend, w, h, params)` constructor is
preserved and delegates through `new_with_memory_mode(.., MemoryMode::Auto)`.
Existing call sites compile and behave the same unless
`ZENMETRICS_VRAM_CAP_BYTES` is set tight enough to force a mode change.

## Modes × metrics support matrix

Which execution modes each metric exposes, on CPU and on GPU, verified
against each crate's public API. Legend: ✓ supported · ✗ not supported
in this release · n/a not applicable to that metric.

- **Full** — whole-image working set.
- **Strip** — vertical strip walker, cold `(ref, dist)` per call.
- **warm_ref** — reference cached once, `score`/`compute` per distorted
  image against whole-image ref state.
- **warm_ref_strip** — reference cached, distorted image walked in
  strips. (iwssim's GPU variant uniquely walks the *ref* in strips too —
  `CachedRefStripPolicy::BothStripped`.)
- **StripPair** — cvvdp-only Mode B: ref + dist walk in strips together,
  no full-ref cache (one-shot CLI path; orchestrator `Backend::GpuStripPair`).
- **CappedPyramid** — cvvdp-only, JOD-shifting depth cap (opt-in safety
  net; not bit-identical to Full, never picked by `Auto`).

### GPU metric crates

| Crate | Full | Strip | warm_ref | warm_ref_strip | StripPair | CappedPyramid |
|---|---|---|---|---|---|---|
| `cvvdp-gpu` | ✓ | ✓ ¹ | ✓ | ✓ | ✓ | ✓ |
| `ssim2-gpu` | ✓ | ✓ | ✓ | ✓ | n/a | n/a |
| `butteraugli-gpu` | ✓ | ✓ ² | ✓ | ✓ | n/a | n/a |
| `dssim-gpu` | ✓ | ✓ | ✓ | ✓ | n/a | n/a |
| `iwssim-gpu` | ✓ | ✓ | ✓ | ✓ ³ | n/a | n/a |
| `zensim-gpu` | ✓ | ✓ | ✓ | ✓ | n/a | n/a |

¹ cvvdp-gpu's `Strip` (Mode E) is the cached-ref path — `warm_reference_srgb`
+ a per-strip dist walker; the one-shot strip is `StripPair`. Verified
[`crates/cvvdp-gpu/src/memory_mode.rs`](crates/cvvdp-gpu/src/memory_mode.rs)
(`MemoryMode::{Full, Strip, StripPair, CappedPyramid}`) +
[`pipeline.rs`](crates/cvvdp-gpu/src/pipeline.rs) (`Cvvdp::new`,
`new_strip`, `new_strip_pair`, `new_capped_pyramid`).
² butteraugli-gpu is the one **strip-preferred** crate — `Auto` picks
Strip even when Full fits.
³ iwssim-gpu's `warm_ref_strip` can keep the ref full or walk it in
strips (`CachedRefStripPolicy`); the other crates keep the ref full and
strip only the dist. Verified
[`crates/zenmetrics-api/src/memory_mode.rs`](crates/zenmetrics-api/src/memory_mode.rs).

### CPU reference crates

| Metric (CPU) | Full | Strip | warm_ref | warm_ref_strip |
|---|---|---|---|---|
| `cvvdp` (in-tree) | ✓ | ✓ | ✓ | ✓ |
| `ssim2` (fast-ssim2 0.8.1) | ✓ | ✓ | ✓ | ✓ |
| `butter` (butteraugli 0.9.4) | ✓ | ✓ | ✓ | ✓ |
| `iwssim` (in-tree) | ✓ | ✓ | ✓ | ✓ |
| `zensim` (zensim 0.3.0) | ✓ | ✓ | ✓ | ✓ |
| `dssim` (dssim-core 3.4) | ✓ | ✗ | ✓ | ✗ |

**dssim CPU has no strip walker** — `dssim-core` 3.4 exposes no strip
API, so `dssim` CPU is Full + warm_ref only (verified
[`crates/zenmetrics-orchestrator/src/cpu_adapter.rs`](crates/zenmetrics-orchestrator/src/cpu_adapter.rs)
`compute_strip` / `compute_warm_ref_strip` return an error for dssim).
On the GPU, dssim-gpu *does* support Strip.

## API surface: invoking each mode

There are three layers. Pick by how many pairs you score:

1. **Umbrella ([`zenmetrics-api`](crates/zenmetrics-api/)) — one cold
   pair, no fallback.** One enum, one constructor, one score:

   ```rust
   use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, MemoryMode};

   // Auto memory mode (the default Metric::new path).
   let mut m = Metric::new(
       MetricKind::Cvvdp, Backend::Cuda, 1024, 1024,
       MetricParams::default_for(MetricKind::Cvvdp),
   )?;
   let score = m.compute_srgb_u8(&ref_rgb, &dist_rgb)?;

   // Force a specific memory mode at construction:
   let mut m = Metric::new_with_memory_mode(
       MetricKind::Ssim2, Backend::Cuda, 4096, 4096,
       MetricParams::default_for(MetricKind::Ssim2),
       MemoryMode::Strip { h_body: None },   // None → resolver auto-sizes the body
   )?;

   // Cache one reference, score many distorted images against it:
   m.set_reference_srgb_u8(&ref_rgb)?;
   let s1 = m.compute_with_cached_reference_srgb_u8(&dist1)?;
   let s2 = m.compute_with_cached_reference_srgb_u8(&dist2)?;
   # Ok::<(), zenmetrics_api::Error>(())
   ```

   The umbrella's `MemoryMode` carries the portable `{ Auto, Full,
   Strip, Tile }` subset; it converts to each crate's own enum at the
   boundary. cvvdp's `StripPair` / `CappedPyramid` are **not** in the
   umbrella subset — reach for the typed crate (below) to use them.

2. **Typed per-crate opaque — a mode the umbrella doesn't expose.**
   Each crate ships `<Metric>Opaque::new` / `new_with_memory_mode` plus
   `set_reference_srgb_u8` + `compute_with_cached_reference_srgb_u8`
   (cvvdp-gpu names these `warm_reference_srgb` +
   `compute_with_warm_ref_srgb`). cvvdp's extra modes:

   ```rust
   use cvvdp_gpu::{CvvdpOpaque, CvvdpParams, MemoryMode, Backend};

   // Mode B one-shot strip-pair (lowest one-shot VRAM):
   let mut s = CvvdpOpaque::new_with_memory_mode(
       Backend::Cuda, 4096, 4096, CvvdpParams::default(),
       MemoryMode::StripPair { h_body: Some(256) },
   )?;

   // JOD-shifting capped pyramid (opt-in; NOT bit-identical to Full):
   let mut s = CvvdpOpaque::new_with_memory_mode(
       Backend::Cuda, 4096, 4096, CvvdpParams::default(),
       MemoryMode::CappedPyramid { levels: 5 },
   )?;
   # Ok::<(), cvvdp_gpu::Error>(())
   ```

   The typed `cvvdp_gpu::Cvvdp` pipeline also offers the matching
   constructors directly: `Cvvdp::new`, `new_strip`, `new_strip_pair`,
   `new_capped_pyramid`.

3. **CPU strip — the in-tree `cvvdp` / `iwssim` crates.** The CPU
   reference crates take an explicit `h_body` on the strip calls:

   ```rust
   use cvvdp::{Cvvdp, CvvdpParams};

   // Strip-shape allocation up front (peak heap bounded to the strip):
   let mut c = Cvvdp::new_strip(4096, 4096, CvvdpParams::default(), 512)?;
   let jod = c.score_strip(&ref_rgb, &dist_rgb, 512)?;

   // Or cache the reference, then strip-walk each distorted image:
   c.warm_reference(&ref_rgb)?;
   let jod = c.score_with_warm_ref_strip(&dist_rgb, 512)?;
   # Ok::<(), cvvdp::Error>(())
   ```

   `h_body` must be a positive power of two — pass `512` when unsure
   (the per-crate default). `iwssim` exposes `iwssim::STRIP_BODY_DEFAULT`
   for the same purpose.

### Orchestrator: automatic mode + backend selection

For batches / sweeps, let [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/)
choose. It owns a persistent benchmark cache and a pure decision
function over it:

```rust
use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, ExecContext, TaskShape};
use zenmetrics_api::MetricKind;

let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;   // bench-on-demand; cache-hit if fresh

let task = TaskShape { metric: MetricKind::Cvvdp, width: 4096, height: 4096 };

// Batch / warm-pool ranking (ranks on warm steady-state ns/px):
let choice = orch.choose_backend_for_task(&task)?;          // ExecContext::Batch

// Single cold call — apply the measured CPU/GPU one-shot crossover:
let choice = orch.choose_backend_for_task_with_context(&task, ExecContext::OneShot)?;
println!("{:?} @ {:.2} ns/px", choice.backend, choice.predicted_ns_per_px);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The chooser's `Backend` enum is the resolved mode: `GpuFull`,
`GpuStrip`, `GpuStripPair` (cvvdp only), or `Cpu`. The `ExecContext`
controls how the cold-start floor is weighed:

- **`Batch`** (default) ranks on the cache's warm `ns_per_px` — correct
  when a persistent warm worker amortizes the GPU context-init floor.
  GPU wins at every measured size for every metric.
- **`OneShot`** consults the measured one-shot crossover: at/below the
  per-metric crossover size (cvvdp / ssim2 / butter / zensim through
  16 MP, dssim through 4 MP, iwssim through 1 MP) it routes to CPU when
  CPU is a feasible candidate, because a single cold GPU call would pay
  the ~181 ms context-init floor that makes CPU faster at that size.

For the full streaming + batch scoring APIs (`submit` / `poll` /
`run_all` / `upload_reference`), the OOM fallback ladder, and cached-ref
auto-detect, see
[`crates/zenmetrics-orchestrator/README.md`](crates/zenmetrics-orchestrator/README.md).

## Performance profile

GPU scoring cost splits into three components. Modelling a workload as

```
total ≈ process_start + Σ_refs(per_ref) + Σ_dists(per_dist)
```

is accurate because each piece is paid in a different scope and each was
measured separately:

- **`process_start`** — paid **once per process**: the CUDA context init
  (`Backend::client()`, a flat ~181 ms floor that is independent of metric
  and image size) plus the first-kernel PTX/JIT load for each metric the
  first time it runs. On the CPU backend this term is ≈ 0 (no device
  handshake — it starts computing immediately).
- **`per_ref`** — paid **once per distinct reference image** you cache via
  `set_reference_srgb_u8` (cvvdp: `warm_reference_srgb`): the metric's
  reference-side precompute. Every new reference re-pays this cost; budget
  one `per_ref` per distinct reference. For five of six metrics this cost is
  flat across references (`setref1 ≈ setref2 ≈ …`); iwssim at 16 MP is the
  exception, where the first reference is *cheaper* (~68 ms) than subsequent
  distinct references (~120–160 ms) — so size the larger value at 16 MP.
- **`per_dist`** — paid **once per scored distorted image** against a warm
  cached reference: `score_with_warm_ref(dist)`, the steady-state per-call
  wall.

The consequence is a ~181 ms one-time GPU floor (plus per-metric JIT). For a
**single small image on a freshly-launched process the CPU wins** — it has
no floor to amortize. As the image grows or the batch lengthens, the GPU's
throughput outruns the CPU even after paying the floor: for **batch / server
use (warm context, reference cached) the GPU is faster at every measured
size** (warm per-call is 10–100× below the CPU wall). The per-process floor
is paid once and shared across every metric and every pair scored in that
process — which is exactly why
[`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/) keeps one
long-lived warm worker. The full warmth-scope analysis (which transitions
re-pay which component) is in
[`docs/GPU_INPROCESS_WARMTH_2026-05-29.md`](docs/GPU_INPROCESS_WARMTH_2026-05-29.md);
the clean per-reference re-measure that settled whether any metric has a
first-ref penalty (none do — the prior iwssim "3×" claim was an n=1
transient) is [`benchmarks/setref_clean_all_2026-05-29.tsv`](benchmarks/setref_clean_all_2026-05-29.tsv)
(task #151).

All numbers below are measured medians; no value is interpolated or
extrapolated. Sizes are 512² (0.262 MP), 1024² (1.049 MP), 2048² / "2K"
(4.194 MP), and 4096² / "16 MP" (16.777 MP).

### `process_start` — CUDA context + first-kernel JIT (once per process)

API: `Backend::client()` then the first `compute_*` on each metric.
Source: [`benchmarks/gpu_coldstart_2026-05-29.tsv`](benchmarks/gpu_coldstart_2026-05-29.tsv)
(`client_init_ms` / `first_compute_ms` / `cold_total_ms`, warm-disk,
7-process medians). Host: RTX 5070 (12 GiB), cuda backend, no
`-C target-cpu=native`.

`cold_total = client_init + metric_new + first_compute`. `client_init`
(the CUDA context) is the shared ~181 ms floor; the rest is per-metric and,
at large sizes, allocation-dominated.

| Metric | `client_init` (ms) | first-kernel JIT `first_compute` 512² (ms) | `cold_total` 512² (ms) | `cold_total` 16 MP (ms) |
|---|---|---|---|---|
| `butteraugli-gpu` | 166.8 | 286.7 | 498.7 | 4923.9 |
| `cvvdp-gpu` | 172.5 | 272.4 | 504.5 | 4282.7 |
| `ssim2-gpu` | 187.1 | 129.4 | 396.2 | 6740.5 |
| `dssim-gpu` | 185.0 | 136.5 | 376.1 | 3949.4 |
| `iwssim-gpu` | 182.5 | 265.1 | 491.4 | 2512.5 |
| `zensim-gpu` | 182.2 | 385.0 | 570.3 | 914.2 |

The `client_init` column is flat across metrics and sizes (measured range
166.8–191.2 ms over all 24 warm rows) — this is the once-per-process floor.
First-ever JIT on an empty PTX disk cache inflates `first_compute` further
(butter 1024² 303 → 1288 ms, +~1050 ms one-shot; zensim 1024² 382 → 506 ms,
+~175 ms — rows 26–27); the figures above are the warm-disk case (process
N>1 after the box has run any GPU job).

### `per_ref` — cache a reference once

API (umbrella): `Metric::set_reference_srgb_u8(ref)`. Per-crate this is
`set_reference_srgb_u8` (butter / ssim2 / dssim / iwssim / zensim) or
`warm_reference_srgb` (cvvdp). Source (all six metrics, clean re-measure):
[`benchmarks/setref_clean_all_2026-05-29.tsv`](benchmarks/setref_clean_all_2026-05-29.tsv)
(task #151 — `setref1` = first `set_reference` on a fully warm instance,
`setref2`/`setref3`/`setref4` = distinct *different-pixel* new references
(the reuse path), each followed by `block_on(client.sync())` **inside** the
timed region, **n=8** samples/phase, median + min + max reported, distinct
pixels every rep). Host: RTX 5070, cuda, no `-C target-cpu=native`. Each
`setref1` phase shows a single rep-1 transient (a one-time first-`set_reference`
allocation spike — iwssim 248 ms, butter up to 4166 ms @16 MP) that the
n=8 median/min reject; that transient is exactly what an n=1 sample would
have mistaken for the phase cost.

| Metric | `setref1` 512² (ms) | `setref2` 512² (ms) | `setref1` 16 MP (ms) | `setref2` 16 MP (ms) |
|---|---|---|---|---|
| `cvvdp-gpu` | 1.65 | 1.59 | 16.98 | 17.17 |
| `ssim2-gpu` | 2.48 | 2.88 | 29.34 | 29.02 |
| `dssim-gpu` | 1.43 | 1.34 | 23.15 | 23.16 |
| `iwssim-gpu` | 2.14 | 2.04 | 68.13 | 120.04 |
| `zensim-gpu` | 0.62 | 0.50 | 14.59 | 14.77 |
| `butteraugli-gpu` | 0.77 | 0.74 | 23.33 | 23.65 |

For five of six metrics (cvvdp / ssim2 / dssim / zensim / butter) the
per-reference cost is **flat**: `setref1 ≈ setref2 ≈ setref3 ≈ setref4`
at every size, so budget one `per_ref` per distinct reference regardless
of which reference it is. The earlier profile recorded a huge butter
first-ref cost (34 ms @512², 3990 ms @16 MP); the task #148 clean
re-measure isolated that to **first-instance allocation + JIT** (which
`process_start` already accounts for), not the per-reference cost.

**iwssim is NOT 3× more expensive on its first reference — the opposite.**
A prior table here reported iwssim @16 MP at 196.5 ms `setref1` vs 67.4 ms
`setref2` and asserted a "~3× first-ref warmup". That row came from task
#144's `gpu_inprocess_warmth` Q3, which was a **single sample (n=1) on a
GPU contaminated by a concurrent zensim eval** — the 196.5 ms was a
transient. The clean n=8 #151 re-measure (two independent 16 MP runs) finds
iwssim's `setref1` (68.1 / 73.6 ms) is the **cheapest** phase; `setref2`–
`setref4` land at 120–163 ms. iwssim alone shows a real first-ref *discount*
at 16 MP (subsequent distinct references cost ~1.8× the first), and is flat
at 512² / 1024² / 2K. Budget the larger ~120–160 ms for every reference
after the first.

### `per_dist` — warm per-call score against a cached reference

API: `score_with_warm_ref(dist)`. Source:
[`benchmarks/gpu_coldstart_2026-05-29.tsv`](benchmarks/gpu_coldstart_2026-05-29.tsv)
(`warm_per_call_ms`, intra-process warm repeats, every call ends in a host
readback so the wall is real GPU execution). Cross-confirmed by the
`warm_ref` cuda rows in
[`benchmarks/gpu_metrics_sweep_2026-05-28.tsv`](benchmarks/gpu_metrics_sweep_2026-05-28.tsv).
Host: RTX 5070, cuda, no `-C target-cpu=native`.

| Metric | 512² (ms) | 1024² (ms) | 2K / 4.2 MP (ms) | 16 MP (ms) |
|---|---|---|---|---|
| `butteraugli-gpu` | 1.54 | 3.61 | 12.93 | 50.20 |
| `cvvdp-gpu` | 4.23 | 6.00 | 11.80 | 41.33 |
| `ssim2-gpu` | 3.96 | 6.50 | 14.16 | 47.70 |
| `dssim-gpu` | 4.14 | 5.21 | 12.17 | 46.81 |
| `iwssim-gpu` | 6.53 | 9.47 | 12.78 | 39.44 |
| `zensim-gpu` | 1.66 | 3.27 | 9.67 | 37.80 |

So scoring a batch of N distorted images against one cached reference at
16 MP on cvvdp is `~504.5 + 16.86 + N×41.33 ms` (process_start512 floor is
size-independent; per_ref and per_dist scale with image size). The
`gpu_metrics_sweep` `warm_ref` cuda column gives the same per-call shape
measured by the independent sweep harness (e.g. cvvdp 4 MP 11.80 ms here vs
7.60 ms there, ssim2 16 MP 47.70 vs 43.98 — same order, different warm-up
counts).

### CPU full-mode wall (`score(ref, dist)`)

API: `score(ref, dist)` (umbrella `zenmetrics-api`, full mode — build +
one cold score per call). Source:
[`benchmarks/cpu_wall_all_metrics_2026-05-29.tsv`](benchmarks/cpu_wall_all_metrics_2026-05-29.tsv)
(`mode=full`, `cold_or_warm=cold`, `mean_ms`). Harness: zenbench 0.1.8
interleaved round-robin (paired stats, loop-overhead compensated — not
criterion). Host: AMD Ryzen 9 7950X, release, no `-C target-cpu=native`
(runtime archmage SIMD dispatch only).

| Metric | 512² (ms) | 1024² (ms) | 2K / 4.2 MP (ms) | 16 MP (ms) |
|---|---|---|---|---|
| `cvvdp` | 32.48 | 128.35 | 607.28 | 3812.26 |
| `ssim2` | 16.67 | 70.05 | 297.76 | 2591.03 |
| `dssim` | 30.53 | 123.48 | 546.16 | 4114.34 |
| `butter` | 12.69 | 62.69 | 347.53 | 1690.87 |
| `iwssim` | 59.81 | 261.88 | 1169.06 | 6665.18 |
| `zensim` | 6.92 | 13.92 | 78.86 | 369.66 |

### Per-mode performance at 16 MP

These tables give the measured **wall** and **peak working-set** of
the four common execution modes (full / strip / warm_ref /
warm_ref_strip — see the
[support matrix](#modes--metrics-support-matrix) for the full set,
including cvvdp's StripPair / CappedPyramid) at a representative large
size. Every cell is a committed-TSV value — no number is interpolated
or extrapolated; unsupported `(metric, mode)` cells say `n/a`.

The modes:

- **full** — `score(ref, dist)`: whole-image working set.
- **strip** — strip-walker, one cold `(ref, dist)` per call.
- **warm_ref** — reference cached once (`set_reference`/`warm_reference`),
  then `score_with_warm_ref(dist)` per distorted image (whole-image
  ref state).
- **warm_ref_strip** — reference cached, distorted image walked in
  strips per call.

**GPU (cuda), 16 MP = 4096².** Wall = `wall_median_ms` (per-call
steady-state); mem = `peak_vram_human`. Source:
[`benchmarks/gpu_metrics_sweep_2026-05-28.tsv`](benchmarks/gpu_metrics_sweep_2026-05-28.tsv)
unless noted. Host: RTX 5070 (12 GiB), no `-C target-cpu=native`.

| Metric | full (ms / VRAM) | strip (ms / VRAM) | warm_ref (ms / VRAM) | warm_ref_strip (ms / VRAM) |
|---|---|---|---|---|
| `cvvdp-gpu` | 45.5 / 3.88 GiB | 203.0 / 2.22 GiB † | 25.9 / 3.88 GiB | 108.9 / 3.88 GiB |
| `butteraugli-gpu` | 62.3 / 3.91 GiB | 81.1 / **481 MiB** | 32.8 / 3.91 GiB | 150.9 / 4.19 GiB |
| `ssim2-gpu` | 50.7 / 6.15 GiB | 205.1 / **1.19 GiB** | 44.0 / 6.19 GiB | 119.7 / 4.06 GiB |
| `dssim-gpu` | 50.5 / 3.16 GiB | 277.8 / **897 MiB** | 52.2 / 3.16 GiB | 161.8 / 2.59 GiB |
| `iwssim-gpu` | 45.3 / 2.16 GiB | 385.0 / **545 MiB** | 42.3 / 2.16 GiB | 99.8 / 802 MiB |
| `zensim-gpu` | 38.1 / 1.16 GiB | 61.1 / **289 MiB** ‡ | 30.9 / 1.16 GiB | 488.3 / 1.22 GiB |

† cvvdp's GPU strip mode is `StripPair` (Mode B, one-shot — ref+dist
walk together); the row is the `strip_pair` cuda row. cvvdp has no
`warm_ref_strip`-distinct VRAM win at 16 MP because its `warm_ref`
keeps full-image ref state on device; the strip win for cvvdp shows up
on the **CPU** path below.
‡ zensim-gpu's standalone cold-strip VRAM at 16 MP is **289 MiB** (vs
1.16 GiB Full — a 4.1× reduction) per the corrected re-measure
[`crates/zensim-gpu/benchmarks/zensim_strip_remeasure_2026-05-28.tsv`](crates/zensim-gpu/benchmarks/zensim_strip_remeasure_2026-05-28.tsv);
the wall (61.1 ms) is from the sweep TSV. The `strip` VRAM rows in
`gpu_metrics_sweep` are flagged superseded (pre-fix code built a
full-image ref pyramid). The `warm_ref_strip` column keeps a device
ref cache, so it stays at 1.22 GiB by design.

**CPU, wall at 16 MP = 4096².** Wall = warm per-call for the two
`warm_ref*` modes, cold per-call for `full`/`strip`. Source:
[`benchmarks/cpu_wall_all_metrics_2026-05-29.tsv`](benchmarks/cpu_wall_all_metrics_2026-05-29.tsv)
(zenbench, 7950X). Peak heap (heaptrack) is reported at the largest
committed heaptracked size — **16 MP (4096²) for cvvdp only**; the
other five were heaptracked at **12 MP (4000×3000)** and are marked
`@12MP`, since no 16 MP heaptrack is committed for them and memory does
not extrapolate across sizes. Heap source:
[`benchmarks/cpu_metrics_full_table_2026-05-28.tsv`](benchmarks/cpu_metrics_full_table_2026-05-28.tsv)
(cvvdp rows corrected to the Path A `new_strip` dispatcher,
[`crates/cvvdp/benchmarks/cpu_path_a_recovered_2026-05-29.tsv`](crates/cvvdp/benchmarks/cpu_path_a_recovered_2026-05-29.tsv)).

| Metric | full (ms / heap) | strip (ms / heap) | warm_ref (warm ms / heap) | warm_ref_strip (warm ms / heap) |
|---|---|---|---|---|
| `cvvdp` | 3812 / 3.66 GB | 2605 / **1.58 GB** | 1790 / 3.15 GB | 2168 / **1.55 GB** |
| `ssim2` | 2591 / 2.01 GB@12MP | 3032 / **0.90 GB**@12MP | 1429 / 1.81 GB@12MP | 2457 / 1.21 GB@12MP |
| `dssim` | 4114 / 2.60 GB@12MP | n/a — dssim-core 3.4 has no strip | 2938 / 2.60 GB@12MP | n/a — no strip |
| `butter` | 1691 / 2.37 GB@12MP | 1624 / **0.80 GB**@12MP | 1472 / 2.31 GB@12MP | 1606 / 1.93 GB@12MP |
| `iwssim` | 6665 / 1.77 GB@12MP | 9954 / **0.70 GB**@12MP | 6203 / 1.77 GB@12MP | 4898 / 0.92 GB@12MP |
| `zensim` | 370 / 0.74 GB@12MP | 368 / 0.69 GB@12MP | 345 / 0.79 GB@12MP | 290 / 0.69 GB@12MP |

**The memory win of strip vs full** is the reason strip mode exists.
At 16 MP on the **CPU** path cvvdp drops from **3.66 GB (full)** to
**1.58 GB (strip)** — a 2.3× reduction — with the bit-identical JOD and
a *faster* wall (Path A `new_strip` is −43 % wall at 16 MP). At 12 MP,
butter (2.37 → 0.80 GB), iwssim (1.77 → 0.70 GB), and ssim2 (2.01 →
0.90 GB) show similar 2.5–3× CPU-heap reductions. On the **GPU** the
standalone strip win is largest for zensim-gpu (1.16 GiB → 289 MiB,
4.1×), butteraugli-gpu (3.91 GiB → 481 MiB, 8.3×), and iwssim-gpu
(2.16 GiB → 545 MiB, 4.1×) — at the cost of more launches, so strip
mode is the OOM-avoidance path, not the throughput path (except butter,
which is strip-preferred). dssim's strip win is GPU-only (3.16 GiB →
897 MiB); the dssim **CPU** path has no strip walker
(`dssim-core` 3.4).

### CPU vs GPU one-shot crossover

The size below which a **single** image on a **cold process** is faster on
CPU than GPU. `gpu_cold_total_ms` is the one-shot GPU floor (context-init +
metric_new + first_compute). Source:
[`benchmarks/cpu_gpu_crossover_2026-05-29.tsv`](benchmarks/cpu_gpu_crossover_2026-05-29.tsv)
+ [`docs/CPU_GPU_CROSSOVER_2026-05-29.md`](docs/CPU_GPU_CROSSOVER_2026-05-29.md).
Hosts: CPU 7950X, GPU RTX 5070, cuda, no `-C target-cpu=native`.

| Metric | one-shot: CPU wins up to | one-shot: GPU wins from | batch (warm) winner |
|---|---|---|---|
| `cvvdp` | 16.8 MP (all measured) | — | GPU at all sizes |
| `ssim2` | 16.8 MP (all measured) | — | GPU at all sizes |
| `butter` | 16.8 MP (all measured) | — | GPU at all sizes |
| `zensim` | 16.8 MP (all measured) | — | GPU at all sizes |
| `dssim` | 4.2 MP (2048²) | 16.8 MP (4096²) | GPU at all sizes |
| `iwssim` | 1.0 MP (1024²) | 4.2 MP (2048²) | GPU at all sizes |

Crossovers stated as a bracket between two measured sizes are interpolated,
never a fabricated MP. GPU-cold was measured only at 512² / 1024² / 2048² /
4096²; the 12 MP and 30 MP CPU rows in the source TSV have no GPU-cold
counterpart and are not given a one-shot winner. For **batch / warm** use
there is no crossover in range — GPU wins everywhere.

### Reproduce these numbers

One runner drives all four measurement harnesses:

```sh
# full grid (512² / 1024² / 2K / 16 MP) — matches the committed TSVs
scripts/perf/reproduce_perf_profile.sh

# quick smoke (512² + 16 MP only)
scripts/perf/reproduce_perf_profile.sh --quick
```

It invokes the existing drivers — no new measurement code:

- **`process_start` + `per_dist`** —
  [`scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py`](scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py)
  (builds each crate's `examples/coldstart_one`,
  e.g. [`crates/cvvdp-gpu/examples/coldstart_one.rs`](crates/cvvdp-gpu/examples/coldstart_one.rs)).
- **`per_ref`** —
  [`scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py`](scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py)
  (builds [`crates/zenmetrics-api/examples/inprocess_warmth.rs`](crates/zenmetrics-api/examples/inprocess_warmth.rs)).
- **CPU full wall** — the `cpu-wall` zenbench binary
  (`cargo build --release -p cpu-profile --bin cpu-wall`).

The GPU harnesses require a CUDA-capable host; the CPU wall runs anywhere.
Outputs land in a timestamped scratch dir and are diffed against the
committed TSVs. See the script header for per-harness flags.

## Documentation

- [`docs/CUBECL_PORTING_GUIDE.md`](docs/CUBECL_PORTING_GUIDE.md) — patterns
  for porting more CUDA / scalar metrics to multi-vendor CubeCL.
- [`docs/CUBECL_GOTCHAS.md`](docs/CUBECL_GOTCHAS.md) — 30-entry catalogue
  of cubecl-0.10-era traps with symptoms / fixes / examples.
- [`docs/SSIMULACRA2_PORTING_PLAN.md`](docs/SSIMULACRA2_PORTING_PLAN.md),
  [`docs/SSIM2_GPU_HANDOFF.md`](docs/SSIM2_GPU_HANDOFF.md) — the per-crate
  porting playbooks.
- [`crates/cvvdp-gpu/docs/PORT_STATUS.md`](crates/cvvdp-gpu/docs/PORT_STATUS.md)
  — ColorVideoVDP per-stage port status against pycvvdp v0.5.4
  (host scalar reference path + GPU composition + parity test
  matrix).
- [`scripts/sweep/cvvdp_backfill/README.md`](scripts/sweep/cvvdp_backfill/README.md)
  — operator runbook for the vast.ai pipeline that backfills cvvdp
  JOD scores onto the zensim training parquet store. Produces side-
  by-side `cvvdp_imazen_*` + `cvvdp_pycvvdp_v054` sidecars with a
  parity gate (`assert_parity.py`) that catches both threshold
  violations and silent-failure flatlines.

## License

Dual-licensed: AGPL-3.0-only (see [`LICENSE-AGPL3`](LICENSE-AGPL3)) or
Imazen commercial (see [`COMMERCIAL.md`](COMMERCIAL.md)). `dssim-gpu`'s
commercial track requires Kornel's upstream DSSIM licensing —
see `COMMERCIAL.md`, but this crate is neither maintained nor warrantied by him.
