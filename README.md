# zenmetrics [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenmetrics/actions/workflows/ci.yml) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

zenmetrics is the Imazen workspace for perceptual image-quality metrics:
multi-vendor **GPU** implementations of the metrics we run in production, the
**CPU** reference crates they are checked against, a unified `zenmetrics` CLI, and
**zenfleet** — the content-addressed job system that drives codec sweeps and
metric backfills across a heterogeneous fleet. Pure Rust, `#![forbid(unsafe_code)]`.

The GPU metrics are built on CubeCL via the
[`zenforks-cubecl`](https://crates.io/crates/zenforks-cubecl) publication of
[tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) (0.10.x — carries
pinned-upload, PTX-cache-widening, and Metal `Atomic<f32>` capability patches for
our use case). A single `#[cube]`-annotated kernel source dispatches across CUDA
(NVIDIA), WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a build-time
CPU fallback.

> Every crate in this workspace is `publish = false` — nothing ships to
> crates.io. Build the CLI and libraries from source (see Quick start), which is
> why the only badges above are CI and license.

## Quick start

The `zenmetrics` CLI scores one or many `(reference, distorted)` pairs on CPU or
GPU. Build it from the workspace:

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
cargo build --release -p zenmetrics-cli       # binary: target/release/zenmetrics

# or install the binary directly
cargo install --git https://github.com/imazen/zenmetrics zenmetrics-cli
```

Score a single pair — CPU SSIMULACRA2, no GPU required:

```sh
zenmetrics score --metric ssim2 --reference ref.png --distorted out.jpg
```

Score one reference against several encoded variants across several metrics —
each unique image decoded once — as a TSV:

```sh
zenmetrics compare \
  --reference ref.png \
  --variant out-q60.jpg --variant out-q80.jpg --variant out.avif \
  --metric ssim2 --metric butteraugli --metric dssim \
  --output tsv
```

In the default build, `--metric` accepts the CPU metrics `ssim2`, `dssim`,
`butteraugli`, and `zensim`; `cvvdp` and `iwssim` need their CPU features
(`--features orchestrator,orchestrator-cpu-cvvdp` / `orchestrator-cpu-iwssim`),
and the GPU variants (`ssim2-gpu`, `dssim-gpu`, `butteraugli-gpu`, `iwssim-gpu`,
`zensim-gpu`, `cvvdp` via `gpu-cvvdp`) need `--features gpu-<metric>`. Run
`zenmetrics list-metrics` to print exactly what your build enabled and which
require a GPU. Other subcommands: `batch` (a TSV of pairs), `sweep` (drive a
codec across a quality × knob grid and score every variant into a Pareto TSV),
`score-pairs` / `assemble` (parquet sidecars + training corpora), `fleet-plan`
(size a sweep's fleet), and `jobexec` (the job-system executor — see below).

For scoring **many** pairs in one process (sweeps, picker training, RD curves),
call [`zenmetrics-orchestrator`](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-orchestrator/README.md)
rather than the CLI per pair. For scoring across a **fleet of machines**, use the
zenfleet job system. Both are covered below.

## Metric crates

Six GPU metric crates plus the two in-tree CPU reference crates the orchestrator's
CPU ladder dispatches to:

| Crate | Metric | Range / shape | Parity reference |
|---|---|---|---|
| [`butteraugli-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/butteraugli-gpu) | Butteraugli | distance, max-norm (default) + libjxl 3-norm | [`butteraugli`](https://crates.io/crates/butteraugli) 0.9.4 |
| [`ssim2-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/ssim2-gpu) | SSIMULACRA2 | 0–100, higher better | [`ssimulacra2`](https://crates.io/crates/ssimulacra2) 0.5 |
| [`dssim-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/dssim-gpu) | DSSIM | distance, 0 = identical | [`dssim-core`](https://crates.io/crates/dssim-core) 3.5 |
| [`iwssim-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim-gpu) | IW-SSIM (Wang & Li 2011) | `[0, 1]`, 1.0 = identical | [`iwssim`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim) (in-tree CPU port) |
| [`zensim-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/zensim-gpu) | zensim feature extractor | 228-feature vector + scalar score 0–100 | [`zensim`](https://github.com/imazen/zensim) 0.3.0 |
| [`cvvdp-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp-gpu) | ColorVideoVDP (still-image, GPU) | JOD ~3–10, higher better | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) 0.5.4 |
| [`iwssim`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim) | IW-SSIM (CPU reference + SIMD) | `[0, 1]`, 1.0 = identical | self (pure-Rust port) |
| [`cvvdp`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp) | ColorVideoVDP (still-image, CPU) | JOD ~3–10 + per-pixel diffmap | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) 0.5.4 |

The metric each GPU crate computes is bit-comparable to its cited reference. The
CPU side of each metric comes from an external reference crate
([`fast-ssim2`](https://crates.io/crates/fast-ssim2) 0.8.1,
[`dssim-core`](https://crates.io/crates/dssim-core) 3.5,
[`butteraugli`](https://crates.io/crates/butteraugli) 0.9.4,
[`zensim`](https://github.com/imazen/zensim) 0.3.0) or an in-tree crate
([`cvvdp`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp),
[`iwssim`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim)).

**Feature gating (important):** the four external-crate CPU backends (ssim2 /
dssim / butteraugli / zensim) ship in the default `cpu-metrics` bundle, but the
two in-tree CPU ports — **`cvvdp` and `iwssim` are NOT in `cpu-metrics`.** Enable
them explicitly (`--features orchestrator,orchestrator-cpu-cvvdp`, resp.
`orchestrator-cpu-iwssim`). A build with neither `gpu-cvvdp` nor `cpu-cvvdp`
reports cvvdp as unavailable — that is a build-config message, not a "cvvdp is
GPU-only" limitation.

### Supporting crates

| Crate | Role |
|---|---|
| [`zenmetrics-api`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-api) | Umbrella: one `MetricKind` enum + one `Metric` type dispatching to every per-crate opaque scorer |
| [`zenmetrics-gpu-core`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-gpu-core) | Shared backend / score / sRGB / stream plumbing for the `*-gpu` crates (CubeCL) |
| [`zenmetrics-orchestrator`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-orchestrator) | Capability-aware backend chooser + persistent benchmark cache + OOM fallback ladder + warm worker pool |
| [`zenmetrics-cli`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-cli) | the `zenmetrics` CLI (`score` / `batch` / `compare` / `sweep` / `score-pairs` / `jobexec` / `assemble` / `fleet-plan`) |
| [`zenstats`](https://github.com/imazen/zenmetrics/tree/master/crates/zenstats) | Paper-correct IQA statistical panel (SROCC / PLCC / KROCC / OR / PWRC + bootstrap-CI A-vs-B) |
| [`zenmetrics-corpus`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-corpus) / [`zenhdr-corpus`](https://github.com/imazen/zenmetrics/tree/master/crates/zenhdr-corpus) | Shared SDR / HDR test-image corpora (test infra) |
| [`cvvdp-conformance`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp-conformance) | pycvvdp conformance fixtures + parity harness for the cvvdp crates |

## In-process scoring entry point: `zenmetrics-orchestrator`

For any caller that scores **more than one** `(ref, dist)` pair — sweeps, picker
training, RD curves, batch comparison — reach for
[`zenmetrics-orchestrator`](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-orchestrator/README.md)
instead of constructing metrics by hand. It adds three things every in-tree
caller used to hand-roll:

1. **Backend selection** — a persistent per-machine benchmark cache picks the
   fastest backend that fits available VRAM, and remembers which `(metric, size)`
   combinations OOM on this machine so it never retries them.
2. **OOM-safe fallback ladder** — `GpuFull → GpuStrip → (cvvdp: GpuStripPair) →
   Cpu`, each downgrade recorded in the cache.
3. **Cached-reference auto-detect** — hashes each task's reference bytes and
   promotes consecutive same-reference tasks to the warm-reference fast path for
   the 1.5–3× speedup sweeps benefit from.

The `zenmetrics` CLI routes scoring through the orchestrator by default. The
legacy direct-dispatch path stays available via `--use-legacy-scheduler` (or
`ZENMETRICS_USE_LEGACY_SCHEDULER=1`) for bit-identical regeneration of archived
parquet sidecars; butteraugli always flows through legacy because its `Auto`
resolves to strip-mode (single-resolution) and diverges from the legacy
always-multires output. The orchestrator path was validated bit-identical to
legacy across all 54 cells (6 metrics × 3 sizes × 3 qs) on RTX 5070 + 7950X. See
the [orchestrator README](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-orchestrator/README.md)
for the streaming + batch APIs, OOM handling, and cached-reference semantics.

## Distributed sweeps: the zenfleet job system

zenfleet is the canonical orchestrator for encode / score / sweep work that spans
many machines — the in-tree system that replaced hand-rolled chunk launchers. It
is content-addressed end to end:

- **Jobs are content-addressed.** A `JobId` is `sha256(kind + sorted inputs)`, so
  declaring the same work twice is a structural no-op.
- **The ledger is the truth, not the queue.** Every finished job writes a row to a
  columnar Parquet ledger in R2 (latest-wins on `(job_id, ts)`); coverage, the
  dashboard, and the reconciler all read the ledger, so a run converges after any
  partial pass or crash.
- **The queue is an R2 conditional-write lease** — a worker claims a job by
  `PutObject` with `If-None-Match: *` on `claims/<job_id>`, so exactly one worker
  wins each job and there is no double execution.
- **Workers are interchangeable and pull-based** (outbound HTTPS to R2 only), so a
  NAT'd basement box is a first-class tier alongside vast.ai / Hetzner / cloud.

Job kinds (`zenfleet_core::JobKind`): `Encode` · `Metric` · `Feature` ·
`Diffmap` · `Resample` · `Bake`, each carrying a resource class for capability
routing and a GC regenerability policy (expensive encodes are kept; cheap
re-scores are LRU-cached).

| Crate | Role |
|---|---|
| [`zenfleet-core`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-core) | Content-addressed job taxonomy, identity, status, blob addressing, and the idle / waste detector |
| [`zenfleet-ledger`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-ledger) | Columnar Parquet ledger + blob index with latest-wins compaction |
| [`zenfleet-ctl`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-ctl) | Declare desired jobs + query coverage / gap from the ledger |
| [`zenfleet-worker`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-worker) | Claims the reconciler's gap, runs a handler via the `ZEN_EXEC` executor, content-addresses outputs, emits ledger rows |
| [`zenfleet-dash`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-dash) | Railway-deployable dashboard + control API (reads the ledger; never runs workers) |
| [`zenfleet-sweep`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-sweep) | Cloud-agnostic sweep worker binary (selects a backend via `--backend`) |
| [`zenfleet-cloud`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-cloud) / [`-local`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-local) / [`-vastai`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-vastai) / [`-hetzner`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-hetzner) | Provider backends behind one common trait |

The thing that does the actual encode/score is `zenmetrics jobexec` — the
`ZEN_EXEC` reference executor: it reads one `DesiredJob` as JSON on stdin and
writes the output bytes (encode) or a JSON score row (metric) to stdout. Drive a
run with the one consolidated command —
[`scripts/jobsys/fleet`](https://github.com/imazen/zenmetrics/blob/master/scripts/jobsys/fleet)
`launch | status | watch | top | kill` (there is no other monitor; `fleet watch`
shows boxes, $/hr burn, per-box GPU/CPU util, idle / failed-to-start boxes, and
ledger progress in one place). Worker images bake every dependency at build time
([`scripts/jobsys/build_executor_image.sh`](https://github.com/imazen/zenmetrics/blob/master/scripts/jobsys/build_executor_image.sh)
copies a precompiled binary in; nothing is apt/pip-installed at boot). Full
runbook: [`docs/RUNNING_JOBS.md`](https://github.com/imazen/zenmetrics/blob/master/docs/RUNNING_JOBS.md);
sweep-plan contract: [`docs/PLAN_SWEEPS.md`](https://github.com/imazen/zenmetrics/blob/master/docs/PLAN_SWEEPS.md).

<!-- crates.io:skip-start -->

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
| `dssim` (dssim-core 3.5) | ✓ | ✗ | ✓ | ✗ |

**dssim CPU has no strip walker** — `dssim-core` 3.5 exposes no strip
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
  one `per_ref` per distinct reference. Measured to 16.777 MP only (40 MP is
  unmeasured — don't extrapolate). **cvvdp / ssim2 / dssim / zensim** are
  roughly flat across references (median `setref1 ≈ setref2 ≈ …`). Two
  exceptions: **butteraugli** eagerly allocates its full reference working set
  on the *first* `set_reference` of a freshly-warmed instance (~250 ms/MP —
  ≈4 s at 16 MP — then flat for later refs), so budget a one-time first-ref
  cost per instance on top of the flat steady state; **iwssim**'s reuse-path
  references cost ~1.8× its first reference at 16 MP (~120–160 ms vs ~68 ms,
  and run-to-run noisy) — its per-reference cost *rises*, so size the larger
  value at 16 MP.
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
the clean per-reference re-measure (task #151,
[`benchmarks/setref_clean_all_2026-05-29.tsv`](benchmarks/setref_clean_all_2026-05-29.tsv))
settled the per-metric first-ref behaviour: cvvdp/ssim2/dssim/zensim are flat
across references; the prior iwssim "3×" was an n=1 transient (its real reuse
cost is ~1.8× its first ref, not 3×); and butteraugli carries a genuine
first-`set_reference` allocation cost (~250 ms/MP on the first call) that the
warmed-instance median in that TSV smooths over — see the raw first-call
samples and
[`docs/GPU_INPROCESS_WARMTH_2026-05-29.md`](docs/GPU_INPROCESS_WARMTH_2026-05-29.md).

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
| `dssim` | 4114 / 2.60 GB@12MP | n/a — dssim-core 3.5 has no strip | 2938 / 2.60 GB@12MP | n/a — no strip |
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
(`dssim-core` 3.5).

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

## GPU CI / Metal

GPU metric kernels are validated on **CUDA** (locally, RTX 5070) and
**Vulkan** (`cubecl-wgpu`, the `gpu-citest` job on Linux) — these are the
backends the kernels ship against, and both run the full parity suites.

The **macOS-Metal CI job is currently disabled** (`if: false` in
`ci.yml`, 2026-06-13). On the 8 GB-unified `macos-latest` runner the
large-image (12 MP / 4000×3000) parity tests wedge the GPU — the
whole-image butteraugli score silently OOMs to 0 and the ssim2 device
hangs until the job timeout — even with serialized test execution. This
is a runner-capacity / large-image-GPU-memory limit, **not** a kernel
correctness problem (CUDA + Vulkan parity is green). Metal coverage will
return once the large-image path is capped or fixed for the 8 GB runner;
work is tracked in the zenmetrics#24 Metal stream. Until then, **Metal is
not a supported/verified backend in CI** — treat Metal results as
unvalidated.

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


<!-- crates.io:skip-end -->

## License

Dual-licensed: AGPL-3.0-only (see [`LICENSE-AGPL3`](https://github.com/imazen/zenmetrics/blob/master/LICENSE-AGPL3))
or Imazen commercial (see [`COMMERCIAL.md`](https://github.com/imazen/zenmetrics/blob/master/COMMERCIAL.md)).
`dssim-gpu`'s commercial track requires Kornel's upstream DSSIM licensing — see
[`COMMERCIAL.md`](https://github.com/imazen/zenmetrics/blob/master/COMMERCIAL.md); this crate is
neither maintained nor warrantied by him.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · **zenmetrics** · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
