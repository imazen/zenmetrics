# zenmetrics-orchestrator

Capability-aware scheduler for `zenmetrics-api`. Sits one layer above the
opaque `Metric` constructor and owns four things that previously every
caller hand-rolled:

1. **Backend selection.** Picks GPU-full / GPU-strip / cvvdp StripPair /
   CPU per task based on a persisted, machine-specific benchmark cache.
2. **OOM recovery.** When the chosen backend OOMs (predicted or
   bubble-up at runtime), the orchestrator walks a fallback ladder
   instead of failing the task.
3. **Cached-reference dispatch.** Auto-detects "many distorted, one
   reference" workloads via xxhash3 and promotes them to the
   `set_reference` + `compute_with_cached_reference` API for the 1.5–3×
   speedup. Callers who want zero overhead can pre-upload references
   explicitly.
4. **Concurrency.** A small worker pool (one GPU worker + N CPU workers)
   handles streaming `submit` / `poll` and batch `run_all` APIs without
   the caller spinning up threads.

> **Status:** the orchestrator is the **recommended entry point** for
> any caller that scores more than one pair at a time. Single-shot
> scoring can still use `zenmetrics-api` directly; everything else —
> sweeps, picker training, RD curves, anything that batches — should
> go through the orchestrator.

## Quickstart

```rust,no_run
use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    Orchestrator, OrchestratorConfig, Task, TaskData,
};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
// Optional but recommended: run the quick-bench so the chooser has
// real perf+VRAM numbers for this machine. `warm()` is idempotent —
// it only runs when the cache is missing or stale.
orch.warm()?;

let task = Task {
    task_id: 0,
    ref_data: TaskData::Srgb8(reference_bytes),
    dist_data: TaskData::Srgb8(distorted_bytes),
    width: 1024,
    height: 1024,
    metric: MetricKind::Ssim2,
    params: None,
    ref_hash: 0,
};
let result = orch.run_single(task);

match result.outcome {
    Ok(score) => println!("score = {}", score.value),
    Err(e) => eprintln!("failed: {e}"),
}
# Ok(()) }
# fn main() {}
# let reference_bytes: Vec<u8> = vec![0; 1024 * 1024 * 3];
# let distorted_bytes: Vec<u8> = reference_bytes.clone();
```

## Choosing between the orchestrator and `zenmetrics-api`

| Caller shape | Use |
| --- | --- |
| One `(ref, dist)` per process, no fallback needed | `zenmetrics-api` |
| Batch of tasks (sweep, RD curve, picker training) | **orchestrator** |
| Streaming workload (submit-as-you-go) | **orchestrator** |
| OOM-tolerant scoring (e.g. unknown image sizes) | **orchestrator** |
| Need to share warm references across many distorts | **orchestrator** |

The orchestrator's overhead on a single task is small (one chooser call
+ one cache touch + a pool spawn if it's the first submit) — typically
< 5 ms above the underlying metric. For workloads with many tasks the
amortised cost is much lower because the pool keeps warm `Metric`
instances across same-signature tasks.

## Batch — `run_all`

```rust,no_run
# use zenmetrics_api::MetricKind;
# use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};
# fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;

let tasks: Vec<Task> = (0..100)
    .map(|i| Task {
        task_id: i as u64,
        ref_data: TaskData::Srgb8(reference_bytes.clone()),
        dist_data: TaskData::Srgb8(distorted_variants[i].clone()),
        width: 1024,
        height: 1024,
        metric: MetricKind::Cvvdp,
        params: None,
        ref_hash: 0,
    })
    .collect();

// Results arrive in COMPLETION order (not submission order). Correlate
// via `task_id`.
for result in orch.run_all(tasks) {
    match result.outcome {
        Ok(score) => println!("task {}: {}", result.task_id, score.value),
        Err(e) => eprintln!("task {} failed: {e}", result.task_id),
    }
}
# Ok(()) }
# let reference_bytes: Vec<u8> = vec![0; 1024 * 1024 * 3];
# let distorted_variants: Vec<Vec<u8>> = (0..100).map(|_| reference_bytes.clone()).collect();
```

## Streaming — `submit` + `poll_any`

When the caller is producing tasks as it goes (e.g. fetching distorted
images from R2), the streaming API lets the orchestrator overlap I/O
with compute.

```rust,no_run
# use zenmetrics_api::MetricKind;
# use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};
# fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;

let mut outstanding = 0;
for next_task in pending_tasks {
    orch.submit(next_task)?;
    outstanding += 1;

    // Drain any completed results without blocking.
    while let Some(result) = orch.poll_any() {
        process_result(result);
        outstanding -= 1;
    }
}

// Drain the tail — block until every submitted task is done.
while outstanding > 0 {
    if let Some(result) = orch.poll_any_blocking() {
        process_result(result);
        outstanding -= 1;
    } else {
        break;
    }
}
# Ok(()) }
# struct StubTask;
# let pending_tasks: Vec<Task> = Vec::new();
# fn process_result(_: zenmetrics_orchestrator::TaskResult) {}
```

## OOM handling — the fallback ladder

Three OOM signals trigger the ladder:

1. **Predictive avoidance.** The chooser consults
   `capability.metrics.<metric>.vram_mib_at` and rejects backends whose
   predicted VRAM exceeds `live_vram_free × 0.85`.
2. **Constructor OOM.** `Metric::new_with_memory_mode` returning
   `Error::TooBigForFull { needed, cap }`. The orchestrator marks the
   `(backend, size_pixels)` cell as failed, persists the cache, and
   re-asks the chooser.
3. **Runtime OOM.** `Metric::compute_srgb_u8` bubbling a cubecl runtime
   OOM. Same recovery as constructor OOM — drop the metric, record the
   cell, retry next backend.

Ladder order (per metric):

```
GpuFull → GpuStrip → (Cvvdp only: GpuStripPair) → Cpu → FullyExhausted
```

Each downgrade updates the persistent capability cache so the same
machine never tries the failing combination twice.

### Strict mode — `OomRetry::NoFallback`

When a caller absolutely needs a specific backend (e.g. a parity test
that wants to fail loudly if GPU isn't available), construct the
orchestrator with `oom_retry_strategy: OomRetry::NoFallback`. The first
OOM bubbles up as `OrchestratorError::FullyExhausted` with one entry in
`backends_attempted` — no ladder, no surprise CPU fallback.

> Note: as of Phase 6, the `OomRetry` knob lives on the design surface
> but the chooser exposes the same selectivity via per-metric
> `KnownOomCell` entries in the cache. A future minor release will plumb
> `OomRetry` end-to-end.

## Cached-reference dispatch

For workloads with many distorted variants of one reference (the
canonical sweep shape), the orchestrator promotes the dispatch to the
metric's `set_reference` + `compute_with_cached_reference` API. This
saves the per-task reference upload (typically 4–8 ms at 4096²) and
re-uses any preprocessed GPU-resident state inside the metric (cvvdp's
XYB-transformed reference cube, ssim2's blurred reference pyramid,
etc.).

### Auto-detect (default)

The orchestrator hashes ref bytes via `xxhash3_64` (~5–15 GB/s on a
modern CPU, ~4–8 ms at 4096²) and consults a sliding window of the last
32 `(metric, w, h, hash)` tuples. On a hit, the next task dispatches
through the cached-ref API.

```rust,no_run
# use zenmetrics_api::MetricKind;
# use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};
# fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;

// 100 distorted variants of the SAME reference → orchestrator
// auto-detects after the first task and switches to cached-ref
// dispatch for the rest.
let tasks: Vec<Task> = (0..100)
    .map(|i| Task {
        task_id: i as u64,
        ref_data: TaskData::Srgb8(reference.clone()),
        dist_data: TaskData::Srgb8(variants[i].clone()),
        width: 1024,
        height: 1024,
        metric: MetricKind::Ssim2,
        params: None,
        ref_hash: 0,
    })
    .collect();

let results: Vec<_> = orch.run_all(tasks).collect();

// Audit the cached-ref auto-detect: 99 / 100 tasks should have hit.
let stats = orch.cached_ref_stats();
println!("cached-ref hits: {} / misses: {}", stats.hit_count, stats.miss_count);
# Ok(()) }
# let reference: Vec<u8> = vec![0; 1024 * 1024 * 3];
# let variants: Vec<Vec<u8>> = (0..100).map(|_| reference.clone()).collect();
```

### Explicit — `upload_reference` + `TaskData::PreUploaded`

When the caller wants zero hashing overhead (e.g. processing 4096²
references where the 8 ms hash adds up across thousands of tasks),
upload once and pass a `TaskRefHandle`:

```rust,no_run
# use zenmetrics_api::MetricKind;
# use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};
# fn run() -> Result<(), Box<dyn std::error::Error>> {
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;

let ref_handle = orch.upload_reference(
    &reference_bytes,
    4096,
    4096,
    MetricKind::Cvvdp,
)?;

for (i, variant) in variants.iter().enumerate() {
    let task = Task {
        task_id: i as u64,
        ref_data: TaskData::PreUploaded(ref_handle.clone()),
        dist_data: TaskData::Srgb8(variant.clone()),
        width: 4096,
        height: 4096,
        metric: MetricKind::Cvvdp,
        params: None,
        ref_hash: 0,
    };
    orch.submit(task)?;
}
// drain results …

orch.drop_reference(ref_handle);
# Ok(()) }
# let reference_bytes: Vec<u8> = vec![0; 4096 * 4096 * 3];
# let variants: Vec<Vec<u8>> = vec![reference_bytes.clone(); 4];
```

## CPU backends

Each per-metric CPU reference adapter is opt-in via a Cargo feature so
downstream consumers don't pay for crates they don't use:

| Feature flag | Pulls | Notes |
| --- | --- | --- |
| `cpu-cvvdp` | `cvvdp-cpu` | In-tree port; full cached-ref via `warm_reference`. |
| `cpu-ssim2` | `ssimulacra2` | Crates.io reference; cached-ref re-uses cached ref bytes (no separate warm-ref API upstream). |
| `cpu-dssim` | `dssim-core` | True cached-ref via `DssimImage`. |
| `cpu-butter` | `butteraugli` | Cached-ref recomputes from bytes (upstream has no separate warm path). |
| `cpu-zensim` | `zensim` | Cached-ref recomputes. |
| `cpu-all` | All five | Convenience bundle for production workers. |

Iwssim has no clean upstream CPU reference — selecting it as the CPU
fallback surfaces `OrchestratorError::CpuMetricUnavailable` and the
ladder advances. See [`docs/CPU_BACKENDS.md`](docs/CPU_BACKENDS.md) for
the per-metric mapping, RAM characteristics, and cached-ref semantics.

## Capability cache lifecycle

The orchestrator writes a per-machine TOML profile to
`$XDG_CACHE_HOME/zenmetrics/capability_<short_hash>.toml` (default
`~/.cache/zenmetrics/`). The `<short_hash>` is the first 16 chars of
`sha256(gpu_model || driver_version || cpu_brand || cpu_features)`, so
the same machine always lands on the same file but different machines
co-exist without stomping each other.

### When the cache re-benches

1. **First run** on a machine with no cache file.
2. **Time-based**: cache > `OrchestratorConfig::cache_validity` old
   (default 7 days). Wall-clock time only — process restarts don't
   invalidate.
3. **Driver / hardware change**: a fresh `detect_gpu()` returns a
   different model or driver version than the cached snapshot.

`warm()` runs the bench only when one of these triggers fires; it's
safe to call on every process startup.

### Manual invalidation

To force a re-bench (e.g. after upgrading a metric crate), call
`Orchestrator::bench()` directly — it unconditionally overwrites the
metric profile table.

### Fleet sharing

A single capability profile can be shared across a fleet of identical
boxes by uploading the TOML to R2 / S3 and pointing
`OrchestratorConfig::cache_dir` at a writable local path that's
pre-populated from the shared key at boot. The orchestrator's
hardware-change check still fires on startup, so a mismatched machine
won't trust a foreign profile — it'll re-bench locally instead.

`scripts/sweep/onstart_orchestrator.sh` in the zenmetrics repo
implements this exact pattern for vast.ai workers.

## `OrchestratorConfig`

```rust,ignore
pub struct OrchestratorConfig {
    pub cache_dir: PathBuf,         // default ~/.cache/zenmetrics/
    pub cache_validity: Duration,   // default 7 days
}
```

Pool-level knobs (`max_parallel_cpu`, `vram_safety_floor_mib`,
`vram_sample_interval_ms`, `vram_stall_ms`) live on the separate
`PoolConfig` struct — pass via `Orchestrator::set_pool_config(cfg)`
before the first `submit` / `run_all` / `upload_reference` call.
Defaults are sensible for desktop GPUs with a display compositor; bare-
metal datacenter GPUs can usually push `vram_safety_floor_mib` lower
than 200.

## Feature flag matrix

```
default        — capability detection + TOML cache only (no scheduling).
bench          — quick-bench harness + chooser. Pulls zenmetrics-api +
                 cvvdp-gpu + xxhash + rayon.
cuda           — single-task executor + worker pool (implies bench).
cpu-cvvdp,
cpu-ssim2,
cpu-dssim,
cpu-butter,
cpu-zensim     — per-metric CPU reference adapters (each implies bench).
cpu-all        — every CPU adapter at once.
```

Production sweep workers typically build with `--features cuda,cpu-all`.
Light callers that just want the capability detection (e.g. a CI sanity
check that the machine has the expected GPU model) build with default
features.

## Dependency on `lilith/cubecl` fork

This crate (and the rest of the zenmetrics GPU stack) pins cubecl to
the `lilith/cubecl` fork via `[patch.crates-io]` in the workspace
root `Cargo.toml`. The fork carries a single patch on top of stock
cubecl 0.10.0: a pinned-host-buffer fast path for `create_from_slice`
uploads.

**Why.** CUDA's `cuMemcpyHtoDAsync` from pageable host memory caps at
~5-6 GB/s because the driver internally stages through a hidden
pinned bounce buffer. Allocating the host buffer with
`cuMemAllocHost_v2` (= "pinned" / "page-locked") lets the driver DMA
directly at 12-25 GB/s on PCIe 4.0. cvvdp-gpu's 12 MP warm-ref bench
goes from 95 ms to 22 ms — a ~4.3× speedup — purely from this
patch. See `docs/CUBECL_GOTCHAS.md` §G6.5 in the workspace root for
the full diagnosis.

The patch:

- Adds `ComputeClient::create_from_slice_pinned(&[u8]) -> Handle` for
  hot per-call uploads that want to skip the
  `caller → pageable Vec<u8> → pinned Bytes` extra memcpy.
- Adds `ComputeClient::reserve_staging(&[usize]) -> Vec<Bytes>` for
  pre-reserving pinned slabs the caller fills in place.
- Adds `ComputeClient::create_tensors_from_slices_pinned` for batch
  variants.
- Transparently routes the existing `create_from_slice` /
  `create_tensor_from_slice` / `create_tensors_from_slices` paths
  through `ComputeServer::staging`, so any caller of the default API
  gets the pinned-upload speedup without source changes.

**Upstream PR.** The patch has been drafted as a PR against
`tracel-ai/cubecl` (referenced as draft PR **#1334**). See
[`../zenmetrics-api/docs/PINNED_UPLOAD_UPSTREAM_PR.md`](../zenmetrics-api/docs/PINNED_UPLOAD_UPSTREAM_PR.md)
for the full diff, bench numbers, and submission steps.

**Workspace pin** (from the root `Cargo.toml`):

```toml
[patch.crates-io]
cubecl         = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-runtime = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-core    = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-common  = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-ir      = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-cuda    = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-cpu     = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-wgpu    = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-hip     = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
cubecl-cpp     = { git = "https://github.com/lilith/cubecl.git", rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e" }
```

**Downstream opt-in.** If you're consuming `zenmetrics-orchestrator`
from a separate workspace, add the same `[patch.crates-io]` block to
your workspace `Cargo.toml`. All ten `cubecl-*` crates must be patched
to the same rev — partial patches mix patched and unpatched code
paths in the dep graph and silently lose the speedup. Backends without
a pinned-memory concept (cubecl-wgpu Metal/Vulkan, cubecl-cpu) ignore
`staging` and behave exactly as stock cubecl, so the patch is safe to
apply unconditionally even when CUDA isn't in use.

**Sunset plan.** Once the upstream PR merges and a cubecl release
ships the change, this crate will drop the fork pin entirely and
return to crates.io versions. The `create_from_slice_pinned` and
`reserve_staging` API symbols are stable across that transition —
they exist on the fork today and will exist on upstream post-merge —
so downstream code paths in cvvdp-gpu / iwssim-gpu / etc. don't need
to change.

## Migration from `zenmetrics-api`

See [`docs/MIGRATION_FROM_API.md`](docs/MIGRATION_FROM_API.md) for
side-by-side code samples.

## License

Dual-licensed under AGPL-3.0 or LicenseRef-Imazen-Commercial; see the
workspace root `LICENSE-AGPL3` and `COMMERCIAL.md` for details.
