# zenmetrics orchestrator — design proposal (2026-05-27)

Sits above `zenmetrics-api::Metric` and the per-crate metric implementations. Owns scheduling, backend selection, OOM recovery, and persistent benchmark caching for a list of `(ref, dist, metric)` tasks.

Status: **DESIGN PROPOSAL — not yet built.** Capturing the architecture before implementation so the open scope questions are explicit.

## Motivation

Sweeps and batch scoring callers today face four pain points:

1. **No cross-metric scheduling.** Each caller picks Full / Strip / CPU manually per metric. No shared knowledge of "what fits, what's fastest" at the current image size on this machine.
2. **OOMs are fatal.** `ssim2-gpu` at 4096² OOMs on 8 GB cards; cvvdp Mode B at huge sizes; iwssim at large strips. Today the call fails; no automatic fallback.
3. **Backend choice depends on host.** cvvdp-gpu wins at 1024² on RTX 5070, loses to pycvvdp at 4096² on smaller GPUs. The right choice depends on machine. No persistent learning.
4. **Cached-reference workloads** (one ref × many dist) are 1.5-3× faster than re-uploading the ref each time, but only if the caller knows to use `set_reference` + `compute_with_cached_reference`. Most callers don't.

## Existing infrastructure

The orchestrator builds on what's already in tree:

- `zenmetrics_api::Metric` — unified `MetricKind { Cvvdp, Butter, Ssim2, Dssim, Iwssim, Zensim }` enum + `new_with_memory_mode` constructor + `compute_srgb_u8` / `set_reference_srgb_u8` / `compute_with_cached_reference_srgb_u8`
- Per-crate `live_vram_probe_bytes()` + `vram_cap_bytes()` + `resolve_auto(width, height, cap) -> ResolvedMode { Full | Strip }`
- Per-crate `estimate_gpu_memory_bytes(width, height) -> Option<usize>` and `estimate_gpu_memory_bytes_for_mode(width, height, mode)` — predictive sizing
- Per-crate `recommend_parallel(width, height) -> usize` — how many concurrent instances fit
- `Error::TooBigForFull { needed, cap }` — predictive OOM signal at constructor time
- The `mem_one_size.rs` example pattern across all `-gpu` crates — known-good measurement harness
- `cvvdp-cpu` crate exists as a true CPU implementation (not just cubecl-cpu backend); other crates' upstream CPU references exist as separate crates (`ssimulacra2`, `dssim-core`, `butteraugli`, `zensim` CPU mode)

## Proposed crate: `zenmetrics-orchestrator`

New crate at `crates/zenmetrics-orchestrator/`, depending on `zenmetrics-api` and the per-crate CPU references where they exist. Public surface:

```rust
pub struct Orchestrator {
    config: OrchestratorConfig,
    capability: CapabilityProfile,   // detected once + cached to disk
    runtime_state: RuntimeState,     // mutable: in-flight tasks, live VRAM, learning
}

pub struct OrchestratorConfig {
    pub cache_dir: PathBuf,           // default ~/.cache/zenmetrics/
    pub bench_on_first_run: bool,     // default true
    pub bench_validity: Duration,     // default 7 days, also invalidate on driver-version change
    pub max_parallel_gpu: Option<usize>,  // default = recommend_parallel at median size
    pub max_parallel_cpu: usize,      // default = num_cpus / 2
    pub oom_retry_strategy: OomRetry, // default Strip→CPU→fail
    pub allow_cpu_fallback: bool,     // default true
}

pub enum OomRetry {
    GpuFullToStrip,                   // fall back to Strip on OOM, keep on GPU
    StripToCpu,                       // if Strip also OOMs, fall back to CPU
    Both,                             // default — Full→Strip→CPU
    NoFallback,                       // strict: bubble OOM up
}

pub struct Task {
    pub ref_data: TaskData,           // bytes OR a TaskRefHandle that the orchestrator can pre-upload + reuse
    pub dist_data: TaskData,
    pub width: u32,
    pub height: u32,
    pub metric: MetricKind,
    pub params: Option<MetricParams>, // None → MetricParams::default_for(kind)
    pub task_id: u64,                 // caller's correlation ID
}

pub enum TaskData {
    Srgb8(Vec<u8>),
    Path(PathBuf),
    PreUploaded(TaskRefHandle),       // for many-dist-one-ref optimization
}

pub struct TaskResult {
    pub task_id: u64,
    pub outcome: Result<Score, OrchestratorError>,
    pub backend_used: BackendChoice,
    pub wall_us: u64,
    pub vram_peak_mb: Option<usize>,
}

pub enum BackendChoice { GpuFull, GpuStrip, Cpu, GpuStripPair /* cvvdp Mode B */ }

impl Orchestrator {
    pub fn new(config: OrchestratorConfig) -> Result<Self>;

    /// Pre-flight + benchmark if needed. Idempotent.
    pub fn warm(&mut self) -> Result<()>;

    /// Streaming API — caller pushes tasks, results come back as they finish.
    /// Internally batches by (metric, backend, size-bucket) for efficiency.
    pub fn submit(&mut self, task: Task) -> Result<TaskHandle>;
    pub fn poll(&mut self, handle: TaskHandle) -> Option<TaskResult>;
    pub fn run_all<I: IntoIterator<Item = Task>>(&mut self, tasks: I)
        -> impl Iterator<Item = TaskResult>;

    /// Inspection.
    pub fn capability(&self) -> &CapabilityProfile;
    pub fn statistics(&self) -> &OrchestratorStats;
}
```

## Capability profile (persistent cache)

Written once per machine to `~/.cache/zenmetrics/capability_<machine_hash>.toml`. Re-validated on every `new()` against driver/CPU fingerprint.

```toml
machine_hash = "sha256(gpu_model + driver_version + cpu_brand + cpu_features_short)"
detected_at = "2026-05-27T05:30:00Z"
last_validated = "2026-05-27T05:30:00Z"

[gpu]
present = true
model = "NVIDIA GeForce RTX 5070"
total_vram_mib = 12288
driver_version = "596.21"
cuda_runtime = "13.2.1"
compute_capability = "8.9"

[cpu]
brand = "AMD Ryzen 9 7950X"
logical_cores = 32
features = ["avx2", "avx512f", "sse4.2"]
ram_mib = 131072

# Per-metric per-backend perf+VRAM profile. Sparse: only measured points.
[metrics.cvvdp.gpu_full]
ns_per_px_at = { "1024" = 5.34, "2048" = 3.10, "4096" = 2.71 }
vram_mib_at = { "1024" = 385, "2048" = 1089, "4096" = 3970 }
last_measured = "2026-05-27T05:30:00Z"

[metrics.cvvdp.gpu_strip_pair]
ns_per_px_at = { "4096" = 2.62 }
vram_mib_at = { "4096" = 2272 }

[metrics.cvvdp.cpu]
ns_per_px_at = { "1024" = 71.22, "2048" = 146.34, "4096" = 98.68 }
ram_mib_at = { "1024" = 75, "2048" = 240, "4096" = 800 }

[metrics.ssim2.gpu_full]
# ... etc
```

**Validity:**
- Stale after `bench_validity` (default 7 days).
- Invalidated if `driver_version` changes (driver bump can change perf substantially).
- Invalidated if `gpu.model` changes (different hardware).
- Self-healing: orchestrator runs a 30-60s quick re-bench in background on stale cache.

## Quick benchmark (run on first use / stale cache)

Ships embedded `synth_pair_with_offset_dist(w, h)` synthetic — same patterns as the existing bench drivers. For each `(metric, backend) × (1024², 2048², 4096²)`:

1. Construct the metric.
2. Run 2 warmup calls (PTX compile + cache fill).
3. Run 5 timed calls; record p50.
4. Sample VRAM peak via cubecl pool stats.
5. Drop the metric → measure VRAM returned to pool.

Total budget: ~30s on a typical machine for the 6 metrics × 3 sizes × 3 backends (some combinations are skipped — e.g., cvvdp Mode B only exists for cvvdp, Strip on butter is much different shape than on cvvdp).

Re-bench triggers:
- First run on a machine (no cache file)
- Cache > 7 days old
- Detected driver/hardware change

## Scheduling algorithm

```
INPUT: Vec<Task>, capability_profile, live_vram_free

GROUP tasks by metric kind        # cvvdp / butter / ssim2 / etc
FOR each metric group:
    GROUP by exact (width, height)  # so we can amortize set_reference
    FOR each (metric, w, h) sub-group:
        # Predict resource usage per backend
        choices = [(backend, ns_per_px, vram_mib) for backend in {GpuFull, GpuStrip, Cpu}]
            FILTER by vram_mib <= live_vram_free * 0.85  # 15% safety margin
            FILTER by feasibility (e.g., StripPair only exists for cvvdp)
            INTERPOLATE ns_per_px from capability cache (or fall back to nearest measured size)

        # Pick best
        best = MIN(choices, key=lambda c: c.ns_per_px)

        # Detect ref-reuse opportunity within sub-group
        if len(tasks) >= REF_REUSE_THRESHOLD (e.g., 4):
            use set_reference + compute_with_cached_reference
        else:
            use compute_srgb_u8

        EMIT batch (tasks, backend=best, use_cached_ref=...)

DISPATCH batches:
    GPU batches → GpuWorkerPool (1 worker per device, max_parallel_gpu cap)
    CPU batches → CpuWorkerPool (rayon thread pool, max_parallel_cpu cap)
    Interleave to keep both pools utilized

CONCURRENT live VRAM watcher:
    Sample every 250ms
    If free VRAM drops below SAFETY_FLOOR (e.g., 200 MB):
        Pause new GPU batch dispatches until free recovers
        Existing batches finish naturally
```

## OOM recovery

Three layers, in order:

1. **Predictive** (best): capability cache + `estimate_gpu_memory_bytes_for_mode` + live VRAM probe. If predicted+safety > free, skip the backend, pick next.

2. **Constructor OOM** (caught at `Metric::new_with_memory_mode`): `Error::TooBigForFull { needed, cap }` is structured and recoverable. Orchestrator marks "this (metric, mode, size) is too big on this machine" in capability cache + retries with next-smaller mode.

3. **Runtime OOM** (caught at `compute_srgb_u8`): cubecl bubbles a runtime OOM error. Orchestrator catches, drops the metric instance, records the failure as a hard "this combination doesn't actually fit" data point in capability cache, retries with next-smaller mode.

Recovery sequence (per metric + size):
```
GpuFull → GpuStrip → (Cvvdp only: GpuStripPair) → Cpu → Error::FullyExhausted
```

Each downgrade updates the capability cache so the orchestrator learns: "on this machine, ssim2 at 4096² Full will OOM — never try it again."

## Architecture decisions (locked 2026-05-27 per user input)

1. **API shape: both streaming + batch.** `Orchestrator::submit(Task) -> TaskHandle` + `Orchestrator::poll(handle)` for incremental result delivery, AND `Orchestrator::run_all(impl IntoIterator<Item=Task>) -> impl Iterator<Item=TaskResult>` for the common batch case. Worker pool runs concurrently behind both surfaces.

2. **CPU backends: all references in initial release.** cvvdp-cpu (already a crate) + ssimulacra2 (CPU) + dssim-core (CPU) + butteraugli (CPU) + zensim (CPU mode). Each is wired as a fallback backend in `BackendChoice::Cpu`. ~1-2 days per crate for the API adapter.

3. **Cached-ref: auto-detect by default with explicit override.** Orchestrator hashes ref bytes (`xxhash3_64`, ~3 GB/s on modern CPUs — fast enough for 4096² at <10ms) and promotes consecutive tasks within the same `(metric, w, h, ref_hash)` to `set_reference` + `compute_with_cached_reference`. Callers who want zero overhead can pre-upload via `TaskData::PreUploaded(TaskRefHandle)`. Both paths coexist.

4. **Crate location: new `crates/zenmetrics-orchestrator/`** sibling to `zenmetrics-api`. Opt-in dependency for callers who want orchestration. `zen-metrics-cli` adopts it for the `sweep` subcommand.

5. **Multi-GPU: single GPU only** (use cubecl's default device). Multi-GPU work-stealing is out of scope for initial release. Capability cache stores the GPU's identity so a multi-GPU machine using device 1 instead of 0 cleanly invalidates / re-benches.

6. **Remote workers: out of scope.** Local-only execution. vast.ai sweep infra in `scripts/sweep/` is a fundamentally different scheduling model.

7. **Result delivery on partial failure:** per-task `Result<Score, OrchestratorError>` in `TaskResult`. The iterator from `run_all` yields one `TaskResult` per task regardless of success/failure. `OrchestratorError::FullyExhausted` indicates all backends were tried and failed; the caller can inspect `backend_used` to see what was attempted.

8. **Persistent cache: machine-local.** `~/.cache/zenmetrics/capability_<hash>.toml`. Sweep-fleet sharing (publish to R2) is out of scope here — would be a separate `zenmetrics-orchestrator-fleet` crate later if needed.

## Phased implementation

Each phase is a separate task with its own JOD-/parity-/perf-test gate, each pushable independently:

**Phase 1 — Skeleton + capability detection** (~1 day):
- New crate `zenmetrics-orchestrator`
- `CapabilityProfile` struct, machine-fingerprint detection (GPU model, VRAM, CPU brand, features)
- TOML serialization to `~/.cache/zenmetrics/capability_<hash>.toml`
- Stale detection (mtime + driver version)
- No scheduling yet

**Phase 2 — Quick-benchmark harness** (~1-2 days):
- Internal bench runner that exercises each metric × backend × size
- Populates `CapabilityProfile.metrics.*` table
- Total runtime budget: ~30-60s
- CHANGELOG entry per measured machine

**Phase 3 — Backend chooser** (~2 days):
- Decision function: `(metric, w, h, vram_free) -> BackendChoice`
- Interpolation from sparse cache points
- Safety margin handling
- Unit tests covering OOM-avoidance scenarios

**Phase 4 — Single-task executor with OOM recovery** (~2 days):
- `run_single(Task)` end-to-end
- Predictive avoidance + constructor-OOM catch + runtime-OOM catch
- Fallback ladder (Full → Strip → CPU)
- Capability cache learning from failures

**Phase 5 — Batch + streaming API** (~3 days):
- Worker pool (GPU + CPU)
- Group-by-(metric, size) batching
- `set_reference` reuse detection
- Streaming `submit/poll` + batch `run_all`

**Phase 6 — CPU backend wiring** (~1-2 days per CPU crate):
- cvvdp-cpu first (already exists as a crate, just needs API adaptation)
- Other CPU references as separate follow-ups

**Phase 7 — Integration + docs** (~1 day):
- Wire into `zen-metrics-cli sweep`
- README + example programs
- Migration guide for existing callers

Total estimate: **~2-3 weeks for Phase 1-5**. Phase 6-7 incremental on top.

## Risks / non-obvious decisions

- **The capability cache will lie eventually.** Driver updates, thermal throttling, background GPU load can all make cached perf numbers wrong. Mitigation: rolling-window measurement (record actual perf during real workloads, blend with cached values).
- **The fallback ladder may surprise callers.** A caller asking for GPU may silently get CPU on OOM. Mitigation: `BackendChoice` returned in `TaskResult` so caller can audit + `OomRetry::NoFallback` opt-out.
- **Per-task overhead matters at small sizes.** A 64×64 task takes 100µs of GPU compute but 5ms of orchestrator decision-making + worker dispatch is silly. Mitigation: a "tiny task" fast path that skips scheduling and just runs CPU directly.
- **VRAM-cap safety margin is heuristic.** 85% is reasonable for desktop GPUs with their own display compositor; bare-metal datacenter GPUs can go higher. Make it configurable.
