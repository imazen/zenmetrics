# Migrating from `zenmetrics-api` to `zenmetrics-orchestrator`

The orchestrator wraps the umbrella's opaque `Metric` API with
scheduling, OOM recovery, and persistent benchmarking. The migration is
straightforward — most call sites change three lines.

## Side-by-side: single-call score

### Before (`zenmetrics-api`)

```rust,no_run
use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
# let reference_bytes: Vec<u8> = vec![0; 1024 * 1024 * 3];
# let distorted_bytes: Vec<u8> = reference_bytes.clone();
let params = MetricParams::default_for(MetricKind::Ssim2);
let metric = Metric::new_with_memory_mode(
    MetricKind::Ssim2,
    Backend::Cuda,
    1024,
    1024,
    params,
    MemoryMode::Full,
)?;
let score = metric.compute_srgb_u8(&reference_bytes, &distorted_bytes)?;
println!("score = {}", score.value);
# Ok(()) }
# fn main() {}
```

Problems:

- Caller picked `MemoryMode::Full` blindly. If 1024² ssim2 doesn't fit
  in available VRAM, this OOMs and the caller has to catch + retry with
  `MemoryMode::Strip` manually.
- No reuse — every call rebuilds the metric.
- No persistence — the next process starts cold.

### After (`zenmetrics-orchestrator`)

```rust,no_run
use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
# let reference_bytes: Vec<u8> = vec![0; 1024 * 1024 * 3];
# let distorted_bytes: Vec<u8> = reference_bytes.clone();
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?; // idempotent — only benches on first/stale machine.

let task = Task {
    task_id: 0,
    ref_data: TaskData::Srgb8(reference_bytes),
    dist_data: TaskData::Srgb8(distorted_bytes),
    width: 1024,
    height: 1024,
    metric: MetricKind::Ssim2,
    params: None,
};
let result = orch.run_single(task);
let score = result.outcome?;
println!("score = {} (backend: {:?})", score.value, result.backend_used);
# Ok(()) }
# fn main() {}
```

What you get:

- The chooser inspects the machine's bench cache + live VRAM and picks
  the fastest feasible backend (typically GPU-full at 1024²; GPU-strip
  at 4096² on a 12 GB card; CPU when GPU is busy).
- Constructor and runtime OOMs auto-fall-back through `GpuFull →
  GpuStrip → (cvvdp StripPair) → Cpu`.
- The machine bench is persisted to `~/.cache/zenmetrics/` so the next
  process starts warm.
- `result.backend_used` audits which backend actually produced the
  score — useful when the ladder kicked in.

## Side-by-side: batch sweep (many `(ref, dist)` pairs)

### Before

```rust,no_run
use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
# let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..100).map(|_| (vec![0; 1024 * 1024 * 3], vec![0; 1024 * 1024 * 3])).collect();
// Hand-roll a per-loop metric reuse — the legacy CLI's sweep handler.
let params = MetricParams::default_for(MetricKind::Cvvdp);
let mut metric = Metric::new_with_memory_mode(
    MetricKind::Cvvdp,
    Backend::Cuda,
    1024,
    1024,
    params.clone(),
    MemoryMode::Full,
)?;

for (i, (ref_bytes, dist_bytes)) in pairs.iter().enumerate() {
    let score = metric.compute_srgb_u8(ref_bytes, dist_bytes)?;
    println!("{i}: {}", score.value);
}
# Ok(()) }
# fn main() {}
```

Problems:

- One OOM kills the entire loop.
- No cached-ref dispatch even when every pair shares the same reference.
- Manual cubecl shared-instance management; gets entangled with worker
  cleanup logic.

### After

```rust,no_run
use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
# let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..100).map(|_| (vec![0; 1024 * 1024 * 3], vec![0; 1024 * 1024 * 3])).collect();
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;

let tasks: Vec<Task> = pairs
    .iter()
    .enumerate()
    .map(|(i, (ref_bytes, dist_bytes))| Task {
        task_id: i as u64,
        ref_data: TaskData::Srgb8(ref_bytes.clone()),
        dist_data: TaskData::Srgb8(dist_bytes.clone()),
        width: 1024,
        height: 1024,
        metric: MetricKind::Cvvdp,
        params: None,
    })
    .collect();

for result in orch.run_all(tasks) {
    match result.outcome {
        Ok(score) => println!("{}: {}", result.task_id, score.value),
        Err(e) => eprintln!("{}: failed: {e}", result.task_id),
    }
}
# Ok(()) }
# fn main() {}
```

What changed:

- A single OOM only kills that one task. The rest finish, with the
  failed task carrying `Err(FullyExhausted)`.
- Cached-ref auto-detect kicks in when pairs share refs — typically a
  1.5–3× speedup on sweep-shaped workloads.
- The pool's GPU worker keeps the cvvdp metric warm across tasks of the
  same `(metric, w, h)` signature.

## Side-by-side: cached-reference pattern

When you have one reference and N variants, the orchestrator's
auto-detect handles it transparently. But you can pre-upload for an
extra 4–8 ms / variant savings:

### After (explicit pre-upload)

```rust,no_run
use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig, Task, TaskData};

# fn run() -> Result<(), Box<dyn std::error::Error>> {
# let reference_bytes: Vec<u8> = vec![0; 4096 * 4096 * 3];
# let variants: Vec<Vec<u8>> = (0..1000).map(|_| reference_bytes.clone()).collect();
let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
orch.warm()?;

let handle = orch.upload_reference(
    &reference_bytes,
    4096,
    4096,
    MetricKind::Cvvdp,
)?;

let mut outstanding = 0;
for (i, variant) in variants.iter().enumerate() {
    orch.submit(Task {
        task_id: i as u64,
        ref_data: TaskData::PreUploaded(handle.clone()),
        dist_data: TaskData::Srgb8(variant.clone()),
        width: 4096,
        height: 4096,
        metric: MetricKind::Cvvdp,
        params: None,
    })?;
    outstanding += 1;
    // Drain non-blocking — keeps the queue from growing without bound.
    while let Some(r) = orch.poll_any() {
        process_result(r);
        outstanding -= 1;
    }
}
// Drain tail.
while outstanding > 0 {
    if let Some(r) = orch.poll_any_blocking() {
        process_result(r);
        outstanding -= 1;
    } else {
        break;
    }
}

orch.drop_reference(handle);
# Ok(()) }
# fn main() {}
# fn process_result(_: zenmetrics_orchestrator::TaskResult) {}
```

## Behaviour differences callers should know

### Backend selection is automatic

The orchestrator picks GPU vs CPU based on the chooser's bench
measurements + live VRAM. Callers used to choosing `MemoryMode` manually
should stop doing so — the chooser typically picks better than a hand-
rolled heuristic because it knows actual per-metric perf and OOM cells
on this specific machine.

Audit the choice with `TaskResult::backend_used`.

### OOM doesn't bubble by default

Where `Metric::compute_srgb_u8` would have surfaced
`Error::TooBigForFull` directly to the caller, the orchestrator catches
it, records the cell in the cache, and walks the ladder. The caller
only sees `Err(FullyExhausted { attempts })` if every backend in the
ladder failed.

For strict-mode callers (parity tests, "fail if GPU isn't available"
checks), the design supports an `OomRetry::NoFallback` knob — see the
orchestrator README's "OOM handling" section for the current status.

### Results arrive in completion order

`run_all` and `poll_any` yield results as workers finish — NOT in the
order tasks were submitted. Correlate via `Task::task_id`. This avoids
unbounded memory growth on long sweeps where one slow task would
otherwise hold every later result in a buffer.

### Worker pool is lazy

`Orchestrator::new` does NOT spawn any threads. The pool initialises on
the first `submit` / `run_all` / `upload_reference`. Callers that only
inspect `Orchestrator::capability()` pay nothing for the pool.

### Capability cache is persistent

The first call to `Orchestrator::new` writes
`~/.cache/zenmetrics/capability_<hash>.toml`. Subsequent processes pick
it up automatically. If you delete the cache the next `warm()` rebuilds
it; if you delete the orchestrator binary mid-sweep the cache survives
unchanged.

## Common pitfalls

### "I get `Err(Chooser(UnknownMetric))` on every task"

You haven't called `warm()` yet. The chooser refuses to pick a backend
when the cache has no measurements for the requested metric. Add
`orch.warm()?` after construction; it's idempotent and only re-benches
when needed.

### "My CPU fallback says `CpuBackendUnavailable`"

The build doesn't include the `cpu-<metric>` feature for that metric.
Rebuild with `--features cpu-cvvdp` (or `cpu-all`) to enable the CPU
reference. Iwssim has no upstream CPU reference and never works —
expect `CpuMetricUnavailable` for iwssim CPU fallback regardless of
feature flags.

### "Two tasks with the same `task_id` come back with the same id"

`task_id` is caller-chosen and never validated for uniqueness. The
orchestrator passes it through unchanged. Use unique IDs if you need to
correlate results to source tasks.

### "Why is the cache file path so long?"

The cache path is `<cache_dir>/capability_<first-16-hex-chars-of-sha256>.toml`.
The 16-char prefix is enough to keep dotfile collisions vanishingly
unlikely across realistic machine populations.

## Where to go next

- [`README.md`](../README.md) — orchestrator overview + every public
  API documented with examples.
- [`docs/CPU_BACKENDS.md`](CPU_BACKENDS.md) — per-metric CPU reference
  adapter mapping + RAM characteristics.
- [`ORCHESTRATOR_DESIGN.md`](../../zenmetrics-api/docs/ORCHESTRATOR_DESIGN.md)
  — original design proposal with the architectural rationale.
- The `examples/` directory in this crate — runnable single-task and
  batch examples driven against a real CUDA device.
