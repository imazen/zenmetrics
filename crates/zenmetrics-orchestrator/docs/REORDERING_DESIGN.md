# Orchestrator task reordering — design (Phase 7.6)

User directive 2026-05-27: **"orchestrator should reorder tasks"**.

Status: **RESOLVED — Phase 7.6 landed 2026-05-27.** Layers 2/3/4
shipped as four commits on top of master. Real-GPU measurement on
the RTX 5070 / 7950X workstation: sorted dispatch reduces
warm-instance constructions on a 60-task mixed chunk from 40 (FIFO)
to 6 — a 6.7× reduction. Cached-ref hit rate on a 50-task
same-ref chunk: 1 miss, 49 hits as designed.

See `CHANGELOG.md`'s "Phase 7.6" entry for the public-API additions
and `MIGRATION_FROM_API.md`'s "Tasks may be reordered before
dispatch" section for caller migration notes.

## Goal

Maximize warm-metric reuse + cached-ref hit rate while keeping peak
VRAM bounded by ONE metric's footprint. Reorder tasks internally so
consecutive dispatches share the warm `Metric` instance and the
device-resident reference.

## Layers (recap)

| Layer | Implements | Phase |
|---|---|---|
| 1. Single warm instance per backend at a time | Pool holds ONE `Metric`; swap on signature change | 7.5 (running) |
| **2. Sort `run_all` by (metric, dims, ref_hash)** | Internal task reorder before dispatch | **7.6** |
| **3. Stream reorder window for submit/poll** | Buffer submissions ~50ms / 16 tasks; sort each window | **7.6** |
| **4. VRAM budget assertion at construction** | Already in Phase 4; add log when budget gates a swap | **7.6** |

## Layer 2 — `run_all` internal sort

```rust
impl Orchestrator {
    pub fn run_all<I: IntoIterator<Item = Task>>(
        &mut self,
        tasks: I,
    ) -> RunAllIter {
        let mut tasks: Vec<Task> = tasks.into_iter().collect();

        // Hash ref bytes for ordering. Hashing is one-shot per task
        // (xxhash3_64 ~3 GB/s; at 4096² u8x3 = 48 MB → ~16ms per task).
        // Tasks with PreUploaded(handle) skip hashing — the handle's
        // metric + dims + a stable ref-id from upload_reference() are
        // used as the sort key.
        for t in &mut tasks {
            t.ref_hash = compute_ref_hash(&t.ref_data);
        }

        // Sort by (metric, w, h, backend_hint, ref_hash). The backend
        // is chosen per-task by the chooser at dispatch time; grouping
        // by the chooser's predicted backend keeps warm-instance
        // signature changes minimal even before chooser runs.
        tasks.sort_by_key(|t| (t.metric, t.width, t.height, t.ref_hash));

        self.dispatch_sorted(tasks)
    }
}
```

**Yield order**: completion order. Tasks complete in roughly the sorted
order (single GPU worker processes them FIFO post-sort), but a CPU
fallback task that interleaves may finish out-of-order. Callers
correlate via `task_id`. Document clearly.

**Pre-uploaded handles** are sorted by `(handle.metric, handle.dims,
handle.ref_id)` — the ref_id is assigned by `upload_reference()` and
acts as the hash equivalent.

## Layer 3 — Streaming reorder window

```rust
pub struct OrchestratorConfig {
    // ... existing fields ...

    /// Streaming `submit()` reorder window. Submissions collected
    /// until either the duration elapses OR the count is reached,
    /// then the window is sorted by (metric, dims, ref_hash) and
    /// dispatched as one batch. Default: 50ms / 16 tasks.
    ///
    /// Set to `(Duration::ZERO, 1)` to disable reordering (strict
    /// submit-order dispatch). Set to `(Duration::MAX, usize::MAX)`
    /// to use `run_all`-style full-batch sort (caller must call
    /// `flush_pending()` explicitly to dispatch).
    pub stream_reorder_window: (Duration, usize),
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            // ... existing ...
            stream_reorder_window: (Duration::from_millis(50), 16),
        }
    }
}

impl Orchestrator {
    pub fn submit(&mut self, task: Task) -> Result<TaskHandle> {
        // Hash ref, append to pending queue
        let task_with_hash = annotate_with_hash(task);
        self.pending.push(task_with_hash);

        // Trigger dispatch if window full
        if self.pending.len() >= self.config.stream_reorder_window.1 {
            self.flush_pending();
        } else if self.pending_started_at.elapsed() >= self.config.stream_reorder_window.0 {
            self.flush_pending();
        }

        // (background timer also drains the window when its duration elapses
        //  even if no new submit() arrives)
    }

    pub fn flush_pending(&mut self) {
        let mut window: Vec<_> = self.pending.drain(..).collect();
        window.sort_by_key(|t| (t.metric, t.width, t.height, t.ref_hash));
        for t in window {
            self.worker_queue.send(t);
        }
    }
}
```

The 50ms window is invisible for sweep workloads (`run_all` ignores
the window since it sorts the whole input). For interactive callers
(e.g., a GUI that scores one image at a time), 50ms is below
perceptual-latency thresholds; if even that's too much, set
`(Duration::ZERO, 1)`.

## Layer 4 — VRAM budget at signature swap

Already implemented in Phase 4's `run_single`. The addition for Phase
7.6 is **observability**: log when the budget check triggers a
backend downgrade or when an instance swap would exceed the budget,
so operators can tune `safety_margin` from real workload patterns.

```rust
// In pool worker, before constructing new instance:
let predicted_mib = estimate_gpu_memory_bytes_for_mode(...);
let free_mib = vram_watcher.snapshot();
if predicted_mib > free_mib * (1.0 - config.vram_safety_margin) {
    log::warn!(
        "vram budget at swap: predict={}, free={}, margin={}, downgrading {:?} -> {:?}",
        predicted_mib, free_mib, margin, requested_backend, fallback_backend
    );
    requested_backend = fallback_backend;
}
```

## Test plan

### Test 1 — warm instance churn on mixed-metric chunk

Synthetic 60-task chunk:
- 3 metrics (cvvdp, ssim2, dssim) × 2 sizes (1024², 2048²) × 3 refs (10 tasks per ref) = 60 tasks
- Submitted in random shuffled order

Assert:
- Internal warm-instance constructions = 6 (one per (metric, size) tuple, after sort)
- Pre-sort dispatch would have been ≥30 constructions

### Test 2 — cached-ref hit rate on multi-dist-one-ref

Synthetic 50-task chunk:
- 1 metric × 1 size × 1 ref × 50 distortions
- Submitted in submit-order

Assert: cached-ref hits = 49 (one `set_reference` on first task; 49
`compute_with_cached_reference` calls)

### Test 3 — peak VRAM on mixed-metric chunk equals max-single-metric

Run test 1's chunk on a real GPU. Sample peak nvidia-smi delta during
execution.

Assert: peak == max(per-metric VRAM at the largest size used) ± 200 MiB
slack for transient buffers. Critically NOT 2× or 3× of that.

### Test 4 — streaming submit-poll with 50ms window

Submit 100 tasks at 200ms intervals (slower than window). Assert each
task's `submit-to-result` wall ≤ 50ms + per-task-compute-time. (Window
doesn't add latency beyond the dispatch lag.)

Submit 100 tasks at 5ms intervals (faster than window). Assert window
fills at N=16 → flushes → all 16 share warm instance.

### Test 5 — explicit flush_pending

Caller submits 50 tasks then calls `flush_pending()`. Assert all
dispatched immediately regardless of window timer.

## Out of scope for 7.6

- Multi-warm pool with LRU eviction (~10× implementation cost vs the
  sort-batching approach for marginal gain on workloads that fit
  the (metric, dims) grouping pattern; defer)
- Per-warm-instance VRAM accounting (the single-warm-at-a-time
  invariant makes this unnecessary)
- Cross-GPU work-stealing (out of scope per original design — single
  GPU only)

## Migration impact

Phase 7.6 only changes internal scheduling. Public API unchanged
except for the new `OrchestratorConfig.stream_reorder_window` field
(non-breaking — has a sensible default).

Callers who depend on submit-order completion order need to opt out:
`stream_reorder_window = (Duration::ZERO, 1)`. Document in
`MIGRATION_FROM_API.md` under "Behavioural differences".

`run_all` callers already get arbitrary completion order; the sort
inside `run_all` doesn't change that contract, just makes the order
more predictable in practice (sorted-input means mostly-sorted
completion).
