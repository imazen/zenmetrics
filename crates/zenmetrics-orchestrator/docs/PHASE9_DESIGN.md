# Phase 9 — orchestrator GPU concurrency

Status: shipped in commits 75902739 (9.1 N-lane pool + 9.3 plumbing),
followed by the 9.2 transfer/compute documentation + 9.4 bench
sweep extensions.

This doc captures the architecture decisions for the four sub-phases
in [`crates/zenmetrics-api/docs/PHASE8_PLAN.md`](../../zenmetrics-api/docs/PHASE8_PLAN.md)
section "Phase 9 — orchestrator GPU concurrency".

## 9.1 — N-lane GPU pool

### What

Replace the single GPU worker with `N` parallel "lanes," each a worker
thread holding its own warm `ExecMetric` and consuming from its own
mpsc queue. `PoolConfig::max_gpu_lanes` (default 1, clamped to 1..=8)
controls the count. Default behaviour is bit-identical to the
pre-Phase-9 single-worker pool.

### Why cubecl streams "for free"

`cubecl-cuda`'s `CudaServer` uses `MultiStream` to map every
`cubecl_common::stream_id::StreamId` to a distinct `cudaStream_t`.
`StreamId` is auto-derived from the OS thread id via thread-local
state (see `crates/cubecl-common/src/stream_id.rs` in our fork). So:

- Each lane thread reads its own thread-local `StreamId` the first
  time it touches the `ComputeClient`.
- `MultiStream::resolve(stream_id, ...)` allocates the `cudaStream_t`
  lazily on first use.
- Multiple threads using the same `ComputeClient` (a cheap handle
  cloned across threads) get N distinct CUDA streams without any
  `unsafe set_stream` calls.

We don't need to expose any new cubecl API for Phase 9.1 — the
existing per-thread stream routing is the right abstraction. The
unsafe `ComputeClient::set_stream(StreamId)` exists for advanced
use cases (Burn's task-graph executor) but is not required for our
lane model.

### Dispatch policy: round-robin

Sorted-batch dispatch (Phase 7.6) sends consecutive tasks to lanes
round-robin via `gpu_next: AtomicUsize` mod `active_lanes`. For a
homogeneous batch (same metric/dims), every lane builds its own warm
instance after the first task — amortized across the batch, instance
construction cost stays low.

Alternative considered: signature-keyed dispatch (hash `(metric, w, h)`
to a lane index). Rejected because it serialises same-signature work
on one lane while leaving others idle — the worst case for the Phase 9
throughput contract.

### VRAM accounting

Each lane independently consumes the per-task footprint. The chooser's
budget check at submit time uses the single-task footprint (it doesn't
know how many lanes are active). The pool's live VRAM watcher catches
runtime exhaustion — when free VRAM drops below
`vram_safety_floor_mib`, every lane that hits the gate stalls briefly,
then dispatches anyway after ~3s of stalls (the chooser already
predicted feasibility; the watcher is the backstop for external VRAM
pressure).

For aggressive workloads with N > 1 lanes, callers should size
`vram_safety_floor_mib` to `N × per-task footprint × safety_margin`,
or accept that OOM-recovery via the executor's ladder will downgrade
some tasks to Strip / Cpu when pressure spikes.

## 9.2 — pipelined transfer + compute overlap

### What

With N lanes active, each lane synchronously runs upload → compute →
download on its own CUDA stream. Multiple lanes run their pipelines
concurrently, so at steady-state:

- Lane 0: computing task K
- Lane 1: uploading task K+1
- Lane 2: downloading task K-1
- Lane 3: about to upload task K+2

This is the textbook three-stage pipeline. The lane abstraction makes
it work without any explicit pipeline machinery — each lane is single-
threaded, but multiple lanes overlap at the device level.

### Pinned-host upload

Phase 8d already shipped the pinned-upload patch
(`feat/pinned-upload` on `lilith/cubecl`). Every metric crate uses
`client.create_from_slice_pinned` for the ref/dist HtoD path. This
brings HtoD from ~5-6 GB/s (pageable, driver-staged bounce buffer) to
~12-25 GB/s (DMA from page-locked memory) — and lets the upload
proceed concurrently with another lane's compute.

### Result download

`compute_srgb_u8` reads back one f32 score (4 bytes). The DtoH
transfer is dwarfed by the kernel launch — overlap with the next
lane's upload is essentially trivial.

### No explicit synchronisation choke point

Spot-checked every metric crate's `*-gpu/src/*.rs` for blocking
`client.sync()` calls. The only matches are in `iwssim-gpu`'s
`pipeline.rs` under `if profile { ... }` gates — instrumentation
only, not on the production path. So lanes don't accidentally
serialise on a device-wide sync.

### Verification

The `throughput_n4_at_least_2_5x_n1_at_4mp` test in
`tests/gpu_concurrency.rs` asserts the empirical 2.5×+ speedup on a
50-task cvvdp batch at 4096². This is the operational acceptance
gate for the 9.2 contract — if the test passes, the pipeline is
overlapping. If it fails, run `nsys profile` to identify which stage
isn't pipelining as expected (see CLAUDE.md "Diagnosing Slow GPU Code"
section for the nsys workflow).

## 9.3 — adaptive lane count from nvidia-smi

### Background watcher

`GpuUtilWatcher::spawn(interval_ms)` polls
`nvidia-smi --query-gpu=utilization.gpu` every `interval_ms` (default
5000 = 5s). The watcher stores:

- Latest utilization percent in `latest_pct: AtomicUsize`
- `consecutive_below_target`: count of consecutive samples < 80%
- `consecutive_above_target`: count of consecutive samples >= 95%

The two counters reset on any sample in the "neutral zone" (80-94%).

### Controller: `Orchestrator::adaptive_lane_tick`

The controller examines the watcher's counters. When
`consecutive_below_target >= 3` and `active_lanes < max_lanes`, it
increments `active_gpu_lanes` (clamped to `adaptive_max_gpu_lanes`).
When `consecutive_above_target >= 3` and `active_lanes > 1`, it
decrements. Both transitions log at debug level.

The threshold of 3 samples is the Phase 9 design doc's
"samples_needed = 3" guard against single-sample noise from
ephemeral GPU load (browser shaders, other processes).

### When does the controller run?

Phase 9.3 ships the API surface (`adaptive_lane_tick`) but does NOT
yet wire it into the dispatch hot path. Callers invoke it from their
own polling loop. Wiring an implicit tick on every `submit()` call
is a future Phase 9.3.1 — the rate-limiting machinery to avoid
spamming `nvidia-smi` in tight submit loops is straightforward but
hadn't matured in the initial Phase 9 wave.

### Lane scale-up is free; scale-down isn't

We always spawn `max_gpu_lanes` threads up-front. Scale-up sets
`active_gpu_lanes` and the dispatcher modulo immediately fans out
across the larger pool. Idle lanes block on `mpsc::recv()` with zero
overhead.

Scale-down sets `active_gpu_lanes` lower; the dispatcher stops
sending new tasks to lanes `[new_count, lane_count)`, but any
in-flight task on those lanes runs to completion. Lanes don't have
explicit "drain" or "drop warm instance on demand" hooks yet —
adding them is Phase 9.3.2. For now, scale-down idles the surplus
lanes; their warm instances hold VRAM until the pool tears down at
`Orchestrator::drop`.

## 9.4 — bench-driven worker count

### What

`Phase 2`'s `bench_with_plan` measures wall_p50 at N=1 worker per
cell. Phase 9.4 extends the bench to also measure with N=2, N=4
concurrent workers of the same signature, recording the
throughput-optimal N per `(metric, size)` cell in the
`MetricProfile`.

### Status

The `MetricProfile` struct gains an `optimal_workers_at: BTreeMap<u64, u8>`
field (per-size optimal N). The bench runner gains a sub-pass that
re-runs each cell with N=2 and N=4 to populate this map.

`Orchestrator` exposes a helper to query the cached optimal N:
`recommend_gpu_lanes_for_signature(metric, size_pixels) -> u8`.
Callers (the chooser, or the user) can plug this into
`PoolConfig::max_gpu_lanes` to size the lane pool per workload.

### Storage cost

The optimal-N map is small (one u8 per measured size, ~10 entries
per metric × 6 metrics = ~60 u8s ≈ 60 bytes). Negligible compared
to the existing `ns_per_px_at` + `vram_mib_at` maps. Serialisable
via the existing `u64_keyed_map_*` serde helpers — TOML round-trips
identically to the existing `BackendBench` shape.

### Decision: don't bench N=8

The hard cap `max_gpu_lanes <= 8` is conservative; in practice
N=4 is the throughput ceiling on a single GPU's compute resources.
Benching N=8 doubles the bench runtime for ~0% added signal. If
future GPUs (e.g., RTX 6090 hypothetical) shift the ceiling, raise
the cap + extend the bench grid then.

## Open questions / future work

- **Phase 9.3.1**: rate-limited implicit controller tick on
  `submit()`. Removes the requirement for callers to drive ticks
  manually.
- **Phase 9.3.2**: explicit lane drain + warm-instance drop on
  scale-down. Frees VRAM proactively when the controller drops a
  lane.
- **Phase 9.5**: cross-lane batch coalescing for tiny tasks. When
  N=4 lanes each see a single 64×64 task, the launch overhead per
  lane is ~50us — coalescing 4 tasks into a single kernel launch on
  one lane is faster. Out of scope for the initial wave; revisit if
  small-task workloads become a measurable bottleneck.
- **Phase 9.6**: NCCL or per-device multi-GPU dispatch. Currently
  one device per process; multi-device is a separate design (likely
  via a `DeviceId`-keyed pool-of-pools).
