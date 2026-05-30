# VRAM lifecycle design — ironclad drop-frees-VRAM + opt-out (tasks #152, #153)

**Status: DESIGN PROPOSAL — no public API changed. Awaits user approval
of a concrete shape before implementation.** Builds directly on the
source-level audit in [`VRAM_DROP_AUDIT_2026-05-29.md`](VRAM_DROP_AUDIT_2026-05-29.md)
(task #150) and two **measured** feasibility spikes:
`crates/cvvdp-gpu/examples/vram_isolation_spike.rs` (CUDA, task #152,
gated `cuda`) and `crates/cvvdp-gpu/examples/wgpu_isolation_spike.rs`
(wgpu/Vulkan as the Metal proxy, task #153 / issue #17, gated `wgpu`) —
both throwaway. The wgpu spike answers issue #17's open item: per-stream
pool isolation holds on wgpu too (§2b).

The question the user posed, verbatim: *design an IRONCLAD way for
dropping a metric to free its VRAM, with an opt-out — or amend the API
to make clear what's reserved.* And specifically: **"do we need a new
context type?"**

**Answer up front: YES, an isolated-stream context type is the only
ironclad option, and the spike proves the mechanism works.** A bare
`impl Drop` on the scorer cannot be ironclad on the shared default
stream (partial-page occupancy + thread-of-drop hazards, both measured
below). The recommended shape (Option B) is a `GpuContext` that owns an
explicit cubecl stream + its private memory pool; dropping the context
reclaims **exactly** that context's VRAM, independent of every other
context, from any thread.

---

## 1. cubecl memory/stream model (file:line — the published `zenforks-cubecl-*` 0.10.1 this workspace depends on)

The structures below were verified identical in the published crates.io
fork (`~/.cache/cargo-read/zenforks-cubecl-*-0.10.1/`, which is what
`Cargo.toml` pins) **and** the working tree at
`/home/lilith/work/zenforks-cubecl-work/`. Line numbers cite the working
tree; the published copies match.

### 1.1 A client can be bound to an explicit stream

- `ComputeClient<R>` carries `stream_id: Option<StreamId>` —
  `cubecl-runtime/src/client.rs:36`.
- `ComputeClient::stream_id()` returns the explicit bound stream if set,
  else `StreamId::current()` (the calling thread's thread-local) —
  `client.rs:85-90`.
- `pub unsafe fn set_stream(&mut self, stream_id: StreamId)` binds it —
  `client.rs:97-99`. Doc: *"highly unsafe and should probably only be
  used by the CubeCL/Burn projects."*
- A **safe** alternative exists: `StreamId::executes(self, f)` swaps the
  thread-local stream id for the duration of a closure and restores it on
  return/unwind via a Drop guard — `cubecl-common/src/stream_id.rs:26-46`.
  An unbound client inside that closure resolves to the swapped id.
- `StreamId { pub value: u64 }` is freely constructible —
  `stream_id.rs:7-10`. `StreamId::current()` lazily allocates a per-thread
  value from a global `AtomicU64` counter — `stream_id.rs:49-56, 89-102`.

### 1.2 Each stream owns its OWN memory pool (per-stream isolation, structurally)

- `Stream { sys: CUstream, memory_management_gpu: MemoryManagement<GpuStorage>,
  memory_management_cpu, drop_queue, errors }` —
  `cubecl-cuda/src/compute/stream.rs:22-29`.
- `create_stream()` calls `cudarc::driver::result::stream::create(NonBlocking)`
  (a single `cuStreamCreate`) and builds a **fresh** `MemoryManagement`
  per stream — `stream.rs:49-83`.
- The CUDA `GpuStorage` holds its own `stream` handle and its own
  `deallocations: Vec<StorageId>` queue —
  `cubecl-cuda/src/compute/storage/gpu.rs:20,55`.

### 1.3 Streams live in a fixed pool keyed by `value % max_streams` — collision-prone, never evicted

- `CudaServer.streams: MultiStream<CudaStreamBackend>` — `server.rs:54`.
  **One server per device**, cached in a global registry
  (`DeviceHandle`), obtained via `CudaRuntime::client(&Default::default())`.
- `MultiStream` stores streams in a `StreamPool` — `stream/event.rs:51`.
- `StreamPool { streams: Vec<Option<F::Stream>>, max_streams }` —
  `stream/base.rs:14-21`. **A bounded ring, NOT a `HashMap<StreamId, …>`.**
- Index = `stream_index(id) = id.value as usize % max_streams` —
  `stream/event.rs:143-145`, `stream/base.rs:101-103`.
- `get_mut_index` lazily creates a stream on first access (None →
  `factory.create()`) then caches it; **never evicts** —
  `stream/base.rs:59-81`. So no use-after-free from eviction. But:
  - **Two distinct `StreamId`s with the same `value % max_streams` alias
    the same physical stream and the same pool.** "Isolated stream per
    context" is only isolated if the context controls `value` to avoid
    collisions.
  - **`max_streams` is a hard cap** on simultaneously-isolated pools.
    Default `default_max_streams() = 128` — `config/streaming.rs:23-25`.
    Configurable via the cubecl runtime config. 128 + 1 special (GC)
    stream.
  - There is **no public API to remove/free an individual stream**. A
    `Stream` + its `MemoryManagement` persist for the life of the server
    (process-global). What a stream *can* do is free its pool's pages
    back to the driver (§1.4) — the `Stream` struct shell stays, but its
    VRAM goes to zero.

### 1.4 `memory_cleanup()` frees only FULLY-FREE pages, on the current stream only

- `ComputeClient::memory_cleanup()` → `server.memory_cleanup(stream_id)`
  — `client.rs:972-976`.
- CUDA server resolves the **current** stream and calls
  `command.memory_cleanup()` → `streams.current().memory_management_gpu.cleanup(true)`
  (`explicit=true`) — `command.rs:78-80`, `server.rs:245-257`.
- `MemoryManagement::cleanup(explicit)` iterates every pool —
  `memory_manage.rs:380-392`.
- `SlicedPool::cleanup(explicit=true)`: for each page, `coalesce()`, and
  **only if `summary.amount_free == summary.amount_total`** call
  `storage.dealloc(id)`; partial pages are **retained and relocated**
  (`update_page`) — `sliced_pool.rs:104-128`.
- `storage.dealloc` only **queues** the id —
  `gpu.rs:204-206`. The `cuMemFreeAsync` fires in `perform_deallocations`,
  invoked by `flush()` — `gpu.rs:63-81, 209-211`. `cleanup` calls
  `storage.flush()` at `memory_manage.rs:89`, and `sync()` flushes the
  stream — so **`memory_cleanup()` + `sync()` is the sequence that
  actually returns pages to the driver.**

**Two hard consequences for the design:**

1. **Partial-page occupancy blocks reclaim.** If a context's metrics
   share a pool page with *anything else live on the same stream*,
   dropping one metric won't free that page. The only way to guarantee a
   metric's pages are fully-free at drop is to **not share its stream
   with anything that outlives it** — i.e. give it its own stream.
2. **Cleanup is stream-scoped, and the stream is selected by the calling
   thread's thread-local unless the client is bound.** `release()` today
   (`metric.rs:979`) cleans the *default thread-local stream* of whatever
   thread calls it — not the metric's own pages, and wrong if the drop
   happens on a different thread than the allocs.

### 1.5 The 2026-05-22 `MetricCache` panic — the use-after-cleanup hazard

`VRAM_DROP_AUDIT_2026-05-29.md §1.4` traced it: `SlicedPool::cleanup`
**relocates** surviving partial pages by rewriting their page index
(`update_page`, sliced_pool.rs:121-123). Any `Binding` captured *before*
the cleanup still carries the stale page index → `find` returns `Err` →
`handle_cursor`'s `.unwrap()` panics on the next dispatch
(`stream.rs:97-102`). **Cleanup is only safe when the calling thread
holds NO live bindings into the stream being cleaned.** Any ironclad
design must structurally guarantee this — not rely on call-site
discipline.

---

## 2. Feasibility spike — MEASURED (not extrapolated)

`crates/cvvdp-gpu/examples/vram_isolation_spike.rs` (throwaway, `#![cfg(feature = "cuda")]`,
registered `required-features = ["cuda"]`). Binds cloned default clients
to explicit `StreamId`s via `set_stream`, allocates real device buffers
(`create_from_slice`, which writes → forces backing pages), and samples
`nvidia-smi --query-gpu=memory.used` **after `client.sync()`** so the
deferred-free queue has drained. Ran natively on the 7950X box (RTX 3060,
12 GiB; native CUDA works — only snap-docker was broken). Two consecutive
runs agreed within noise; representative run:

```
baseline (no cubecl ctx)            used=  2848 MiB
per-context target ~= 1536 MiB (24 x 64 MiB)

A allocated (stream 101)            used=  5313 MiB  (Δbase +2465)
A+B allocated (stream 202)          used=  7617 MiB  (Δbase +4769)   <- peak ≈ SUM
after drop+cleanup A                used=  5345 MiB  (Δbase +2497)   <- A's 2272 MiB freed
  (B client view, B alive)          used=  5345 MiB                  <- B untouched
after drop+cleanup B                used=  3009 MiB  (Δbase +161)    <- B's 2336 MiB freed

CONTROL: two 1-MiB allocs share a page, drop one
  control: both small alive         used=  3073 MiB  (Δbase +225)
  control: dropped 1of2 + cleanup   used=  3073 MiB  (Δbase +225)   <- 0 MiB freed (partial page)
  control: dropped 2of2 + cleanup   used=  3009 MiB  (Δbase +161)   <- whole 64-MiB page freed

THREAD MOBILITY: alloc on worker thread, reclaim on main
  worker-thread alloc (stream 404)  used=  5312 MiB  (Δbase +2464)
  main-thread reclaim of 404        used=  3008 MiB  (Δbase +160)   <- 2304 MiB freed cross-thread

VERDICT:
  drop+cleanup A freed      +2272 MiB  <- isolated reclaim of A
  B still resident after A    +2497 MiB (B untouched)
  drop+cleanup B freed      +2336 MiB  <- isolated reclaim of B
  ISOLATION: CONFIRMED — A's pool freed to driver independently while B stayed resident, then B freed
  THREAD-MOBILE: explicit stream_id overrides thread-local
```

**Findings (all measured):**

1. **Per-stream isolation is REAL.** Two contexts on distinct streams
   (101, 202) hold independent pools. Cleaning stream 101 returned
   ~2.27 GiB to the driver while stream 202's ~2.5 GiB stayed fully
   resident. Then cleaning 202 returned its ~2.34 GiB. This is the exact
   "drop A frees A, B untouched; drop B frees B" behaviour the new
   context type needs.
2. **The shared-page hazard is REAL (control).** Two 1-MiB allocs
   co-resident on one pool page: dropping one + cleanup freed **0 MiB**
   (page still partially occupied → retained). Only dropping the second
   freed the whole page. This is precisely why a bare `Drop` on a
   *shared* stream cannot be ironclad.
3. **Explicit-stream contexts are thread-mobile.** Buffers allocated on
   a worker thread (stream 404) were reclaimed in full from the **main
   thread** by binding a client to the same explicit stream. The explicit
   `stream_id` overrides the thread-local, so a context that owns an
   explicit stream is `Send`-safe and can be dropped/reclaimed from any
   thread — not pinned to its allocating thread.
4. **Stream creation is cheap.** Each context = one `cuStreamCreate` +
   one `MemoryManagement` struct (lazy, on first use). This is *not* the
   ~181 ms device-context init (that's the process-global CUDA context +
   PTX, created once per device and shared by all streams).

---

## 2b. wgpu / Metal backend — does the isolation hold there too? (task #153) — MEASURED

The §2 spike proved isolation on the **CUDA** backend. The open question
for issue **imazen/zenmetrics#17** was whether `MetricSession`'s per-stream
pool isolation is *ironclad on wgpu/Metal too*, or whether it must fall
back to best-effort `release()` there. The wgpu spike answers it.

**HARDWARE CAVEAT (stated up front, no fabrication).** This was run on the
same WSL2 / Windows host with an **NVIDIA RTX 5070** — there is **no Apple
GPU here**, so Metal cannot run on this box. cubecl-wgpu's memory layer
(`WgpuMemManager` + the `SchedulerMultiStream` stream pool) is
**backend-agnostic within wgpu** — the *same* code path serves
Metal/Vulkan/DX12; the only thing that differs per backend is the shader
compiler and the underlying `wgpu::Device`. So the spike runs wgpu via its
**Vulkan** backend (auto-selected by `AutoGraphicsApi` on this host;
recorded at runtime via `client.info()` → `wgpu::Backend::Vulkan`), and
that Vulkan result is the **load-bearing proxy for Metal**: if isolation
works through cubecl-wgpu's abstraction it works on Metal; if it didn't,
Metal would be out regardless. **Metal *hardware* confirmation needs a
Mac — no Metal numbers are fabricated.**

### 2b.1 Structural model on wgpu (file:line, `zenforks-cubecl-* 0.10.1`, fork tree 970ad5b5)

The wgpu backend uses the **same** backend-agnostic `ComputeClient<R>`,
`SchedulerMultiStream`, `StreamPool`, and `MemoryManagement` as CUDA —
only the storage and the per-stream container differ:

- `ComputeClient::set_stream(StreamId)` — **same method, same crate**
  (`cubecl-runtime/src/client.rs:97`). It is defined once on the generic
  client, not per backend. `memory_usage`/`memory_cleanup`/`sync` likewise
  (`client.rs:930 / 972 / 899`).
- **Pools are PER-STREAM on wgpu too.** Each `WgpuStream` *owns* a
  `WgpuMemManager` (`cubecl-wgpu/src/compute/stream.rs:30`), which owns
  **three** `MemoryManagement<WgpuStorage>` pools — main / staging /
  uniforms (`cubecl-wgpu/src/compute/mem_manager.rs:20-22`). So an isolated
  stream → an isolated set of pools, exactly mirroring CUDA's
  per-`Stream` `MemoryManagement`.
- **Stream selection is identical:** the `WgpuServer` holds a
  `SchedulerMultiStream<ScheduledWgpuBackend>` (`server.rs:48`); every op
  routes through `self.scheduler.stream(&stream_id)` to fetch the
  per-stream `WgpuStream`. Streams are stored in the same `StreamPool`
  keyed by `stream_id.value % max_streams`
  (`cubecl-runtime/src/stream/base.rs:101-103`, default `max_streams=128`,
  `config/streaming.rs:24`). `WgpuStreamFactory::create()`
  (`schedule.rs:80`) builds a fresh `WgpuStream` (fresh pools) per slot.
- `memory_usage(stream_id)` → `stream.mem_manage.memory_usage()`
  (`server.rs:428`) reports **that stream's pool** `MemoryUsage`
  (`bytes_in_use`, `bytes_reserved`). This is the per-stream in-API truth.
- `memory_cleanup(stream_id)` → `stream.mem_manage.memory_cleanup(true)`
  (`server.rs:434`) → the same `sliced_pool.rs:104-128` "free fully-free
  pages" code → `WgpuStorage::dealloc` (`storage.rs:113`), which drops the
  backing `wgpu::Buffer`.

**The one real difference from CUDA — driver-level return is lazy.** On
CUDA, dropping a buffer enqueues `cuMemFreeAsync` and the spike `sync()`
flushes it, so the page returns to the driver deterministically (and
nvidia-smi sees it drop). On wgpu, `WgpuStorage::dealloc` just removes the
`wgpu::Buffer` from its map (drop), and `WgpuStorage::flush` is a no-op
with the comment *"We don't wait for dealloc"* (`storage.rs:117-119`) —
wgpu reclaims the underlying allocation **lazily**, on its own schedule.
Also note **all wgpu streams share ONE `wgpu::Device` + ONE `wgpu::Queue`**
(cloned in `schedule.rs:84-85`), whereas CUDA streams are genuinely
separate streams. So on wgpu the *pool bookkeeping* is per-stream, but the
*device/queue* underneath is shared.

### 2b.2 The measurement problem (and how it was solved)

The #133 GPU sweep found **nvidia-smi cannot see wgpu/Vulkan allocations**
per-PID (`--query-compute-apps` lists nothing for graphics/Vulkan
contexts). The spike confirmed this extends to the **total** card counter
on WSL2: with **3072 MiB** of wgpu buffers live (two 1536-MiB contexts),
nvidia-smi `memory.used` total moved **+3 MiB**. (Contrast: the §2 CUDA
spike on this same box saw nvidia-smi track every allocation, because CUDA
contexts *are* enumerated.) So nvidia-smi is **not** a usable signal for
wgpu here.

The spike therefore uses cubecl-wgpu's **own per-stream
`memory_usage().bytes_reserved`** as the primary signal — the in-API truth
of what each stream's pool holds on the device — and reports nvidia-smi
total only as a (negative) secondary corroboration. This is **pool-level**
evidence (what the cubecl pool reports), which is exactly the level the
`MetricSession` design needs; **driver-level** OS return is wgpu's job and
could not be independently confirmed on this host (nvidia-smi blind +
wgpu defers the free).

### 2b.3 Measured result (Vulkan; raw log: `crates/cvvdp-gpu/benchmarks/wgpu_isolation_spike_2026-05-30.txt`)

Per-context target = 1536 MiB (24 × 64 MiB), pool `bytes_reserved` in MiB:

| step | A-pool reserved | B-pool reserved |
|---|---|---|
| A allocated | **1536** | — |
| B allocated | 1536 (unchanged) | **1536** |
| A-pool view, both alive | **1536** (A only, NOT 3072) | 1536 |
| drop+cleanup A | **0** | **1536** (untouched) |
| drop+cleanup B | 0 | **0** |

1. **Per-stream POOL isolation is REAL on wgpu.** While both A and B were
   live, A's pool reported **1536 MiB (A's footprint only, not 3072)** —
   B's allocations went into B's pool, invisible to A's accounting.
   Cleaning stream 101 dropped A's pool to **0** while stream 202's pool
   stayed fully resident at **1536**. Then cleaning 202 dropped its pool
   to 0. This is the exact "drop A frees A, B untouched; drop B frees B"
   behaviour — same as CUDA, observed at the pool level.
2. **The shared-page hazard reproduces (control).** Two 1-MiB allocs
   co-resident on one pool page: dropping one + cleanup freed **0 MiB**
   (page still partially occupied → retained); dropping the second freed
   the page (8 MiB). Same page-granular hazard as CUDA.
3. **Explicit-stream contexts are thread-mobile on wgpu too.** Buffers
   allocated on a worker thread (stream 404) were reclaimed in full
   (1536 MiB pool-reserved → 0) from the **main thread** by binding a
   client to the same explicit stream. The explicit `stream_id` overrides
   the thread-local (`stream_id.rs`), so a wgpu context that owns an
   explicit stream is `Send`-safe / thread-mobile, identical to CUDA.
4. **Driver-level (nvidia-smi total) was uninformative** (+3 MiB peak vs
   1536 MiB allocated) — the #133 invisibility extends to the total
   counter under WSL2. Pool-level is the load-bearing signal and it is
   unambiguous.

### 2b.4 Verdict for #17

- **Per-stream pool isolation: CONFIRMED on wgpu (Vulkan).** Same
  structural mechanism as CUDA — `set_stream` + per-`WgpuStream`
  `WgpuMemManager` + `stream_id % max_streams` pool routing — proven by
  the in-API `memory_usage()` pool accounting. **By backend-agnostic
  construction within wgpu, this carries to Metal** (Metal shares the
  identical `WgpuMemManager`/`SchedulerMultiStream` code; only the shader
  compiler differs). **Metal hardware confirmation still needs a Mac.**
- **`memory_cleanup` on wgpu reclaims at the POOL level** (releases the
  pool's pages; `bytes_reserved → 0`), via `WgpuStorage::dealloc` dropping
  the `wgpu::Buffer`. **Driver/OS-level return is lazy on wgpu** (no
  `cuMemFreeAsync`-style flush; `storage.rs` "We don't wait for dealloc")
  and was not independently observable here. This is weaker than CUDA's
  confirmed driver-level return, but it is still the cubecl pool releasing
  its claim — which is what `MetricSession` controls.

**Implication for `MetricSession` (Option B):** the isolation design is
**ironclad at the pool level on wgpu**, the same as on CUDA — a
`MetricSession` that owns an explicit stream can drop+cleanup its own pool
without disturbing a sibling session's pool, on Vulkan/DX12 and (by
construction) Metal. The one honest asterisk is that on wgpu the final
hand-back to the OS is wgpu's lazy decision rather than a deterministic
flush; for VRAM-pressure purposes the cubecl pool is emptied (reusable by
the next allocation on any stream), but a tool measuring *driver* free
VRAM may see it return later than on CUDA. This does **not** require a
best-effort `release()` fallback on wgpu — `memory_cleanup()` on the
session's stream is the correct, isolated reclaim on wgpu just as on CUDA;
the fallback would only matter if pool-level isolation had failed the
spike, which it did **not**.

---

## 3. What is "reserved" vs "freed" (the user's second ask)

Regardless of which option ships, this is the honest accounting an API
user should be able to rely on:

| State | Lives where | Freed by |
|---|---|---|
| Process-global CUDA device context (~181 ms init) + compiled PTX module cache | `DeviceHandle` registry, one per device, process-lifetime | **Never** during a run — shared by all metrics/contexts. Released at process exit. |
| The `Stream` struct shell + empty `MemoryManagement` (a few KB host-side) | `StreamPool` slot, process-lifetime once created | **Never** — cubecl has no remove-stream API. But its *pool pages* go to zero on cleanup. |
| A metric's device working-set pages (the GiBs) | the metric's stream's pool | `drop(metric handles)` → pool free list; then `memory_cleanup()` + `sync()` on **that stream** → driver. |
| Shared LUT/const buffers (e.g. blur LUTs) | the metric instance's own handles (per-crate audit §2: each metric owns its LUTs, not cross-shared) | dropped with the metric. **No cross-metric shared device buffers exist today** (audit §2.1–2.7). |
| Pinned host staging buffers | per-stream `memory_management_cpu` | same cleanup path (cleanup hits all pools). |

So "reserved forever" = the device context + PTX + the stream shells
(bytes, not GiBs). "Freed on drop" = everything that costs real VRAM,
**iff** the metric's pages are fully-free on its stream at cleanup time.
Option B makes that "iff" structural.

---

## 4. Options

### Option A — `impl Drop` on the GPU scorer + opt-out flag

**Shape:** add `impl Drop for CvvdpOpaque` (and each `*Opaque`) that, on
drop, calls `client.memory_cleanup()` + `sync()`. Opt-out via a
`leak_on_drop: bool` field / a `Metric::into_pooled()` that sets it, so
batch/warm callers who *want* the pool retained can skip the cleanup.

**Ironclad analysis — NOT ironclad. Three measured/sourced hazards:**

1. **Shared-page partial occupancy (measured, §2 control).** Every
   metric today is built on the **default thread-local stream**
   (`CudaRuntime::client(&Default::default())`, opaque.rs:364). Two
   metrics alive at once on thread T share T's pool. Dropping one runs
   `memory_cleanup` but frees only pages that are *fully* free — any page
   shared with the still-alive metric is retained. So "drop frees this
   metric's VRAM" is **false** whenever a sibling metric (or a warm
   cached ref) is co-resident. The control case measured exactly 0 MiB
   freed in this situation.
2. **Use-after-cleanup panic (sourced, §1.5; the 2026-05-22 incident).**
   `Drop` runs `memory_cleanup`, which relocates surviving partial pages.
   If *another* live metric on the same thread/stream holds bindings into
   a relocated page, its **next dispatch panics** (`stream.rs:101`
   "page doesn't exist"). A `Drop` that cleans a shared pool can crash an
   unrelated, still-alive metric. This is not hypothetical — it already
   happened once and is why the orchestrator cache stopped calling
   cleanup.
3. **Thread-of-drop ≠ thread-of-alloc (sourced, §1.4).** `Drop` cleans
   the *dropping thread's* default stream. If a metric allocated on
   thread T1 is dropped on T2 (trivially possible — `Metric` is `Send`),
   the `Drop` cleans T2's pool (freeing nothing relevant) and leaves T1's
   pages resident forever. Silent leak, no error.
4. **Perf cost on the hot path.** `sync()` on drop is a full device
   barrier. Batch loops that construct→score→drop per cell would eat a
   sync per cell. The opt-out exists for this, but it makes "frees on
   drop" the surprising *non-default* for the performance-sensitive path,
   inverting the safe default.

**Public API delta:** add `Drop` impls (behavioural change — observable
sync on drop) + a `leak_on_drop`/`into_pooled` opt-out on every `*Opaque`
and on `Metric`. **Migration impact:** existing batch/warm callers
silently get a per-drop sync unless they opt out — a latent perf
regression. **Verdict: REJECT as the primary mechanism.** It is ironclad
*only* in the degenerate case of exactly one live metric on its thread —
which is not the orchestrator's reality (warm cached refs, lane reuse).

### Option B — new context type owning an isolated stream + pool (RECOMMENDED)

**Shape:** a `GpuContext` (working name) that, at construction, picks a
**unique** `StreamId` (from a process-global allocator that hands out
collision-free `value`s within `max_streams`) and owns it. Metrics are
created *through* the context, which binds their client to the context's
stream. Dropping the context runs `memory_cleanup()` + `sync()` **on its
own stream**, reclaiming exactly that context's pages.

**Why this is ironclad — each §1/§2 hazard is structurally eliminated:**

- **Shared-page occupancy → eliminated.** The context's stream is private
  to that context. Nothing else allocates on it. So at context drop, the
  context's metrics are the *only* occupants of its pool pages; once their
  handles drop, every page is fully-free and `memory_cleanup` returns all
  of it. (Spike §2 finding 1 is exactly this: stream 101 freed fully
  because nothing else lived on it.)
- **Use-after-cleanup panic → eliminated.** Cleanup runs on the context's
  stream, which no *other* live metric references (other contexts own
  other streams; same-context metrics are dropped first as part of the
  context teardown). No foreign live binding can point at a relocated
  page on this stream. The 2026-05-22 failure mode cannot occur across
  contexts.
- **Thread-of-drop hazard → eliminated.** The context cleans by binding a
  client to its **explicit** stream id, which overrides the thread-local
  (spike §2 finding 3, measured cross-thread reclaim). So the context can
  be dropped from any thread and still reclaim its own pages — it is not
  pinned to the allocating thread. `GpuContext: Send`.
- **Opt-out is trivial and natural:** *don't drop the context* (keep it
  alive to retain the warm pool across many scores — the batch fast
  path), or call `ctx.forget()` / `core::mem::forget(ctx)` to leak the
  stream's pool deliberately, or a `ctx.into_leaked()` that drops the
  metrics but skips the cleanup. The *safe default* (drop → reclaim) is
  the ironclad one; the opt-out is the explicit, named exception.

**What's "reserved" under Option B:** the process-global device context +
PTX (never freed mid-run) and the stream shell (a `StreamPool` slot —
bytes). What the context frees on drop: its entire pool (the GiBs).
Documented per §3.

**Cost / limits:**

- One `cuStreamCreate` per context (cheap, §2 finding 4). The ~181 ms
  device init is *not* re-paid — it's process-global.
- **`max_streams = 128` cap.** At most 128 simultaneously-isolated
  contexts (minus a couple of reserved lanes). The orchestrator uses a
  handful of lanes, not hundreds, so this is comfortable — but the
  context allocator MUST recycle freed `value`s and MUST refuse (or
  document) past 128 to avoid silent aliasing (two live contexts on the
  same `value % 128` would share a pool — re-introducing the shared-page
  hazard). Recycling is safe because a dropped context's stream pool is
  empty (cleaned), so reusing its `value` for a new context starts from a
  clean pool.
- Warm-batch perf is *better* than Option A: the context holds the pool
  across N scores (no per-score sync), and reclaim happens once at context
  drop. This matches the orchestrator's "warm per signature, reclaim on
  signature change" need (audit §3.4) — a `GpuContext` per signature, kept
  warm, dropped (→ reclaim) on swap.

**Public API delta (the part the user must approve — sketch in §6):** one
new public type `GpuContext` + constructor methods on it (`ctx.cvvdp(…)`,
`ctx.metric(kind, …)`) + `Drop for GpuContext` + an opt-out
(`ctx.leak()` / `into_leaked`). The existing free-stream constructors
(`Metric::new`, `CvvdpOpaque::new`) can stay (back-compat) but would be
documented as "uses the shared default stream; reclaim is best-effort via
`release()`, not ironclad — prefer `GpuContext` for guaranteed reclaim."

**Migration impact:** additive. Existing code keeps working on the
default stream; new code that wants ironclad reclaim opts into
`GpuContext`. The orchestrator's `gpu_worker_main` swap path becomes
"drop the old context (→ guaranteed reclaim of exactly its VRAM), build a
new one" — strictly simpler and safer than the current "drop metric +
reclaim_pooled_vram on the default stream" which only works because each
lane is single-metric-at-a-time.

### Option C — keep `release()`, harden + document (the honest minimum if B is rejected)

`release(self, backend)` already exists (metric.rs:979) and
`reclaim_pooled_vram(backend)` (the §150 fix). Option C is: keep them,
fix their correctness envelope, and *document precisely what's reserved*
rather than promise ironclad per-metric reclaim.

**Hardening needed (none of it makes it fully ironclad):**

1. **Thread-correctness:** today `reclaim_pooled_vram` cleans the calling
   thread's default stream. Document loudly that `release()` MUST be
   called on the thread that constructed the metric (the audit already
   says this; it's not enforced). Still breaks silently if violated.
2. **A `reclaim_all_streams()` helper** that iterates known lane stream
   ids and cleans each (the orchestrator knows its lanes). Useful for an
   idle/chunk-end sweep, but requires the caller to *enumerate* streams —
   cubecl exposes no "clean every stream" call, so this is a
   zenmetrics-side registry.
3. **Doc the reserved set** (§3 table) in the `release`/`reclaim` rustdoc
   so users stop expecting a single metric's drop to drop GiBs while a
   sibling is alive.

**Why it's not ironclad:** it still cleans a *shared* default stream, so
the partial-page hazard (§2 control) and the use-after-cleanup panic
(§1.5) remain whenever more than one metric (or a warm cached ref) is
live on the thread. It is the right *fallback* only if per-stream
isolation had failed the spike — which it did **not**.

**Public API delta:** add `reclaim_all_streams()` (+ a stream registry),
rustdoc updates. **Migration impact:** minimal. **Verdict:** ship the doc
clarifications regardless (they're true and cheap), but C alone does not
satisfy "ironclad."

---

## 5. Recommendation

**Adopt Option B — a `GpuContext` that owns an isolated cubecl stream +
pool.** The spike proves the one load-bearing assumption (per-stream
isolation frees independently, cross-thread, at the page granularity a
private stream guarantees). Every hazard that makes Option A non-ironclad
(shared-page occupancy, use-after-cleanup panic, thread-of-drop) is
structurally removed by giving each context its own stream. The opt-out
is the most natural possible: keep the context alive (warm pool) or
explicitly `leak()` it.

**Answer to "do we need a new context type?": Yes.** Not because cubecl
*forces* it — `set_stream` + a unique `StreamId` is enough mechanically —
but because the *only* way to make "drop frees exactly this metric's
VRAM" ironclad is to own the stream the metric lives on, and a context
type is the right place to own that stream + enforce the
metrics-die-before-the-stream-is-cleaned ordering. A bare `Drop` on the
scorer can't own the stream (it's the shared default), so it can't be
ironclad.

**Ship alongside:** Option C's doc clarifications (§3 reserved-set table
in the `release`/`reclaim_pooled_vram` rustdoc) — they're true today and
cost nothing, and they make the back-compat free-stream path honest about
being best-effort.

**Cross-backend (task #153 / issue #17):** Option B is **ironclad at the
pool level on wgpu too** (Vulkan measured §2b; Metal carries by
backend-agnostic construction — needs a Mac to confirm hardware-side). The
SAME `set_stream` + per-stream `WgpuMemManager` + `stream_id % max_streams`
routing gives `GpuContext` isolated pool reclaim on wgpu. **No best-effort
`release()`-only fallback is required on wgpu/Metal** — `memory_cleanup()`
on the session's own stream is the correct isolated reclaim there as on
CUDA. The single honest asterisk for wgpu: the final hand-back to the OS is
wgpu's *lazy* decision (no `cuMemFreeAsync`-style flush), so the cubecl pool
is emptied immediately (reusable by the next alloc) but a *driver*-VRAM
probe may see the return later than on CUDA. Document this in the wgpu/Metal
rustdoc for `GpuContext` when Option B ships.

**Keep for back-compat:** the existing `Metric::new` / `*Opaque::new`
free-stream constructors and `release()` — re-documented as best-effort.

---

## 6. Concrete API sketch for Option B (for user approval — NOT yet implemented)

This is the shape to approve. Names are provisional (`GpuContext` vs
`MetricSession` — flagged for the user). **No public API has been changed
in this task; this is a proposal.**

```rust
// zenmetrics-api/src/context.rs  (NEW module, proposed)

/// An isolated GPU execution context. Owns a private cubecl stream and
/// its memory pool. Every metric built through this context allocates on
/// that stream; dropping the context reclaims **exactly** this context's
/// device VRAM back to the driver, independent of every other context.
///
/// Reserved (NOT freed by drop): the process-global device context, the
/// compiled-kernel (PTX) cache, and the stream shell. Freed by drop: the
/// context's entire working-set pool (see crate docs "VRAM lifecycle").
///
/// `GpuContext: Send` — it may be dropped from a thread other than the
/// one that built its metrics (the explicit stream id overrides cubecl's
/// thread-local stream selection).
pub struct GpuContext {
    backend: Backend,
    stream_id: StreamId,   // unique within max_streams; recycled on drop
    // ... a bound client clone for cleanup ...
}

impl GpuContext {
    /// Acquire a fresh isolated context on `backend`.
    ///
    /// # Errors
    /// - `Error::TooManyContexts` if `max_streams` (default 128) live
    ///   contexts already exist on this backend (refusing avoids silent
    ///   stream aliasing → shared-pool reclaim hazard).
    pub fn acquire(backend: Backend) -> Result<Self>;

    /// Build a metric on this context's isolated stream. Mirrors
    /// `Metric::new` but pins the metric to this stream.
    pub fn metric(
        &self,
        kind: MetricKind,
        width: u32,
        height: u32,
        params: MetricParams,
    ) -> Result<Metric>;

    // Optional metric-specific sugar, e.g.:
    // pub fn cvvdp(&self, w: u32, h: u32, p: CvvdpParams) -> Result<Metric>;

    /// Reclaim this context's pooled VRAM to the driver WITHOUT dropping
    /// the context (idle hook; safe only when no metric on this context
    /// is mid-score — i.e. between scores, all handles dropped).
    pub fn reclaim(&self);

    /// Opt-out: consume the context but DO NOT reclaim — leave its pool
    /// resident (e.g. handing the warm pool to a successor, or a
    /// deliberate leak for a short-lived process). The stream slot is
    /// NOT recycled (its pool is non-empty), counting against the cap.
    pub fn leak(self);
}

impl Drop for GpuContext {
    /// Reclaim exactly this context's VRAM: the owned metrics are dropped
    /// first (handles → pool free list), then `memory_cleanup()` +
    /// `sync()` run on this context's explicit stream (→ driver). The
    /// stream's `value` is returned to the allocator for reuse (its pool
    /// is now empty, so reuse starts clean). Ironclad because nothing
    /// else ever allocated on this stream.
    fn drop(&mut self) { /* bind client to self.stream_id; cleanup; sync; recycle value */ }
}
```

**Open questions for the user to decide before implementation:**

1. **Type name:** `GpuContext` vs `MetricSession` vs `MetricContext`?
2. **Metric ownership:** does the context *hand out* `Metric`s that the
   caller drops independently (as sketched — metrics pinned to the stream
   but owned by the caller), or does the context *hold* its metrics and
   expose `ctx.score(...)` (stronger ordering guarantee — metrics
   provably die with the context, but less flexible)? The stronger form
   makes the "metrics die before cleanup" invariant un-violable.
3. **Cap behaviour at 128:** hard error (`TooManyContexts`) vs fall back
   to the shared default stream with a documented best-effort downgrade?
   (Recommend hard error — silent downgrade re-introduces the hazard.)
4. **Keep `release()`/`reclaim_pooled_vram`** as-is (best-effort default
   stream) or deprecate once `GpuContext` lands?

---

## 7. Acceptance-gate status (this task)

- [x] cubecl isolation feasibility answered — §1 (file:line) + §2
      (measured): a context **can** own an isolated stream+pool that
      frees independently. Constraints: `max_streams=128` cap,
      `value % max_streams` aliasing, stream shell never removed (only
      its pool emptied).
- [x] Spike: isolated-context drop frees its VRAM to driver, isolated
      from another context — **CONFIRMED** (stream 101 freed 2272 MiB
      while 202 stayed resident; then 202 freed 2336 MiB). Plus measured
      partial-page control (0 MiB freed when sharing a page) and
      cross-thread reclaim (2304 MiB freed from a foreign thread).
- [x] **wgpu/Metal isolation feasibility (task #153 / issue #17)** — §2b
      (file:line) + measured on **Vulkan** (no Apple GPU on this WSL2 host;
      cubecl-wgpu's `WgpuMemManager`/`SchedulerMultiStream` is
      backend-agnostic within wgpu, so Vulkan is the load-bearing proxy
      for Metal). **POOL-LEVEL ISOLATION CONFIRMED**: A-pool 1536→0 on
      cleanup while B-pool stayed 1536, then B-pool 1536→0; partial-page
      control reproduced; cross-thread reclaim 1536 MiB. Measured via
      cubecl per-stream `memory_usage()` (nvidia-smi is blind to wgpu/
      Vulkan here — per-PID AND total, extending #133). Driver-level OS
      return is wgpu's *lazy* decision (no `cuMemFreeAsync` flush) and was
      not independently observable. **Metal HARDWARE confirmation needs a
      Mac — no Metal numbers fabricated.** Spike:
      `crates/cvvdp-gpu/examples/wgpu_isolation_spike.rs` (gated `wgpu`);
      raw log: `crates/cvvdp-gpu/benchmarks/wgpu_isolation_spike_2026-05-30.txt`.
- [x] Design doc with options A/B/C, ironclad analysis each, API sketch
      for the recommendation, "new context type?" answered (yes).
- [x] NO public API changed; spikes are throwaway/gated (`cuda` and `wgpu`
      features, `examples/`).
- [ ] Implementation — deferred to user approval of the §6 shape.
      `MetricSession` is **ironclad at the pool level on wgpu/Metal** (no
      `release()`-only fallback needed) per §2b; document the wgpu lazy-OS-
      return asterisk in the `GpuContext` rustdoc when Option B ships.
```
