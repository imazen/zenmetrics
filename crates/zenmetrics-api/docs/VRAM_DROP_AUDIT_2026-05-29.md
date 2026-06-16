# VRAM drop / pool-reclaim audit — 2026-05-29

Task #150 (`claude-vram-drop`). Audit-first pass over the cubecl memory
model and every `-gpu` crate's Drop / cached-reference lifecycle, plus the
orchestrator's metric-swap path, to answer two questions raised by #147:

1. When an API user drops a `Metric` (or clears a cached reference), does
   the device VRAM actually come back to the **driver**, or only to the
   cubecl pool's free list (the ~3830 MiB plateau #147 measured)?
2. When the orchestrator swaps metrics (signature change), does the old
   metric's pooled VRAM linger while the new metric allocates — pushing
   peak toward SUM(metrics) instead of MAX(single metric)?

Every claim below cites `file:line`. Runtime peak numbers are deferred to
STEP 3 (`benchmarks/vram_cleanup_2026-05-29.tsv`) — this doc is the
source-level model.

---

## 1. cubecl memory model — pool vs driver

**Fork under audit:** `/home/lilith/work/zenforks-cubecl-work/` (the
`zenforks-cubecl-work` colocated repo our crates depend on).

### 1.1 Dropping a `Handle` returns to the pool, not the driver

A `cubecl::server::Handle` is a reference-counted binding into a pool
*slice*. When the last clone drops, the slice is marked free in its page
but the page's device allocation (the `cuMemAlloc*` backing it) is **not**
freed — it stays in `MemoryManagement.pools[*]` for reuse. This is the
plateau #147 saw: 30 construct→score→drop cycles at 16 MP hold ~3830 MiB
because each cycle's handles return to the same pooled pages.

- Pool storage layer: `crates/cubecl-runtime/src/memory_management/memory_pool/`
  (`sliced_pool.rs`, `exclusive_pool.rs`).
- The CUDA storage `dealloc` is itself **deferred**: it only pushes the
  storage id onto a `deallocations: Vec<StorageId>` queue —
  `crates/cubecl-cuda/src/compute/storage/gpu.rs:204-206`.
- The actual `cuMemFreeAsync` / `cuMemFree` happens only in
  `perform_deallocations()` —
  `crates/cubecl-cuda/src/compute/storage/gpu.rs:63-81` — which is invoked
  by `flush()` (gpu.rs:209-211).

### 1.2 `ComputeClient::memory_cleanup()` IS the driver-reclaim API

Public API: `ComputeClient::memory_cleanup(&self)` —
`crates/cubecl-runtime/src/client.rs:972-976`. Doc: *"Ask the client to
release memory that it can release. Results will vary on what the
allocator deems beneficial."* The chain:

1. `client.memory_cleanup()` → `server.memory_cleanup(stream_id)`
   (client.rs:972-976).
2. CUDA server: `crates/cubecl-cuda/src/compute/server.rs:245-257` →
   `command.memory_cleanup()`.
3. `Command::memory_cleanup` →
   `streams.current().memory_management_gpu.cleanup(true)` —
   `crates/cubecl-cuda/src/compute/command.rs:78-80`. **`explicit=true`.**
4. `MemoryManagement::cleanup(explicit=true)` —
   `crates/cubecl-runtime/src/memory_management/memory_manage.rs:380-392`
   — calls each pool's `cleanup(storage, alloc_reserve_count, true)`.
5. With `explicit=true` each pool drains its pages and calls
   `storage.dealloc(id)` on every **fully-free** page:
   - `SlicedPool::cleanup` — `sliced_pool.rs:104-128`: drains all pages,
     `page.coalesce()`, and for any page where `amount_free ==
     amount_total` calls `storage.dealloc(id)`. Partially-used pages are
     **retained** (and **relocated** via `update_page` — see §1.4).
   - `ExclusivePool::cleanup` — `exclusive_pool.rs:193-227`: with
     `explicit`, deallocs every free page immediately (bypasses the
     `free_count >= ALLOC_AFTER_FREE` gate).
6. `storage.dealloc` only **queues** the id (gpu.rs:204-206). The
   `cuMemFree*` fires on the next `flush()`.

**Therefore: to observe VRAM return to the driver after a metric drop,
the sequence must be `drop metric` (handles → free pool slices) →
`client.memory_cleanup()` (free pages → storage dealloc queue) →
`client.sync()` / a flush (queue → `cuMemFree*`).** A drop alone leaves
the pool full (the plateau). `memory_cleanup` alone queues but does not
free until a flush. The STEP 3 measurement must `sync()` after cleanup
before sampling, otherwise it will read the pre-free figure.

NB `server.memory_cleanup` uses `StreamErrorMode { flush: false }`
(server.rs:248-251) and `Command` drop does not flush the storage —
`ResolvedStreams::drop` only registers a GC task for slice analysis
(`crates/cubecl-runtime/src/stream/event.rs:166-187`); it does not call
`storage().flush()`. Only `flush()` (server.rs:166-180) and `sync()`
(server.rs:182+) run `memory_management_gpu.storage().flush()`. So an
explicit `sync()` after `memory_cleanup()` is required to realize the
free.

### 1.3 The pool is PER-STREAM, and a stream is PER-THREAD

This is the safety crux for the orchestrator fix.

- Each `Stream` owns its **own** `memory_management_gpu:
  MemoryManagement<GpuStorage>` —
  `crates/cubecl-cuda/src/compute/stream.rs:23-27`.
- `memory_cleanup(stream_id)` operates on `streams.current()`, i.e. the
  stream selected by the calling thread's `stream_id`
  (command.rs:78-80, server.rs:245-257).
- `ComputeClient::stream_id()` falls back to `StreamId::current()` when
  the client wasn't bound to an explicit stream —
  `crates/cubecl-runtime/src/client.rs:85-88`.
- `StreamId::current()` is **thread-id based** (a `thread_local`) —
  `crates/cubecl-common/src/stream_id.rs:16` (`std::thread_local!`),
  `:49-51` (`fn current`).

**Consequence:** `client.memory_cleanup()` called from thread T cleans
**only T's stream pool**. It cannot touch another lane's stream pool, so
it cannot invalidate a *different* thread's live bindings. The danger is
strictly intra-thread: cleaning while **this thread** still holds live
bindings into pages that get deallocated/relocated.

### 1.4 Why the 2026-05-22 `memory_cleanup` attempt panicked

Documented at `crates/zenmetrics-cli/src/metrics/cache.rs:225-235` and
`:526-537`: an earlier orchestrator-cache version called
`cubecl_runtime_memory_cleanup()` and it panicked at
`cubecl-cuda/src/compute/stream.rs:101` ("Memory page 0 doesn't exist").

Root cause, traced in source: `stream.rs:97-102` is `handle_cursor`,
which does `stream.memory_management_gpu.get_cursor(binding.memory).unwrap()`.
`get_cursor` → `find(binding)` (memory_manage.rs:395-421) looks the slice
up by its descriptor's **page index**. `SlicedPool::cleanup` **relocates**
surviving partially-used pages by rewriting their page index
(`update_page`, sliced_pool.rs:121-123). Any `Binding` captured **before**
the cleanup still carries the **stale** page index → `find` returns
`Err`/None → `.unwrap()` panics on the next dispatch that dereferences
that binding.

So the failure was **not** "memory_cleanup is unsafe in general" — it was
"memory_cleanup was called while live bindings from the same stream were
still in flight / still referenced by a cached metric." The fix the
orchestrator-cache took (drop the metric, skip the hint) avoided the
panic but **also left the freed pages in the pool** — i.e. it traded the
panic for the SUM-not-MAX plateau. The correct fix (STEP 2) is to call
`memory_cleanup` **only at the point where this thread holds NO live
bindings** — i.e. immediately after the old metric is fully dropped and
**before** the new metric is constructed, on the worker's own thread.

---

## 2. Per-crate Drop + cached-reference lifecycle

**No `-gpu` crate has an explicit `impl Drop`** (verified:
`grep -rn "impl Drop" crates/{cvvdp,ssim2,butteraugli,dssim,zensim,iwssim}-gpu/src`
→ zero matches). Drop is the derived field-by-field drop. Each pipeline
struct holds `client: ComputeClient<R>` (a cheap Arc-backed clone; e.g.
`crates/butteraugli-gpu/src/pipeline.rs:109`,
`crates/ssim2-gpu/src/pipeline.rs:206`) plus a **fixed** set of
`cubecl::server::Handle` fields. Dropping the struct drops every Handle
(→ pool free list) and drops the client clone (the global device/server in
`DeviceHandle`'s registry persists — clients are obtained via
`CudaRuntime::client(&Default::default())`, e.g.
`crates/butteraugli-gpu/src/opaque.rs:258`). So **drop releases handles to
the pool but never to the driver** — consistent with §1.1.

### 2.1 butteraugli-gpu

- Drop: derived. `Butteraugli<R>` holds fixed Handle fields (sRGB
  staging, planar linear/XYB, blur, freq bands, accumulators, mask,
  diffmap, blur LUTs) — **no growing `Vec<Handle>`**
  (`crates/butteraugli-gpu/docs/VRAM_LEAK_CHECK_2026-05-29.md:83-86`).
  Nested: `half_res: Option<Box<Butteraugli<R>>>` (one sibling, built at
  construction) and `ref_cache_full: Option<Box<Butteraugli<R>>>` (the
  strip-mode Mode-E whole-image cache, lazily built **once** and reused).
- Cached-ref: `set_reference_with_options` (whole-image) overwrites the
  existing planes **in place** — no new plane handles
  (VRAM_LEAK_CHECK doc :96-98, confirmed by #147's `many_new_refs` cell:
  100 distinct refs hold flat at 3840 MiB, returning to exactly 3840).
  `clear_reference` exposed via opaque (metric.rs:929).
- Verdict: **no accumulation**, **plateau-not-leak** (this is #147's
  finding — `create_drop` 16 MP `full` holds ~3830 MiB precisely because
  the pool retains dropped pages).

### 2.2 ssim2-gpu

- Drop: derived. Fixed Handle fields; caches are fixed-slot:
  `cached_ref` arrays + `strip_cached_ref: Option<Vec<StripCachedRefScale>>`
  (`crates/ssim2-gpu/src/pipeline.rs:236-268`).
- Cached-ref: `set_reference` (pipeline.rs:811) overwrites the fixed
  `cached_ref` arrays in place. `set_reference_strip_mode`
  (pipeline.rs:853-890) reallocates the strip cache **only when the shape
  changes** (`v.len() != dims.len()`, :879-884); otherwise reuses slots.
  `clear_reference` sets `has_cached_reference = false` (:552). Opaque
  exposes `clear_reference` / `has_cached_reference` (metric.rs:931,952).
- Verdict: **no accumulation** across `set_reference` calls.

### 2.3 dssim-gpu

- Drop: derived. `ref_full: Option<RefFullState>`
  (`crates/dssim-gpu/src/pipeline.rs:280`).
- Cached-ref: whole-image `set_reference` (pipeline.rs:679-693) writes
  into pre-allocated per-scale planes in place. Strip-mode
  `set_reference_full_for_strip` (pipeline.rs:706-870) allocates a fresh
  `RefFullState` and assigns `self.ref_full = Some(...)` (:870) — the
  prior `Some` is dropped on reassignment (→ pool). `clear_reference`
  sets `self.ref_full = None` (:926), dropping the strip ref handles to
  the pool. The transient `ref_lin`/`temp1`/`temp2` allocated inside
  that fn drop at end of call (:702-705 doc).
- Verdict: **no accumulation** (overwrite-on-reassign). Strip-mode
  `set_reference` does churn the pool (alloc fresh, drop old) but bounded.

### 2.4 iwssim-gpu

- Drop: derived. `scales[s].lp_ref` is a per-scale Handle;
  `cached_strip_ref: Option<CachedStripRefState>` holds the strip-mode
  `lp_ref[strip_idx][scale]: Vec<Vec<Handle>>`
  (`crates/iwssim-gpu/src/pipeline.rs:467, 581`).
- Cached-ref: `set_reference` (pipeline.rs:969+) reassigns
  `self.scales[s].lp_ref = alloc(...)` (:1610) — overwrite, old handle
  drops. Strip mode builds a fresh `CachedStripRefState` and assigns
  `self.cached_strip_ref = Some(...)` (:1787) — prior `Some` dropped.
  `clear_reference` sets `self.cached_strip_ref = None` (:887, :954).
- Verdict: **no accumulation** (overwrite-on-reassign). Strip-mode
  `set_reference` churns the pool (fresh per-strip-per-scale alloc, old
  dropped) but bounded.

### 2.5 zensim-gpu

- Drop: derived. `persist_planes_ref: Vec<[Handle; 4]>`
  (`crates/zensim-gpu/src/pipeline.rs:375`) is built **once** at
  construction (`Scale::new` loop, :736-750 in `new`).
- Cached-ref: `set_reference` (pipeline.rs:920+) populates the
  pre-allocated `persist_planes_ref` in place; there is **no
  per-set_reference `push`**. Host-side `cached_ref_strip_srgb: Vec<u8>`
  fallback is cleared on each set (:930-932). The opaque shim does **not**
  expose `clear_reference` (umbrella `clear_reference` is a no-op for
  zensim — metric.rs:924-925) and `has_cached_reference` returns `false`
  (metric.rs:963-964).
- Verdict: **no accumulation**.

### 2.6 cvvdp-gpu

- Drop: derived. Fixed per-level scratch (`weber_scratch`, `bands_ref`,
  `gauss_ref`, etc.) built once at construction; warm-ref state is a
  small host scalar `warm_ref_baseband_log_l_bkg`
  (`crates/cvvdp-gpu/src/pipeline.rs:2584, 3035, ...`).
- Cached-ref: `warm_reference_srgb` (the umbrella's `set_reference` path
  for cvvdp, metric.rs:806-807) writes into the pre-allocated
  `bands_ref[k].planes[c]` / `weber_scratch[k].log_l_bkg` **in place**
  (pipeline.rs:4090-4149); the `out.push`/`bands_out` there are
  **host-side readback Vecs**, not device handles. `warm_ref_*` is
  cleared (set to `None`) at the head of each warm-ref build
  (:4090, :4242, etc.). Opaque has no `clear_reference` accessor
  (umbrella no-op — metric.rs:921); `has_warm_reference` works
  (metric.rs:962).
- Verdict: **no accumulation** (overwrite-in-place).

### 2.7 Umbrella `Metric` (`zenmetrics-api`)

- `Metric` (`crates/zenmetrics-api/src/metric.rs:194-213`) is an enum
  wrapping each `*Opaque`. **No `impl Drop`** — drop forwards to the
  inner opaque (which has no Drop either; derived field drop releases the
  pipeline's Handles to the pool).
- There is **no `release()` / `memory_cleanup` exposure** anywhere in
  `zenmetrics-api` or the metric crates (verified by grep). So an API
  user who drops a `Metric` gets handles→pool but **cannot** reclaim the
  pooled VRAM to the driver — the §1.1 / #147 plateau. **This is the gap
  STEP 2 fix #1 closes.**

---

## 3. Orchestrator swap / idle cleanup gap

### 3.1 GPU worker swap path

`gpu_worker_main` (`crates/zenmetrics-orchestrator/src/pool.rs:750-758`)
keeps one warm `current_metric: Option<ExecMetric>` per lane. On a
signature change (`metric, w, h, backend`):

```text
pool.rs:794   let signature_changed = current_signature != Some(sig);
pool.rs:795   if signature_changed {
pool.rs:826       current_metric = None;     // drop old metric → handles to POOL
pool.rs:827       cached_ref_hash = None;
pool.rs:828       construct_pub(...)         // new metric allocates FRESH pages
```

**The gap:** at line 826 the old metric's handles return to the cubecl
pool's free list (per §1.1) but the pages stay resident. At line 828 the
new metric (different signature → different working-set shape) allocates
from the pool, but its shape rarely matches the freed pages exactly, so
it grows new pages. **Peak across the swap = old pooled pages + new
allocated pages ≈ SUM, not MAX.** There is **no `memory_cleanup()`** call
between the drop (826) and the construct (828). (The comment at
pool.rs:825 acknowledges "Drop the old metric first to release device
buffers" — but "release" here means "to the pool", not "to the driver".)

This is **exactly** the position where cleanup IS safe (§1.3/§1.4): it is
the worker's own thread, and after `current_metric = None` this thread
holds **no** live bindings (the just-dropped metric was the only holder;
`cached_ref_hash` is also cleared). So a `memory_cleanup()` + `sync()`
here cannot hit the stale-binding panic that bit the 2026-05-22 cache
attempt — that attempt called cleanup from a *different* call site where
a cached metric's bindings were still live.

### 3.2 The OOM-recovery path has the same gap

`pool.rs:976-977` (runtime OOM) and `:1202` (CPU worker) do
`current_metric = None; current_signature = None;` to force a fresh
construct on the next task — but again without `memory_cleanup`, so the
OOM'd pool stays full.

### 3.3 Idle / chunk-end

There is no "worker went idle" hook in `gpu_worker_main` — the loop
blocks on `rx.recv()` (pool.rs:761). At chunk-end the pool is dropped
(its lane senders drop → workers exit). The legacy CLI path
(`zenmetrics-cli/src/metrics/cache.rs`) has `cleanup_all`
(:238-246) called between source images when
`SWEEP_CLEANUP_BETWEEN_SOURCES=1`, but it deliberately does **not** call
`memory_cleanup` (the 2026-05-22 note). So even the legacy cleanup leaves
the pool resident.

### 3.4 Same-signature warm path must NOT be cleaned

Within one signature, consecutive tasks reuse the warm metric (no
construct, no drop) — pool.rs:886-948. Calling `memory_cleanup` between
same-signature scores would deallocate the warm metric's own pages while
its bindings are live → the §1.4 panic, **and** would re-pay the per-dist
allocs (#145's warm-path win). **Cleanup must fire only on
signature-change (after drop, before construct) and on idle/OOM — never
between same-signature scores.** This is the hard constraint the fix must
honor.

---

## 4. Fix plan (STEP 2) derived from this audit

1. **Expose a safe pool-reclaim entry point.** Add a small free function
   in `zenmetrics-api` (e.g. `reclaim_pooled_vram(backend)`) that obtains
   the current thread's global client for the backend and calls
   `client.memory_cleanup()` then `client.sync()`. Per §1.3 this is
   thread/stream-scoped, so it only frees the caller thread's pool. Also
   document/expose it so an API user who drops a `Metric` and wants VRAM
   back can call it (closes §2.7).
2. **Orchestrator swap cleanup.** In `gpu_worker_main`, after
   `current_metric = None` (pool.rs:826) and **before** `construct_pub`,
   call the reclaim on signature-change only. Same on the OOM path
   (:976-977). This is safe per §3.1 (no live bindings on this thread at
   that point). Target: swap peak ≈ MAX(single metric).
3. **Do NOT** call reclaim between same-signature tasks (§3.4) — measure
   that the warm per-call wall is unchanged.

All three gated by STEP 3 before/after peak measurement on a real GPU.
