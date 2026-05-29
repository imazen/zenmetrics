# Orchestrator cost-model audit (task #146, 2026-05-29)

Audit of the `zenmetrics-orchestrator` scheduler against the measured
cost model. **AUDIT FIRST — every "current behavior" claim cites
`file:line`.** The goal is to find where the orchestrator diverges from
optimal for the measured numbers, fix the high-value gaps, and
explicitly document what is already optimal so a later session does not
churn it.

## Cost model (measured, committed)

Sources (parent repo `benchmarks/`):

- `benchmarks/gpu_coldstart_2026-05-29.tsv` — per-metric cold-start
  breakdown. Columns: `client_init_ms` (CUDA context init),
  `metric_new_ms` (per-signature construct), `first_compute_ms`,
  `cold_total_ms` (= the three summed), `warm_per_call_ms`.
- `benchmarks/cpu_wall_all_metrics_2026-05-29.tsv` — CPU full-mode
  zenbench wall (7950X, no `target-cpu=native`).
- `benchmarks/cpu_gpu_crossover_2026-05-29.tsv` — the synthesized
  one-shot + batch crossover table (task #141).

### Levers, with measured magnitudes

1. **GPU per-process-start ≈ 181 ms** — `client_init_ms` column in
   `gpu_coldstart_2026-05-29.tsv` ranges 166–191 ms across all six
   metrics. Paid **once per process** (CUDA context init), independent
   of metric or size. Optimal handling = a persistent warm worker pays
   it once and reuses the context for every subsequent task and every
   metric.

2. **per-ref = `set_reference` precompute** — paid once per distinct
   reference; cvvdp/butter/ssim2/zensim all expose a precompute API the
   adapters cache. Optimal handling = cache the precomputed reference
   and order same-ref tasks consecutively so the precompute is hit once
   per distinct ref, not once per (ref, dist) pair.

3. **per-dist = warm per-call** — `warm_per_call_ms` column. Cheap once
   the process + ref are warm (1.5–62 ms across the grid).

4. **CPU-vs-GPU one-shot crossover** — `cpu_gpu_crossover_2026-05-29.tsv`
   `one_shot_winner` column. GPU pays the `cold_total_ms` floor
   (~376–6740 ms) on a one-shot call; CPU pays only `cpu_full_ms`. So
   for a SINGLE cold call:

   | metric | CPU faster one-shot through | GPU faster one-shot at |
   |---|---|---|
   | cvvdp  | 16 MP (4096²) | >16 MP (extrapolated) |
   | ssim2  | 16 MP (4096²) | >16 MP (extrapolated) |
   | butter | 16 MP (4096²) | >16 MP (extrapolated) |
   | zensim | 16 MP (4096²) | >16 MP (extrapolated) |
   | dssim  | 4 MP (2048²)  | 16 MP (4096²) |
   | iwssim | 1 MP (1024²)  | 4 MP (2048²) |

   **Batch** (`batch_winner` column, GPU compared on `warm_per_call_ms`):
   GPU wins everywhere, every metric, every measured size — the warm
   per-call number always beats CPU once the cold floor is amortized.

5. **cross-metric context sharing** — one process shares the CUDA
   context across metrics (task #144 `gpu_inprocess_warmth` Q1). After
   the first metric pays `client_init_ms`, a second metric in the same
   process pays only its own `metric_new_ms`, not another context init.

## Per-lever audit

### Lever 1 — Persistent warm process: OPTIMAL, no change

**Current behavior:** the GPU worker thread
(`gpu_worker_main`, `pool.rs:750`) is a long-lived loop
(`while let Ok(task) = rx.recv()`, `pool.rs:761`) that holds
`current_metric: Option<ExecMetric>` (`pool.rs:758`) and only rebuilds
it when the `(metric, w, h, backend)` signature changes
(`signature_changed`, `pool.rs:794-795`; rebuild block `pool.rs:828-839`).
The CUDA context lives inside the metric's cubecl `ComputeClient`, which
is created on the worker thread's first construct and persists for the
thread's lifetime. `warm_instance_construction_count()` (`pool.rs:734`)
exposes the construct count for tests.

**Optimal:** pay `client_init_ms` (~181 ms) once per worker thread,
reuse for all subsequent tasks. ✔ matches.

**Gap:** none. The N-lane pool (Phase 9.1, `PHASE9_DESIGN.md`) spawns
lanes up-front and idles surplus lanes on `mpsc::recv()` with zero
overhead — each lane pays its own one-time context init, which is the
correct cost for genuine concurrency.

### Lever 2 — Cached-ref reuse + same-ref ordering: OPTIMAL, no change

**Current behavior:** two cooperating mechanisms.

- *Reuse:* the GPU worker tracks `cached_ref_hash: Option<u64>`
  (`pool.rs:759`). On each task it computes
  `need_install = cached_ref_hash != Some(task.ref_hash) || signature_changed`
  (`pool.rs:925`); only on `need_install` does it call
  `m.set_reference(ref_bytes)` (`pool.rs:927`) and update the cached
  hash. Subsequent same-ref tasks take
  `m.compute_with_cached_reference_with_extras` (`pool.rs:940`),
  skipping the precompute. The CPU worker mirrors this (`pool.rs:1160`).
  The adapters back this with real precompute APIs — fast-ssim2
  `Ssimulacra2Reference::new` (`cpu_adapter.rs:424`), butter
  `ButteraugliReference` (`cpu_adapter.rs:329`), zensim
  `precompute_reference` (`cpu_adapter.rs:475`).
- *Ordering:* Phase 7.6 sorts tasks by `(metric, w, h, ref_hash)` before
  dispatch — `run_all` internal sort (`REORDERING_DESIGN.md` Layer 2) and
  the streaming reorder window (`pool.rs:1686-1689`,
  `OrchestratorConfig::stream_reorder_window`, `lib.rs:119`). This places
  same-ref tasks consecutively so the worker's `need_install` fires once
  per distinct ref.

**Optimal:** precompute once per distinct ref, order same-ref
consecutively. ✔ matches. Phase 7.6's real-GPU measurement
(`REORDERING_DESIGN.md` header) recorded 49/50 cached-ref hits on a
single-ref chunk and a 6.7× reduction in warm-instance constructions on
a mixed chunk.

**Gap:** none.

### Lever 3 — per-dist warm per-call: OPTIMAL, no change

**Current behavior:** the warm per-call path is exactly
`compute_with_cached_reference_with_extras` (`pool.rs:940`) on the
cached-ref branch and `compute_with_extras` (`pool.rs:942,947`) on the
no-cached-ref branch — both reuse the warm `current_metric`. No
re-construction, no re-init.

**Optimal:** ✔ matches.

### Lever 4 — butter expensive-first-ref awareness: PARTIAL (amortization optimal; chooser unaware)

**Current behavior:** butter's expensive reference precompute is
amortized by the cached-ref machinery (lever 2) — the first same-ref
task pays the `ButteraugliReference` build (`cpu_adapter.rs:329`,
`compare_strip` reuse documented `cpu_adapter.rs:737-749`), subsequent
same-ref tasks reuse it. So *within a warm batch* butter is handled
optimally.

**Gap (shared with lever 4's interaction with the chooser):** the
chooser ranks butter purely on warm `ns_per_px` (see Lever 4-bis /
the cross-cutting gap below). The cost cache (`BackendBench`, `lib.rs:201`)
has no slot for the cold floor or the first-ref precompute cost, so for a
*one-shot* butter call the chooser still prefers GPU (lower warm ns/px)
even though `cpu_gpu_crossover` says CPU is ~3× faster one-shot at 16 MP
(1690 ms CPU vs 4923 ms GPU `cold_total`). This is the same root cause as
the CPU-vs-GPU routing gap — it is not butter-specific. Documented under
the cross-cutting gap; no butter-only fix is warranted.

### Lever 5 — cross-metric context sharing in mixed chunks: OPTIMAL, no change

**Current behavior:** each worker lane holds ONE `ExecMetric` swapped on
signature change (`pool.rs:826-838`). When the metric changes within a
lane, only the old metric is dropped and a new one constructed
(`construct_pub`, `pool.rs:828`) — the cubecl `ComputeClient` /CUDA
context underneath is NOT torn down (it is process/thread-global state in
cubecl-cuda's `MultiStream` server, `PHASE9_DESIGN.md` §9.1). So a mixed
chunk pays `client_init_ms` once for the lane, then only `metric_new_ms`
per distinct metric — exactly the task #144 Q1 finding.

**Optimal:** ✔ matches.

## Cross-cutting GAP — chooser routes on warm `ns_per_px`, ignores the one-shot cold floor

**This is the one real gap (task Step 1 lever #3, "likely GAP").**

**Current behavior:** `Orchestrator::choose_backend_with_config`
(`chooser.rs:540`) interpolates a single per-pixel cost
`ns_per_px` per backend from the bench cache
(`interpolate_ns_per_px`, `chooser.rs:302`) and selects the backend with
the **lowest `ns_per_px`** (`chooser.rs:599-620`, ranking loop). The
cached `ns_per_px` is the **warm, steady-state p50** — the bench discards
`warmup_iters` (default 2, `bench.rs:58,648`) and times the median of the
post-warmup iterations (`bench.rs:672-675`). So the GPU's
`client_init_ms` + `metric_new_ms` + `first_compute_ms` (the cold floor)
is **never** represented in the number the chooser ranks on.

Consequence: for a one-shot small/medium image the chooser picks the GPU
backend (its warm ns/px is lower than CPU's) even though the GPU has to
pay the ~376–4923 ms cold floor that makes CPU strictly faster for that
single call. `cpu_gpu_crossover_2026-05-29.tsv` `one_shot_winner` =
**CPU** for cvvdp/ssim2/butter/zensim through 16 MP, dssim through 4 MP,
iwssim through 1 MP — the chooser disagrees with all of these on a cold
one-shot.

`run_single` (`executor.rs:775`) is the entry point that pays the full
cold floor on **every** call (it constructs a metric fresh per task,
`executor.rs:836`), so the gap bites hardest there. The pool path
amortizes the floor across a batch, so for batch/`run_all` the warm
`ns_per_px` ranking is already correct (matches `batch_winner` = GPU
everywhere).

**Why this is the high-value fix:** it is the only lever where current
behavior diverges from the measured optimum. Levers 1/2/3/5 already match.

**Optimal behavior:** when a task is known to be **one-shot** (no warm
GPU worker will amortize the floor) and the image is at/below the
per-metric crossover size, route to CPU. When the task is part of a
**batch** (the warm pool amortizes the floor) OR the image is above the
crossover size, route to GPU as today.

### Fix design (implemented in Step 2)

Add a *one-shot crossover table* encoding the measured
`cpu_gpu_crossover` `one_shot_winner` per `(metric, pixels)`, with
provenance pointing at the TSV + commit. Expose a new chooser entry that
takes an **execution-context hint** (`OneShot` vs `Batch`):

- `Batch` (default for `submit` / `run_all`, where the pool amortizes the
  ~181 ms floor): unchanged — rank on warm `ns_per_px`. This preserves
  the verified Phase 7.6 / Phase 9 behavior and matches `batch_winner`.
- `OneShot` (opt-in for `run_single`-style callers and the first task of
  a tiny chunk): if the per-metric crossover says CPU wins at this size
  **and** the CPU backend is feasible (feature on, not OOM-listed),
  prefer CPU; otherwise fall through to the existing warm-ns/px ranking.

The crossover threshold is stored as the **largest pixel count at which
CPU still wins one-shot** per metric, read directly from the TSV:

| metric | CPU-wins-one-shot ≤ (pixels) | first GPU-wins size |
|---|---|---|
| cvvdp  | 4096² = 16 777 216 | >16 MP |
| ssim2  | 4096² = 16 777 216 | >16 MP |
| butter | 4096² = 16 777 216 | >16 MP |
| zensim | 4096² = 16 777 216 | >16 MP |
| dssim  | 2048² = 4 194 304  | 4096² |
| iwssim | 1024² = 1 048 576  | 2048² |

Defaults must stay backwards-compatible: the existing
`choose_backend` / `choose_backend_for_task` keep ranking on warm ns/px
(Batch semantics), so no current caller's behavior changes. The one-shot
routing is a new, additive code path.

## Other observations (no action)

- **OOM ladder (Phase 4/8i):** `known_oom_cell` (`chooser.rs:424`) +
  `record_oom_and_persist` (`executor.rs:863`) handle VRAM exhaustion
  with a cascade hypothesis + stale-OOM defeat. Orthogonal to the time
  cost model; correct as-is.
- **GPU concurrency (Phase 9):** N-lane round-robin + adaptive lane
  count + bench-driven optimal N. Each lane's one-time context init is
  the right cost for true parallelism; not a cost-model gap.
- **`extrapolation_pessimism` (1.20, `chooser.rs:59`)** inflates warm
  ns/px above the largest measured size — a per-pixel correction, not a
  fixed-floor correction. It does not address the one-shot gap (which is
  a *fixed* cost, not a per-pixel one).

## Summary

| Lever | Status | Action |
|---|---|---|
| 1. Persistent warm process | OPTIMAL | none |
| 2. Cached-ref reuse + ordering | OPTIMAL | none |
| 3. per-dist warm per-call | OPTIMAL | none |
| 4. butter first-ref | amortization OPTIMAL; chooser blind (= gap below) | none butter-specific |
| 5. cross-metric context sharing | OPTIMAL | none |
| **X. CPU-vs-GPU one-shot routing** | **GAP** | **add one-shot crossover routing (Step 2)** |
