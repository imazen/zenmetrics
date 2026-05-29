# GPU metric cold-start wall — 2026-05-29

Task #140. Measures the **fixed one-shot overhead** a fresh process
pays before any per-pixel work — CUDA context init + cubecl kernel
JIT/PTX load + first host→device upload + first compute + readback —
versus the **warm per-call wall** of subsequent calls in the same
process. This fixed cost is what decides the GPU-vs-CPU crossover for a
one-shot CLI scoring a single small image.

The committed warm-only sweep
(`benchmarks/gpu_metrics_sweep_2026-05-28.tsv`, task #133) does NOT
capture this — it discards the first call as warm-up and reports only
steady-state per-call wall.

- **Data:** `benchmarks/gpu_coldstart_2026-05-29.tsv`
  (+ `.meta` provenance)
- **Harness:** `scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py`
- **Drivers:** `crates/<metric>-gpu/examples/coldstart_one.rs` (one per
  metric; timer starts BEFORE `Backend::client()` so context init is
  captured — the existing `mem_one_size` `warm_ms` excludes it)
- **Host:** RTX 5070 (12 GiB), driver 596.21, CUDA SDK 13.2.1, cubecl
  0.10.1 (zenforks), `git=0caf36d5`
- **Method:** subprocess-per-cold-sample (cold = fresh CUDA context);
  MEDIAN over **7 fresh processes** per (metric, size) per phase;
  `warm_per_call` = median of 10 intra-process calls. Every timed call
  ends in a host readback (`client.read_one` inside the score
  reduction), forcing a GPU sync — the wall is real execution, not
  async submission.
- **Sizes:** 512² (0.26 MP), 1024² (1 MP), 4 MP (2048²), 16 MP (4096²)
- **Metrics:** butteraugli-gpu, cvvdp-gpu, ssim2-gpu, dssim-gpu,
  iwssim-gpu, zensim-gpu (cuda backend)

## The one-line answer

**A one-shot GPU score pays ~370–570 ms of fixed overhead at 512²
before any per-pixel work** (rising to several seconds at 16 MP,
dominated by buffer allocation), versus a warm per-call wall of
1.5–6.5 ms at 512². The CPU-vs-GPU crossover therefore lives **far
above** the warm regime: for a single small image the GPU's
hundreds-of-ms context+JIT floor is unrecoverable, so CPU wins; for a
server scoring many images, that floor is amortized to zero and GPU
wins at every size.

## Three components of cold-start (measured, not estimated)

The cold one-shot decomposes into three timed phases. Their behavior
across size/metric is what makes the crossover analyzable:

| phase | what it is | scales with | shared? |
|---|---|---|---|
| `client_init_ms` | cubecl `Backend::client()` — CUDA context init via dlopen | **flat** (size- and metric-independent) | **yes** — pay once per process |
| `metric_new_ms` | `Metric::new()` — GPU buffer / pyramid allocation | **size** (for eager-alloc metrics) | no |
| `first_compute_ms` | kernel JIT/PTX load + first upload + first compute + readback | metric + mild size | no (per metric) |

### 1. Context init is a flat ~181 ms floor — the unavoidable tax

`client_init_ms` over all 24 cells: **mean 180.7 ms, range
166.8–191.2 ms, median 182.5 ms.** It does not move with image size or
which metric is loaded — it is the cost of cubecl dlopen-ing the CUDA
driver and standing up a context. A one-shot GPU score can *never* beat
this floor; even a 1-pixel image pays ~180 ms before the first kernel.
On a server this is paid once at startup and amortized to zero.

### 2. Allocation (`new`) is the size-scaling term — and it dominates at large sizes

For metrics that allocate their GPU working set eagerly in `new()`, the
allocation cost scales steeply with image size and becomes the
dominant cold term at 16 MP:

| metric | new_ms @512 | @1024 | @4mp | @16mp |
|---|---|---|---|---|
| butteraugli-gpu | 40 | 181 | 762 | **3850** |
| ssim2-gpu | 65 | 288 | 1092 | **5734** |
| cvvdp-gpu | 52 | 175 | 716 | **3398** |
| dssim-gpu | 49 | 178 | 674 | **3065** |
| iwssim-gpu | 41 | 118 | 410 | **1920** |
| zensim-gpu | **0.1** | 0.1 | 0.1 | 0.1 |

zensim allocates lazily (its `new` is ~0; the work lands in
`first_compute`). The eager-alloc metrics pay multi-second `new()` at
16 MP — this is the cubecl pool reserving large transient scratch, and
it is the single largest cold-start term above ~4 MP.

### 3. Kernel JIT + first compute — per-metric, mildly size-scaled

`first_compute_ms` (warm PTX disk cache) is the JIT/PTX-load +
first-upload + first-compute + readback. It ranges ~130–915 ms and is
mostly a per-metric constant (kernel count/complexity) with a mild
size contribution from the first upload + the compute itself:

| metric | first_compute_ms @512 | @1024 | @4mp | @16mp |
|---|---|---|---|---|
| ssim2-gpu | 129 | 130 | 196 | 822 |
| dssim-gpu | 137 | 138 | 204 | 701 |
| iwssim-gpu | 265 | 228 | 246 | 413 |
| cvvdp-gpu | 272 | 234 | 301 | 755 |
| butteraugli-gpu | 287 | 303 | 401 | 914 |
| zensim-gpu | 385 | 383 | 444 | 725 |

## Cold-start totals vs warm per-call

`cold_total_ms = client_init + metric_new + first_compute` — the true
one-shot wall. `coldstart_overhead_ms = cold_total − warm_per_call`.

| metric | size | cold_total (ms) | warm_per_call (ms) | overhead (ms) |
|---|---|---|---|---|
| butteraugli-gpu | 512  | 499  | 1.54 | 497 |
| butteraugli-gpu | 1024 | 653  | 3.61 | 650 |
| butteraugli-gpu | 4mp  | 1331 | 12.9 | 1318 |
| butteraugli-gpu | 16mp | 4924 | 50.2 | 4874 |
| cvvdp-gpu | 512  | 504  | 4.23 | 500 |
| cvvdp-gpu | 1024 | 589  | 6.00 | 583 |
| cvvdp-gpu | 4mp  | 1188 | 11.8 | 1177 |
| cvvdp-gpu | 16mp | 4283 | 41.3 | 4241 |
| ssim2-gpu | 512  | 396  | 3.96 | 392 |
| ssim2-gpu | 1024 | 610  | 6.50 | 603 |
| ssim2-gpu | 4mp  | 1475 | 14.2 | 1461 |
| ssim2-gpu | 16mp | 6741 | 47.7 | 6693 |
| dssim-gpu | 512  | 376  | 4.14 | 372 |
| dssim-gpu | 1024 | 506  | 5.21 | 501 |
| dssim-gpu | 4mp  | 1060 | 12.2 | 1047 |
| dssim-gpu | 16mp | 3949 | 46.8 | 3903 |
| iwssim-gpu | 512  | 491  | 6.53 | 485 |
| iwssim-gpu | 1024 | 527  | 9.47 | 517 |
| iwssim-gpu | 4mp  | 835  | 12.8 | 822 |
| iwssim-gpu | 16mp | 2512 | 39.4 | 2473 |
| zensim-gpu | 512  | 570  | 1.66 | 569 |
| zensim-gpu | 1024 | 574  | 3.27 | 571 |
| zensim-gpu | 4mp  | 635  | 9.67 | 625 |
| zensim-gpu | 16mp | 914  | 37.8 | 876 |

Read off: at 512², the cold one-shot is **240–380× the warm per-call**
for the lighter metrics. The warm per-call is single-digit ms; the cold
one-shot is several hundred ms. That gap is the entire crossover story.

## Cold PTX disk cache (worst case, first-ever run)

cubecl/NVIDIA caches compiled PTX in `~/.nv/ComputeCache` (1.1 GiB on
this box). All 24 rows above were measured with a **warm disk cache** —
the realistic deployed case (process N>1 after the box has run any GPU
job). The truly-first-ever process (empty cache, forcing PTX JIT from
scratch) is materially slower, by a **metric-dependent factor**:

| metric | first_compute warm-disk | first_compute cold-disk | factor | extra one-shot |
|---|---|---|---|---|
| butteraugli-gpu @1024 | ~303 ms | ~1288 ms | **~4.2×** | +~1050 ms |
| zensim-gpu @1024 | ~383 ms | ~506 ms | **~1.3×** | +~175 ms |

The penalty scales with kernel count/complexity, **not** image size, and
is paid exactly once: the second process onward sees warm-disk numbers
(measured: butteraugli sample B dropped from 1288 → 611 ms once the
cache repopulated). The original cache was renamed aside (never
deleted) for this experiment and restored afterward. Cold-disk rows in
the TSV carry `disk_cache_state=cold_disk`, `n_samples=1`.

**Worst-case one-shot for a freshly-provisioned box** (cold disk + cold
context) is therefore ~`client_init + new + cold_first_compute`. For
butteraugli at 1024² that measured ~1706 ms; for a fleet worker that
fetches the image and scores once, the first image eats the full JIT
tax and every image after is warm.

## Implication for the GPU↔CPU crossover

- **Server / batch (warm process, warm disk):** the ~181 ms context +
  per-metric JIT are paid once at startup, amortized to zero. The
  per-image wall is the warm per-call (1.5–50 ms across 512²–16 MP from
  the table above, and the dedicated warm sweep). GPU wins at every
  size — exactly what `gpu_metrics_sweep_2026-05-28.tsv` shows.
- **One-shot CLI (fresh process):** the GPU pays **~370–570 ms before
  any per-pixel work** at 512² (warm disk), or **~1.7 s** worst-case
  with a cold disk cache. A CPU metric on a 512² image finishes in tens
  of ms with no context tax. So **CPU wins for one-shot small images**
  by the full cold-start margin, and the crossover size — where GPU
  per-pixel throughput finally overcomes the fixed ~370–570 ms floor —
  sits well into the multi-MP range. (Computing the exact crossover
  pixel count requires the per-metric CPU per-pixel slope from the CPU
  benches; this doc supplies the GPU fixed-overhead side of that
  equation.)

The actionable number: **a one-shot GPU score pays ~180 ms of CUDA
context init plus ~130–900 ms of kernel JIT + first compute (warm disk)
before producing a single useful pixel** — and for the eager-alloc
metrics, multi-second allocation on top of that above 4 MP.

## Caveats

- cuda backend only. wgpu (Vulkan) cold-start was not measured this
  run; the harness supports `--backends wgpu` if a wgpu cold number is
  wanted later.
- `metric_new_ms` and `first_compute_ms` have run-to-run jitter from
  the cubecl pool and PTX-cache validation (visible in the per-sample
  log); the reported values are 7-sample medians.
- The intercept/slope split here is observed per phase, not an OLS fit
  — context init is the flat term, allocation the size term, JIT the
  per-metric term. A formal `α + β·pixels` fit would need a denser size
  sweep (the 4-bucket grid spans 64× in pixels, too sparse for a clean
  line) — out of scope for the fixed-overhead question this task asked.
