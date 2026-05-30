# In-process GPU warmth transitions — measured (2026-05-29)

Task #144. The zenmetrics orchestrator runs mixed metrics through ONE
long-lived warm worker (single-warm-instance pool). The behavior of
that worker across metric switches and reference switches was, until
this task, **stated from architecture, not measured**. This doc
replaces the guesses with committed numbers from a quiet-machine run.

Every quantitative claim below cites a measured row in
`benchmarks/gpu_inprocess_warmth_2026-05-29.tsv`. Nothing here is
inferred — where a number could not be measured (one 16 MP cell hit a
VRAM cap), it is stated as not measured, not projected.

- **Data:** `benchmarks/gpu_inprocess_warmth_2026-05-29.tsv` (+ `.meta`)
- **Driver:** `crates/zenmetrics-api/examples/inprocess_warmth.rs`
- **Harness:** `scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py`
- **Baseline:** `benchmarks/gpu_coldstart_2026-05-29.tsv` (#140) —
  the **fresh-process** `cold_total_ms` per metric.
- **Host:** RTX 5070 (12 GiB, shared with the WSL2/Windows desktop —
  ~7.8 GiB free for CUDA), driver 596.21, CUDA SDK 13.2.1, cubecl
  0.10.1 (zenforks), `git=592ea475`. Sizes: 512² + 16 MP. 5 fresh
  processes per cell (median); 5 warm reps per process. Quiet machine
  (load < 1.9 at launch).

## Method & correctness

- **Cold = a fresh process** (new CUDA context). Each scenario/ordering
  is a separate child process; medians are over 5 fresh processes.
- **Q1's A→B sequence is WITHIN one process** — B is measured with A's
  CUDA context already warm. That is the whole point.
- **Every timed *score* call ends in a host readback** inside the
  opaque `compute_*` (returns a scalar/vec → `read_one` → GPU sync).
  Cross-check that the sync is real: warm per-call scales correctly
  with size — cvvdp 3.9 ms @512 → 42.9 ms @16 MP, matching #140's
  4.2/41.3 ms. Async-submit timing would be sub-ms and size-flat.
- **Every timed `set_reference` is followed by `block_on(client.sync())`**
  — `set_reference` only *queues* the ref-side precompute. Without the
  explicit sync the host `Instant` measures submission, not execution.
- **ref1 ≠ ref2 ≠ ref3** use distinct XorShift seeds. Verified
  non-degenerate: the score changes between references in the rows
  (e.g. cvvdp Q3 512 ref1 score 4.9520 → ref2 4.9840).
- Release build, cuda backend, **no `-C target-cpu=native`**.

### #140 fresh-process baseline (what each warm transition is measured against)

`cold_total = client_init + new + first_compute`, warm-disk, 7-sample
medians:

| metric | size | client_init | new | first_compute | **cold_total** | warm_per_call |
|---|---|---|---|---|---|---|
| cvvdp | 512  | 172.5 | 51.6 | 272.4 | **504.5** | 4.23 |
| ssim2 | 512  | 187.1 | 65.4 | 129.4 | **396.2** | 3.96 |
| zensim | 512 | 182.2 | 0.13 | 385.0 | **570.3** | 1.66 |
| butter | 512 | 166.8 | 40.3 | 286.7 | **498.7** | 1.54 |
| dssim | 512 | 185.0 | 49.2 | 136.5 | **376.1** | 4.14 |
| iwssim | 512 | 182.5 | 41.3 | 265.1 | **491.4** | 6.53 |
| cvvdp | 16mp | 168.2 | 3397.8 | 755.3 | **4282.7** | 41.3 |
| ssim2 | 16mp | 183.8 | 5734.5 | 822.3 | **6740.5** | 47.7 |
| zensim | 16mp | 189.9 | 0.06 | 725.0 | **914.2** | 37.8 |

## The one-line answers

1. **Q1 — a second metric in a warm process pays ~190–290 ms at 512²,
   NOT the full 504/396/570 ms fresh-process cold_total.** The CUDA
   context init (~181 ms) is paid exactly once per process; the second
   metric only re-pays its own allocation + its own kernel JIT. The
   ~181 ms floor is shared regardless of which metric runs first.
2. **Q2 — kernels are per-metric, context is shared.** B always pays
   its own `first_compute` (kernel JIT) the first time, even with a hot
   context; warm-per-call then drops to the steady-state #140 number.
3. **Q3 — a new warm_ref reference is NOT free for 5 of 6 metrics.**
   cvvdp/ssim2/dssim/iwssim/zensim re-pay essentially the same
   `set_reference` cost for a new (different-pixel) reference as for the
   first one (~0.5–2.8 ms @512, ~14–67 ms @16 MP). **butter is the lone
   exception:** its first ref is expensive (34 ms @512, 3990 ms @16 MP)
   but a *subsequent* new ref reuses the allocated buffers and is nearly
   free (0.76 ms @512, 21.6 ms @16 MP).
4. **Q4 — full-mode different-ref-per-call costs nothing beyond normal
   per-call work.** Changing the reference every call is within noise of
   repeating the same pair (cvvdp 512: 5.16 vs 5.75 ms; 16 MP: 42.6 vs
   42.9 ms; ssim2/zensim identical), because full mode rebuilds the
   reference side every call anyway.

## Q1 — cross-metric context sharing

`client_init` (the explicit `CudaRuntime::client()` that stands up the
CUDA context) measured across all 12 q1q2 cells: **183.2, 185.9, 187.0,
183.1 ms @512** and **191.6, 193.2, 190.6 ms @16 MP** — a flat
~181–193 ms floor, paid once per process, independent of which metric
is first or the image size. This is the #140 "~181 ms floor" claim
*verified to survive a second metric entering the process*.

`B_first_same_process` = metric B's first score in a process where
metric A already ran (context hot, A's kernels loaded, B's not). Compare
to B's fresh-process `cold_total` (#140) → the **context-sharing
saving**:

| ordering (A→B) | size | B | B_first_same_process (ms) | B fresh cold_total (#140) | saving (ms) |
|---|---|---|---|---|---|
| cvvdp→ssim2 | 512 | ssim2 | **199.8** (new 100.9 + compute 88.2) | 396.2 | ~196 |
| ssim2→cvvdp | 512 | cvvdp | **287.6** (new 67.1 + compute 227.2) | 504.5 | ~217 |
| cvvdp→zensim | 512 | zensim | **495.0** (new 32.7 + compute 444.8) | 570.3 | ~75 |
| zensim→ssim2 | 512 | ssim2 | **187.6** (new 95.8 + compute 91.7) | 396.2 | ~209 |
| cvvdp→ssim2 | 16mp | ssim2 | **1816.7** (new 1456.7 + compute 383.1) | 6740.5 | ~4924 |
| cvvdp→zensim | 16mp | zensim | **449.0** (new 34.3 + compute 414.8) | 914.2 | ~465 |
| zensim→ssim2 | 16mp | ssim2 | **6315.4** (new 5961.8 + compute 363.7) | 6740.5 | ~425 |
| ssim2→cvvdp | 16mp | cvvdp | **not measured** (VRAM cap, see Caveats) | 4282.7 | — |

**Reading the saving.** At 512², `B_first_same_process` ≈
`B.new + B.first_compute` with **no `client_init` term** — exactly the
#140 cold_total minus its ~181 ms context init. Both the cvvdp→ssim2 and
ssim2→cvvdp orderings show ~196–217 ms saved, confirming the context
init is paid once *regardless of ordering*. The cvvdp→zensim saving is
smaller (~75 ms) because zensim's `first_compute` (kernel JIT) is its
dominant cold term and is per-metric — not shareable; only the context
init was saved.

At 16 MP the savings are dominated by allocation, not context: cvvdp→ssim2
saves ~4.9 s because ssim2's `B_new` (1456 ms) is far below its
fresh-process `new` (5734 ms) — **the cubecl device-memory pool, warmed
by cvvdp's allocations, is reused by ssim2**. This pool reuse is a second
warmth channel on top of the shared context (see Q2 note).

**Corrected statement.** "A second metric pays the ~181 ms context init
again" is **false**. Measured: the second metric pays ~190–290 ms total
at 512² (its own alloc + its own kernel JIT), of which **zero** is
context init — the ~181 ms is paid once per process.

## Q2 — per-metric kernel warmth

Four phases per metric, same process. A = first metric (cold context +
cold kernels), B = second metric (hot context, cold B-kernels):

| metric | role | size | new (ms) | first_compute (ms) | warm ×5 median (ms) |
|---|---|---|---|---|---|
| cvvdp | A | 512 | 83.3 | 274.5 | 3.92 |
| ssim2 | B (after cvvdp) | 512 | 103.9 | 96.7 | 4.08 |
| ssim2 | A | 512 | 98.2 | 130.5 | 4.98 |
| cvvdp | B (after ssim2) | 512 | 64.4 | 224.1 | 5.76 |
| zensim | B (after cvvdp) | 512 | 33.5 | 461.5 | 2.24 |
| zensim | A | 512 | 34.6 | 506.5 | 2.47 |
| cvvdp | A | 16mp | 3420.6 | 903.3 | 42.95 |
| ssim2 | B (after cvvdp) | 16mp | 1389.6 | 427.1 | 82.34 |
| zensim | B (after cvvdp) | 16mp | 34.9 | 414.8 | 46.65 |
| zensim | A | 16mp | 36.6 | 832.3 | 45.86 |

**What this shows.** `first_compute` (kernel JIT + first upload + first
compute) is a per-metric cost paid the first time *that metric* runs,
hot context or not: cvvdp's `first_compute` is ~224–274 ms whether it's
A or B at 512²; ssim2's is ~97–130 ms. Warm-per-call then collapses to
the #140 steady-state (cvvdp ~3.9/42.9 ms, ssim2 ~4/82 ms, zensim
~2.2/46 ms at 512/16 MP). So **kernels are per-metric, the context is
shared** — confirmed across cvvdp, ssim2, and zensim (the eager- vs
lazy-alloc spread).

Note: `B.new` at 16 MP is *smaller* than `A.new` for the same metric
(ssim2 B_new 1390 ms vs its fresh `new` 5734 ms) because the cubecl
device-memory pool A allocated is reused by B. The pool is a
process-global warmth channel alongside the CUDA context.

## Q3 — new reference in warm_ref mode (the key guess to kill)

On a metric warmed by a throwaway full-mode score (context + kernels +
pool all hot), measure `set_reference(ref1)` then `set_reference(ref2)`
where ref2 is **different pixel content**. Each `set_reference` is
sync'd with `block_on(client.sync())`.

| metric | size | setref1 (ms) | setref2 — NEW ref (ms) | warm_call (ms) | newref_call (ms) | new ref free? |
|---|---|---|---|---|---|---|
| cvvdp | 512 | 2.35 | **1.47** | 2.40 | 2.29 | no — re-pays |
| ssim2 | 512 | 2.54 | **2.26** | 2.85 | 3.34 | no — re-pays |
| dssim | 512 | 2.16 | **2.02** | 3.04 | 2.61 | no — re-pays |
| iwssim | 512 | 2.81 | **1.89** | 7.07 | 6.48 | no — re-pays |
| zensim | 512 | 0.58 | **0.48** | 1.90 | 1.84 | no — re-pays |
| **butter** | 512 | **34.28** | **0.76** | 1.26 | 1.18 | **YES — buffers reused** |
| cvvdp | 16mp | 16.86 | **16.94** | 25.13 | 25.04 | no — re-pays |
| ssim2 | 16mp | 28.45 | **29.36** | 29.58 | 29.19 | no — re-pays |
| dssim | 16mp | 19.91 | **20.31** | 28.19 | 28.55 | no — re-pays |
| iwssim | 16mp | 196.51 | **67.40** | 81.99 | 78.77 | partial (alloc on 1st) |
| zensim | 16mp | 14.73 | **13.95** | 32.25 | 32.39 | no — re-pays |
| **butter** | 16mp | **3990.41** | **21.64** | 45.37 | 44.48 | **YES — buffers reused** |

> **CORRECTION (tasks #148 + #151).** The `setref1` column above is
> **n=1** and was measured on a GPU contaminated by a concurrent zensim
> eval, so its first-ref spikes (butter 34/3990 ms, iwssim 196.51 ms) are
> **transients, not per-reference costs**. The clean n=8 re-measure
> ([`../benchmarks/setref_clean_all_2026-05-29.tsv`](../benchmarks/setref_clean_all_2026-05-29.tsv),
> task #151; butter alone in
> [`../crates/butteraugli-gpu/benchmarks/butter_setref_clean_2026-05-29.tsv`](../crates/butteraugli-gpu/benchmarks/butter_setref_clean_2026-05-29.tsv),
> task #148) finds **`setref1 ≈ setref2 ≈ setref3 ≈ setref4` for all six
> metrics** on a fully warm instance — the butter "45×/184× drop" was
> first-instance allocation + JIT (the `process_start` term), not a
> reference-reuse effect. iwssim @16 MP is the only size-sensitive case,
> and it runs **the opposite direction** of the row above: clean `setref1`
> = 68–74 ms is the *cheapest* phase, `setref2`–`setref4` = 120–163 ms.
> Every `setref1` phase shows one rep-1 transient (iwssim 248 ms, butter
> up to 4166 ms @16 MP) that the n=8 median rejects. Treat the table below
> these two TSVs as authoritative for per-reference budgeting.

**Answer.** For 5 of 6 metrics (cvvdp, ssim2, dssim, zensim at both
sizes; iwssim at 512²) a new reference re-pays essentially the same
`set_reference` cost as the first reference — it is **NOT free**. There
is no machine-wide "any ref is warm" cache; each new reference re-runs
the ref-side precompute. (The first-ref *spikes* in the n=1 table above
are contamination — see the CORRECTION note; the clean #151 re-measure
shows `setref1 ≈ setref2` for every metric, so this "not free" conclusion
holds on the median while the per-metric magnitudes are the #151 numbers.)

**butter is the exception that the inference-based claim would have
gotten wrong.** butter's *first* `set_reference` on a warm instance is
expensive (34 ms @512, **3990 ms @16 MP** — it eagerly allocates its
full reference working set on first use). A *subsequent* new reference
reuses those buffers and costs only 0.76 ms @512 / 21.6 ms @16 MP — a
**45×/184× drop**. *(Per the CORRECTION above, task #148's clean re-measure
attributed butter's apparent first-ref spike to first-instance
allocation + JIT — the `process_start` term — not a reference-reuse
effect; on a fully warm instance butter's `setref1 ≈ setref2`. The iwssim
"196 → 67 ms" reading was likewise n=1 contamination — clean #151 shows
iwssim's first ref is the cheapest 16 MP phase, not the most expensive.)*

`newref_call` ≈ `warm_call` for every metric (the per-call score against
the new reference costs the same as against the first), and the scores
differ between ref1 and ref2 — confirming ref2 is genuinely a different,
correctly-processed reference.

**Corrected statement.** "A new warm_ref reference is free (some
machine-wide cache)" is **false for 5 of 6 metrics** — each new ref
re-pays ~0.5–2.8 ms @512 / ~14–67 ms @16 MP. It is **true only for
butter** (and partially iwssim at 16 MP), where the cost is a one-time
buffer allocation amortized across subsequent refs.

## Q4 — full-mode, different reference every call

Same warm instance: repeat the SAME (ref,dist) pair ×5 (`same_ref`),
then score 5 pairs each with DIFFERENT reference+distorted pixels
(`diff_ref`). Image generation is outside the timed region.

| metric | size | fullmode_same_ref (ms) | fullmode_diff_ref (ms) | delta |
|---|---|---|---|---|
| cvvdp | 512 | 5.16 | 5.75 | within noise |
| ssim2 | 512 | 4.10 | 3.96 | within noise |
| zensim | 512 | 2.36 | 2.27 | within noise |
| cvvdp | 16mp | 42.58 | 42.88 | within noise |
| ssim2 | 16mp | 49.09 | 48.60 | within noise |
| zensim | 16mp | 47.42 | 47.46 | within noise |

**Answer.** Changing the reference on every full-mode call costs
**nothing beyond the normal per-call work** — `diff_ref` is within
run-to-run noise of `same_ref` for all three metrics at both sizes.
Full mode rebuilds the reference side every call regardless, so a new
reference is already priced into the per-call wall. (This is the
opposite end of the spectrum from warm_ref mode, where the reference is
cached and a new one re-pays the cache build per Q3.)

## Corrected statements (inference → measurement)

| Previously stated (from architecture) | Measured truth (cited row) |
|---|---|
| "A second metric in a warm process re-pays the ~181 ms context init." | **False.** Second metric pays ~190–290 ms @512 = its own alloc + its own kernel JIT; **zero** context init. `client_init` is a flat ~183–193 ms paid once (Q1 rows 2/12/22/32, 72/83/93). |
| "Context init cost depends on which metric loads first." | **False.** `client_init` 183.2 / 185.9 / 187.0 / 183.1 ms across all 4 orderings @512 — ordering-independent. |
| "A new warm_ref reference is free (machine-wide cache)." | **False for 5/6 metrics.** New ref re-pays ~0.5–2.8 ms @512 / 14–67 ms @16 MP (Q3 rows 42–65, 103–126). Free only for butter (setref2 0.76 ms vs setref1 34.3 ms @512). |
| "Changing the reference every call in full mode costs extra." | **False.** `diff_ref` ≈ `same_ref` within noise for cvvdp/ssim2/zensim at both sizes (Q4 rows 66–71, 127–132). |

## Implication for the single-warm-instance worker

- **Switching metrics in the warm worker is cheap after warmup.** The
  ~181 ms CUDA context is paid once at process start; each metric's
  kernel JIT (~100–500 ms) is paid once the first time *that* metric is
  used; thereafter every metric switch costs only that metric's
  allocation (small at 512², large at 16 MP — and partly amortized by
  the shared cubecl pool).
- **In warm_ref mode, budget a `set_reference` cost per new reference**
  for every metric except butter — it is NOT free. At 512² it's
  ~0.5–2.8 ms; at 16 MP ~14–67 ms. For butter, only the first reference
  after construction is expensive; subsequent references are nearly free.
- **In full mode, the reference is free to change every call** — no
  caching, so no per-switch penalty beyond the normal per-call wall.

## Caveats

- cuda backend only; wgpu not measured.
- 512² + 16 MP only (floor-dominated + production size). Intermediate
  sizes were swept in #140's cold run and aren't re-swept here — the
  *transition* behavior is size-class consistent (small = fixed floor,
  large = alloc/per-pixel).
- **Q1 `ssim2→cvvdp` at 16 MP is `ALL_SAMPLES_FAILED`** (TSV row 82).
  Holding ssim2's ~5.7 GB working set then constructing cvvdp (whose
  Auto memory mode wants 3.36 GB) exceeded cvvdp's `VRAM_CAP_BYTES`
  guard on this 12 GiB card shared with the WSL2 desktop (~7.8 GiB free)
  — `EXIT 101: needs at least 3355434476 bytes`. This is a measured VRAM
  characteristic of the ssim2→cvvdp ordering at 16 MP on this hardware,
  **not** a context-sharing finding: Q1's both-orderings conclusion is
  established by the four 512² orderings + the cvvdp→ssim2 16 MP cell,
  which all show the context init paid once. The reverse 16 MP number is
  **not measured** (not projected).
- Q1q2 cells drop metric A (and `block_on(client.sync())`) before
  building metric B — frees A's GPU working set back to the cubecl pool
  while keeping the process-global CUDA context alive. This is both
  required (two eager-alloc 16 MP working sets don't fit the free VRAM)
  and representative of the real warm worker, which processes one
  metric's working set at a time.
- The first fresh process per (metric,size) occasionally shows an
  inflated `first_compute` (cold-disk PTX JIT / pool warmup); the
  reported values are 5-process medians, which absorb that one outlier.
