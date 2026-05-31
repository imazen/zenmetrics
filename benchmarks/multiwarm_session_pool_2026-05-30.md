# Multi-warm session pool — measured perf (task #155, 2026-05-30)

Data: [`multiwarm_session_pool_2026-05-30.tsv`](./multiwarm_session_pool_2026-05-30.tsv).
Host: 7950X / RTX 5070 (12 GiB), CUDA 13.2. Workload: 6 refs × 8 dists = 48 tasks,
round-robin (FIFO, no sort) so consecutive tasks switch reference. Single-warm caches
1 reference (in-place `set_reference` reuse); multi-warm caches up to `max_entries`
(budget-bounded), evicting LRU.

| size | metric | single | multi | speedup | multi set_ref |
|------|--------|--------|-------|---------|---------------|
| 256² | cvvdp | 292.9 ms | 225.0 ms | **1.30×** | 6 |
| 1024² | cvvdp | 684.2 ms | 709.4 ms | 0.97× | 6 |
| 4096² | cvvdp | 82.4 s | 193.2 s | **0.43×** | **48** |
| 256² | ssim2 | 458.9 ms | 311.4 ms | **1.47×** | 6 |
| 1024² | ssim2 | 718.6 ms | 1345 ms | 0.53× | 6 |
| 4096² | ssim2 | 25.8 s | 326.3 s | **0.08×** | **48** |

## Verdict: a conditional win, not a uniform one

The multi-warm pool is a **strict win only when the batch's distinct-reference working
set fits the VRAM budget** — small/medium images, where all 6 refs stay warm and each is
precomputed once (`set_ref = 6`). There it lands **+1.30–1.47×** at 256².

At 4096² it **regresses 2.3× (cvvdp) to 12.6× (ssim2)**. The `set_ref` column is the
tell: at 4096² each warm entry is GiB-scale, the auto budget holds ~1 entry, and the
round-robin reference order means the LRU evicts the entry you're about to need again →
**every task misses → rebuild** (`set_ref = 48`, one per task). And a multi-warm miss is
expensive: evict = `memory_cleanup()` + stream `sync()` + driver free, then driver
re-allocate + rebuild + re-precompute. Single-warm, by contrast, re-runs `set_reference`
**on the same warm instance in place** — no teardown, the pool stays resident — so under
thrash it is far cheaper.

This is classic cache thrashing, amplified by per-entry teardown cost.

## Consequence

`PoolConfig::multiwarm_session_pool` is therefore shipped **default OFF (opt-in)** — a
naive always-on multi-warm would regress large-image interleaved workloads out of the box.

## Follow-up: capacity / thrash guard

To make multi-warm a safe always-on default, the lane must route oversized working sets
back to the single-warm in-place path. Options:
- **Capacity gate:** if `budget_mib / est_entry_mib < 2` (the pool can't hold a useful
  working set), use single-warm. Cheap, static, catches the 4096² case directly.
- **Thrash detector:** track eviction rate; once evictions-per-task crosses a threshold,
  disable the pool for the remainder of the run and fall back to single-warm.
- **Working-set awareness:** when the batch is sortable (Phase 7.6), the orchestrator
  already knows the distinct-reference count; only enable multi-warm if that count fits.

The "fall back to 1 entry" branch already in `WarmSessionPool` is **not** sufficient — a
1-entry pool still evicts+rebuilds (full teardown) every task, which is exactly the
expensive path. The fallback must hand off to the single-warm `ExecMetric` (in-place
reuse), not shrink the pool to 1.
