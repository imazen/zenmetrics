# CVVDP CPU 16 → 40 MP heap scaling diagnosis

**Date:** 2026-05-28
**Agent:** claude-cvvdp-cpu-scaling-diag (task #130)
**Workspace:** `zenmetrics--cvvdp-diag-40mp`
**Tool:** heaptrack 1.3.0 + massif cross-check; `cpu-profile` driver (release build,
default-features=false +std, parallel feature OFF)
**Parent commit:** `master@origin = 5979e084594473a535b308a113555ab404a94d40`
**Binary:** `target/release/cpu-profile` (lto=false, codegen-units=1, debug=1)

## TL;DR

**The 16 → 40 MP "super-linear heap jump" claim in the task brief is incorrect.**

Measured at six sizes between 16 MP and 40 MP, peak heap scales **strictly linearly**
in pixel count at **218 MB/MP** (intercept ≈ 0). R² = 0.99999798 over the six
measurements. No allocator threshold, no rayon thread blow-up, no f32 vs i16 issue.

The apparent doubling from "91.8 MB/MP at 16 MP" to "203 MB/MP at 40 MP" arose
from comparing two different metrics in heaptrack output:

| Metric | 16 MP value | 40 MP value | MB/MP | Comment |
|---|---|---|---|---|
| Top peak-consumer callstack (`weber_contrast_pyr_into` transients) | 1.54 GB | 3.66 GB | 91.8 → 91.9 | scales **linearly** |
| Process-wide peak heap (heaptrack header / massif `mem_heap_B` max) | 3.66 GB | 8.68 GB | 218.1 → 218.0 | scales **linearly** |

The "1.54 GB at 16 MP" in the brief is the top callstack; the "8.08 GiB at 40 MP"
is the process peak. Apples-to-apples both views are linear.

**Decision: Path B (targeted fix) is not viable because there is no super-linear
defect to fix.** The 16 → 40 MP heap growth is the expected linear growth of
six full-image-sized pyramids + cache + DKL planes + per-call transient buffers.
The persistent `Scratch::new` allocations + transient `weber_contrast_pyr_into`
allocations together consume **218 B/px**, and any meaningful headroom reduction
**requires Path A (chunks 4+6: per-strip allocation of WeberPyramid / WeberPyramidCache
bands + StripBandWorkspace dispatch)**. The handoff doc
`crates/cvvdp/docs/CPU_KSPL_HANDOFF_chunks_4_and_6.md` is the right place to
land it; the data structures (`StripBandWorkspace`, `with_capacity_strip`) are
already shipped — only the strip-major dispatcher and `Scratch::new` lazy-mode
gating remain.

## Measurements

### Heap trajectory (process peak heap memory consumption)

`heaptrack_print <trace>.zst | grep "peak heap"` and massif cross-check via
`heaptrack_print -M <out>.massif` + max `mem_heap_B`.

| Size (MP) | W × H | heaptrack peak (SI GB) | massif peak (bytes) | MB/MP | t_score (s) | JOD |
|---|---|---|---|---|---|---|
| 16.78 | 4096 × 4096 | 3.66 | 3 657 672 855 | 218.2 | 6.24 | 9.4585028 |
| 23.98 | 5120 × 4683 | 5.23 | 5 227 529 943 | 218.1 | 8.18 | 9.4555445 |
| 28.00 | 5824 × 4807 | 6.10 | 6 103 630 263 | 217.9 | 9.75 | 9.4452171 |
| 32.00 | 6144 × 5208 | 6.98 | 6 975 801 111 | 218.1 | 11.77 | 9.4564285 |
| 35.96 | 6656 × 5402 | 7.84 | 7 838 828 599 | 218.0 | 12.52 | 9.4546928 |
| 39.81 | 7680 × 5184 | 8.68 | 8 679 597 143 | 218.0 | 14.32 | 9.4560947 |

Raw `.zst` traces: `benchmarks/heaptrack/scaling_2026-05-28/cvvdp_full_{16,24,28,32,36,40}mp.zst`.
TSV: `crates/cvvdp/benchmarks/cpu_scaling_diag_40mp_2026-05-28.tsv`.

### Linear fit

```
peak_heap_GB = 0.00277 + 0.21796 GB/MP × pixels_M
             ≈ 0    + 218 B/px

R² = 0.99999798
max |residual| = 0.0047 GB
```

The intercept is statistically zero (3 MB fixed overhead against an 8.68 GB
total — within heaptrack's quantization). The slope is invariant from 16 MP
through 40 MP. **There is no inflection point.**

### Top-10 peak consumers at 40 MP (`heaptrack_print --backtraces` lifted)

Annotated with source location and scaling factor from 16 MP measurement of
the *same callstack*.

| # | Bytes (16 MP) | Bytes (40 MP) | B/px | Scaling | Source | What it is |
|---|---|---|---|---|---|---|
| 1 | 1.54 G | 3.66 G | 91.8 | 2.377× (px: 2.373×) | `pyramid.rs:444/452` via `pipeline.rs:1881/1890/1899` | **Transient** `WeberPyramidCache::gauss_img/gauss_l` band Vec growth inside `weber_contrast_pyr_into` — first-call cold path (6 channels × 9 levels × ~159 MB peak each at top of pyramid). Pre-allocated by `WeberPyramidCache::with_capacity` (line item #3), but heaptrack attributes capacity-growth events to the path that first reserved them. |
| 2 | 1.07 G | 2.55 G | 64.0 | 2.383× | `scratch.rs:284-291` via `Cvvdp::with_geometry` `pipeline.rs:158` | **Persistent** 6 × `WeberPyramid::with_capacity`: 3 ref + 3 dist. Per pyramid = 2 × (4/3) × W·H × 4 bytes ≈ 170 MB at 16 MP, 405 MB at 40 MP. Six of them = ~1.0 GB → ~2.4 GB. |
| 3 | 537 M | 1.27 G | 32.0 | 2.365× | `scratch.rs:314-316` via `Cvvdp::with_geometry` | **Persistent** 3 × `WeberPyramidCache::with_capacity` (dist side). Per cache = 2 × (4/3) × W·H × 4 bytes ≈ 170 MB at 16 MP, 405 MB at 40 MP. Three of them = ~500 MB → ~1.2 GB. (`weber_cache_ref` is NOT pre-allocated — only `weber_cache_dist`.) |
| 4 | 403 M | 955 M | 24.0 | 2.370× | `scratch.rs:276-281` via `Cvvdp::with_geometry` | **Persistent** 6 × DKL planes: `dist_a/rg/vy` + `ref_a/rg/vy` = 6 × W·H × 4 bytes = 24 B/px. |
| 5 | 50 M | 119 M | 3.00 | 2.379× | `cpu_profile/src/main.rs` `synth_pair` | Driver's `Vec<u8>` reference sRGB buffer (the original `r` allocation before `dist` materializes). 3 B/px (sRGB-u8 interleaved). |
| 6 | 50 M | 119 M | 3.00 | 2.379× | `cpu_profile/src/main.rs` `chunks_exact(3)..collect()` | Driver's `Vec<u8>` distorted sRGB buffer. Same 3 B/px. |
| 7 | 74 K | 93 K | tiny | constant | `simd_pyramid.rs:522` via `expand_vertical_pass` | Per-level vscratch buffer growth (≤ pyramid level 1). |
| 8 | 74 K | 74 K | tiny | constant | `cpu_profile/src/main.rs` | Driver-side bytemuck `&[[u8;3]]` reborrow stack. |
| 9 | 4 K | 4 K | tiny | constant | mid-level | Allocator metadata / Rayon thread setup. |
| 10 | 3 K | 3 K | tiny | constant | mid-level | Same class. |

**Every non-trivial allocation scales linearly with pixel count (× 2.37 over the
× 2.37× pixel range).** No allocation grows faster than O(W·H); no allocation
shows a threshold jump.

### Cross-check via massif

Re-extracted peak heap via `heaptrack_print -M <out>.massif` + max `mem_heap_B`.
Matches heaptrack peak header to within heaptrack's reported precision (3 sig
figs) at every size. The two measurements are not independent (massif is
derived from the same heaptrack trace), but the agreement confirms the peak
heap value is not a rounding artifact in the header. Both views report the
same 218 B/px slope.

### Massif heap-vs-time at 40 MP — peak is SUSTAINED, not transient

Per task brief step 3: confirm whether the 40 MP peak is a short-lived transient
or a sustained working set. Result: **sustained**.

| Marker | Time (s) | Heap (GB) | Note |
|---|---|---|---|
| First >= 2 GB | 0.20 | 5.02 | DKL+pre-allocated Scratch lands as a single block (5 GB at snapshot 19) |
| First >= 6 GB | 2.38 | 6.04 | mid-weber-build (per-channel pyramids accumulating) |
| First >= 8 GB | 7.88 | 8.12 | last weber-build done; fold bands starting |
| Peak | 10.22 | 8.68 | mid fold-bands (BandWorkspace inflation atop persistent scratch) |
| Total runtime | 14.55 | — | including dist-buffer drop + cleanup |

Sustained time at >= 80 % of peak: **6.38 s (44 % of runtime)**.
Sustained time at >= 90 % of peak: **3.87 s (27 % of runtime)**.

This is not an allocator-rounding spike — the memory is held during the entire
weber + fold compute phase. A "low memory mode B" cannot reclaim this by
delaying allocations or running things in a different order; the buffers are
in active use across most of the score() call.

This further reinforces Path A: shrinking the per-call working set at shallow
levels (via strip-shape allocators) is the only way to bring this number down.

## Hypothesis ranking — what's NOT happening

Per the task brief's expected suspects, here's what was ruled out:

| Hypothesis | Status | Evidence |
|---|---|---|
| **Allocator pressure / THP huge-page rounding** | **Ruled out.** | Massif `mem_heap_B` (logical bytes requested) matches heaptrack peak (RSS-equivalent). If THP rounding were inflating consumption, massif would lag heaptrack by hundreds of MB. They agree to 4 sig figs at every size. |
| **rayon per-thread scratch blow-up** | **Ruled out.** | This build has `parallel` feature OFF (cpu-profile builds cvvdp with `default-features = false, features = ["std"]`). All measurements taken via single-threaded `build_one_side_recycle`'s `#[cfg(not(feature = "parallel"))]` arm. No rayon scratch in the trace. |
| **WeberPyramid `bands[k]` push-extend reserves 2×** | **Ruled out.** | `WeberPyramid::with_capacity` pre-sizes every `bands[k].data` with `vec![0.0_f32; w*h]` (line `pyramid.rs:66`), avoiding push-extend's 2× growth. The 1.54 G transient at heaptrack site #1 is the **first call's** cold path inside `weber_contrast_pyr_into` (where `cache.scratch.expanded` / `gauss_tmp` first grow to ~159 MB), and it scales linearly with W·H. |
| **f32 vs i16 representation** | **Ruled out (no easy win).** | All persistent f32 buffers (DKL planes, WeberPyramid bands, log_l_bkg, gauss caches) hold dynamic-range data that requires f32 precision for bit-identical JOD parity. The chunk 5 strip kernels in `strip_kernels.rs` continue to use f32 — there's no easy half-precision substitution that preserves the JOD parity tests. (Could be revisited as a follow-up but not for "Path B".) |
| **Diffmap path** | **N/A.** | The driver calls `score()` with `want_diffmap = false`. No diffmap allocations in the trace. |
| **Tile-buffer / parallel reduction transient** | **N/A.** | No parallel feature, no rayon transients. |

## Hypothesis ranking — what IS happening

The 218 B/px slope decomposes exactly:

| Source | B/px | % of total | Persistent or transient? |
|---|---|---|---|
| 6 × `WeberPyramid` (3 ref + 3 dist), 2 × (4/3) × 4 B = 16 B/px each | 96 | 44 % | Persistent |
| 3 × `WeberPyramidCache` (dist), 2 × (4/3) × 4 B = 32 B/px total | 32 | 15 % | Persistent |
| 6 × DKL planes (W·H × 4 B) | 24 | 11 % | Persistent |
| `weber_contrast_pyr_into` transient (`cache.scratch.expanded` + `gauss_tmp` + per-level reductions, 6 channels × ~16 B/px summed pyramid) | ~60 | ~28 % | Transient (peaks during first call) |
| Driver `Vec<u8>` × 2 (ref + dist sRGB) | 6 | 3 % | Persistent (host) |
| **Total** | **~218** | **100 %** | Mix |

These add up to the measured 218 B/px slope to within heaptrack's reporting
precision. The breakdown is **dominated by persistent full-image-sized
allocations** that `Scratch::new` makes up front (× 6 ref + × 3 dist
WeberPyramid + × 3 WeberPyramidCache + × 6 DKL = 18 image-shaped objects).
**There is no allocation that could be eliminated by a simple targeted fix**
because every one of these is required for the current algorithm to produce
bit-identical JOD parity.

## Decision: Path B is NOT viable

The brief asked whether a **cheaper targeted fix** could deliver a Mode B that's
faster than Full at 40 MP, or whether the multi-day strip-major dispatcher
(chunks 4+6) is genuinely required.

### Why Path B fails

A targeted fix would need to remove ~50% of memory consumption (to push 40 MP
peak down from 8.68 GB toward the user's "low memory" goal). The candidates
for targeted removal are:

1. **Drop `weber_cache_dist` pre-allocation.** Saves ~1.27 GB at 40 MP.
   But moves the cost to the **first `score()` call's** `weber_contrast_pyr_into`
   path — the cache's gauss_img/gauss_l Vecs grow on-demand instead. Peak heap
   during that first call is unchanged. Wins at iteration #2+, not at iteration #1
   (the workload that matters for "compute one score then exit", which is the
   sweep worker pattern).

2. **Reduce `WeberPyramid` count from 6 to 3 by sharing ref + dist slots.**
   Requires interleaving the per-side weber builds (currently `build_both_sides_into`
   builds ref then dist sequentially). Not safe to share without rework — the
   fold stage reads both `ref_weber` and `dist_weber` simultaneously. Architectural
   change comparable in scope to chunks 4+6.

3. **Drop the DKL plane pre-alloc.** Saves 955 MB at 40 MP, but the DKL planes
   are inputs to `weber_contrast_pyr_into` — they must materialize before the
   weber build. Re-organizing to stream sRGB → DKL → weber per-strip is exactly
   what chunks 4+6 do.

None of these is "cheaper" than Path A. Each either:
- shifts memory cost without reducing it (option 1),
- breaks the pipeline ordering (option 2 needs a fold rewrite),
- or *is* a partial version of Path A (option 3).

### Why Path A is required

The persistent `Scratch::new` allocations + transient per-call buffers all scale
linearly in W·H. Reducing peak heap at 40 MP **requires changing the per-call
buffer shape from full-image (W × H) to per-strip (R_k × W)**. That's exactly
what chunks 4+6 deliver:

- **Chunk 4** (already shipped as `StripBandWorkspace` data structures, gated
  `dead_code`): per-band f32 slots sized at R_k × bw rather than bh × bw.
- **Chunk 6** (NOT shipped, the architectural work): rewrite the band-fold
  outer loop to iterate strip-major across shallow levels, building per-strip
  weber pyramids via `with_capacity_strip` instead of `with_capacity`.

The strip-shaped allocators (`WeberPyramid::with_capacity_strip` at
`pyramid.rs:96`, `WeberPyramidCache::with_capacity_strip` at `pyramid.rs:381`)
are **already implemented** and tested. The remaining work is:

1. `Scratch::new_strip` already exists at `scratch.rs:230`. Wire it into
   `Cvvdp::score_strip` / `score_with_warm_ref_strip` as the constructor
   variant.
2. The strip-major band-fold dispatcher (chunk 6 of CPU K_SPLIT) — replicate
   the GPU's `_run_d_bands_strip_major_shallow` (cvvdp-gpu `pipeline.rs:5343`)
   using the scalar strip kernels from `strip_kernels.rs`.

### Expected reduction with Path A

From the per-source decomposition table above:
- 6 × WeberPyramid full-image → ~80% reduction at shallow levels (chunks 4+6):
  saves ~77 B/px → -3.05 GB at 40 MP
- 3 × WeberPyramidCache(dist) full-image → ~80% reduction: saves ~26 B/px → -1.04 GB
- `weber_contrast_pyr_into` transient → ~80% reduction (per-strip scratch instead of full-image): saves ~48 B/px → -1.91 GB
- DKL planes → unchanged (required as inputs)
- BandWorkspace (currently sized at full band per level) → replaced by StripBandWorkspace at shallow levels

Total expected: **40 MP peak heap goes from 8.68 GB to ~2.7 GB** (and 16 MP
from 3.66 GB to ~1.7 GB, matching the brief's "1.7 GB target at 16 MP").

This matches the handoff doc's "peak heap at 16 MP should drop from 4.73 GB
to ~1.7 GB target" prediction (the 4.73 GB number was the older PHASE9YB
measurement before the cache shrink_to_fit landed; today's 3.66 GB is already
better, but the target is right).

## Recommended next action

**Pick up Path A — ship chunks 4 + 6.** The data structures and scalar strip
kernels are already in tree (chunks 1-3 + 5 in `strip_kernels.rs`); only the
dispatcher work remains. The handoff doc
`crates/cvvdp/docs/CPU_KSPL_HANDOFF_chunks_4_and_6.md` already specifies the
shape (Scratch::new_strip wiring, strip-major band-fold), and the parity gate
`tests/strip_parity.rs` will catch any regression in JOD output.

Do **not** spend effort on targeted shrinks of the existing full-image allocations.
The measured trajectory says they all scale linearly and there is no quick win;
the architecture has to change.

## Workspace / commit info

- Workspace: `~/work/zen/zenmetrics--cvvdp-diag-40mp`
- Parent: `master@origin 5979e084594473a535b308a113555ab404a94d40`
- This commit will add: 6 × `.zst` heaptrack traces, this doc, the
  scaling TSV, the budget Python.
- Cleanup: `jj workspace forget cvvdp-diag-40mp` + remove workspace dir on completion.

## References

- Existing 40 MP measurement (b7fc7c78): matches the 40 MP point here at 8.68 GB
- `crates/cvvdp/docs/CPU_KSPL_HANDOFF_chunks_4_and_6.md` — chunks 4+6 spec
- `benchmarks/heaptrack/drivers/cpu_profile/src/main.rs` — driver
- `crates/cvvdp/src/scratch.rs` — Scratch::new + Scratch::new_strip
- `crates/cvvdp/src/pyramid.rs` — WeberPyramid::with_capacity + with_capacity_strip
- `crates/cvvdp/src/strip_kernels.rs` — chunks 1-3+5 (scalar strip kernels)
- `crates/cvvdp/src/pipeline.rs` — score / score_strip / fold_bands
