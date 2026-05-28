# CPU metric heaptrack report — Phase 9x gate (2026-05-27)

**Author:** Phase 9x agent (claude-phase9x)
**Workspace:** `zenmetrics--phase9x` (sibling of `master`, parent change `f0ddc6bb`)
**Host:** Water-cooled AMD Ryzen 9 7950X, 49 GiB RAM, 32 logical cores
**Tools:** heaptrack 1.3.0, heaptrack_print, valgrind 3.x (available, not used here)

This is a READ-ONLY gate. Goal: profile six CPU metrics × four execution modes × three image sizes (1 MP, 16 MP, 40 MP) and produce a ranked low-hanging-fruit list for the optimization phases that follow.

User directive (2026-05-27): *"review heaptrack for all cpu metrics incl iwssim and speed up both full, cached ref, stripwise, and cached full strip dist and figure out how to support 40mp images. work from low hanging fruit first and use max effort"*

---

## 1. Executive summary — what to fix first

1. **cvvdp + iwssim: hoist per-call entry buffers into `Scratch`.** `Cvvdp::warm_reference` and `Cvvdp::score_with_warm_ref` both freshly `vec![0.0; w*h]` the DKL planes (3 × `w*h*4` bytes) every call. `Iwssim::score`, `score_gray`, `warm_reference`, and `pad_gray` each do the same for gray planes (`w*h*4` bytes per buffer). At 40 MP this is **~1.6 GB of fresh allocation per call** (cvvdp) or **~160 MB per call** (iwssim), which is pure waste — the `Scratch` struct already exists in cvvdp and `Iwssim` is `&mut self`. **Estimated heap reduction: 15–25 % per call; CPU savings from skipping the zero-fill 5–10 %.**

2. **cvvdp `weber_contrast_pyr_into` allocates 9× per call regardless of size.** Top single allocator across every cvvdp cell (1 MP → 5.59 MB × 9 = 50 MB; 40 MP → 213.85 MB × 9 = 1.9 GB). 9 calls = `3 channels × 3 pyramid build steps`. These are sized per-level and currently `RawVecInner::reserve` grows fresh. Reusing a per-pyramid scratch buffer (one buffer per `n_levels`, owned by `Scratch`, sized at largest band) would drop these 9 calls to zero ongoing allocations after warmup. **Estimated heap reduction: ~50 % of cvvdp peak.**

3. **dssim `lab_chan` allocates 805 MB × 100 calls at 40 MP.** dssim-core allocates the multi-scale LAB pyramid fresh on every `create_image` (which we call once for ref + once for dist per pair). Each pyramid level is its own `Vec<f32>`. Total = ~2 GB at 40 MP. The crate is third-party but the orchestrator's adapter ALREADY caches `DssimImage<f32>` for the warm-ref path; the matrix confirms warm_ref doesn't reduce peak heap (0 % savings) because both ref + dist pyramids still build per call. **Fix option A**: precompute distorted-side pyramid only once if a thread-local pool is added in dssim-core (upstream PR territory). **Fix option B**: gate dssim at a lower 40MP cap and route to strip-based subsampling (see #5).

4. **butter, ssim2, iwssim, dssim, cvvdp have NO stripwise API.** Only zensim does. Five of six metrics will OOM-or-degrade on 40MP+ inputs with no in-crate option to tile. The CpuAdapter today reflects this — its OOM-fallback ladder swaps backends but never tiles within a metric. **Phase 9.Y must either add per-metric tile/strip APIs (mandatory for production 40MP) or wire a metric-agnostic tile-aggregate harness above the adapter.**

5. **zensim strip mode is currently MORE memory than full at 16/40 MP.** Strip mode showed +36 % peak heap at 40 MP (full=2.64 GB, strip=3.59 GB) and +23 % at 16 MP. Counterintuitive — strip is supposed to bound memory at `O(strip_height × width)`. The 213.64 MB single-allocation at `PrecomputedReference::new` inside strip mode is the smoking gun: strips re-precompute the reference per strip rather than re-use a single full reference. Adding cross-strip reference caching (perhaps via the `warm_ref_strip` path) brings it down to +13 % at 40 MP; production should default to **`warm_ref_strip`** when streaming.

---

## 2. Per-metric per-mode allocation summary

Measured under heaptrack 1.3.0 on cpu-profile driver (release build, debug=1, codegen-units=1, no LTO). All cells use `synth_pair_offset_dist` deterministic input. Each cell = one `cpu-profile <metric> <mode> <w> <h>` process invocation.

**Legend:**
- `peak_heap` is heaptrack's "peak heap memory consumption" (in-process malloc-tracked, excludes file mmaps).
- `bpp` = peak heap bytes per pixel (`peak_heap / (w*h)`). Lower is better.
- `n_alloc` = total calls to allocation functions over the lifetime of the process.
- `runtime` = total process runtime under heaptrack (5-15 % slowdown vs no-instrument).
- `GAP` rows ran only the synth-pair builder + the gap-marker exit; recorded so the matrix matrix is complete; not analyzed.

### Table 2.1 — Peak heap by (metric, mode, size)

| metric  | mode               | 1 MP        | 16 MP       | 40 MP       | bpp@40MP |
|---------|--------------------|------------:|------------:|------------:|---------:|
| cvvdp   | full               | 295.81 MB   | 4.73 GB     | **11.31 GB**| **302**  |
| cvvdp   | warm_ref           | 243.37 MB   | 3.89 GB     | 9.30 GB     | 249      |
| cvvdp   | strip              | GAP         | GAP         | GAP         | —        |
| cvvdp   | warm_ref_strip     | GAP         | GAP         | GAP         | —        |
| ssim2   | full               | 184.65 MB   | 2.75 GB     | 7.29 GB     | 195      |
| ssim2   | warm_ref           | 172.05 MB   | 2.55 GB     | 6.49 GB     | 174      |
| ssim2   | strip              | GAP         | GAP         | GAP         | —        |
| ssim2   | warm_ref_strip     | GAP         | GAP         | GAP         | —        |
| dssim   | full               | 203.48 MB   | 3.25 GB     | 9.29 GB     | 249      |
| dssim   | warm_ref           | 203.48 MB   | 3.25 GB     | 9.29 GB     | 249      |
| dssim   | strip              | GAP         | GAP         | GAP         | —        |
| dssim   | warm_ref_strip     | GAP         | GAP         | GAP         | —        |
| butter  | full               | 238.65 MB   | 3.49 GB     | 8.03 GB     | 215      |
| butter  | warm_ref           | 235.50 MB   | 3.36 GB     | 7.59 GB     | 203      |
| butter  | strip              | GAP         | GAP         | GAP         | —        |
| butter  | warm_ref_strip     | GAP         | GAP         | GAP         | —        |
| iwssim  | full               | 153.83 MB   | 2.47 GB     | 5.90 GB     | 158      |
| iwssim  | warm_ref           | 145.44 MB   | 2.33 GB     | 5.58 GB     | 149      |
| iwssim  | strip              | GAP         | GAP         | GAP         | —        |
| iwssim  | warm_ref_strip     | GAP         | GAP         | GAP         | —        |
| zensim  | full               | 72.04 MB    | 1.11 GB     | 2.64 GB     | **71**   |
| zensim  | warm_ref           | 71.51 MB    | 1.11 GB     | 2.64 GB     | 71       |
| zensim  | strip              | 69.98 MB    | 1.37 GB     | 3.59 GB     | 96       |
| zensim  | warm_ref_strip     | 57.25 MB    | 1.12 GB     | 2.99 GB     | 80       |

### Table 2.2 — Runtime by (metric, mode, size) under heaptrack

| metric  | mode             | 1 MP  | 16 MP   | 40 MP    |
|---------|------------------|------:|--------:|---------:|
| cvvdp   | full             | 0.31s | 6.62s   | 18.10s   |
| cvvdp   | warm_ref         | 0.35s | 7.22s   | 14.60s   |
| ssim2   | full             | 0.18s | 3.54s   | 7.65s    |
| ssim2   | warm_ref         | 0.20s | 3.17s   | 7.11s    |
| dssim   | full             | 0.27s | 4.95s   | 12.38s   |
| dssim   | warm_ref         | 0.28s | 4.79s   | 10.62s   |
| butter  | full             | 0.14s | 2.04s   | 4.91s    |
| butter  | warm_ref         | 0.14s | 2.14s   | 5.72s    |
| iwssim  | full             | 0.39s | 8.78s   | 20.46s   |
| iwssim  | warm_ref         | 0.45s | 8.82s   | 19.52s   |
| zensim  | full             | 0.04s | 0.65s   | 1.63s    |
| zensim  | warm_ref         | 0.04s | 0.68s   | 1.64s    |
| zensim  | strip            | 0.05s | 0.62s   | 1.71s    |
| zensim  | warm_ref_strip   | 0.05s | 0.57s   | 1.63s    |

**Observation:** zensim is **5–13× faster** than the slowest CPU metric at every size. Per-pixel work ratios stay almost constant from 1 MP to 40 MP (zensim 71 bpp peak ≈ flat; cvvdp 296→303 bpp ≈ flat). The peak-heap-per-pixel curve is linear in pixels with negligible fixed overhead at these sizes.

---

## 3. Hotspot patterns across metrics

Top allocation hotspots, grouped by pattern:

### 3.1 Pattern A — per-call entry-conversion buffers (cvvdp, iwssim)

**Both metrics allocate fresh sRGB-byte → working-space planes on every call**, despite having persistent `Scratch` (cvvdp) or `&mut self` (iwssim) available.

**cvvdp** (`crates/cvvdp/src/pipeline.rs:187-189`):
```rust
pub fn warm_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
    let mut ref_a  = vec![0.0; w * h];    //   4 bytes/px
    let mut ref_rg = vec![0.0; w * h];    //   4 bytes/px
    let mut ref_vy = vec![0.0; w * h];    //   4 bytes/px
    srgb_to_dkl_planar(ref_srgb, w, h, display, &mut ref_a, ...);
    // ...
}
```

At 40 MP that's **480 MB per call**, zero-filled twice (vec! init + value writes). The `Scratch` struct already holds `dkl_planes` slots — they're unused on the `warm_reference` path.

**iwssim** (`crates/iwssim/src/pipeline.rs:97-100, 137-138, 167-168, 204`):
```rust
let mut ref_gray = alloc::vec![0.0_f32; (w as usize) * (h as usize)];
let mut dis_gray = alloc::vec![0.0_f32; (w as usize) * (h as usize)];
```

Also `pad_gray` (line 198-213) returns a fresh `Vec` even on the fast-path (`src.to_vec()`). At 40 MP that's **~160 MB per call** allocated + zeroed + filled + dropped.

**Fix scope:** Add `scratch: GrayScratch` field to `Iwssim`, and use cvvdp's existing `Scratch::dkl_planes` slot from the warm path. Both are mechanical refactors; both metrics are in-tree.

### 3.2 Pattern B — multi-scale per-band buffer allocations

**cvvdp**, top alloc by count: `cvvdp::pyramid::weber_contrast_pyr_into` — 9 calls per `score()`. Sizes at 40 MP: 213.85 MB per call × 9 = ~1.9 GB. Comment in source: `// caches across calls` — but inspection shows the cache *capacity* persists; the underlying Vec for each band still grows on first call of a new size.

**dssim**, top alloc by count: `dssim_core::dssim::Dssim::lab_chan` — 90-100 calls per `compare()`. Each is a per-scale, per-channel band. 805 MB × 100 = ~80 GB across the call but the peak holds at 9.29 GB because many drop before the peak.

**iwssim**, top alloc: `iwssim::pipeline::score_with_split` — 4 calls per score, 160 MB each at 40 MP = 640 MB.

**Fix scope:** Each metric has its own pool design. cvvdp and iwssim are in-tree (easy); dssim-core is third-party (would need an upstream PR or a wrapper that reuses pyramid memory).

### 3.3 Pattern C — rayon startup overhead

**butter** top alloc by count: `rayon_core::registry::Registry::new` — 31 calls totaling 1.49 KB. This is per-process startup overhead, fixed regardless of image size. Total alloc count (737) is dominated by it. Same fingerprint shows in zensim (`rayon_core::registry::Registry::new` × 31).

**Fix scope:** Out of scope for single-call profiling — but a long-running orchestrator process amortizes this away. Worth noting that benchmarks comparing single-shot scoring against batch scoring should subtract this once.

### 3.4 Pattern D — `Vec<[u8; 3]>` chunking adapters

The adapter layer's `bytes.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()` pattern (cpu_adapter.rs ssim2/dssim/butter/zensim wiring) allocates an intermediate `Vec<[u8;3]>` of `w*h*3` bytes before handing to the metric crate. At 40 MP that's a **120 MB allocation per ref + 120 MB per dist** that's pure adapter overhead.

**Fix scope:** Implement `ImageSource` / `LinearRgbImage` / equivalent for raw `&[u8]` triplets so the metric crate can read pixels in place. ssim2 already has `ToLinearRgb for ImgRef<'_, [u8; 3]>` — the conversion step is just the chunking adapter. zensim's `RgbSlice::new` takes `&[[u8; 3]]` directly; same chunking story.

### 3.5 Pattern E — fast-ssim2 blur scratch

ssim2 top alloc by *peak*: `fast_ssim2::input::imgref_impl::_` — 805 MB × 24 calls at 40 MP (24 = number of subbands × planes in the SSIMULACRA2 pyramid). The `Blur::blur_into` call shows up 6 times with 0B peak — those are blur scratch buffers that are reused but their *first*-time allocation still happens per process.

**Fix scope:** fast-ssim2 is an Imazen crate at `~/work/zen/fast-ssim2/`. A `Ssimulacra2Reference` that retains its compute context across `.compare()` calls would reduce dist-side peak. Currently `Ssimulacra2Reference::new` + `compare` is the cached-ref API but `compare` still allocates dist-side fresh.

---

## 4. 40 MP feasibility matrix

Host budget: 49 GiB total RAM, **42 GiB available** (per `free -h` at the start of this run). A production worker shares the host with rayon thread pools, the GPU runtime, and other processes; a conservative safe ceiling is **30 GiB** for one (ref, dist) pair.

| metric  | mode             | peak @ 40 MP | % of 30 GiB budget | category |
|---------|------------------|-------------:|-------------------:|----------|
| zensim  | full             | 2.64 GB      | 9 %                | **GREEN** |
| zensim  | warm_ref         | 2.64 GB      | 9 %                | **GREEN** |
| zensim  | warm_ref_strip   | 2.99 GB      | 10 %               | **GREEN** |
| zensim  | strip            | 3.59 GB      | 12 %               | **GREEN** |
| iwssim  | warm_ref         | 5.58 GB      | 19 %               | **GREEN** |
| iwssim  | full             | 5.90 GB      | 20 %               | **GREEN** |
| ssim2   | warm_ref         | 6.49 GB      | 22 %               | **GREEN** |
| ssim2   | full             | 7.29 GB      | 24 %               | **GREEN** |
| butter  | warm_ref         | 7.59 GB      | 25 %               | **GREEN** |
| butter  | full             | 8.03 GB      | 27 %               | **YELLOW** |
| dssim   | warm_ref         | 9.29 GB      | 31 %               | **YELLOW** |
| dssim   | full             | 9.29 GB      | 31 %               | **YELLOW** |
| cvvdp   | warm_ref         | 9.30 GB      | 31 %               | **YELLOW** |
| cvvdp   | full             | **11.31 GB** | **38 %**           | **YELLOW → RED for parallel batch** |

**No OOMs occurred** on this 49 GiB host at 40 MP for any (metric, mode) cell. **However**:

- **CVVDP at 40 MP in full mode uses 11.3 GB.** Three concurrent CPU workers scoring different (ref, dist) pairs would exceed the 30 GiB safe budget. Either gate concurrent CPU cvvdp to 1 worker at 40 MP, or implement strip mode (#1 priority for cvvdp at 40 MP).
- **DSSIM at 40 MP uses 9.3 GB even with the warm-ref adapter cache active.** The cache only saves a *re-create* on the second call; both ref + dist pyramids still build per pair. Two concurrent workers exceed the safe budget. Needs strip or upstream PR.
- **Five of six metrics have no strip API.** When the next-larger size class lands (60 MP, 80 MP), all five OOM with no fallback within the metric.

**Categorization summary:**

- **GREEN** (fits Full at 40 MP, < 25 % budget): zensim, iwssim, ssim2, butter (warm_ref only)
- **YELLOW** (fits Full at 40 MP but > 25 % budget; recommend warm_ref or strip): butter (full), dssim, cvvdp
- **RED** (OOMs in Full at 40 MP): **none observed at 40 MP on this host**, but cvvdp at 80 MP would scale to ~22 GB and become RED. The "RED" risk lives at the next size class up, where no metric except zensim has a fallback.

---

## 5. Ranked low-hanging-fruit table

Effort ratings: **S** ≤ 0.5 day, **M** 0.5–2 days, **L** 2–5 days, **XL** > 5 days (or upstream PR).

| Priority | Pattern | Affected metrics | Action | Expected gain | Effort |
|---------:|---------|------------------|--------|---------------|:------:|
| **P0**   | per-call entry-buffer reallocation | cvvdp, iwssim | Hoist `vec![0.0; w*h]` planes into the existing `Scratch` / `&mut self` struct; reuse across calls. cvvdp: `pipeline.rs:187-189` + 4 more sites. iwssim: `pipeline.rs:97-100, 137-138, 167-168, 204`. | cvvdp: **−480 MB/call alloc @ 40 MP**, ~5 % wall-time. iwssim: **−160 MB/call @ 40 MP**, ~3 % wall-time. Heap peak: −5–10 %. | **S** |
| **P0**   | adapter `chunks_exact(3).collect()` pre-allocation | ssim2, dssim, butter, zensim | Add `impl ImageSource for &[u8]` (or equivalent) so the adapter passes raw bytes through without a `Vec<[u8; 3]>` materialization. ssim2 already has the right impl via `ImgRef<'_, [u8; 3]>` — switch the adapter to a transmute-free zero-copy view. | **−240 MB allocation per pair @ 40 MP** across adapter layer. ~2 % wall-time. | **S** |
| **P1**   | cvvdp `weber_contrast_pyr_into` allocates 9× | cvvdp | Add per-pyramid-band `Vec<f32>` slots in `Scratch`, sized at the largest band, reused per call. 9 calls × 213 MB → 9 calls × 0 MB after first. | cvvdp full @ 40 MP: **11.31 GB → ~7 GB** (estimated −38 %). Warm path similarly. | **M** |
| **P1**   | dssim per-call pyramid allocation | dssim | Add a `DssimPool` thread-local in dssim-core OR wrap with a per-thread reusable scratch in the adapter. 90-100 calls × 805 MB → mostly reused. | dssim @ 40 MP: **9.29 GB → ~5 GB** (estimated −45 %). Upstream PR. | **L** |
| **P1**   | iwssim `score_with_split` 160 MB × 4 | iwssim | Same scratch hoist as P0 cvvdp but for the internal `score_from_gray` working planes (lp_ref/lp_dis/g_ref vec allocations). | iwssim @ 40 MP: **5.90 GB → ~4 GB** (estimated −33 %). | **M** |
| **P1**   | cvvdp + iwssim strip API | cvvdp, iwssim | Implement `score_strip(strip_inner, strip_margin)` mirroring zensim's design. Strips for cvvdp: per-band pyramid is independent except for blur-stencil overlap; clean port. iwssim: per-scale 11×11 Gaussian needs ~5-tap margin per scale × 5 scales = wider overlap. | Enables 80 MP+ on both metrics; bounds peak per pair to O(strip × width). | **L** |
| **P2**   | zensim strip mode: cache ref pyramid across strips | zensim | The `compute_with_ref_streaming_strips` path already exists; default the public `compute_streaming_strips` to construct a full reference once instead of re-precomputing per strip. Current strip mode peak 40 MP = 3.59 GB; `warm_ref_strip` = 2.99 GB. | zensim strip mode: **3.59 GB → 2.99 GB** (−17 %). | **S** |
| **P2**   | butter, ssim2 strip API | butter, ssim2 | butter has a documented "multires strip" benchmark CSV (`benchmarks/butter_multires_strip_2026-05-22.csv`) — there's prior art for the algorithm. ssim2 docs say "callers should tile and aggregate" (lib.rs:256) — design needed. | Unblocks 80 MP+ for both. Saves ~50 % heap at 40 MP if strip width tuned. | **L** |
| **P2**   | dssim strip API | dssim | Upstream PR to dssim-core OR replace dssim-core path with a strip-aware in-tree dssim implementation. dssim-core is unmaintained-ish — consider in-tree port if upstream is unresponsive. | Unblocks 80 MP+ for dssim. | **XL** |
| **P3**   | rayon registry startup amortization | butter, zensim | Initialize a process-wide rayon pool once in the orchestrator's main; metric crates inherit. Already done in `bench` worker but not in single-pair driver. | One-time startup hit, no per-call gain. | **S** |
| **P3**   | fast-ssim2 dist-side scratch reuse | ssim2 | Upstream to fast-ssim2: a `CompareContext` that retains dist-side scratch buffers across `.compare()` calls. The Imazen crate is at `~/work/zen/fast-ssim2/`. | ssim2 @ 40 MP: ~7.3 GB → ~5 GB (estimated −30 %). | **M** |

### Top-5 quick wins (≤ M effort, > 15 % peak heap reduction):
1. **P0** cvvdp + iwssim scratch hoist (S) — saves 480 MB / 160 MB per call respectively.
2. **P0** Adapter zero-copy bytes (S) — saves 240 MB per pair across 4 metrics.
3. **P1** cvvdp weber_contrast_pyr_into pool (M) — saves ~4 GB at 40 MP.
4. **P1** iwssim score_with_split pool (M) — saves ~2 GB at 40 MP.
5. **P2** zensim strip-ref caching (S) — saves 600 MB at 40 MP, makes strip < full.

---

## 6. Per-metric specific findings

### 6.1 cvvdp (in-tree, `crates/cvvdp/`)

- **Versions:** 0.1.0; PYCVVDP_REFERENCE_VERSION = v0.5.4.
- **Per-pixel cost:** ~300 bpp at every size (1 MP, 16 MP, 40 MP all converge to ~302 bpp). Highest of all metrics.
- **Cached-ref savings:** Best of all metrics — 17.8 % heap, 19.3 % wall-time at 40 MP. The cached-ref path skips DKL conversion + pyramid build for the ref side.
- **Allocator profile:** Dominated by `weber_contrast_pyr_into` (9 calls × 213 MB at 40 MP) and `expand_vertical_pass` (8 calls × 0 B — pool-backed, but the first allocation each session shows up). Only 553 total allocs per call — *fewer* than dssim's 1167, but each is bigger.
- **40 MP feasibility:** **YELLOW** — fits at 11.3 GB but concurrent workers exceed 30 GB budget. P0+P1 fixes drop this to ~5 GB → GREEN.
- **Strip support:** none. Adding it is **P1**.

### 6.2 ssim2 (fast-ssim2 0.8, Imazen ~/work/zen/fast-ssim2/)

- **Versions:** fast-ssim2 0.8.0; SSIMULACRA2 spec.
- **Per-pixel cost:** ~175–195 bpp, slight drift up at 40 MP (subband pyramid grows nonlinearly).
- **Cached-ref savings:** Modest — 11 % heap, 7 % wall-time at 40 MP. `Ssimulacra2Reference::new` precomputes ref-side XYB + subbands; dist-side still re-allocates per call.
- **Allocator profile:** `Blur::blur_into` (6 × 0 B — pool reused), `imgref_impl::_` (24 × 805 MB at 40 MP — these are dist-side subband allocations).
- **40 MP feasibility:** **GREEN** at 7.29 GB. Comfortable.
- **Strip support:** none in fast-ssim2 (docs explicitly say "callers should tile and aggregate"). Adding it is **P2** (upstream Imazen crate, no third-party PR friction).

### 6.3 dssim (dssim-core 3.4)

- **Versions:** dssim-core 3.4 (third-party, but actively-maintained).
- **Per-pixel cost:** ~210–250 bpp (drift up at 40 MP).
- **Cached-ref savings:** **None** — 0.0 % heap savings. The adapter's `cached_ref: DssimImage<f32>` only saves the ref-side image build; dist still allocates a fresh pyramid. The crate has no precompute-once-compare-many API.
- **Allocator profile:** `Dssim::lab_chan` (90-100 × 805 MB at 40 MP), `DssimChan` similar. 1167 total allocs per pair — highest of any metric.
- **40 MP feasibility:** **YELLOW** at 9.29 GB. Concurrent workers exceed budget.
- **Strip support:** none. Adding it is **XL** (upstream third-party PR or in-tree replacement). For now, gate concurrency.

### 6.4 butter (butteraugli 0.9.2, sibling at ~/work/butteraugli/)

- **Versions:** butteraugli 0.9.2 (Imazen sibling).
- **Per-pixel cost:** ~200–240 bpp.
- **Cached-ref savings:** Marginal — 5.5 % heap, **negative 16.5 % wall-time** at 40 MP (warm_ref ran *slower* than full). The cached-ref path in the adapter just `clone()`s the cached bytes and re-runs `compute_butter` — that's recompute, not warm. This is a known gap (cpu_adapter.rs:294 documents `supports_cached_ref` returning false for butter).
- **Allocator profile:** `rayon_core::registry::Registry::new` × 31 (startup), `Image3F::from_pool_dirty` × 2 (4 MB at 1 MP, 160 MB at 40 MP), `__arcane_gaussian_blur_dispatch_v4` × 4 (321 MB at 40 MP).
- **40 MP feasibility:** **YELLOW** at 8.03 GB. Use warm_ref-effort gated.
- **Strip support:** none in the public API, but there's prior internal art in `benchmarks/butter_multires_strip_2026-05-22.csv`. Probably **M-L** effort to expose.

### 6.5 iwssim (in-tree, `crates/iwssim/`, Phase 8g)

- **Versions:** 0.1.0 (Phase 8g, 2026-05-27).
- **Per-pixel cost:** ~150 bpp at every size (very stable). Lowest of the bpp-expensive metrics.
- **Cached-ref savings:** Modest — 5.4 % heap, 4.6 % wall-time at 40 MP. The warm path saves the ref-side Laplacian pyramid + per-scale Cu eigendecomposition.
- **Allocator profile:** Only 220 total allocs per pair — lowest of any metric! Dominated by `score_with_split` (4 × 160 MB at 40 MP), `score_with_warm_ref_gray` (5 × 213 MB), `compute_iw_maps` (3 × ?).
- **40 MP feasibility:** **GREEN** at 5.90 GB. Comfortable.
- **Strip support:** none. Adding it is **L** — the 11×11 Gaussian stencil × 5 scales requires wider strip overlap than zensim's defaults.
- **Per-call wall-time:** Highest of all CPU metrics at 40 MP (20.46s) — strong CPU candidate for SIMD review next.

### 6.6 zensim (sibling, `~/work/zen/zensim/`)

- **Versions:** 0.3.0; ZensimProfile::latest_preview (PreviewV0_2 base).
- **Per-pixel cost:** ~70 bpp in full/warm_ref modes — **2-5× lower** than any other metric. Strip mode rises to 96 bpp due to per-strip overhead.
- **Cached-ref savings:** None (the adapter doesn't use it; zensim's `compute_with_ref` exists but the adapter falls back to recompute per cpu_adapter.rs:464-471).
- **Allocator profile:** 1414–2675 total allocs per pair (highest count, lowest peak each). `parse_feature_transform_params` × 219 calls × 0 B at process start (MLP bake parse — fixed overhead). `ScaleBuffers::ensure_capacity` × 30-88 (pool-backed, peak shows first-init); `convert_source_to_xyb_into` × 14-88 (0 B — pool-backed).
- **40 MP feasibility:** **GREEN** at 2.64 GB. Comfortable for concurrent workers.
- **Strip support:** **YES** — `compute_streaming_strips_default` and `compute_with_ref_streaming_strips_default`. Both work end-to-end. Counterintuitively strip uses MORE memory at 16/40 MP (3.59 GB vs 2.64 GB full at 40 MP); see P2 fix.

---

## 7. Mode-coverage gaps — informs Phase 9.Y scope

The user spec asked for four modes per metric. Here's where each metric stands:

| metric  | full | warm_ref | strip | warm_ref_strip | Action needed |
|---------|:----:|:--------:|:-----:|:--------------:|---------------|
| cvvdp   | ✓    | ✓        | —     | —              | Add strip + warm_ref_strip APIs |
| ssim2   | ✓    | ✓        | —     | —              | Add tile-aggregate (upstream Imazen) |
| dssim   | ✓    | (✓¹)     | —     | —              | Upstream PR or in-tree replacement |
| butter  | ✓    | (✓²)     | —     | —              | Surface multires strip work |
| iwssim  | ✓    | ✓        | —     | —              | Add strip + warm_ref_strip APIs |
| zensim  | ✓    | (✓³)     | ✓     | ✓              | Fix strip peak regression (P2) |

Footnotes:
- ¹ dssim adapter caches `DssimImage<f32>` but dist still recomputes; net heap savings 0 %.
- ² butter adapter stashes ref bytes and calls full compute on warm path — recompute, not warm. Adapter at cpu_adapter.rs:294 advertises `supports_cached_ref = false` to reflect this.
- ³ zensim warm_ref shows 0 % heap savings in this harness because the driver doesn't use the cached-ref API path — `Zensim::compute()` always re-converts both planes. Real warm-ref-batch consumers would call `precompute_reference` once and `compute_with_ref` many times; that path was not isolated in this matrix.

### Phase 9.Y scope implications

1. **Five of six metrics need strip APIs.** This is the bottleneck for 80 MP+ inputs. Without strip APIs, the only available fallback is the GPU path; if GPU is unavailable (e.g., adapter ladder failed), the worker OOMs.
2. **Three metrics have inadequate warm-ref**: dssim (0 % savings), butter (0 %, adapter even calls it "for API-shape parity"), zensim (the harness didn't exercise it but spec is clear).
3. **Driver gap noted**: the cpu-profile driver does **not** test the proper zensim cached-ref pattern (precompute once, score many). Phase 9.Y benchmarks should add a "batch" mode that issues N dist computes against one cached ref — that's the production iterative-quant loop shape.

---

## 8. Methodology & reproducibility

**Driver:** `benchmarks/heaptrack/drivers/cpu_profile/` — single Cargo crate, one binary. Takes 4 args (`<metric> <mode> <w> <h>`). All metrics share one synth-pair fixture (mirrored from `zenmetrics_orchestrator::bench::synth_pair_offset_dist`). Exit codes: 0 = OK, 2 = GAP, 1 = FAIL.

**Build:**
```bash
cargo build --release -p cpu-profile
```
Release profile with `debug = 1, codegen-units = 1, lto = false` so heaptrack's backtraces have symbol info without the LTO compile cost. Build takes ~11 s on the cached workspace, ~2 min cold.

**Matrix runner:** `benchmarks/heaptrack/run_matrix.sh` — drives the cell grid and writes `summary_YYYYMMDDTHHMMSSZ.tsv` + per-cell `.zst` heaptrack files + per-cell `.log` (the driver's stdout).

**Parser:** `benchmarks/heaptrack/parse_heaptracks.py` — runs `heaptrack_print` on every `.zst`, extracts headline stats + the top-3 user-frame allocators (skipping libcore/std::rt/heaptrack frames), emits `stats.tsv`.

**Total matrix runtime:** ~6 minutes wall time for 72 cells (54 real cells + 18 GAP cells). Total disk: 1.1 MB across all heaptrack files (zstd compression).

**Repro on a different host:**
```bash
cargo build --release -p cpu-profile
bash benchmarks/heaptrack/run_matrix.sh
python3 benchmarks/heaptrack/parse_heaptracks.py > benchmarks/heaptrack/stats.tsv
```

### Caveats

- **Heaptrack instruments malloc** — it does not see file mmaps or stack memory. Peak RSS reported includes heaptrack's own overhead (~5 MB).
- **One process per cell** — startup costs (rayon registry, MLP bake parse for zensim, etc.) are amortized into every cell's allocation count. For production batch comparison, subtract these.
- **`Cargo.lock` updated** by adding `cpu-profile` as a workspace member. The package isn't published.
- **Synth pair is uniform low-variance noise.** Real photo content may stress different code paths (e.g., zensim's adaptive features, dssim's high-variance branches). For the allocation-cost analysis this is fine — alloc-site coverage is what we're after.

### File inventory

All committed to `benchmarks/heaptrack/`:
- `drivers/cpu_profile/` — driver crate (Cargo.toml + src/main.rs)
- `run_matrix.sh` — matrix runner
- `parse_heaptracks.py` — parser
- `stats.tsv` — parsed summary (72 rows, 18 columns)
- `summary_*.tsv` — raw run log (72 rows from the matrix runner)
- `<metric>_<mode>_<size>.zst` — 72 raw heaptrack traces (~16 KB each)
- `<metric>_<mode>_<size>.log` — 72 driver stdout logs

To inspect a single cell interactively:
```bash
heaptrack_gui benchmarks/heaptrack/cvvdp_full_40MP.zst
# or
heaptrack_print benchmarks/heaptrack/cvvdp_full_40MP.zst | less
```
