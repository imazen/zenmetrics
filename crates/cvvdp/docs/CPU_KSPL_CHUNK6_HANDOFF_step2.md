# cvvdp CPU K_SPLIT chunk 6 — Step 2 Handoff (2026-05-28)

**Status:** chunks 1, 2, 3, 4 (wiring), 5 SHIPPED. Chunk 6 step 1 + 2 SHIPPED. Chunk 6 step 3 (dispatcher) NOT yet shipped.

## What shipped in step 1 + 2 (this session)

Two commits on master:

- `4e6195c2` — `phase9zf chunk 6 step 1: WeberPyramid::with_capacity_strip + WeberPyramidCache::with_capacity_strip`
- `3bd25014` — `phase9zf chunk 6 step 2: Scratch::new_strip constructor for strip-mode allocations`

These add:

- `WeberPyramid::with_capacity_strip(sw, sh, n_levels, h_body)` in `crates/cvvdp/src/pyramid.rs`
- `WeberPyramidCache::with_capacity_strip(sw, sh, n_levels, h_body)` in `crates/cvvdp/src/pyramid.rs`
- `Scratch::new_strip(width, height, n_levels, h_body)` in `crates/cvvdp/src/scratch.rs`

All three are `#[allow(dead_code)]` until step 3 (the dispatcher) wires them in.

### Why no memory win YET

The current `weber_contrast_pyr_into` build path writes full-image-shape outputs via
`out.bands[k].data.resize(n_px, 0.0)` (pyramid.rs:482). Calling `Scratch::new_strip` and
then `score()` would resize the strip-shape buffers back to full-image on the first
build call, defeating the optimization.

**Memory wins land when step 3 (the strip-major dispatcher) calls a strip-aware build
path that does NOT resize-up.**

## Critical measurement: chunk 4 wiring already did most of the work

Heaptrack measurements taken in this session (driver: `target/release/cpu-profile cvvdp <mode> <w> <h>`):

| Size | Mode | Peak heap | Brief target | Gap |
|---|---|---|---|---|
| 4 MP | full | 743.84 MB | — | — |
| 4 MP | strip | **479.59 MB** | 420 MB | 60 MB |
| 16 MP | full | 2.98 GB | — | — |
| 16 MP | strip | **1.73 GB** | 1.7 GB | 30 MB |

Compare to the brief's stated baseline (pre-chunk-4-wiring):

> Heaptrack measured 4.73 GB peak [at 16 MP] suggests another ~2.8 GB is transient

The chunk 4 wiring **already cut 3 GB at 16 MP** by routing shallow-level CSF + masking
through `process_shallow_strip_band` with `StripBandWorkspace` (which uses `R_k × bw`
slots instead of `bh × bw`).

The remaining gap to the brief targets is small (~30-60 MB) and comes from the
persistent allocations in `Scratch::new`:

- `WeberPyramid::with_capacity` (6 instances): **1.07 GB peak at 16 MP**
- `WeberPyramidCache::with_capacity` (3 instances): **537 MB peak at 16 MP**

Step 3's strip-major dispatcher can shrink these via the strip-shape constructors
already in place.

### Expected peak heap after step 3 (estimated)

Strip-shape compression ratio at `h_body = 512, n_levels = 9`:

- Total shallow rows (full): 4096 + 2048 + 1024 + 512 + 256 + 128 = 8064
- Total shallow rows (strip): 1148 + 572 + 284 + 140 + 68 + 32 = 2244
- Compression: ~28% of full → savings ~72% at shallow levels

Estimated savings at 16 MP:

- WeberPyramid: 1.07 GB × ~72% = ~770 MB saved
- WeberPyramidCache: 537 MB × ~72% = ~390 MB saved
- Total: ~1.16 GB saved

Projected 16 MP strip peak after step 3: **~570 MB** (well below 1.7 GB target).

Projected 4 MP strip peak: **~190 MB** (well below 420 MB target).

## Step 3: the dispatcher work (NOT shipped)

### Required changes

1. **Strip-major outer dispatcher**: replace `fold_bands_parallel`'s shallow-band processing
   with strip-major outer iteration. Currently shallow processing is band-major-outer
   (one `process_shallow_strip_band` call per band, which iterates strips internally).

2. **Per-(s, k) weber band builder**: a new function that builds the weber pyramid band
   at level `k` for one strip `s`, writing into the strip-shape persistent buffer at
   `weber_dist[c].bands[k].data` (sized `bw × R_k`). Uses the chunk-5 strip kernels:
   - `downscale_strip_into` for the gauss pyramid (built fresh per strip from level-0 strip)
   - `upscale_v_strip_into` + `upscale_h_strip_into` for the expand chain
   - `subtract_weber_3ch_strip_into` for the contrast computation

3. **Per-(s, k) CSF + masking + pool**: feed the strip-shape weber band through the
   existing CSF + masking + pool chain. The CSF + masking helpers are already
   strip-degenerate (see AUDIT_2026-05-28.md §A rows 8-11).

4. **Deep levels stay full-image**: for `k >= k_split`, the existing
   `build_one_side_recycle` + level-major fold path applies as today.

### Reference implementation

GPU dispatcher at `crates/cvvdp-gpu/src/pipeline.rs:5343` —
`_run_d_bands_strip_major_shallow`. Read the strip-major outer loop carefully; the
CPU port should follow the same shape but using scalar strip kernels from chunk 5
instead of cubecl dispatches.

GPU per-strip weber builder: `_dispatch_dist_weber_csf_strip_s_for_level` and
`_dispatch_ref_weber_strip_s_for_level` at `cvvdp-gpu/src/pipeline.rs:5764` and
`:6144` respectively.

### Acceptance gate

1. **Strip mode peak heap at 16 MP ≤ 1.7 GB** (currently 1.73 GB; needs only
   ~30 MB improvement, but step 3 should drop it to ~570 MB)
2. **Strip mode peak heap at 4 MP ≤ 420 MB** (currently 479 MB; step 3 → ~190 MB)
3. **270-cell strip_parity grid: bit-identical pass** (no regression)
4. **Wall regression < 22% vs Full**

### Files to touch

- `crates/cvvdp/src/pipeline.rs` — new `score_internal_strip_major` method, new
  `fold_bands_strip_major` dispatcher, integration with `score_strip`
- `crates/cvvdp/src/pyramid.rs` — new `weber_contrast_pyr_strip_into` helper that
  writes strip-shape weber band data per (s, k)
- `crates/cvvdp/src/scratch.rs` — wire `new_strip` into `Cvvdp::new_strip_mode` (already
  exists as `new_strip` constructor; need a `Cvvdp::new_strip_mode` entry point)

### Risk

- **Bit-identical parity** is the load-bearing invariant. The strip-major outer loop
  changes iteration order; the pool accumulator must see strips in the same row-order
  sequence as the band-major outer (current chunk 4 wiring) for bit-identical f32 add
  ordering. This is true because at each level k, strip s = 0, 1, 2, ... walks the
  band in row order regardless of outer loop nesting — but the per-strip weber build
  must produce bit-identical band data vs the full-image build sliced to the strip's
  window.
- **The chunk-5 kernels are individually parity-tested** (12 bit-identical tests in
  `strip_kernels.rs`), so the per-(s, k) weber build should produce bit-identical
  band data per strip when the kernels are wired correctly. The risk is in the
  wiring (especially the gauss pyramid sourcing — the strip walker reads from full
  gauss with strip windowing).

## Why ship step 1 + 2 separately

Step 1 + 2 add infrastructure that future sessions can verify, document, and build
on without risk. Step 3 carries bit-identical parity risk and is best landed in a
focused session with the full 270-cell parity grid running on every iteration.

Step 1 + 2's value:
- Establishes the strip-shape allocator API
- Documents the size table via `mode_b_strip_h_at_level` integration
- Provides the `Scratch::new_strip` constructor as a wire-up point

When step 3 lands, the changes should be:
- Add `Cvvdp::new_strip_mode(w, h, h_body, params)` that calls `Scratch::new_strip`
- Modify `score_strip` to detect strip-mode-allocated `Cvvdp` instances and route
  through the new dispatcher
- Add the per-(s, k) builder + dispatcher orchestration

## Heaptrack baseline data (preserved for delta comparisons)

Raw heaptrack files (NOT committed; in /tmp at session end):

- `/tmp/cvvdp_strip_4mp_baseline.zst` — strip mode, 2048×2048, 479.59 MB peak
- `/tmp/cvvdp_strip_16mp_baseline.zst` — strip mode, 4096×4096, 1.73 GB peak
- `/tmp/cvvdp_full_4mp_baseline.zst` — full mode, 2048×2048, 743.84 MB peak
- `/tmp/cvvdp_full_16mp_baseline.zst` — full mode, 4096×4096, 2.98 GB peak

Step 3's measurements should compare against these as the post-chunk-4-wiring,
pre-dispatcher baseline.

## Strip-parity tests

`crates/cvvdp/tests/strip_parity.rs` continues to pass bit-identical on master
HEAD after step 1 + 2 (verified):

```
running 4 tests
test strip_walker_dispatches_n_strips_at_default_size ... ok
test strip_jod_invariant_across_h_body_at_seed_0 ... ok
test strip_parity_default_grid_cold ... ok
test strip_parity_default_grid_warm ... ok
```

180-cell default grid (90 cold + 90 warm). The 270-cell big grid (`cvvdp-strip-parity-big`
feature) is gated for CI; should be run on step 3 landing.
