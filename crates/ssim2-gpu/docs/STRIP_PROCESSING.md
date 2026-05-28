# Strip processing for ssim2-gpu — design

## Status

**Phase 1 (aliasing): SHIPPED 2026-05-22.** Replaced 30 per-scale plane
buffers (`*_v` and `*_t` for {sigma11, sigma22, sigma12, mu1, mu2}, 5
plane names × 3 channels × 2 orientations) with a single
`(v_scratch, t_scratch)` pair per channel. Reuse safe because each
two-pass blur sequentially writes `v_scratch` then reads it via
transpose then writes `t_scratch` then reads it via the second blur —
in-order GPU launches mean a later blur's overwrite of `v_scratch` cannot
race the earlier blur's transpose-read. The batched pipeline
(`pipeline_batch.rs::BatchScale`) already used this idiom; Phase 1
brought the unbatched pipeline in line.

Saving: 24 plane handles per scale, ~30% reduction in the variable-cost
intermediates. Across 6 pyramid scales the working set drops from
~10.4 GB to ~7.3 GB at 24 MP.

**Phase 2 (strip processing): SHIPPED 2026-05-22.** `Ssim2::new_strip`
+ `Ssim2::compute_stripped` ship the per-strip allocation + driver
described below. Measured at 24 MP (6000×4000) via
`examples/bench_strip_vs_whole.rs`: working set 2.87 GB (strip,
body=1024) vs 7.49 GB (whole) — a 62% reduction. Whole-image OOMs the
default 8 GB cap at 24 MP; strip mode fits with ~5 GB headroom.

Per-call wall time at 12 MP: 30.7 ms whole vs 52.1 ms strip (1.7×
overhead). At 24 MP strip is ~100 ms (whole skip-OOM). The overhead is
the per-strip halo recompute + extra reduction launches; it's the
price for the memory bound.

## Why

`Ssim2::new(client, w, h)` pre-allocates 57 working f32 buffers per
scale × 6 scales (post-Phase-1). For a 1024² image at scale 0 that's
228 MB; for a 24 MP image (e.g. 6000×4000) it's ~5.5 GB at scale 0
alone, ~7.3 GB across all scales. A production sweep worker running
SSIM2 + IW-SSIM + zensim-WithIw + butteraugli + cvvdp concurrently at
24 MP would need 30+ GB of working-set VRAM, overflowing 24 GB
consumer GPUs and crowding RTX 5090-class.

Strip processing bounds peak memory to a function of strip size, not
image size. With 1024-row strips on a 6000×4000 image, working set
stays at ~1.5 GB regardless of total height.

## Stencil reach budget

The cumulative reach at the finest scale, walking through the pipeline:

| Stage | Per-axis radius (at its own scale) | Cumulative at scale 0 |
|---|---|---|
| `downscale_2x` LP downsample (scale s → s+1) | ~1 | 1 × 2^s |
| `blur_pass` IIR Gaussian, radius N=4 (scale s) | 4 | 4 × 2^s |

Worst case is scale 5 (coarsest in the 6-level pyramid):
- `blur_pass` at scale 5 needs ±4 rows at scale 5 = ±128 rows at scale 0
- LP build from scale 0 to scale 5 cascades downscale reflections —
  each cascade adds ±1 row at the input scale, doubled at finer scales

Computing the full cumulative reach symbolically:

```
H(s)   = halo at scale s in scale-0 rows
H(0)   = 4                         # blur_pass at scale 0
H(s)   = 2^s × (4 + 1)             # blur_pass + LP-build reach at scale s
H_max  = max over s ∈ {0..5}
       = max(4, 10, 20, 40, 80, 160)
       = 160
```

**Total finest-resolution halo: ~160 rows per side.** Round up to
**256 rows per side** for safety + alignment with iwssim's halo budget.

Note: ssim2-gpu has no cross-scale read (unlike iwssim's `parent_band`
that reads from the coarser scale), so the halo is purely the blur
and downscale stencil — about 35% smaller than iwssim's.

## Strip sizing

Per-strip budget at 24 MP (image 6000×4000), with H_strip = body rows + halo:

| H_body | H_strip (body + 2× halo) | strips needed | overhead | scale-0 plane size |
|---|---|---|---|---|
| 512 | 1024 | 8 | 100% | 6000×1024×4B = 24 MB |
| 1024 | 1536 | 4 | 50% | 36 MB |
| 2048 | 2560 | 2 | 25% | 60 MB |
| 4000 (full) | n/a | 1 | n/a | 94 MB (no strips needed) |

Sweet spot: **H_body = 1024**, **H_strip = 1536**. 24 MB per plane at
scale 0 × 57 planes ≈ 1.4 GB working set at scale 0 alone. Across all
6 pyramid scales the geometric series adds ~37% on top, giving ~2.87 GB
total per strip on a 24 MP image — measured 2026-05-22 in
`examples/bench_strip_vs_whole.rs`. Still fits comfortably on 8 GB
GPUs with multiple metrics live (vs ~7.5 GB whole-image).

Per-scale strip dimensions:
- Scale 0: 6000 × 1536
- Scale 1: 3000 × 768
- Scale 2: 1500 × 384
- Scale 3: 750 × 192
- Scale 4: 375 × 96
- Scale 5: 188 × 48 (rounded up)

At scale 5 the strip is 48 rows tall but the IIR blur with radius 4
needs at least 9 rows of context. That's fine — 48 > 9 + 2× scale-5-halo
(which is ~6). If H_body is dropped below 512 the scale-5 strip
becomes too small to support the IIR boundary handling without
zero-pad contamination of the body region; that's the lower bound.

## IIR boundary handling per strip

ssim2-gpu's IIR Gaussian zero-pads outside the image (libjxl reference
behaviour — `kernels::blur.rs::blur_pass_kernel` initialises
`prev_{1,3,5}` and `prev2_{1,3,5}` to zero, treats out-of-frame `src`
reads as zero). This is the **simplest possible** boundary handling for
strip processing: the kernel already zero-initialises state at the top
of each launch, so launching it on a strip of size `(image_w,
h_body + 2*halo)` gives correct results within the body region as long
as the halo extends ≥ 4 rows beyond the body on each side at the
finest scale (8 × halo_floor for cumulative cross-scale reach).

Importantly, **no per-strip IIR state save/restore is needed.** Each
strip launch is independent. The whole-image and strip-processed
results converge as the halo grows; with the proposed halo of 256 rows
at scale 0 (= 8 rows at scale 5), the IIR response within the body
matches whole-image computation to f32 noise.

## Per-strip accumulation

The final score is `Σ d` and `Σ d⁴` per (scale, channel, error-map),
folded host-side into the SSIMULACRA2 sigmoid. Reductions today sum
the full plane: `kernels::reduction::fused_sum_p4_kernel` iterates
`0..plane.len()` with no row-range awareness.

Per-strip contract:
- Each strip writes per-thread partial sums to its `partials` buffer.
- The reduction kernel must accept a `(body_y_start, body_y_end)`
  row range so it only sums body rows (NOT halo rows).
- Host accumulates partial sums across strips, dividing only at the end.

This requires adding `(row_start: u32, row_end: u32, width: u32)`
parameters to `fused_sum_p4_kernel` and the matching portable-path
kernel. Body row indices are in the strip's local coordinate system.

## API shape

```rust
impl<R: Runtime> Ssim2<R> {
    /// Strip-processing constructor. Allocates working set for a
    /// single (h_body + halo) × w strip; reuses across strips for
    /// the same (image_w, image_h, h_body) configuration.
    pub fn new_strip(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
    ) -> Result<Self>;

    /// Score a pair via strip processing. The host slices `ref_srgb`
    /// and `dist_srgb` into strips with halo overlap, computes
    /// per-strip partials, accumulates into per-scale sums, and
    /// finalizes once at the end.
    pub fn compute_stripped(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<GpuSsim2Result>;
}
```

Backwards-compatible: `Ssim2::new` keeps the whole-image path. Strip
path is opt-in via `new_strip`.

### Cached-reference path

Out of scope for the first version. RD-search hot loops can fall back
to the whole-image path until v2. Open question: cache the per-strip
reference state on R2 / persistent VRAM, or recompute ref per strip
(loses the RD-search speedup)?

## Implementation order (suggested)

1. **Add row-range params to reduction kernels** so partial sums can
   ignore halo rows. Touch: `kernels/reduction.rs` (both fast and
   portable paths). ~half day.
2. **Refactor `Ssim2::compute_with_mode` into
   `compute_inner_with_mode(strip_dims, body_range)`** that operates
   on a single strip-shaped working set. The existing whole-image
   path becomes `compute_inner_with_mode(full, 0..full_h)`. ~1 day.
3. **Add `Ssim2::new_strip` constructor** sizing buffers to strip
   dimensions instead of full image. ~half day.
4. **Add `compute_stripped` driver** that loops over strips, calls
   `compute_inner_with_mode` per strip with the right halo offsets,
   accumulates results. ~1-2 days.
5. **Tests**: at 256×256 and 1024×1024, strip path vs whole-image
   path must agree within f32 noise (5e-5 rel for accumulated sums).
   Add `tests/strip_parity.rs`. ~1 day.

**Total estimate for ssim2 alone: ~3-5 working days** (smaller than
iwssim because there's no cross-scale dependency and the IIR's
zero-pad boundary means no state-save/restore complexity).

## Tradeoffs and risks

- **Reduction order shift**. Per-strip sums reorder f32 adds.
  Cross-tile-size parity will drift ~1e-5 rel — well above
  cross-backend tolerance (5e-4) so no test loosening needed.
- **Halo overhead at 24 MP**: ~25% wasted compute per strip vs
  whole-image (border pixels computed twice). Acceptable price for
  ~5× memory reduction.
- **Strip-boundary body rows must NOT be summed twice**. The reduction
  kernel's row-range parameter is the gate; misindexing here would
  give double-counted regions and silently wrong scores. Strong test
  coverage required.
- **Halo bound is theoretical**. The 4-radius IIR has infinite
  impulse response in principle; zero-padding truncates it. With a
  halo of 4 rows at the finest scale, response within the body should
  match whole-image computation to better than 1e-5 absolute. Verify
  empirically before shipping the strip path.

## What this design does NOT cover

- `Ssim2Batch` (the batched-against-cached-reference path). Strip
  processing for batch is a separate problem.
- The strip-aware `set_reference` cache.
- f16 / bf16 intermediates — orthogonal to strips.

## Status: SHIPPED (2026-05-22)

The Phase 2 work is now landed. Key components:

- `Ssim2::new_strip(client, image_w, image_h, h_body)` constructor in
  `src/pipeline.rs`. Allocates buffers sized for one strip
  (`image_w × (h_body + 2 × STRIP_HALO_ROWS)`).
- `Ssim2::compute_stripped` / `compute_stripped_with_mode` driver in
  `src/pipeline.rs`. Loops over strips, runs the per-strip pipeline,
  reads per-strip sums back, accumulates host-side in f64, folds
  through the SSIMULACRA2 weight table.
- Strip-aware reduction kernels in `src/kernels/reduction.rs`:
  `launch_sum_p4_rows` filters by transposed-buffer column index
  (= original frame y-axis) to drop halo rows from the sum.
- `MemoryMode::Strip { h_body }` now routes to `new_strip` in
  `new_with_memory_mode`. `MemoryMode::Auto` falls back to Strip when
  the Full estimate exceeds the VRAM cap.
- `Error::CachedRefNotSupportedInStripMode` returned when
  `set_reference` is called on a strip-mode instance.
- `tests/strip_parity.rs`: 30 tests, all running on both CUDA and
  wgpu (task #53, 2026-05-28: the 4096² tests previously gated to
  cuda-only were unblocked by splitting `pipeline.rs::cube_count_1d`
  into a 2D dispatch when cubes > 32768, which keeps each dim under
  wgpu's 65535-per-dim cap). Covers parity vs whole at 256² / 1024²
  / 2048² / 4096², cross-tile-size agreement, uneven last strip,
  single-strip degenerate case, error paths, KernelMode dispatch
  (Full / Lossless / Fast), identical-pair sanity, IIR boundary in
  halo.
- `examples/bench_strip_vs_whole.rs`: wall-time + working-set sweep
  at 1 / 4 / 12 / 24 MP. Results land in `benchmarks/`.
