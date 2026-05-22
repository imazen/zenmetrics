# Strip processing for iwssim-gpu — design

## Why

`Iwssim::new(client, w, h)` pre-allocates 19 working f32 buffers per scale × 5 scales (see `pipeline.rs::struct Scale`). For a 1024² image at scale 0 that's 76 MB; for a 24 MP image (e.g. 6000×4000) it's ~1.8 GB at scale 0 alone, ~2.6 GB across all scales. A production sweep worker running IW-SSIM + zensim-WithIw + SSIM2 + butteraugli + cvvdp concurrently at 24 MP would need 6–8 GB of working-set VRAM, overflowing 8 GB consumer GPUs and crowding 12 GB ones.

Strip processing bounds peak memory to a function of strip size, not image size. With 1024-row strips on a 6000×4000 image, working set stays at ~256 MB regardless of total height.

## Stencil reach budget

The cumulative reach at the finest scale, walking through the pipeline:

| Stage | Per-axis radius (at its own scale) | Cumulative at scale 0 |
|---|---|---|
| `binom5` LP downsample (scale s → s+1) | 2 | 2 × 2^s |
| `gauss11` 11×11 valid blur (scale s) | 5 | 5 × 2^s |
| `imenlarge2` 2× upsample with binom5 (scale s+1 → s for parent_band) | ~4 input pixels | 4 × 2^(s+1) = 8 × 2^s |
| `3×3` box statistics (scale s, on `cs` map) | 1 | 1 × 2^s |

Worst case is scale 4 (coarsest, 5-level pyramid):
- `gauss11` at scale 4 needs ±5 rows at scale 4 = ±80 rows at scale 0
- LP build from scale 0 to scale 4 cascades binom5 reflections — each cascade adds ±2 rows at the input scale, doubled at finer scales

Computing the full cumulative reach symbolically:

```
H(s)   = halo at scale s in scale-0 rows
H(0)   = 5                         # gauss11 at scale 0
H(s)   = 2^s × (5 + 2)             # gauss11 + LP-build reach at scale s
H_max  = max over s ∈ {0..4}
       = max(5, 14, 28, 56, 112)
       = 112
```

Add the imenlarge2 cross-scale read (`parent_band` at scale s reads scale s+1's LP, which adds ~4 rows at scale (s+1) = 8 × 2^s scale-0 rows). At scale 3 that's 64 additional rows. The 3×3 box on the cs map is negligible (≤ 16 scale-0 rows at scale 4).

**Total finest-resolution halo: ~180 rows per side** (112 worst-case `gauss11` + ~64 cross-scale parent_band + ~16 box). Round up to **256 rows per side** for safety + alignment.

## Strip sizing

Per-strip budget at 24 MP (image 6000×4000), with H_strip = body rows + halo:

| H_body | H_strip (body + 2× halo) | strips needed | overhead | scale-0 plane size |
|---|---|---|---|---|
| 512 | 1024 | 8 | 100% | 6000×1024×4B = 24 MB |
| 1024 | 1536 | 4 | 50% | 36 MB |
| 2048 | 2560 | 2 | 25% | 60 MB |
| 4000 (full) | n/a | 1 | n/a | 94 MB (no strips needed) |

Sweet spot: **H_body = 1024**, **H_strip = 1536**. 24 MB per plane at scale 0 × 19 planes = ~460 MB working set per strip — fits comfortably on 8 GB GPUs with multiple metrics live.

Per-scale strip dimensions:
- Scale 0: 6000 × 1536
- Scale 1: 3000 × 768
- Scale 2: 1500 × 384
- Scale 3: 750 × 192
- Scale 4: 375 × 96

At scale 4 the strip is 96 rows tall but the 11×11 gauss11 valid blur needs 11 rows. That's fine — 96 > 11 + 2× scale-4-halo (which is ~10). If H_body is dropped below 512 the scale-4 strip becomes too small to support the cross-scale reads without negative-rows underflow; that's the lower bound.

## Cross-scale dependency: parent_band

The IW-SSIM info-weight at scale s needs `imenlarge2(LP[s+1])` cropped to scale-s shape. With strips, scale (s+1)'s LP for this strip must cover scale-s strip's full extent at the next-finer scale. Concretely:

- Each strip processes one logical image region.
- Build scale-0 LP for the strip (with halo). Build scale 1, 2, 3, 4 from it cascading.
- For each scale s ∈ {0..3}, compute `parent_band[s] = imenlarge2(LP[s+1])` using the SAME strip's scale-(s+1) data — works because LP[s+1] was just computed for this strip and covers the strip's full region (with proportionally smaller halo at the coarser scale).

No inter-strip data exchange needed for the parent_band path — each strip is self-contained.

## Per-strip accumulation

The final score is `wmcs_j = Σ(cs_j · iw_j) / Σ(iw_j)` for j∈{0..3} and `wmcs_4 = mean(cs_4 · l_5)`. These are pixel-summed over the full image.

Per-strip contract:
- Each strip writes per-scale partial sums to its `partials` buffer (the existing `partials` field on `Iwssim<R>` already does this).
- Host accumulates partial sums across strips, dividing only at the end.
- The reduction kernel needs to NOT include halo rows in its sum (only `body` rows count). Add a row-range parameter to `reduction::launch_weighted_sum` and friends.

The `sums` buffer (final per-slot reductions) lives across strips and accumulates incrementally — host reads it after the last strip's finalize launch.

## API shape

```rust
impl<R: Runtime> Iwssim<R> {
    /// Strip-processing constructor. Allocates working set for a
    /// single (h_body + halo) × w strip; reuses across strips for
    /// the same (image_w, image_h, h_body) configuration.
    pub fn new_strip(client: ComputeClient<R>, image_w: u32, image_h: u32, h_body: u32) -> Result<Self>;

    /// Score a pair via strip processing. The host slices `ref_gray`
    /// and `dis_gray` into strips with halo overlap, computes
    /// per-strip partials, accumulates into per-scale sums, and
    /// finalizes once at the end.
    pub fn compute_gray_stripped(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<GpuIwssimResult>;
}
```

Backwards-compatible: `Iwssim::new` keeps the whole-image path. Strip path is opt-in via `new_strip`.

### Cached-reference path

`set_reference_stripped(ref_gray)` pre-computes and caches per-strip ref-side LP pyramids on the GPU. Cache size scales with `image_h × image_w × 4 (binom5 cascades)` — at 24 MP that's ~400 MB just for cached ref LP. May want a smaller cache that keeps only the current strip plus the previous strip's coarsest-scale data for cross-scale boundary continuity. **Open question.**

For RD-search hot loops where the same ref is scored against many distorted images, this matters; for one-shot scoring, just skip the cache.

## Implementation order (suggested)

1. **Add row-range params to reduction kernels** so partial sums can ignore halo rows. ~half day.
2. **Refactor `Iwssim::compute_gray` into `compute_gray_inner(strip_h, body_start, body_end)`** that operates on a single strip-shaped working set. The existing whole-image path becomes `compute_gray_inner(full_h, 0, full_h)`. No memory change yet — just routing. ~1 day.
3. **Add `Iwssim::new_strip` constructor** that sizes buffers to strip dimensions instead of full image. ~half day.
4. **Add `compute_gray_stripped` driver** that loops over strips, calls `compute_gray_inner` per strip with the right halo offsets, accumulates results. ~1-2 days.
5. **Tests**: at 256×256 and 1024×1024, strip path vs whole-image path must agree within f32 noise (5e-5 rel for accumulated sums). Existing parity tests cover whole-image; add strip-specific tests. ~1 day.
6. **Cached-reference strip path**: punt on first version, add later if RD-search throughput matters. Up to ~3 days when needed.

**Total estimate for iwssim alone: ~5-7 working days.** zensim-WithIw, ssim2-gpu, butteraugli-gpu need parallel refactors; each is similar scope (~3-5 days).

## Tradeoffs and risks

- **Reduction order shift**. Per-strip sums reorder f32 adds. Cross-tile-size parity will drift ~1e-5 rel — well above cross-backend tolerance (5e-4) so no test loosening needed.
- **Halo overhead at 24 MP**: ~25% wasted compute per strip vs whole-image (border pixels computed twice). Acceptable price for 5× memory reduction.
- **Strip-boundary cs/iw rows must NOT be summed twice**. The reduction kernel's row-range parameter is the gate; misindexing here would give double-counted regions and silently wrong scores. Strong test coverage required.
- **The `set_reference` cache becomes proportional to image size, not strip size**. Either accept that (RAM is cheap, VRAM isn't), recompute ref LP per strip (loses the RD-search speedup), or punt on cached-ref support for the strip path.

## What this design does NOT cover

- ssim2-gpu, butteraugli-gpu, zensim-gpu — they have analogous strip-processing needs but different stencils. Each gets its own design doc once iwssim ships.
- cvvdp-gpu — pipeline already has streaming-ish characteristics via its multi-tier pyramid; may or may not need strips.
- Multi-GPU striping (distribute strips across multiple GPUs). Out of scope.
- f16 / bf16 intermediates — separate precision-vs-memory project; orthogonal to strips.
