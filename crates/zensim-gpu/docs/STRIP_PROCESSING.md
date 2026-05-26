# zensim-gpu strip-mode processing

`MemoryMode::Strip` walks the image in `h_body + 2 × halo` row strips,
reusing the existing per-scale fused / persist / masked-IW kernels with
strip-sized device buffers. Peak working set drops from `O(image_h)` to
`O(strip_alloc_h)`. Landed 2026-05-26.

## Constructor

```rust
let z = Zensim::<R>::new_strip(client, image_w, image_h, h_body)?;
// or with explicit halo/regime:
let z = Zensim::<R>::new_strip_with_halo_and_regime(
    client, image_w, image_h, h_body, halo, regime
)?;
```

- `h_body` MUST be a multiple of `STRIP_ALIGN` (= `2^(SCALES-1)` = 8).
- `halo` MUST be ≥ `STRIP_DEFAULT_HALO` (= 40) AND a multiple of
  `STRIP_ALIGN`. The 40-row floor guarantees that at scale 3 (the
  deepest pyramid level) the halo still covers the 11×11 V-blur
  diameter `R = 5`, so V-blur at body rows reads strip-buffer data
  rather than triggering the strip-local mirror.
- `regime` ∈ {`Basic`, `Extended`, `WithIw`} works identically to
  `new_with_regime`.

`MemoryMode::Auto` picks Strip when the Full estimate exceeds the VRAM
cap; otherwise it picks Full. `MemoryMode::Strip { h_body: None }`
auto-sizes the body to the largest multiple of `STRIP_ALIGN` that fits
the cap.

## Strip layout

Given an `image_h`-row image and `h_body` body rows per strip, the
walker emits `ceil(image_h / h_body)` strips. Each strip exposes four
row ranges in image coordinates:

```
strip k: body = [k × h_body, min((k+1) × h_body, image_h))
         upload = [max(body_lo - halo, 0), min(body_hi + halo, image_h))
```

The body region "belongs to" that strip — it owns the per-feature
contribution to the final score. The halo region is overlap with
adjacent strips' uploads; halo rows drive the V-blur sliding window so
mu1/mu2/ssq/s12 at body rows are computed against image-correct
inputs, but halo rows do NOT contribute to the per-pixel feature sums.

## Body-row gate in kernels

`fused_features_kernel`, `fused_features_kernel_persist`, and
`masked_iw_strip_kernel` each accept `y_body_start, y_body_end` as
parameters. Full-image callers pass `(0, height)` — every row
contributes. Strip-mode callers pass the body row range in
strip-buffer coordinates.

Inside the WALK Y loop:

```text
let is_body = y >= y_body_start && y < y_body_end;
let mask = if is_body { 1.0 } else { 0.0 };
// every per-pixel feature value is multiplied by `mask` before
// accumulating into a0..a16 / peak0..peak2.
```

Multiplicative mask (not branch-predicated accumulation) keeps the
CFG identical to the pre-body-gate kernel, which preserves the
existing Basic-vs-Persist bit-parity test (`cpu_gpu_feature_sweep`'s
1e-9 abs floor) — branching introduces FMA reordering that breaks
parity by ~6e-6 in the worst case.

## Per-strip orchestration

For each strip:

1. **Update per-scale h.** `set_scale_h_for_strip(actual_strip_h)`
   resets `scales[s].h` to `actual_strip_h.div_ceil(2)^s` (boundary
   strips can have actual_strip_h < strip_alloc_h).
2. **Upload.** Ref + dist sRGB rows for `[upload_start, upload_end)`
   pack into u32-packed strip buffers (sized for strip_alloc_h).
3. **XYB pyramid.** `run_xyb_pyramid` runs the sRGB → positive-XYB
   color kernel + 2× downscale on strip-sized buffers.
4. **Fused features.** `launch_blur_and_features{_persist}_with_body`
   runs with `(y_body_start, y_body_end)` mapped per scale via
   `div_ceil`. Both ends use `div_ceil` so consecutive strips' body
   ranges at scale s are contiguous (no overlap, no gaps).
5. **Masked + IW.** `launch_masked_iw_with_body` (only when regime
   needs it) runs on the strip-sized persist planes with the same body
   gate.
6. **Reduce.** `launch_reduction` + `launch_reduction_ext` produce
   per-(scale, ch, slot) finals on device.
7. **Read back + accumulate.** Strip finals add to host-side
   `acc_f64 / acc_max / acc_ext_f64`; peaks take max.

After all strips:

8. **Pack feature vector.** `pack_feature_vector(acc_f64, acc_max,
   acc_ext_f64, scale_image_h)` divides by the FULL image's pixel
   count at each scale (= `pw × scale_image_h[s]`). Strip body row
   counts sum exactly to the full image's row count at every scale.

## Cached reference (`set_reference`)

In Full mode, `set_reference` uploads the ref sRGB and pre-builds the
ref XYB pyramid on the device, then `compute_with_reference` only
uploads/builds the dist side.

In Strip mode the per-scale buffers are sized for one strip — caching
the **full** ref pyramid in `Scale.ref_xyb` is not possible. Mode E
(strip + cached-ref) instead keeps a separate **full-image** ref XYB
pyramid in `RefFullXybState` on device; the strip walker copies the
strip's row range from there into `Scale.ref_xyb` per `compute_with_reference`
call, skipping ref re-upload + ref XYB rebuild every strip.

### Mode E device cache (task #75, 2026-05-26)

`set_reference` allocates `RefFullXybState` lazily on first call,
runs `srgb_to_positive_xyb_kernel` + `downscale_2x_3ch_kernel` on
full-image-sized scratch buffers, then keeps the 3-channel × 4-scale
XYB planes resident across `compute_with_reference` calls. Each strip
runs `install_ref_xyb_from_full_cache(up_lo)`, which launches a
`copy_rows_kernel` blit per (scale, channel) — total `3 × 4 = 12`
blit kernel dispatches per strip, each copying `padded_w × Scale.h`
f32 values.

**Bit-exact parity** vs the per-strip rebuild path: the 2× downscale
operates on consecutive `(2r, 2r+1)` row pairs. Since `up_lo` is
always a multiple of `STRIP_ALIGN = 8 = 2^(SCALES-1)`, scale-s row
`r` of the strip buffer equals scale-s row `((up_lo >> s) + r)` of
the full-image cache. The blit copies that row range directly; no
boundary computation differs from what the strip walker would have
produced itself. Verified by `tests/strip_parity.rs::host_cached_only_matches_device_cached_512x512`.

**Memory cost** (3 channels × `Σ pyramid_pixels_at_full_h` × 4 bytes):

| image | Mode E device cache | strip working set | total |
|-------|---------------------|-------------------|-------|
| 1024² Basic    | 16 MB | 70 MB  | 86 MB  |
| 2048² Basic    | 64 MB | 130 MB | 194 MB |
| 4096² Basic    | 256 MB | 290 MB | 546 MB |
| 8192² WithIw   | 1024 MB | 705 MB | 1729 MB |

(Working-set numbers from `STRIP_PROCESSING.md` § "Memory footprint".)

The cache costs 12 channels × `Σ pyramid_pixels` × 4 bytes vs the full
mode's 24 channels (ref + dist), so it's ~50% of the full-mode ref+dist
XYB allocation. For 4096² Basic that's ~256 MB on top of strip's
290 MB working set = 546 MB total — still 2.2× smaller than Full's
1186 MB.

**Speedup** (RTX 5070, CUDA 13.2.1, `examples/strip_cached_ref_speedup`;
mean of 10-15 timed iters; `benchmarks/zensim_strip_device_cache_2026-05-26.csv`):

| image | h_body | host-cache ms | device-cache ms | speedup |
|-------|--------|---------------|------------------|---------|
| 1024² | 256 | 10.36 | 5.47  | **1.89×** |
| 2048² | 256 | 24.38 | 16.78 | **1.45×** |
| 4096² | 256 | 66.04 | 61.54 | 1.07×   |

The win is largest at smaller resolutions where per-strip ref XYB
rebuild overhead is a bigger fraction of the dispatch budget. At
4096² the per-strip kernel dispatches are large enough that the
saved ref upload + ref pyramid rebuild only pulls 4-5 ms out of a
66 ms iter. For warm-loop callers (encoder quant search, N dist
iters vs one ref), 1.07-1.89× is significant.

**Opt-out**: callers with extreme VRAM pressure or few dist iters
can call `set_reference_host_cached_only` instead. This keeps the
old behaviour — host-side `cached_ref_strip_srgb` cache, per-strip
ref re-upload + ref XYB rebuild.

**Lifecycle**:
- `set_reference` allocates `RefFullXybState` lazily; subsequent
  `set_reference` calls for the same resolution reuse the same
  per-scale handles (overwrite the xyb content).
- `clear_reference` drops `RefFullXybState`; the next
  `set_reference` re-allocates.
- The `src_u8_full` staging buffer stays allocated across
  set_reference calls (alloc happens once on the first call).

## Diffmap in strip mode (task #76)

Per `docs/DIFFMAP_DIVERGENCES.md` §2, the per-pixel diffmap
production is delegated to the canonical CPU
`zensim::Zensim::compute_with_ref_and_diffmap_linear_planar` path
in Phase 1. **This is intentional and works identically in strip
mode** — the diffmap entry-points
(`score_with_diffmap`, `score_with_warm_ref_diffmap`,
`score_from_linear_planes_with_*_diffmap`) route entirely through
`Zensim::diffmap_state`'s CPU `Zensim` driver and never touch the
GPU strip walker.

The CPU is deterministic, so strip-mode diffmap output is bit-identical
to full-mode diffmap output for the same `(reference, distorted)` pair.
Verified by `tests/strip_parity.rs::diffmap_strip_matches_full_bit_for_bit_512x512`.

`set_reference` in strip mode mirrors the reference into
`diffmap_state.warm_ref` exactly as Full mode does — the device
cache for the GPU feature pipeline (`RefFullXybState`) and the
CPU diffmap state are independent caches.

**Out of scope for now**: a GPU-native diffmap kernel chain (Phase 2).
When that ships, it will need a strip-aware per-scale diffmap output
buffer + a `copy_rows_kernel`-style stitch to combine per-strip
diffmap slices back into a full-image output. The CPU-routing
design is the simplest correct solution that ships today.

## Parity vs Full mode

Strip output matches Full output within these bounds (measured on
`tests/strip_parity.rs`, RTX 5070, CUDA backend):

| image | h_body | regime | max abs rel | notes |
|-------|--------|--------|------------|-------|
| 512×512 | 128 | Basic    | 1.80e-3 | aligned |
| 512×512 | 128 | Extended | 1.80e-3 | aligned |
| 512×512 | 128 | WithIw   | 1.80e-3 | aligned |
| 768×384 | 64  | Basic    | 4.64e-3 | aligned |
| 1024×768 | 256 | Basic   | (smoke)   | aligned |
| 400×300 | 120 | Basic    | 1.73e-2 | UNALIGNED (height not multiple of 8) |

Drift sources:

- **f32 V-blur sliding sums.** Strip mode's kernel call uses
  `n_strips=1` (one GPU strip per image strip), so the V-blur slide
  covers up to ~50 rows in a single per-thread accumulator. Full mode
  splits the same image into ~4 GPU strips at the same scale, so each
  slide is ~13 rows. Different slide histories → different f32
  rounding paths through `sum_m1, sum_m2, sum_sq, sum_s12`.
- **Reduction order across strips.** Each strip's finals add to the
  host accumulator; the order is well-defined but differs from the
  Full-mode reduction tree on device.

Both contributions are < 1% rel for pyramid-aligned image sizes,
~2% rel for unaligned boundary cases (image_h not a multiple of 8).

## Memory footprint

The strip estimator returns the working-set bytes for one
`h_body + 2 × halo` strip:

```rust
let bytes = estimate_strip_gpu_memory_bytes_with_regime(width, h_body, regime)?;
```

Empirical scaling (matches `estimate_gpu_memory_bytes`'s per-pyramid-
pixel coefficients applied to the strip's allocation height):

- Basic: `41 × pyramid_pixels(width, strip_alloc_h)` B
- Extended: `38 MB + 139 × pyramid_pixels` B
- WithIw: `71 MB + 136 × pyramid_pixels` B

### Measured (nvidia-smi peak delta vs process start)

Measured on RTX 5070 + CUDA 13.2.1 via
`examples/strip_measure_actual.rs`. Values **include** the
~193 MB cubecl runtime pool overhead. See
`benchmarks/zensim_strip_vs_full_2026-05-26.csv`.

| image | regime | Full MB | Strip MB | reduction |
|-------|--------|---------|----------|-----------|
| 4096² (16 MP) | Basic    | 1186 | 290 | 4.1× |
| 4096² (16 MP) | Extended | 2722 | 482 | 5.6× |
| 4096² (16 MP) | WithIw   | 2722 | 482 | 5.6× |
| 8192² (67 MP) | Basic    | 3490 | 513 | 6.8× |
| 8192² (67 MP) | WithIw   | 10205 | 705 | **14.5×** |

The 8192² × WithIw case is the cardinal-direction fix for issue #16:
Strip WithIw (705 MB) is now within 1.4× of Strip Basic (513 MB),
mirroring the CPU zensim's structurally-flat memory cost across
regimes. Full WithIw at 8192² needs 10.2 GB and won't fit on a 12 GB
fleet box; Strip WithIw fits with room to spare.

## Out of scope (follow-ups)

- GPU-native diffmap kernel chain (currently CPU; see § "Diffmap in
  strip mode" above for the path).
- Tile mode (2D strips). The CPU zensim doesn't use 2D tiling
  internally either; if needed it would require a separate kernel
  pass.
