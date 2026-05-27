# ssim2-gpu optimization review — 2026-05-27

Companion to the GPU memory audit at
`benchmarks/gpu_memory_audit_2026-05-27.{csv,md}`. This document
answers: **does ssim2-gpu have the same 3-36× speedup left on the
table that cvvdp-gpu just delivered?** Short answer: yes, but for
different reasons. cvvdp's win was lifting per-band uniform uploads
out of compute loops (Option C, commit 152a6924). ssim2's
inefficiency is structural: **dozens of separate per-channel
kernel launches per scale** where zensim-gpu fuses everything into
a single tile-fused mega-kernel per scale.

Expected speedup if all five fixes below land: ~3-5× at 12 MP based
on launch-count reduction + the LDS-reuse pattern zensim already
exploits. Strong: the worst hot loop (blur) goes 27→3 launches per
scale; the entire CWR per-scale ledger goes from ~49 launches to
~5-8.

All file:line citations are at master `b1d080a`
("test(gpu-audit): add per-crate mem_one_size bench drivers +
orchestrator") — direct parent of canonical master with the
cvvdp-gpu fix landed.

## What 52 HtoD/iter actually means

cubecl emits one `cuMemcpyHtoDAsync` per kernel launch to upload
the kernel's scalar uniforms (the `width`, `height`, `slot`, etc.
arguments that aren't `Array<T>` handles). Counting kernel launches
in ssim2's `compute_with_reference_with_mode` (default
`Ssim2Mode::Faster`, NUM_SCALES=6, image-pyramid for 12 MP):

| Stage                                     | Per scale (full) | Per scale (CWR / skip-Faster) |
|-------------------------------------------|------------------|-------------------------------|
| `launch_zero_fill_f32` partials           | 1 (call-prefix)  | 1 (call-prefix)               |
| `srgb_u8_to_linear_planar_kernel`         | 1 (call-prefix)  | 1 (call-prefix)               |
| `downscale_2x_plane_kernel`               | 3 ch × (S−1)     | 3 ch × (S−1)                  |
| `linear_to_xyb_planar_kernel`             | 1 (3-ch fused)   | 1                             |
| `pointwise_mul_kernel` (sigma11/22/12)    | 9 (3 prods × 3ch)| 6 (sigma22 + sigma12; 2 × 3ch)|
| `blur_pass_kernel` + `transpose_kernel`   | **45**           | **27**                        |
|   = 5 planes × 3 ch × 3 launches (full)   |                  |                               |
|   = 3 planes × 3 ch × 3 launches (CWR)    |                  |                               |
| `transpose_kernel` for raw xyb pair       | 6 (ref + dis)    | 3 (dis only)                  |
| `error_maps_kernel`                       | 3 (per channel)  | 3                             |
| `launch_sum_p4` reductions                | up to 9          | 0-9 depending on skip-map     |
| Per-scale total                           | **~78**          | **~50**                       |

Multiplied across 6 scales (skip-map drops some at coarse levels)
gives the **52 HtoD/iter** number observed in nsys. Skip-Faster
eliminates ~70% of slot-level work at the coarsest scales but
leaves the **per-channel kernel-launch tax** untouched at active
scales.

The cure isn't lifting uploads — there ARE no per-band uploads to
lift. The cure is **fusing channels and fusing passes**, the same
playbook zensim-gpu uses. A zensim-style `fused_features_kernel`
collapses the 27-launch blur-dis-only block into a single launch
per scale (operating on all 3 channels via shared-memory tile
walking).

## Findings — every `client.create` in the hot path

`grep -n "create_from_slice\|create_from_bytes\|client.create" crates/ssim2-gpu/src/pipeline.rs`:

| Line | Call                                       | Context              | Verdict |
|-----:|-------------------------------------------|----------------------|---------|
| 161  | `alloc_plane` (Scale::new constructor)    | `Ssim2::new`         | OK — constructor only. |
| 360  | `client.create_from_slice` `src_u8_a`     | `Ssim2::new`         | OK — constructor only. |
| 361  | `client.create_from_slice` `src_u8_b`     | `Ssim2::new`         | OK — constructor only. |
| 363  | `client.create_from_slice` `partials`     | `Ssim2::new`         | OK — constructor only. |
| 364  | `client.create_from_slice` `sums`         | `Ssim2::new`         | OK — constructor only. |
| 466  | `client.create_from_slice` `src_u8_a`     | `Ssim2::new_strip`   | OK — constructor only. |
| 467  | `client.create_from_slice` `src_u8_b`     | `Ssim2::new_strip`   | OK — constructor only. |
| 469  | `client.create_from_slice` `partials`     | `Ssim2::new_strip`   | OK — constructor only. |
| 470  | `client.create_from_slice` `sums`         | `Ssim2::new_strip`   | OK — constructor only. |
| 654  | `self.client.create(bytes)` (packed)      | `pack_srgb_into_packed_u32_handle` | OK — Phase-4 helper, not in default hot path. |
| 908  | `self.client.create(bytes)`               | `set_reference` strip helper | OK — once per `set_reference`, not per CWR. |
| 1658 | `self.client.create(bytes)`               | strip-mode CWR helper | Per-call — strip-mode only. |
| 1818 | (comment) `create_from_slice_pinned`      | upload comment       | n/a |
| 1846 | `self.client.create(bytes)`               | `upload_and_srgb_to_linear` | **Per-call.** Pinned-staging path — already optimised; ~48 MB DMA at 12 MP. |

**Verdict: ssim2-gpu does NOT have a "constants re-uploaded inside
the compute loop" anti-pattern.** Every persistent buffer is
allocated in `Ssim2::new` (lines 360-364) or `Scale::new` (line
161). Per-call uploads (line 1846 / 1658) are unavoidable — the
distorted-side bytes change on every CWR call by definition. The
pinned-staging fast path is in place (T_x.O 2026-05-17, see line
1815-1846 comments).

So unlike cvvdp's 36× win, ssim2-gpu has no upload-side fat to
trim. The wins live in the **kernel launch count and per-scale
DRAM round-trip**, which the next five sections address.

## Fix #1 — Fuse the H-pass blur output into a tile-fused V-pass + features kernel (zensim pattern)

**Current behaviour** (`pipeline.rs:2030-2076`,
`blur_plane_two_pass_iir`): blur each plane in 3 launches —
`blur_pass_kernel` (v-pass on rows), `transpose_kernel`, second
`blur_pass_kernel` (v-pass on transposed). At 5 planes (sigma11,
sigma22, sigma12, mu1, mu2) × 3 channels × 3 launches = **45
launches per scale** for blur alone in `compute` (or 27 in CWR /
`run_blur_dis_only_masked` at lines 2385-2407).

**Required change.** Replicate zensim-gpu's `fused_features_kernel`
(`crates/zensim-gpu/src/kernels/fused.rs:95`). That kernel does:

- H-blur kept in shared-memory circular buffer (DIAM=11 rows)
- V-blur consumed directly from shared memory, never materialised
  to DRAM
- All 3 channels processed in the same block (block.z dimension)
- All 5 sigma/mu products computed in the same V-blur pass
- Per-thread partials written directly, no separate reduction kernel

Net launches per scale go from 45 → 1. At 6 scales × 27 = 162
HtoD/iter become 6, and ~570 MB of intermediate buffers (the
`mu1_full`, `mu2_full`, `sigma11_full`, `sigma22_full`,
`sigma12_full` per-scale × 3-channel families) become
shared-memory-only. The Scale struct's `sigma11_full / sigma22_full
/ sigma12_full / mu1_full / mu2_full` (lines 144-148) can be
dropped entirely.

**Estimated speedup at 4096²:** 3-4×. Memory savings: ~3 GB at
4096² (eliminating the five `_full` buffer families). This is the
biggest single fix and the one that wipes out the "ssim2 uses 6.2
GB" headline.

## Fix #2 — 3-channel fused downscale kernel

**Current behaviour** (`pipeline.rs:1903-1931`,
`build_linear_pyramid_until`): 3 independent
`downscale_2x_plane_kernel` launches per scale (`for ch in 0..3`).
At 6 scales = 15 launches.

**Required change.** Port zensim's
`downscale_2x_3ch_kernel`
(`crates/zensim-gpu/src/kernels/downscale.rs`, called at
`pipeline.rs:1789` / `2823`). That kernel reads 3 source planes and
writes 3 destination planes in a single dispatch. The Cubecl signature
extends from `(src, dst, src_w, src_h, dst_w, dst_h)` to
`(src_r, src_g, src_b, dst_r, dst_g, dst_b, src_w, src_h, dst_w,
dst_h)` — trivial change.

**Estimated savings:** 15 launches → 5 per CWR call (one per scale
transition). HtoD/iter drops by 10. Combined with Fix #1 this
brings the per-call HtoD count well into the single digits.

## Fix #3 — 3-channel fused pointwise products (sigma11/22/12)

**Current behaviour** (`pipeline.rs:1955-1976`, `pointwise_mul` and
`pipeline.rs:2262-2286` `pointwise_mul_masked`): `for ch in 0..3 { launch
pointwise_mul_kernel(a[ch], b[ch], out[ch]) }` — 3 launches per
product, 3 products per scale × 6 scales = 54 launches per `compute`
call (27 in CWR after Fix #1 fuses blur).

**Required change.** Write a `pointwise_mul_3ch_kernel(a_r, a_g,
a_b, b_r, b_g, b_b, out_r, out_g, out_b)`. Better: fold sigma11 /
sigma12 / sigma22 production INTO the fused-features kernel from
Fix #1 — they're cheap pointwise ops on inputs the kernel already
has in registers, so they cost nothing extra in the fused path.

**Estimated savings:** subsumed by Fix #1 once the fused-features
kernel produces sigma products directly. If shipped standalone:
27 → 9 launches per CWR call.

## Fix #4 — Drop the explicit transpose for the second blur pass

**Current behaviour** (`pipeline.rs:2055-2063` in
`blur_plane_two_pass_iir`): tiled-transpose 32×32 LDS kernel
between the two v-passes. This is necessary in the current design
because the blur kernel walks columns and the two passes need
different access patterns.

**Required change.** zensim's fused kernel keeps the H-blur output
in **shared memory** (lines 16-30 of
`crates/zensim-gpu/src/kernels/fused.rs`), so the transpose
disappears: H-blur writes to a `[DIAM × TX]` shared-memory tile,
V-blur reads from the same tile. No DRAM transpose needed.

**Estimated savings:** 5 plane-blurs × 3 channels = 15 transpose
launches per scale collapsed to 0. Also kills the `v_scratch`,
`t_scratch` buffer families (3 channels × 2 buffers × 6 scales =
36 plane buffers, ~430 MB at 4096²). Note this is **already
subsumed by Fix #1** — listing separately because if Fix #1 ships
as a partial fuse, the transpose elimination is the next-biggest
chunk.

## Fix #5 — Eliminate the post-blur transpose of `ref_xyb_t` / `dis_xyb_t`

**Current behaviour** (`pipeline.rs:2146-2181`,
`run_transpose_raw_xyb_pair` and `:2302-2344`
`run_transpose_raw_xyb_pair_masked`): the raw XYB planes are
transposed AFTER the blurs because the blur output lives in
transposed orientation and the error-maps kernel reads
ref/dis/blurred in matching layouts.

**Required change.** After Fix #1, the fused kernel can produce
error-map outputs (or their pre-aggregation sums) directly in
either orientation, eliminating the need to keep raw planes in two
layouts. The simplest patch: have the fused-features kernel emit
its outputs in the same orientation as the input planes, so no
transpose is needed before the error-maps kernel. Alternatively,
the error-maps math can be integrated into the fused kernel too
(zensim does its per-feature math in the same V-blur pass — see
`fused.rs:46-55`).

**Estimated savings:** 3 channels × 6 scales × 1-2 transposes =
~12-18 launches per CWR call eliminated. Memory:
`ref_xyb_t` / `dis_xyb_t` × 3 ch × 6 scales (~290 MB at 4096²) goes away.

## Cached-ref / strip mode interaction

ssim2-gpu's strip mode (`Ssim2::new_strip`, `compute_stripped`,
strip-mode `set_reference` via mode-E) **uses the same per-channel
per-pass kernel launches** — see
`crates/ssim2-gpu/src/pipeline.rs:1670-1816` for the strip
process_scale; the inner blur path is the same
`blur_plane_two_pass` (line 2002) as the whole-image path.

The measured audit confirms strip helps with VRAM (4096² Full =
6232 MiB vs strip = 1357 MiB, **4.6× reduction**), but **not with
HtoD launch count** — strip just walks the same kernel sequence
over smaller tiles, multiple times per call. The HtoD figure goes
**up** in strip mode (more strips, same per-strip launch ledger).

After the fixes above land, strip mode benefits proportionally:
the fused kernel does the same amount of work, just bounded by
the strip's halo'd window. The strip-mode `set_reference` (mode-E,
lines 834+) which currently allocates full-image-sized
`StripCachedRefScale` buffers (3 channels × 6 scales × ~24 MB at
4096² = 432 MB) becomes much smaller if the fused kernel reads its
inputs straight from the strip-mode body buffers.

## Summary

| Fix | HtoD launches saved | VRAM saved @ 4096² | Risk |
|-----|--------------------:|-------------------:|------|
| #1 fused features kernel (zensim port) | ~27/call | ~3 GB | Med — new mega-kernel, needs care with tile geometry |
| #2 3-channel downscale | ~10/call | 0 (compute-only) | Low — straightforward port |
| #3 3-channel pointwise mul | ~18/call (subsumed by #1) | 0 | Low — trivial port; redundant after #1 |
| #4 drop transpose between blur passes (subsumed by #1) | ~15/call | ~430 MB | n/a — part of #1 |
| #5 drop raw-xyb transpose | ~12/call | ~290 MB | Low |

Total: **52 HtoD/iter → ~5-8 HtoD/iter** and **6.2 GB → ~2-3 GB**
at 4096². Steady-state wall-time win estimated 3-5× at 12 MP based
on (a) kernel-launch overhead at 30-50 µs each × ~45 saved
launches = 1.4-2.3 ms saved on launches alone, plus (b) DRAM
bandwidth from elimination of intermediate buffer writes = the
larger win at 12+ MP where each blur-intermediate DRAM round-trip
is ~50 MB.

The single biggest win is **Fix #1** — it's the structural change
that closes the gap with zensim-gpu. If only one fix lands, this
is the one. The other fixes accumulate naturally once the fused
kernel ships.
