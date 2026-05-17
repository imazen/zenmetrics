# CubeCL wishlist (from imazen perf work, May 16-17 2026)

Synthesized from `docs/CUBECL_GOTCHAS.md`, this session's perf chase across 6 GPU crates (cvvdp / butter / ssim2 / dssim / iwssim / zensim), the pinned-upload fork (lilith/cubecl), the upstream PRs (#1030 merged, #1324 merged, #1334 draft), and the Kanetaka et al. IWAIT 2026 SSIMULACRA2 optimization work.

Priority is **our** priority — what would most accelerate zenmetrics. Upstream priority may differ.

## Tier 1 — biggest leverage, cross-crate impact

### W1. Persistent pinned-buffer write API

`client.write_to_handle(handle: &Handle, bytes: &[u8])` so a pre-allocated GPU buffer can be re-uploaded per call without re-allocation. Today `create_from_slice_pinned` allocates a new device handle every call; the memory pool reuses slabs internally but `cuMemAllocAsync` still fires (~600 µs avg, but ~26 ms × 5 iters = 131 ms wasted per warm-ref bench at 12 MP). **Affects all 6 zenmetrics GPU crates.** This is probably the single highest-ROI cubecl improvement for us.

Sketch:
```rust
let scratch_handle = client.empty(n_bytes);  // once at construction
// per call:
client.write_to_handle_pinned(&scratch_handle, bytes);
```

The pinned-upload fork already has the pinned-allocation machinery — `reserve_staging` returns pinned `Bytes`, `create` consumes them. Need the missing piece: write into an existing handle without changing its identity.

### W2. `Array<u8>` storage on CUDA backend

Today every byte-input kernel (sRGB color stages on all 6 crates) pre-packs 3 bytes per u32 to satisfy the cross-backend Array<u32> constraint. WGSL has no u8 storage type so this is fine for that backend, but CUDA could read `Array<u8>` directly — would drop the 12 MP sRGB upload from 48 MB to 36 MB (another 25% on top of T4.L).

Implementation hint: feature-gate `Array<u8>` behind a "cuda-i8-storage" cap that the wgpu/metal backends don't advertise. Crates that opt in pick the fast path on CUDA, fall back to packed u32 elsewhere.

### W3. CUDA graph capture

Capture a pipeline once, replay as a single graph launch. cvvdp-gpu launches ~110 kernels per JOD call — at 5-10 µs CUDA launch overhead each, that's ~600 µs floor purely in launch dispatch (matters at 1 MP, dominates at 256²). A graph-captured pipeline would replay all 110 in microseconds.

Currently gotcha G3.5 in `docs/CUBECL_GOTCHAS.md` notes this is unexposed.

### W4. Bool comptime generics on `#[cube]`

Vship's SSIMU2 skip-map uses 8 template specializations on `<bool SkipSSIM, bool SkipArtifact, bool SkipDetailloss>` (`src/HIP/ssimu2/score.hpp:329-358`). We ported via runtime mask (G1.7 workaround) — branches inside the kernel. Comptime spec would generate dead-code-eliminated variants. Probably 5-15% speedup on the skip-map path.

Gotcha G1.7 currently notes "split into two kernel entry points sharing a non-launch-able `#[cube]` helper" as the workaround — clunky at 8 variants.

## Tier 2 — significant friction, manageable workarounds exist

### W5. CUDA stream priorities (full surface)

Partially landed via lilith/cubecl PR #1324 (merged upstream as `feat(streaming): add stream priority hint`). The remaining piece: multi-stream concurrency for ref/dist pipelines. Vship runs reference and distorted on separate CUDA streams with event-based sync at convergence; our `Cvvdp::compute_dkl_jod` serializes them. Estimated 1.3-1.5× upper bound at 12 MP.

Needs: convenient `client.with_stream(StreamId::A) { ... }` scope or equivalent.

### W6. `cubecl-cpu` atomic<f32> support

`Atomic<f32>::fetch_add` panics on cubecl-cpu (`compiler/visitor/elem.rs:38` not-yet-implemented). cvvdp-gpu had to ship `compute_dkl_jod_host_pool` as a workaround. Affects portability of any kernel that reduces via atomics. Tracel-mlir / cpp backend feature parity.

### W7. WGPU/Vulkan subgroup operations

Current LDS pool reduction does 8 pointer-jumping passes (256 → 1 in shared memory). On hardware with subgroup ops (Vulkan `VK_KHR_shader_subgroup_arithmetic`, Metal `simd_sum`), the last 5 passes (32 → 1) could collapse to a single `subgroup_sum`. ~40-50% reduction-stage speedup at the LDS pool kernel level.

cubecl 0.10 supports `subcube_sum` on CUDA. wgpu backend doesn't yet.

### W8. GPU event timing primitives

`client.event_record() -> EventHandle; let ms = client.event_elapsed(start, end);` for honest per-stage GPU timing without dropping to raw cudarc. Currently `Instant::now()` lies (async submit), `CUDA_LAUNCH_BLOCKING=1` isn't always honored by cubecl's queue, and the only honest path is `nsys profile` (slow + cold-cache).

`CVVDP_TRACE=1` exists in cvvdp-gpu as a host-side host-trace approximation but the numbers don't reconcile with bench wall time (3 ms traced vs 244 ms benched). Real GPU events would close this gap.

## Tier 3 — paper cuts on every kernel author

### W9. `f32::exp` as registered cube op

Gotcha G1.1. Currently workaround `f32::powf(2.0, x * LOG2_E)`. Cosmetic but every numerics-heavy kernel hits this.

### W10. `u32::abs_diff` registered

Gotcha G1.4. Manual workaround. Trivial to add.

### W11. `Atomic<f32>::fetch_max` on CUDA

Gotcha G1.2. CUDA's hardware `atomicMax` is integer-only. Currently we bit-cast to u32 atomic-max (works for non-negative f32 by IEEE-754 ordering). cubecl could synthesize the CAS loop and present a uniform API.

### W12. Typed-zero idiom for if/else expressions

Gotcha G1.5 sibling. Code that wants a "zero of the right NativeExpand<T> type" has to write `idx - idx` because `0usize` literal doesn't auto-promote in mixed-arm if expressions:

```rust
let safe_idx = if in_range { idx } else { 0usize };  // E0308: expected NativeExpand<usize>
let zero_idx = idx - idx;  // Workaround
let safe_idx = if in_range { idx } else { zero_idx };
```

Either auto-promote literals in cube context, or document a typed-zero macro.

### W13. Block-as-expression in `#[cube]` if-branches

Multi-line if/else arms sometimes refuse to compile cleanly. Forces extraction of helper functions or sequential assignments. Rust supports block expressions; cube macro should too.

### W14. Better diagnostics on missing-op codegen failures

`f32::exp` fails with "unknown intrinsic" at JIT, not at compile. Several gotchas (G1.1, G1.2, G1.4) would be caught earlier with a compile-time check that the cube body only uses ops registered in the IR.

A `cargo cube-check` lint pass would also catch the bool-generics issue, the typed-zero literal issue, the SharedMemory size-vs-index type mismatch (G1.5), etc.

## Tier 4 — library-level primitives we keep re-implementing

### W15. Separable convolution helpers

Every GPU image crate writes separable Gaussian / box / FIR convolution. Could be:

```rust
cubecl::stencil::separable_fir::<f32, 5>(
    input, output, kernel: &[f32; 5], boundary: Boundary::Reflect,
);
```

Would standardize the LDS-tile + integral-table border + halo-load pattern.

### W16. Block-level reductions as a library

Every reduction kernel has the same pointer-jumping pattern. Could be:

```rust
let workgroup_sum: f32 = cubecl::reduce::sum_workgroup(value);  // returns to thread 0
```

Would eliminate the boilerplate we re-write in every crate's reduction kernel.

### W17. 2D tile loaders as a primitive

`cubecl::tile::load_with_halo(src, tile_origin, halo_size, boundary)` returns a `SharedMemory<f32>` with the tile loaded cooperatively. Captures the common LDS-tile pattern (downscale, blur, malta, masking). Would have saved the multi-page kernels in T1.B / T_x.A / ssim2 transpose etc.

## Tier 5 — build + dev experience

### W18. Faster cold builds

5-9 min cold compile of cubecl-cuda (G6.1). Mostly tracel-mlir deps. Hard to fix from outside.

### W19. Doctest configuration support

`crates/ssim2-gpu` has 9 doctests that fail under `--no-default-features --features cuda` because they import `cubecl::wgpu`. We treat as pre-existing acceptable but it's noise on every test run. Need a cubecl convention for "this doctest requires feature X" gating.

### W20. Migration guide for 0.10 → 0.11 mega-refactor

Upstream PR #1322 "mega-refactor" changed the frontend substantially (alloc→layout rename, frontend ref/value distinction, etc.). We're holding on 0.10 partly because of this; would happily migrate with a guide.

## Tier 6 — speculative / longer-term

### W21. fp16 / packed math on `#[cube]`

Vship doesn't use fp16 either (it's portable HIP+CUDA single-source). But our cvvdp 4-channel temporal filter could benefit from `float2`-packed FMA (2 channels per instruction). vship uses `fmaf(float2, ...)` for this in `src/HIP/cvvdp/temporalFilter.hpp:153-178`. Would need cubecl native f16x2 / f32x2 vector types.

### W22. Pinned-host buffer pool

Even better than W1: a pool of N pre-allocated pinned buffers that the runtime cycles through. Application asks "give me a 144 MB pinned buffer", gets one back without allocation. Would let us drop the per-call `create_from_slice_pinned` cost to near zero.

### W23. Cross-stream barrier / event API

For W5's multi-stream concurrency to be useful, need `let event = client.record_event(stream_a); client.wait_event(stream_b, event);` so the dist-side stream can wait on the ref-side stream's pyramid completion.

## What's already shipped (from our fork)

For reference — these are NOT wishlist items, they're done:

- ✅ `create_from_slice_pinned` (PR #1334 draft against upstream main; merged in our fork at `pr/pinned-v0.10.0` / `08d34ac0`)
- ✅ `reserve_staging` for pre-allocated pinned slabs (same PR)
- ✅ Default `create_from_slice` routes through pinned staging (same PR)
- ✅ CUDA stream priority hint (upstream PR #1324, merged)

## Cross-references

- Full gotchas catalog: `~/work/zen/zenmetrics/docs/CUBECL_GOTCHAS.md` (G1.1–G6.8, G7.x)
- Porting guide: `~/work/zen/zenmetrics/docs/CUBECL_PORTING_GUIDE.md`
- vship analysis (with cubecl gap callouts per metric): `~/work/zen/zenmetrics-refs/vship-acceleration-analysis-2026-05-16.md`
- Kanetaka paper notes (skip-map → bool comptime + FIR → separable convolution primitive): `~/work/zen/zenmetrics-refs/kanetaka-iwait-2026-paper-notes.md`
- Our cubecl fork branches:
  - `pr/pinned-v0.10.0` (`de2f9857`) — what zenmetrics consumes via patch.crates-io
  - `pr/pinned-upload-rebased-2026-05-17` (`2ab56e7b`) — upstream PR #1334 against post-mega-refactor main
- Upstream merged contributions: PR #1324 (stream priority), PR #1030 (base pinned `staging()` API in v0.10.0)
