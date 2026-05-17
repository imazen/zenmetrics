# Vship → zenmetrics acceleration analysis

**Date:** 2026-05-16
**Vship rev:** `96b2750` (codeberg.org/Line-fr/Vship, MIT NON-AI)
**Zenmetrics rev:** `1fc417cd` on `feat/cvvdp-gpu-scaffold` (cvvdp-gpu tick 415)
**Author:** Claude (Opus 4.7, 1M context)
**Scope:** how does vship achieve its GPU speed and what's portable to our CubeCL stack?

Architecture comparison + a prioritized porting plan, with a measured baseline on RTX 5070 (this box).

---

## Measured: vship CVVDP on RTX 5070

Built vship via `make buildcuda` (nvcc 13.2, sm_75 with JIT to sm_120 Blackwell). Source: `~/work/refs/Vship`, library: `~/work/refs/Vship/libvship.so`. Bench harness saved at `~/work/zen/zenmetrics-refs/vship-bench-2026-05-16/`.

**Vship CVVDP, RTX 5070, 5 warmup iters dropped, 20 measured:**

| Size | MP | median ms/call | median ns/px |
|---|---|---|---|
| 256×256 | 0.07 | 2.51 | 38.2 |
| 512×512 | 0.26 | 3.46 | 13.2 |
| 1024×1024 | 1.05 | 4.28 | 4.08 |
| 2048×1536 | 3.15 | 8.71 | 2.77 |
| **4000×3000** | **12.00** | **28.82** | **2.40** |

Linear fit `t = α + β·pixels`: **α ≈ 2.4 ms fixed overhead, β ≈ 2.2 ns/px steady-state slope.**

**Comparison on the same hardware** (cvvdp-gpu's 12 MP numbers from `crates/cvvdp-gpu/benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`, RTX 5070, CUDA 13.0):

| Path | ns/px @ 12 MP | vs vship |
|---|---|---|
| **vship CVVDP** | **2.40** | **1.00× (baseline)** |
| pycvvdp v0.5.4 (PyTorch+cuDNN) | 14 | 5.8× slower |
| cvvdp-gpu warm-ref | 34 | 14.2× slower |
| cvvdp-gpu cold | 62 | 25.8× slower |

The gap is real. The architectural reasons (LDS tiling, pointer-jumping reductions, cross-stream concurrency, min-abs in shared-load) are what create this gap and they're all portable to CubeCL.

**Caveats on the measurement:**
- Vship CVVDP is temporal. With `Vship_ResetCVVDP` between iters the temporal history is empty, so Y_transient stays 0 and Y_sustained uses only the current frame. This means vship is doing slightly less work than the full temporal pipeline — but cvvdp-gpu is still-only, so this is the fair single-image comparison.
- Random uint8 input. Kernel work on the GPU is content-independent (no early-out per pixel), so the timing isn't sensitive to image content.
- Same machine for vship's measurement (this RTX 5070); cvvdp-gpu's numbers are from their committed benchmark on this same hardware.
- Single-frame, single-stream, single-handler. No `recommend_parallel` benefit applied to either side.

The 15× steady-state gap is the structural ceiling. Closing it to within 2× (i.e. cvvdp-gpu at ~5 ns/px) is achievable from the porting plan below.

---

## TL;DR

Vship is **fast** because it does five things that cvvdp-gpu does not:

1. **Tile-load + LDS reuse** for every spatial kernel. Each gaussian blur, malta-diff, downscale, and masking blur loads a halo-padded tile into `groupshared` once per workgroup. cvvdp-gpu currently reloads every tap from global memory per thread (the `downscale_kernel` does 25 uncoalesced loads per output pixel with **zero** shared memory reuse).
2. **Pointer-jumping LDS reductions**, not atomics. Vship's `sumreduce`, `pooling::reduceSum`, and `diffnorms::sumreducenorm` all reduce within a workgroup via LDS, optionally fused (3 or 6 reductions in one kernel). cvvdp-gpu uses `Atomic<f32>::fetch_add` from every pixel into 27 global slots — at 12 MP that's 324M atomic adds per JOD call.
3. **Cross-stream concurrency.** Reference and distorted pipelines run on independent CUDA/Vulkan queues with event-based sync at convergence points. cvvdp-gpu has `recommend_parallel` for cross-call parallelism but no intra-call stream split.
4. **Min-abs / sum-diff / etc. fused into LDS load.** When you'd otherwise need a separate kernel to produce `min(|R|, |T|)`, vship computes it during the shared-mem tile fill — zero added bandwidth.
5. **Compile-time skip-map dispatch.** 8 SSIMU2 specializations prune the (plane, scale) cells whose final-score weight is ≤ 0.01. About a 30% kernel work reduction on SSIMU2 cheap cells.

Vship is **not** faster because of any AMD-specific magic. There's no wavefront/subgroup primitives anywhere (deliberate portability — same source runs on CUDA via preprocessor macros, on Vulkan via SLANG→SPIR-V). No fp16. No texture units. No descriptor sets in the Vulkan path (uses `bufferDeviceAddress`). All speed comes from kernel structure and memory layout.

For cvvdp-gpu specifically, the steady-state 2.4× gap vs pycvvdp (which itself loses to vship on equivalent ssimu2/butter cases) is dominated by:
- the 25-load-per-thread downscale,
- the 13-tap PU blur with the same pattern,
- and the atomic pool reduction.

A focused 3-4 week port of vship's LDS+pointer-jumping pattern to those three kernels should close most of the gap.

---

## How vship is laid out (one-page version)

```
src/
├── VshipAPI.h          ← stable C ABI: handler-per-metric, opaque IDs
├── VshipColor.h        ← color enums (matrix, transfer, primaries)
├── VshipLib.cpp        ← single TU; #ifdef VULKANBUILD picks backend
├── FFVship.cpp         ← FFmpeg-based CLI (producer/consumer pipeline)
│
├── HIP/                ← HIP source = also CUDA source via preprocessor
│   ├── util/           ← preprocessor.hpp aliases hipMalloc→cudaMalloc etc.
│   ├── gpuColorToLinear/   ← YUV→linear RGB, chroma upsample, transfer fns
│   ├── ssimu2/         ← SSIMULACRA2: downsample, makeXYB, gaussianblur, score
│   ├── butter/         ← Butteraugli: opsin, freq-separate, malta, mask
│   └── cvvdp/          ← CVVDP: temporal, csf, lpyr, masking, pooling
│
└── Vulkan/             ← parallel tree using SLANG shaders
    ├── util/           ← vulkanDeviceManager, VMA allocator
    ├── ssimu2/
    │   ├── shaders/*.slang   ← source
    │   └── *.hpp             ← host dispatch
    │ (libvshipSpvShaders/*.spv compiled offline, committed to repo)
    ├── butter/
    └── cvvdp/
```

Single source for the kernels via the preprocessor trick at `src/HIP/util/preprocessor.hpp:22-80`:

```c
#ifdef __CUDACC__
#define hipMalloc cudaMalloc
#define hipStream_t cudaStream_t
// ... ~60 more aliases
#endif
```

The Makefile compiles the SAME `VshipLib.cpp` as `.cpp` for HIP (`hipcc`) or as `.cu` for CUDA (`nvcc -x cu src/VshipLib.cpp -arch=native -shared`). The Vulkan path is a parallel implementation, not a fallback.

License: **MIT NON-AI**. Cannot be included in any ML training dataset and cannot be used to train ML models. Algorithms are upstream-derivable (CVVDP from gfxdisp/ColorVideoVDP BSD-3, SSIMU2 from cloudinary Apache-2.0, butter from libjxl BSD-3) — re-implementing from upstream is fine, line-for-line copying from vship is not.

---

## How cvvdp-gpu is laid out

```
crates/cvvdp-gpu/
├── src/
│   ├── lib.rs              ← public surface, Error, MAX_LEVELS=9
│   ├── params.rs           ← DisplayModel, DisplayGeometry, SRGB_LINEAR_TO_DKL
│   ├── host_scalar.rs      ← all-host reference (parity ground truth)
│   ├── pipeline.rs         ← 2955 lines — Cvvdp<R>, all dispatch
│   └── kernels/
│       ├── color.rs        ← srgb_to_dkl_kernel
│       ├── pyramid.rs      ← downscale, upscale_v, upscale_h, subtract_weber_3ch
│       ├── csf.rs          ← csf_apply_6ch_kernel + 32x32x3 LUT
│       ├── masking.rs      ← pu_blur_h_3ch, pu_blur_v_3ch_scaled, mult_mutual_3ch
│       └── pool.rs         ← pool_band_3ch_kernel (atomic-f32)
└── benches/, tests/, docs/
```

**Every kernel uses `CubeDim::new_1d(64)` with one thread per pixel.** No LDS. No subgroup ops. The only reduction primitive is `Atomic<f32>::fetch_add`. CubeCL 0.10.0-pre.4 is the entire GPU dependency.

Backend matrix:
- CUDA / Vulkan / DX12 / HIP via cubecl → all kernels supported, including atomic add
- `cubecl-cpu` → panics on `Atomic<f32>::fetch_add` (workaround: `compute_dkl_jod_host_pool`)
- Metal → silently no-ops on the atomic (same workaround)

Current perf (RTX 5070, from `benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`):
| Path | per-pixel | vs pycvvdp |
|---|---|---|
| pycvvdp v0.5.4 CUDA | 14 ns/px | 1.00× |
| cvvdp-gpu cold | 62 ns/px | **4.4× slower** |
| cvvdp-gpu warm-ref | 34 ns/px | **2.4× slower** |

Parity is locked at ≤0.005 JOD vs pycvvdp v0.5.4 across all q={1,5,20,45,70,90} fixtures — the speed gap is purely a GPU-engineering gap, not an algorithm gap.

---

## Where vship's tricks come from, kernel by kernel

This is the meat of the document. For each technique, I cite the vship file and explain what it does, then describe the analogous cvvdp-gpu site and what would change.

### 1. LDS-tiled separable gaussian (the big one)

**Vship's pattern** (`src/HIP/ssimu2/gaussianblur.hpp:24-77`):

- Workgroup: 16×16 threads = 256
- LDS: `__shared__ float tampon[32*32 * 3]` (12 KiB total, 3 tiles for ref+dist+temp)
- Each thread issues 4 conditional loads to populate the 32×32 region (covers the 16×16 valid interior + 8-pixel halo on each side)
- Horizontal blur: 17-tap, kernel and integral-table precomputed
- `__syncthreads()` then vertical blur
- Output: central 16×16 region written back

**Key sub-trick: integral table for border weights** (`gaussianblur.hpp:8-22`):

```c
// gaussiankernel_integral[k] = sum_{i<k} gaussiankernel[i]
// border-aware renormalization is then:
float weight = gaussiankernel_integral[end] - gaussiankernel_integral[beg];
output = sum / weight;
```

O(1) edge handling instead of a per-thread sum-of-weights loop. Classic technique, applied at the right level.

**Equivalent in cvvdp-gpu**: `kernels::pyramid::downscale_kernel` (`pyramid.rs:567-704`) is a 5×5 stencil (25 taps), but it's **fully unrolled per thread** — each output thread reads 25 input samples directly, with no LDS sharing across the workgroup. Neighboring threads' tap regions overlap by 20/25, so 80% of reads are redundant.

**Port:**
```rust
// New: kernels::pyramid::downscale_tiled_kernel
// CubeDim: (16, 16, 1)
// LDS: tile<f32, (20, 20)>  // 16×16 outputs + 4-px halo each side
// Per thread: 1 fused load with reflection indexing, sync, then 25-tap separable read from LDS
```

Expected speedup on the downscale stage: 3-5× from amortizing the 25-load fan-out. CubeCL 0.10 supports LDS via `SharedMemory<f32>`.

This is also the explicit recommendation from `crates/burn-conv-spike/README.md` post-mortem — and explains why burn's im2col→GEMM CMMA path regressed 4.32× on 1-channel work: it can't tile separably in LDS because cuDNN's path is dense-matmul shaped.

### 2. Hierarchical workgroup reduction (kill the atomics)

**Vship's pattern** (`src/HIP/ssimu2/score.hpp:144-188`, `pooling.hpp:5-78`):

- Workgroup: 1024 threads, 1D
- LDS: `__shared__ float sum[N_outputs * 1024]` (24 KiB for 6 outputs)
- Each thread loads 1 input value (with power transform if Lp pooling)
- Pointer-jumping reduction: `for (next = 1; next < 1024; next *= 2) { if (thx < next) sum[thx] += sum[thx + next]; __syncthreads(); }`
- Thread 0 writes the per-block sum to global output
- For total reductions, the host re-dispatches with the previous output as input until `n ≤ 1024`

Notable: vship's reduction loop uses `if (thx % (next*2) == 0)` (the textbook-suboptimal predicate; warp-divergent in the last passes). The well-known better form is `if (thx < next) sum[thx] += sum[thx+next]` which keeps active threads contiguous — ~2× speedup on the last 5 levels of the reduce. We should use the better form in our port; vship's choice is an easy improvement.

**Fused 6-output reduction** (`score.hpp:144-188`): one kernel produces 6 different per-block sums (3 SSIM-component maps × 2 norm orders). LDS layout is 6 stripes of 1024 floats; pointer-jumping operates on each stripe independently. Saves 5 separate reduction passes.

**Fused 3-norm reduction** (`butter/diffnorms.hpp:11-46`): one kernel produces `pow(x, Qnorm)`, `|x|³`, `|x|` simultaneously — sum/sum/max in interleaved LDS. Eliminates 2 separate reduction passes per diffmap.

**Equivalent in cvvdp-gpu**: `kernels::pool::pool_band_3ch_kernel` (`pool.rs:202-235`) does:

```rust
// every thread, every pixel:
let v_a = safe_pow(abs(ref_a - dist_a), beta);
let v_rg = safe_pow(abs(ref_rg - dist_rg), beta);
let v_vy = safe_pow(abs(ref_vy - dist_vy), beta);
Atomic::<f32>::fetch_add(&partials_h[slot_a], v_a);
Atomic::<f32>::fetch_add(&partials_h[slot_rg], v_rg);
Atomic::<f32>::fetch_add(&partials_h[slot_vy], v_vy);
```

At 12 MP and 9 levels that's `12M × 3 channels × 9 bands ≈ 324M atomic adds`, all contending on 27 global locations. Even if the GPU is good at non-blocking atomics, contention serializes them per slot.

**Port:**
```rust
// New: kernels::pool::pool_band_3ch_tiled_kernel
// CubeDim: (256, 1, 1)
// LDS: SharedMemory<f32, 256 * 3>  // 3 KiB
// Per thread: compute 3 powers, write to LDS[3*thx..3*thx+3]
// sync
// Pointer-jump reduction over 256 threads (8 levels)
// Thread 0: 3 atomic adds (per workgroup, not per pixel)
```

At 12 MP that drops from 324M atomics to `(12M / 256) × 3 × 9 ≈ 1.27M atomics` — **255× reduction in atomic traffic.** Expected pool-stage speedup: 5-10×.

### 3. Min-abs (or sum, diff, etc.) fused into LDS load

**Vship's pattern** (`HIP/cvvdp/gaussianBlur.hpp:28-40`, `GaussianSmartSharedLoadMinAbs`):

The masking model needs `min(|R|, |T|)` before blurring. Naive approach: one kernel produces `min(|R|, |T|)` to a temp buffer, second kernel blurs. Vship instead fuses:

```c
// During the LDS tile fill, each thread does:
float r = ref_buf[idx];
float t = dist_buf[idx];
tampon[lds_idx] = (fabsf(r) < fabsf(t)) ? fabsf(r) : fabsf(t);
__syncthreads();
// then blur the tampon directly
```

The fusion costs zero bandwidth (we'd have loaded both R and T anyway for the blur) and eliminates one whole kernel + one whole temp buffer pass.

**Equivalent in cvvdp-gpu**: `kernels::masking::min_abs_3ch_kernel` is a separate launch that writes a temp buffer that the next blur reads. Fusing it into the blur saves one launch and ~`n_px * 3 * 4 bytes` of write+read traffic per band.

**Port:** when porting the tiled PU blur (Tier 2 below), do the min-abs in the same kernel during LDS tile fill. Net savings per band per call: ~144 MB R/W at 12 MP × 3 channels.

### 4. Cross-stream concurrency

**Vship's pattern** (`HIP/ssimu2/main.hpp:32-50`, `HIP/cvvdp/main.hpp:24-94`):

Each handler owns N CUDA streams (typically 2-4). The orchestrator splits work that's independent between reference and distorted:

```c
// Stream 0: ref pipeline
// Stream 1: dist pipeline  (downsample, pyramid, csf)
// hipEventRecord on stream 1's completion
// hipStreamWaitEvent on stream 0 to merge
// Stream 0: cross-channel masking, pool (needs both)
```

On RTX 5070, the front of the pipeline (downscale, color, weber) is fully bandwidth-bound. Splitting ref and dist onto separate streams lets the GPU overlap them — the SM scheduler keeps both warps in flight.

**Equivalent in cvvdp-gpu**: there is no intra-call stream split. `_dispatch_d_bands_dist_and_band_loop` (`pipeline.rs:1792-2085`) runs everything on one default stream. `recommend_parallel` exposes cross-call parallelism (different image pairs in different threads) but not within one call.

**Port:** CubeCL 0.10 supports multiple queues via `ComputeStream::new` (or whatever the equivalent — check cubecl-cuda's API). Refactor to assign ref-side dispatches to stream A and dist-side to stream B, sync at the band-loop merge. Lower-priority because it's complex and the LDS wins above are bigger.

### 5. Float2-packed FMA

**Vship's pattern** (`HIP/cvvdp/temporalFilter.hpp:153-178`):

```c
// 4 temporal channels: Y_sustained, RG, YV, Y_transient
// Pack 2 channels per float2 across the FMA:
float2 kernelTemp = make_float2(filter[0][k], filter[3][k]);  // Y_sustained, Y_transient
float2 valueTemp = make_float2(frame[x], frame[x]);
resY_Y = fmaf(valueTemp, kernelTemp, resY_Y);  // 2 channels per FMA
```

Doubles the FMA throughput on hardware that issues `float2` as one instruction (most desktop NVIDIA + AMD).

**Equivalent in cvvdp-gpu**: cvvdp-gpu doesn't have temporal channels (intentionally still-only). But the same trick applies to the 3 DKL channels in `csf_apply_3ch_kernel`, `pu_blur_*_3ch_kernel`, and the pool kernel: pack 2 channels per `vec2<f32>` where the third lives separately. Or use `vec3<f32>` directly if CubeCL/the target backend emits packed instructions for it.

**Port:** experimental — try once the bigger wins land, measure on RTX 5070.

### 6. CSF LUT pre-reduced at init

**Vship's pattern** (`HIP/cvvdp/csf.hpp:23-58`):

The CSF table is 32×32×4 (luminance × frequency × channel). At init, vship interpolates along the frequency axis on CPU (which is fixed per band — one frequency per pyramid level), producing per-band 32×4 strips. The GPU only ever searches the 32-entry luminance axis.

**Equivalent in cvvdp-gpu**: `kernels::csf::csf_apply_6ch_kernel` does the same pre-reduction (`precompute_logs_row` at `csf.rs:211-220`), but reads the resulting `logs_row` from global memory per pixel. The whole CSF LUT for a band is 32 floats × 3 channels = 384 bytes per band — easily LDS-cacheable.

**Port:** load the per-band `logs_row` into LDS once at the start of `csf_apply_6ch_kernel`, then per-pixel reads hit LDS instead of global. Net: ~`n_px × 3 × 4 bytes` of global reads per band eliminated. ~3 µs at 12 MP. Small but free.

### 7. `pow(2, xcm_weights[i])` precomputed

**Both vship AND cvvdp-gpu** do this at runtime:

- Vship: `HIP/cvvdp/maskingModel.hpp:57-60` does `powf(2.0, xcm_weights[i])` 16 times per pixel for the 4×4 cross-channel matrix.
- cvvdp-gpu: `kernels::masking::mult_mutual_3ch_with_blurred_kernel` has 9 `safe_pow(2.0, XCM[i])` calls per pixel for the 3×3 matrix.

**`xcm_weights` are compile-time constants.** They should be precomputed once and stored as `XCM_LINEAR_3X3 = [2^XCM[0], 2^XCM[1], ...]`. Even at fast `pow` (~20 cycles), 9 unnecessary `pow` calls per pixel × 12M pixels × 9 bands = 9.7 billion cycles wasted per JOD call.

**Port:** trivial — add 9-element `const XCM_LINEAR_3X3: [f32; 9]` next to `XCM_3X3` in `masking.rs:69-76`, use it directly. **This is the cheapest win in the entire document.**

### 8. Sum-and-diff SSIM fusion (SSIMU2-specific, but ports to butter)

**Vship's pattern** (`HIP/ssimu2/score.hpp:233-256`):

Standard SSIM needs `s11 = blur(im1²)` and `s22 = blur(im2²)`. Vship instead computes:

```c
// One extra-blur of the product (a*b)
s12 = blur(im1 * im2);
// One blur of (im1 + im2)²
sumsquared = blur((im1 + im2) * (im1 + im2));
// Derive the rest algebraically:
s11_plus_s22 = sumsquared - 2 * s12;
```

Saves one 17-tap separable blur per (scale, plane). On SSIMU2 with 18 scale×plane combinations that's 18 fewer expensive blurs per call.

**Equivalent in cvvdp-gpu**: N/A directly (cvvdp doesn't use SSIM). But useful intuition for any future ssimu2-gpu port from CubeCL.

### 9. In-place plane recycling

**Vship's pattern** (`HIP/butter/main.hpp:46-47, 96-98`):

After a buffer is no longer read, the orchestrator aliases it to the next buffer needed:

```c
src1_d = block_diff_dc;  // alias src1 → dc buffer once src1 unused
src2_d = block_diff_ac;
// later:
temp = mask;
lf2 = temp3;
mf2 = temp4;
```

Reduces working set from 23 planes to ~10 planes — meaningful at 12 MP (450 MB → 200 MB).

**Equivalent in cvvdp-gpu**: scratch is managed by `DBandsScratch`, `WeberScratch` structs (`pipeline.rs` Level/DBandsScratch). Each level has its own slab. There's room to recycle baseband-level scratch into deeper-level scratch since deeper levels are smaller — likely minor gain on VRAM, no perf gain.

### 10. Skip-map dispatch (SSIMU2-specific)

**Vship's pattern** (`HIP/ssimu2/score.hpp:329-358`): 8 template specializations of the same kernel that skip computing whichever of (ssim, artifact, detailloss) has weight ≤ 0.01 for a given (plane, scale). LDS budget drops from 12 KB to 8 KB on the cheapest combinations.

**Equivalent in cvvdp-gpu**: N/A — cvvdp doesn't have this kind of weighted-skip structure. The pool weights are non-zero across all bands/channels.

---

## Prioritized port plan

| # | Change | Impact | Effort | Where |
|---|---|---|---|---|
| ~~T1.A~~ | ~~Precompute `2^xcm` constants~~ | **Already done before this analysis** — `XCM_3X3` (`masking.rs:69-76`) is pre-exponentiated, and `pu_scale = 10^MASK_C` and `d_max_lin = 10^D_MAX` are baked into kernels as `f32::new(...)` literals. The initial survey agent misread the call sites. | n/a | done |
| **T1.B** ✅ | LDS-tiled `downscale_kernel` | **Landed 2026-05-16** as `downscale_tiled_kernel`. 16×16 workgroup, 36×36 LDS tile (5.2 KB). | 1 day | `kernels/pyramid.rs:706-879` |
| **T1.C** ✅ | LDS-reduction pool kernel | **Landed 2026-05-16** as `pool_band_3ch_lds_kernel`. 256-thread workgroup, pointer-jumping reduce, 1 atomic per workgroup per channel. 255× atomic-traffic reduction at 12 MP. | 1 day | `kernels/pool.rs:251-358` |
| **T2.D** | LDS-tiled PU blur (h + v) | 3-5× on masking blur | 2-3 days | `kernels/masking.rs:281-533` |
| **T2.E** | Min-abs fused into PU blur LDS load | saves 1 launch + 144 MB R/W per band | 0.5 day | merge into T2.D |
| **T2.F** | CSF logs_row into LDS | ~3 µs / call at 12 MP | 0.5 day | `kernels/csf.rs:401-475` |
| **T2.G** | Subgroup reduction for final 5 pool levels | ~10% on pool stage | 0.5 day | `kernels/pool.rs` |
| **T3.H** | Multi-stream ref/dist concurrency | up to 1.4× end-to-end | 3-5 days | `pipeline.rs::_dispatch_d_bands_*` |
| **T3.I** | Tiled fused upscale_v+h (single kernel) | merges 2 launches per level | 2 days | `kernels/pyramid.rs:727-927` |
| **T3.J** | LDS-tile reuse across consecutive levels | hard, may not work | 5+ days | pyramid stages |
| **T4.K** | Per-size pool dispatch (small bands → atomic kernel, large → LDS) | unblocks T1.C's 256² regression | 0.5 day | pool dispatch site |

### T1.B + T1.C + T4.K + T4.L measured on RTX 5070 (2026-05-16)

Commits on `master`:
- `c649f135` — T1.B + T1.C (LDS-tiled downscale + LDS-reduction pool)
- `2a0f4bd0` — T4.K (per-size pool dispatch, fixes T1.C's 256² regression)
- `e05ec2e3` — T4.L (pack RGBA u32 upload — the biggest single win, 3.2× on warm-ref)

### T4.L: profiling-driven win

After T4.K, `nsys profile -t cuda --stats=true` on the warm-ref harness revealed the dominant bottleneck was **NOT GPU compute** — it was `cuMemcpyHtoDAsync` at **55.7% of wall time** (301 ms across 5 iters), dominated by the per-iter dist sRGB upload widened from u8 to u32 (36 MB → 144 MB at 12 MP).

The fix: pack 3 sRGB bytes per pixel into ONE u32 (R | G<<8 | B<<16) instead of widening each byte. Kernel unpacks with 3 shifts + 3 ANDs, which is free vs the upload time saved. Cuts upload 3× per call.

GPU kernel time per iter is now ~9-10 ms (was already small but invisible behind upload); upload per iter is ~40 ms (still pageable-memory limited at ~1.25 GB/s vs PCIe 4×16 ~32 GB/s — full PCIe needs pinned host memory which CubeCL 0.10 doesn't expose).

### Cumulative measurements vs master baseline 2610ae9c

| Path | Baseline | After T4.L | Cumulative speedup |
|------|---------|------------|--------------------|
| 12 MP cold | 655.9 ms | 180.3 ms | **3.64×** |
| 12 MP warm-ref | 351.7 ms | 95.3 ms | **3.69×** |
| 1 MP cold | 30.5 ms | 10.2 ms | **2.99×** |
| 1 MP warm-ref | 17.5 ms | 4.84 ms | **3.62×** |
| 256² cold | 4.7 ms | 3.7 ms | 1.27× |
| 256² warm-ref | 2.6 ms | 2.9 ms | 0.90× (noise at this size) |

12 MP warm-ref **ns/px: 29.3 → 7.94**.

**Gap to vship 2.40 ns/px on the same RTX 5070 closed from 14.2× to 3.3×.**

### What's left between us and vship parity

nsys post-T4.L shows the next-biggest cost is *still* cuMemcpyHtoDAsync (~53% of remaining time), at ~40 ms per iter for the ~48 MB packed dist upload. Closing it requires:

1. **Pinned host memory for the dist upload** — would unlock full PCIe bandwidth (~1.5 ms per iter vs 40 ms). CubeCL 0.10 has no pinned-memory API; either patch CubeCL or use a different runtime.
2. **Persistent `src_ref` buffer reuse across calls** — avoid the ~9 ms/iter `cuMemAllocAsync` overhead. Same constraint as (1).
3. **Possibly Array<u8> input to drop another 25% upload** — needs cubecl 8-bit storage support verification.

After those, the remaining ~10 ms/iter is actual GPU compute, putting us within ~3× of vship's 2.40 ns/px (i.e., ~5-7 ns/px). Closing that last ~2-3× requires either multi-stream concurrency (T3.H) or kernel-mega-fusion (vship-style 1-launch-per-band masking), both multi-day efforts.

### T1.B + T1.C + T4.K measurements (mid-session)

Earlier measurements as the changes landed:

| Path | Baseline | T1.B+T1.C | T1.B+T1.C+T4.K | Cumulative |
|------|---------|-----------|----------------|------------|
| 12 MP cold | 655.9 ms | 547.1 | **479.5** | **1.37×** |
| 12 MP warm-ref | 351.7 ms | 277.3 | **244.3** | **1.44×** |
| 1 MP cold | 30.5 ms | 20.9 | **16.2** | **1.88×** |
| 1 MP warm-ref | 17.5 ms | 10.8 | **9.2** | **1.90×** |
| 256² cold | 4.7 ms | 5.1 (regression) | **3.8** | **1.24×** |
| 256² warm-ref | 2.6 ms | 3.7 (regression) | **2.2** | **1.18×** |

12 MP warm-ref **gap to vship closed from 14.2× to 8.5×**.

ns/px at 12 MP warm-ref: 29.3 (baseline) → 23.1 (T1.C) → **20.4 (T4.K)**.

T4.K was discovered by accident: T1.C regressed 256² because at small bands the LDS-pool kernel's 8 syncs + 256 atomics costs more than the per-pixel-atomic kernel's 65K atomics on 3 slots. Routing bands below 16K pixels through the original atomic kernel not only fixed 256² but also further improved 12 MP — the 9-level pyramid's deepest 4-5 bands are below 16K pixels regardless of starting size and were also paying sync overhead under unconditional LDS.

All 278 cvvdp-gpu tests pass through each commit. Parity ≤ 0.005 JOD vs pycvvdp v0.5.4 preserved.

**Combined expected speedup** if T2.D + T2.F + T2.E land next: cvvdp-gpu warm-ref from 20.4 ns/px toward **~8-10 ns/px**, beating pycvvdp v0.5.4 (14 ns/px).

**Reaching vship's 2.4 ns/px** requires the Tier 3 changes too (multi-stream concurrency + tile-reuse across pyramid levels). Multi-month effort but the structural ceiling is clear: vship proves the metric can be 8× faster than where we are today (which is already 1.44× faster than this morning), on this exact hardware.

---

## Bench setup (done)

1. **Build vship CUDA.** `nvcc -x cu src/VshipLib.cpp -std=c++17 -I src -arch=sm_75 -shared -Xcompiler -fPIC -o libvship.so` works cleanly with nvcc 13.2. sm_75 binary JITs to sm_120 (RTX 5070 Blackwell) at first run. Library is 5.0 MB.
2. **Bench harness.** `~/work/zen/zenmetrics-refs/vship-bench-2026-05-16/vship_cvvdp_bench.c` — minimal C, links libvship, calls `Vship_CVVDPInit3` + `Vship_PinnedMalloc` + `Vship_ComputeCVVDP` against random uint8 RGB pairs. Compile: `gcc vship_cvvdp_bench.c -O2 -I /home/lilith/work/refs/Vship/src -L /home/lilith/work/refs/Vship -lvship -Wl,-rpath,/home/lilith/work/refs/Vship`.
3. **cvvdp-gpu reference.** Used the committed `crates/cvvdp-gpu/benchmarks/pycvvdp_12mp_cuda_2026-05-14.md` numbers (same RTX 5070, CUDA 13.0). Did not re-run to avoid touching the actively-claimed cvvdp worktree.

**Open follow-up benches** when capacity allows:
- Run cvvdp-gpu in `~/work/zen/zenmetrics--cvvdp-new` once the active agent finishes, to confirm the committed numbers still hold post-tick-415.
- Side-by-side same-image comparison (matching JOD against the same content) — vship operates on BT.709+display-model sRGB inputs, cvvdp-gpu on the same. Output JOD should match within ~0.05 on natural content.
- Vulkan path benchmark requires WSL2 Vulkan→GPU passthrough (currently llvmpipe only). Skipped.

---

## License and attribution notes

Vship is **MIT NON-AI**. Don't copy SLANG/HIP source line-for-line. Don't include vship binaries in any picker training pipeline. The mathematical algorithms are upstream-derivable:

- CVVDP from gfxdisp/ColorVideoVDP (BSD-3-clause) — the CSF LUT `csf_lut_weber.hpp` was almost certainly regenerated from gfxdisp's `data/` directory.
- SSIMU2 from cloudinary/ssimulacra2 (Apache-2.0).
- Butteraugli from libjxl/libjxl (BSD-3-clause).
- The Burt-Adelson 5-tap kernel is published 1981 — public knowledge.
- The GPU optimization patterns (LDS tiling, pointer-jumping reduce, integral tables for borders) are CUDA-cookbook common — public knowledge.

What we should NOT do:
- Translate vship .hpp/.slang files line-for-line into Rust.
- Use vship's CSF LUT values directly — regenerate from gfxdisp's reference.

What we CAN do:
- Re-implement the **structural choices** (block size 16×16, 32×32 LDS tiles, separable-then-fused approach, etc.) — those are facts about GPU efficiency, not creative expression.
- Cite vship as inspiration in commit messages.
- Re-derive constants from upstream BSD-3 / Apache-2.0 sources.

---

## Open questions for the maintainer

1. **Pool reduction priority.** Is T1.C (atomic→LDS+per-workgroup) high enough priority to land before vast.ai backfill completes, or should it ship after to avoid mid-flight parity risk?
2. **Tile size choice.** Vship uses 16×16=256 for spatial work, 1024 for reductions. CubeCL defaults to 64. Want to standardize on 16×16 for tiled kernels or follow Vulkan's 1024?
3. **Wgpu native vs cubecl-cuda.** Does the project want LDS-tiled WGSL too, or focus the perf push on cubecl-cuda only? WGSL `workgroup` memory works; the win is the same.
4. **Subgroup ops.** CubeCL 0.10 supports `subcube_sum` on CUDA. Want to require it for the production cvvdp-gpu path or keep the LDS-only path for portability?
5. **Multi-stream timing.** T3.H needs CubeCL multi-queue. Has that been used in zenmetrics yet?

---

## Where to find the source material

Vship (local clone):
- `~/work/refs/Vship/src/HIP/cvvdp/main.hpp` — orchestrator
- `~/work/refs/Vship/src/HIP/cvvdp/maskingModel.hpp` — masking core (the model to port)
- `~/work/refs/Vship/src/HIP/cvvdp/csf.hpp` — CSF LUT plumbing
- `~/work/refs/Vship/src/HIP/cvvdp/lpyr.hpp` — Burt-Adelson pyramid + L_bkg
- `~/work/refs/Vship/src/HIP/cvvdp/pooling.hpp` — LDS pointer-jump reduction
- `~/work/refs/Vship/src/HIP/cvvdp/temporalFilter.hpp` — temporal (not needed for stills)
- `~/work/refs/Vship/src/HIP/ssimu2/score.hpp` — fused 6-reduce + skip-map
- `~/work/refs/Vship/src/HIP/butter/maltaDiff.hpp` — 16-orientation correlator
- `~/work/refs/Vship/src/Vulkan/cvvdp/shaders/*.slang` — SLANG mirrors of HIP
- `~/work/refs/Vship/LICENSE` — MIT NON-AI

zenmetrics:
- `~/work/zen/zenmetrics--cvvdp/crates/cvvdp-gpu/src/kernels/pyramid.rs:567-704` — downscale (T1.B target)
- `~/work/zen/zenmetrics--cvvdp/crates/cvvdp-gpu/src/kernels/pool.rs:202-235` — pool (T1.C target)
- `~/work/zen/zenmetrics--cvvdp/crates/cvvdp-gpu/src/kernels/masking.rs:69-76` — XCM constants (T1.A target)
- `~/work/zen/zenmetrics--cvvdp/crates/cvvdp-gpu/src/kernels/masking.rs:281-533` — PU blur (T2.D target)
- `~/work/zen/zenmetrics--cvvdp/benchmarks/pycvvdp_12mp_cuda_2026-05-14.md` — baseline numbers
- `~/work/zen/zenmetrics--cvvdp/docs/PORT_STATUS.md` — parity status
- `~/work/zen/zenmetrics--cvvdp/docs/CHROMA_DRIFT_INVESTIGATION.md` — parity history
- `~/work/zen/zenmetrics--cvvdp/crates/burn-conv-spike/README.md` — burn-conv-spike post-mortem (same conclusion: LDS-tile direct stencil)
