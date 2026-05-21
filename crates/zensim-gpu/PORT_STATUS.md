# zensim-gpu port status

Multi-vendor GPU port of `zensim-cuda` using CubeCL. Algorithmic parity
target is the published `zensim` v0.2.8 crate with
`ZensimProfile::latest()` (= `WEIGHTS_PREVIEW_V0_2`, 228 features = 4
scales × 3 channels × 19 features).

## Module status

| Module | Source | LOC | Status | Notes |
|---|---|---|---|---|
| `kernels::color` | `zensim-cuda-kernel/src/color.rs` | ~110 | ✅ ported | sRGB packed-u8 → planar positive XYB. 256-entry LUT uploaded as `Array<f32>` (cubecl 0.10 can't index host-side `[f32; 256]` constants from `#[cube]`). `cbrt` substituted with `f32::powf(_, 1.0/3.0)` — the magic-constant Newton seed in CPU's `cbrtf_fast` requires `reinterpret_cast<u32>(K_B0)` which cubecl-cuda's codegen rejects for literal-folded constants. Drift vs CPU `cbrtf_fast` is a few ULPs, well below the SSIM normalisation threshold. Same call as dssim-gpu's Lab cbrt. |
| `kernels::pad` | `zensim-cuda-kernel/src/pad.rs` | ~30 | ✅ ported | Mirror-fill SIMD-padded columns; precomputed offset table on device. |
| `kernels::downscale` | `zensim-cuda-kernel/src/downscale.rs` | ~40 | ✅ ported | 2×2 box average with edge clamp on the **padded** plane (CPU zensim does not re-pad after downscaling — pad columns simply downscale along with everything else). |
| `kernels::blur` | `zensim-cuda-kernel/src/blur.rs` | ~70 | ✅ ported | Fused horizontal box-blur producing 4 outputs (`mu1`, `mu2`, `sigma_sq`, `sigma12`) per pixel. Mirror-x logic inlined in pure u32 (cubecl 0.10's `#[cube]` macro fights mixed signed-unsigned arithmetic). |
| `kernels::features` | `zensim-cuda-kernel/src/features.rs` | ~190 | ✅ ported | Fused V-blur + per-pixel feature extraction. **One thread per column** writes 17 f64 sums + 3 f32 maxes to per-column slots — no atomics needed (each column owns a unique slot). Host-side fold across columns produces the per-channel feature scalars. Avoids `Atomic<f64>` (cubecl 0.10 doesn't expose it) and `Atomic<f32>::fetch_max` (broken on Metal per gotcha G3.x). |
| Pipeline (`pipeline::Zensim`) | `zensim-cuda/src/lib.rs` | ~440 | ✅ wired | 4-scale pyramid with cached-reference state. SIMD padding matches `simd_padded_width` exactly so feature footprints stay aligned with CPU's `accum.n = padded_w × h`. |
| `Zensim::set_reference` / `compute_with_reference` | same | (above) | ✅ implemented | Reference pyramid cached after `set_reference`; subsequent `compute_with_reference` reads it without re-running the ref-side sRGB→XYB / pad / downscale chain. Cached-vs-direct drift ≤ 1e-3 on the noisy-gradient lock test. |

## Validated parity (RTX 5070, CUDA 13.2, host-side `zensim` v0.2.8)

Full test suite — **19 / 19 pass on CUDA** (verified 2026-05-20):

| Test file | Tests | Coverage |
|---|--:|---|
| `tests/cpu_parity.rs` | 3 | Basic + peak 228-feat per-slot parity (identical / noisy-gradient 64² / checkerboard 128² multi-strip) |
| `tests/extended_parity.rs` | 6 | Extended 300-feat (incl. masked block 228..300) + WithIw 372-feat structural |
| `tests/parity_lock.rs` | 8 | Aggregate score: synthetic edges + cached-vs-direct + JPEG corpus q70/q90 |
| `tests/weights_parity.rs` | 1 | Byte-for-byte CPU/GPU weights match `WEIGHTS_PREVIEW_V0_2` |
| `tests/opaque.rs` | 1 | `Zensim::compute_features_srgb_u8` opaque-API path |

`tests/parity_lock.rs` aggregate-score numbers (2026-05-20 re-run):

| Case | CPU score | GPU score | rel error |
|---|---|---|---|
| 32×32 identical gradient | 100.0 | 99.9531 | 4.7e-4 |
| 64×64 black vs white | -208.4879 | -208.4871 | 4.0e-6 |
| 64×64 noisy gradient (±8) | 63.6834 | 63.6830 | 6.3e-5 |
| `dssim-cuda` corpus q70.jpg | 80.9018 | 80.8850 | 2.1e-4 |
| `dssim-cuda` corpus q90.jpg | 91.3509 | 91.3486 | 2.5e-5 |
| Cached-vs-direct drift | (n/a) | (n/a) | ≤ 1e-3 score |

`tests/extended_parity.rs` per-feature parity numbers
(2026-05-20 re-run):

| Case | max \|gpu − cpu\| | Slot budget |
|---|---|---|
| 64² identical input (300 slots) | 6.80e-4 | 5e-2 abs |
| 64² noisy gradient (300 slots) | within budget on every slot | basic 2e-3 rel · peak/max 3e-2 rel · L8 5e-3 rel · **masked 5e-3 rel** |
| 128² checkerboard multi-strip (300 slots) | within budget on every slot | same as above |
| 64² WithIw[0..300] vs Extended[0..300] | 0.0 (bit-identical) | 5e-3 abs |
| 64² WithIw IW block (300..372) on noisy input | 72/72 non-zero, max \|val\| 0.236 | finite + magnitude < 1e3 |

Synthetic edge cases (grayscale, polar opposite, low-magnitude X
channels) sit comfortably under 1e-4 relative error. Real-image
corpus parity at q70 is ≈ 2.1e-4 (0.021 %) — within the cross-arch
FMA contraction floor that bounds CUDA-PTX vs CPU AVX-512 (the
`zensim-cuda` crate documents the same regime as "~ULP of cross-arch
FMA drift"). q90 and the synthetic cases land at ≤ 6e-5.

## Principled per-channel H-blur activity (2026-05-17, masked + IW blocks)

The masked-block (slots 228..300) and IW-block (slots 300..372) both
compute a per-channel "activity map" as
`activity[c] = box_blur(|src[c] - mu1[c]|)` and weight per-pixel
SSIM by `1/(1+k·a)` (masked) or `1+k·a` (IW). The CPU's pre-2026-
05-17 implementation had an accidental cross-channel cascade at
strip-overlap rows — `bufs.mu1` was reused across X→Y→B channels
via `std::mem::swap(&mut bufs.mu1, &mut bufs.mask)`, and the fused
V-blur only wrote inner rows of `mu1`, leaving overlap rows holding
the previous channel's stale state (or zero for X / prior-strip B
mask state for strip K≥1). See zensim's
`docs/PRINCIPLED_ACTIVITY.md` for the full RCA.

CPU was redesigned (commit `caf52d36` on
`feat/principled-activity`, shipped 2026-05-17 as
`2dab8f3` on zensim main) to use a per-channel strip-local
`H_blur(src)` as the activity-map reference at ALL strip rows
(inner + overlap). Channels are decoupled; the activity for each
channel sees only its own H-blurred source.

GPU `kernels::masked_iw_strip` was re-aligned (zenmetrics `1b8ccab`,
2026-05-17). The pre-fix host-side `populate_carryover` simulator,
the `carry: Array<f32>` kernel parameter, and the strip-K-vs-strip-0
branch in `masked_iw_strip_kernel` are all **deleted**. The new
kernel loads a DIAM-wider `wide_src[TX + 4R]` per (row, channel)
into shared memory and computes `H_blur(src)` on-the-fly into
`mu1_row[TX + 2R]`.

### Parity result after the redesign

All masked-block (228..300) and IW-block (300..372) features match
CPU within **5e-3 rel at every scale and every fixture size**,
including multi-strip scales (128² scale 0 + scale 1; 12 MP scale
0 + scale 1 + scale 2). No per-channel, per-scale tolerance
widening needed.

### Perf side effect

12 MP RTX 5070 WithIw 372-feature steady-state:

- Pre-redesign (with carryover plane + cross-channel branches):
  ~26.92 ms / iter.
- Post-redesign (principled per-channel H-blur, no carry, no
  cross-channel cascade): **~22.9 ms / iter**.
- **~15 % faster.** The kernel does DIAM extra src loads per row
  to compute H-blur on the fly but loses all per-channel branches
  in the hot path AND drops the ~23 MB device buffer that was
  being read every iteration.

### Caveat for downstream

Any sweep parquet data that pre-dates the 2026-05-17 fix and
includes masked/IW features (slots 228..372) was scored against
the **old cascade semantics**. The magnitude of shift in the
affected slots is bounded by the pre-fix 1.5-4 % rel GPU residual
that the fix eliminated. Re-bake any V_X model whose training
corpus consumed pre-2026-05-17 masked/IW values where those
features are load-bearing. CPU and GPU runtimes agree on the new
semantics, so current production scoring is consistent across the
two paths.

## The HF feature thresholds

The pipeline's host-side feature extraction matches CPU
`zensim::streaming::compute_features` exactly, including the
**per-pixel-variance threshold** that gates the HF ratios:

```rust
hf_energy_loss = if var_src > 1e-10 { (1.0 - var_dst / var_src).max(0.0) } else { 0.0 };
hf_energy_gain = if var_src > 1e-10 { (var_dst / var_src - 1.0).max(0.0) } else { 0.0 };
hf_mag_loss   = if mad_src > 1e-10 { (1.0 - mad_dst / mad_src).max(0.0) } else { 0.0 };
```

CPU's threshold is `var_src > 1e-10` (per-pixel variance), NOT
`den.abs() > 0.0`. Without this the f32 cancellation residue in
`Σ (s − mu1)²` for constant-colour channels (e.g., the B channel of a
grayscale image, where the XYB transfer collapses to a fixed value)
blows up the ratio and dominates the score. CPU and GPU agree on this
threshold so the feature output is bit-exact across the boundary
where the HF ratios fold to 0.

## FMA fusion match

The kernels use `cubecl::prelude::fma()` explicitly to replicate CPU's
`f32::mul_add` chains in:
- The opsin matrix multiply (`m00*r + (m01*g + (m02*b + K_B0))`)
- The `cbrtf_fast` Halley iterations
- The H-blur sums (`sum_sq = fma(s, s, fma(d, d, sum_sq))`)
- The per-pixel SSIM math (`num_m`, `num_s`, `denom_s`)

`absorbance_bias = -cbrtf_fast(K_B0)` is precomputed on the host using a
direct port of CPU's `cbrtf_fast` (magic-constant Newton seed + 2 Halley
iterations) and passed to the kernel as a runtime scalar — the bit-cast
inside `cbrtf_initial` triggers cubecl-cuda's
`reinterpret_cast<u32 const&>(literal)` codegen failure when applied to
a const-folded `K_B0` literal.

## Performance

Wall-clock measurements on RTX 5070 + CUDA 13.2 (Ryzen 9 7950X CPU
reference). `examples/bench.rs` runs N=8 iterations after 2 warm-ups.
`gpu_cwr` is the cached-reference path (`set_reference` once, then
`compute_with_reference` per call); `gpu_full` includes both phases
each call.

| Size       | CPU      | GPU (cached-ref) | GPU (full)  | GPU vs CPU (cwr) |
|------------|----------|------------------|-------------|------------------|
| 64×64      |  1.49 ms |   0.68 ms        |   0.81 ms   | **2.2× faster**  |
| 256×256    |  4.16 ms |   1.13 ms        |   1.28 ms   | **3.7× faster**  |
| 512×512    |  9.46 ms |   2.42 ms        |   2.26 ms   | **3.9× faster**  |
| 1024×1024  | 16.35 ms |   6.33 ms        |   7.54 ms   | **2.6× faster**  |
| 2048×2048  | 44.59 ms |  15.78 ms        |  24.91 ms   | **2.8× faster**  |
| 4096×4096  | 248.6 ms |  95.57 ms        | 179.49 ms   | **2.6× faster**  |

GPU now beats CPU at every resolution. Per-MP timing (lower is better):

| Size       | gpu_cwr (ms/MP) |
|------------|-----------------|
| 1024²      | 6.0             |
| 2048²      | 3.8             |
| 4096²      | 5.7             |

Best per-MP at 2 K. The 1 ms/MP target is **PCIe-bound, not compute-
bound**, on the WSL2 reference host:

- **set_reference** at 1 MP is ~2.5 ms — dominated by the 4 MiB H2D
  upload of packed sRGB. WSL2's virtualised PCIe runs at ~3 GiB/s; on
  native Linux this would be ~10 GiB/s and the upload would drop to
  ~0.4 ms.
- **compute_with_reference** at 1 MP is ~5 ms total. Of that:
  ~1.3 ms is the dis-side upload (same WSL2 PCIe constraint), ~1 ms
  is the dis-side sRGB→XYB+pyramid kernel work, ~1 ms is the fused
  H+V+features work (4 launches, one per scale), ~0.2 ms is the
  on-device reduction, ~0.1 ms is the 1.6 KiB final read-back, and
  the rest is variance / launch overhead.

To hit 1 ms/MP from here would need:
1. Native PCIe (or unified memory on Grace Hopper / DGX) — the
   biggest cost we can't shave with code.
2. CUDA graph capture — would amortise the ~6 launches × 50 µs of
   per-call overhead. cubecl 0.10 doesn't expose
   `cuStreamBeginCapture`; the CUDA crate uses `cudarse_driver`
   directly. Tracked at tracel-ai/cubecl#1319.
3. Async upload overlap — cwr(c2)'s upload starts while cwr(c1)'s
   compute runs. Needs a multi-stream / async API, not in cubecl 0.10.

The tile-fused kernel is now the steady-state path; further kernel
work would need to go below cubecl into raw PTX or the cudarc
driver.

Optimisations applied since the initial port:
- **Tile-fused H-blur + V-blur + features kernel** (`kernels::fused`).
  One kernel per scale (was H-blur + V-blur as two kernels). H-blur
  outputs live in shared memory across the V-blur slide instead of
  round-tripping through DRAM. Eliminates the 12 H-blur scratch planes
  per scale (~50 MiB at 1 MP) entirely. Block dim 64 with 12 KiB
  shared per block — ~10 blocks resident per SM.
- **Persistent partials buffers** (`Zensim::new`-time allocation, no
  per-call alloc churn). 12 small allocations / call → 0.
- **Single batched read-back of finals only**. After the on-device
  reduction the host reads ~1.6 KiB instead of ~5.7 MiB per call at
  1 K resolution.
- **3-channel-per-launch H-blur and V-blur+features kernels**.
  Reduces 24 per-call launches → 8.
- **3-channel-per-launch downscale**. Saves 6 launches across the
  pyramid build.
- **Column-strip parallelism in V-blur+features**. Each column is split
  into `n_strips` Y-strips, each processed by its own thread —
  `padded_w × n_strips × 3` parallel threads at the SM-occupancy floor
  (lifts 1 K perf 2× over the old per-column-only kernel).
- **On-device reduction kernel** (`reduce_scale_kernel`). Folds per-
  (col, strip, channel) partials into per-(scale, channel, slot)
  finals on the GPU. One launch per scale (4 total at SCALES = 4)
  with a 60-cube grid (3 channels × 20 slot kinds). Cuts the post-
  compute D2H from a multi-MiB read to a 1.6 KiB read.
- **Fused sRGB → XYB + mirror-pad in one launch**. The kernel covers
  the padded plane and reads from the mirror source column when
  `x ≥ width`. Eliminates the separate per-channel pad pass (was 3
  launches).
- **Packed sRGB-RGB upload**: one u32 per pixel (`R | G<<8 | B<<16`)
  instead of 3 widened u32s. Cuts H2D bandwidth 3× — load-bearing on
  WSL2 where virtualised PCIe is the dominant cost.
- **Persistent host pack scratch**: reused across calls so the u8 →
  u32 packing doesn't re-allocate.
- **No partials zeroing** between calls. Every column thread writes
  all 17 + 3 of its slots in `fused_vblur_features_kernel`, so the
  previous call's contents are fully overwritten before any reduction
  reads them.

## Backend coverage

| Backend | Build | Tests | Notes |
|---|---|---|---|
| CUDA (NVIDIA, native) | ✅ | ✅ 7/8 | Validated on RTX 5070 + CUDA 13.2. |
| WGPU (cross-vendor) | ✅ | ⚠ untested in WSL2 | WSL2 has no Vulkan ICD by default (gotcha G3.2). Validate on native Linux / Mac / Windows. |
| HIP (AMD ROCm) | ✅ (compiles) | ⚠ untested | Same shape as dssim-gpu / ssim2-gpu HIP path. |
| CPU (cubecl-cpu) | ✅ (compiles) | ❌ build-only | cubecl-cpu 0.10 doesn't yet support `Array<f64>` indexing reliably; we use the published `zensim` crate as the CPU reference instead. |

## Known gotchas applied

- **G1.x / `cbrt`** → substituted `f32::powf(_, 1.0/3.0)`. The CPU's
  `cbrtf_fast` magic-constant Newton seed via `reinterpret_cast<u32>` is
  not reachable on cubecl-cuda for compile-time-folded constants.
- **G1.5 / SharedMemory sizing** — n/a (no shared memory; kernels are
  per-pixel / per-column).
- **G2.1 / `CubeCount`/`CubeDim` not Copy** — every launch site
  recomputes the count.
- **G3.2 / WSL2 no Vulkan ICD** — wgpu backend is build-only on the
  reference host; CUDA validates the algorithm.
- **G3.3 / cubecl-cpu no atomics + f64 indexing limitations** —
  cubecl-cpu is build-only.
- **Cancellation in degenerate inputs** — see "HF-ratio noise floor"
  above.
- **Per-column partials, not per-block atomics** — sidesteps the lack
  of `Atomic<f64>` and Metal's broken `Atomic<f32>::fetch_max`. Cost:
  `padded_w × 17 × 8` bytes of scratch per (scale, channel), 557 KiB at
  4 K, well within budget.

## Followups (not blocking)

- **Tighten black-vs-white parity** by adopting the same FMA fusion
  order as the CPU AVX-512 path. cubecl 0.10 has `fma()` (see
  `cubecl::prelude::fma`); using it explicitly may close the 12-point
  gap. Test won't reflect on real images either way.
- **Per-kernel parity examples** modelled on `ssim2-gpu`'s set
  (`color_parity.rs`, `blur_parity.rs`, `features_parity.rs`). The
  integration tests already validate the full pipeline against
  `zensim` v0.2.8; per-kernel diagnostics would only be needed on
  regression.
- **Batched scoring (`ZensimBatch`)** on the dssim-gpu /
  butteraugli-gpu / ssim2-gpu shape. Useful for encoder rate-distortion
  sweeps. Not implemented yet because `zensim-cuda` itself doesn't
  expose a batched API; would be a workspace-level addition rather than
  a port.
- **Tighten `cached_reference_matches_direct` to 1e-5** once the
  fma-ordering match is in place. Currently `< 1e-3`.

`zen-metrics-cli` integration is **done** — zensim-gpu is wired
alongside `dssim-gpu`, `ssim2-gpu`, `butteraugli-gpu` in
`crates/zen-metrics-cli/src/metrics/mod.rs` (see the
`#[value(name = "zensim-gpu")]` variant on the Metric enum).
