# cvvdp-gpu

Multi-vendor GPU implementation of [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)
(still-image mode), built on [CubeCL](https://github.com/tracel-ai/cubecl).
One Rust kernel source, runs on:

- **CUDA** (NVIDIA) via `cubecl-cuda`
- **WGPU** (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- **HIP** (AMD ROCm) via `cubecl-hip` (`hip` feature)
- **CPU** via `cubecl-cpu` through `Cvvdp::compute_dkl_jod_host_pool`
  (host-side Minkowski fold) — the GPU pool kernel uses
  `Atomic<f32>::fetch_add`, which `cubecl-cpu` 0.10 doesn't support.

Algorithmic parity target is the published
[`ColorVideoVDP`](https://github.com/gfxdisp/ColorVideoVDP) Python
reference **v0.5.4**, still-image code path only. Video / temporal
channels (sustained + transient) are intentionally out of scope for
v0 — defer until still-mode parity is locked.

## Single-image usage

```rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let ref_srgb: Vec<u8> = vec![128u8; (w * h * 3) as usize];
let dis_srgb: Vec<u8> = vec![128u8; (w * h * 3) as usize];
let jod = cvvdp.score(&ref_srgb, &dis_srgb)?;
println!("JOD = {jod:.4}");
# Ok::<(), cvvdp_gpu::Error>(())
```

Swap `cubecl::cuda::CudaRuntime` for `cubecl::wgpu::WgpuRuntime` to
target Metal / Vulkan / DX12 / WebGPU, or `cubecl::hip::HipRuntime`
for AMD ROCm. On `cubecl::cpu::CpuRuntime` use
`Cvvdp::compute_dkl_jod_host_pool` instead of `Cvvdp::score` — see
the CPU backend section below.

## Per-pixel diffmap

`Cvvdp::score_with_diffmap` returns the same JOD scalar as
`Cvvdp::score` AND fills a caller-owned `Vec<f32>` with a
per-pixel error signal (row-major, `width * height` non-negative
f32 values, large where the masked error is large in any DKL
channel or pyramid band). The diffmap is the per-pixel input the
jxl-encoder CVVDP-loop fork uses for its per-block 8×8 median + MAD
heuristic. See `src/kernels/diffmap.rs` for the recipe and
`docs/DIFFMAP_DIVERGENCES.md` for the relationship between the
diffmap and the scalar JOD.

```rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let ref_srgb = vec![128u8; (w * h * 3) as usize];
let dis_srgb = vec![100u8; (w * h * 3) as usize];

// Pre-allocate once; reuse across many score calls.
let mut diffmap: Vec<f32> = Vec::with_capacity((w * h) as usize);
let jod = cvvdp.score_with_diffmap(&ref_srgb, &dis_srgb, &mut diffmap)?;
assert_eq!(diffmap.len(), (w * h) as usize);
println!(
    "JOD = {jod:.4}, peak pixel error = {:.4}",
    diffmap.iter().cloned().fold(0.0_f32, f32::max)
);
# Ok::<(), cvvdp_gpu::Error>(())
```

The diffmap is also available from the warm-ref + linear-planes
families (`score_with_warm_ref_diffmap`,
`score_from_linear_planes_with_diffmap`,
`score_from_linear_planes_with_warm_ref_diffmap`). Identical
inputs (`ref ≡ dist`) produce an all-zero diffmap to 1e-7 absolute
— the buttloop consumer relies on this invariant.

## Linear-RGB planes entry points

For callers that already have linear-light sRGB primaries in planar
f32 (e.g. JPEG XL encoder buttloops),
`Cvvdp::score_from_linear_planes` (and the warm-ref / diffmap
variants) skips the host-side sRGB pack + sRGB→linear LUT lookup.
Mirrors butteraugli-gpu's
`compute_with_reference_from_linear_planes` (W44-PHASE3-B4 pattern):

```rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let n_pix = (w * h) as usize;
let ref_r = vec![0.5_f32; n_pix];
let ref_g = vec![0.5_f32; n_pix];
let ref_b = vec![0.5_f32; n_pix];
let dis_r = vec![0.4_f32; n_pix];
let dis_g = vec![0.5_f32; n_pix];
let dis_b = vec![0.5_f32; n_pix];

let jod = cvvdp.score_from_linear_planes(
    &ref_r, &ref_g, &ref_b, &dis_r, &dis_g, &dis_b,
)?;
# Ok::<(), cvvdp_gpu::Error>(())
```

The display model (`y_peak`, `y_black`, `y_refl`) and DKL matrix
still apply on GPU — the caller is responsible for the sRGB EOTF
only.

## Display models

`DisplayModel` carries the per-display photometric + colorimetric
configuration that drives the EOTF, primaries, and ambient-reflection
math:

- `y_peak`, `y_black`, `y_refl` — historical fields; preserved for
  back-compat with v1 callers and the canonical sRGB parity tests.
- `eotf` — sRGB / PQ / HLG / Linear / BT.1886 / Gamma(g). Defaults
  to `Eotf::Srgb`.
- `primaries` — BT.709 / BT.2020 / Display P3 / DCI-P3. Defaults
  to `Primaries::Bt709`.
- `e_ambient_lux`, `k_refl` — ambient illuminance and screen
  reflectivity; the host derives `y_refl = e_ambient_lux / π *
  k_refl` per cvvdp's `vvdp_display_photo_eotf`.

Construct from upstream-style parameters:

```rust
use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};

let standard_4k = DisplayModel::new(
    200.0,                 // y_peak (cd/m²)
    1000.0,                // contrast (y_black = y_peak / contrast)
    250.0,                 // e_ambient_lux
    0.005,                 // k_refl
    Eotf::Srgb,
    Primaries::Bt709,
);
assert_eq!(standard_4k.y_peak, DisplayModel::STANDARD_4K.y_peak);
```

Or load a named preset from the vendored `display_models.json`:

```rust
use cvvdp_gpu::params::{DisplayModel, DisplayGeometry, Eotf, Primaries};

let hdr_pq = DisplayModel::by_name("standard_hdr_pq").unwrap();
assert_eq!(hdr_pq.y_peak, 1500.0);
assert_eq!(hdr_pq.eotf, Eotf::Pq);
assert_eq!(hdr_pq.primaries, Primaries::Bt2020);

let geo = DisplayGeometry::by_name("standard_4k").unwrap();
assert_eq!(geo.resolution_w, 3840);
```

The registry mirrors every preset shipped in pycvvdp v0.5.4 main
(`standard_4k`, `standard_hdr_pq`, `standard_hdr_hlg`,
`standard_hdr_linear`, `standard_fhd`, `iphone_*`,
`macbook_pro_16`, `lg_oled_*`, `65inch_hdr_pq_*`, …). See
[`docs/DISPLAY_SPECS.md`](docs/DISPLAY_SPECS.md) for the full list
and what each preset routes to.

Note: as of this release the EOTF + primaries dispatch is wired
through the host-scalar pipeline (`display_byte_to_dkl_scalar`,
`display_linear_rgb_to_dkl_scalar` in `kernels::color`). The GPU
fast path (`Cvvdp::score`, `compute_dkl_jod`, etc.) still assumes
sRGB + BT.709 for the kernel uploads — it reads the
`y_peak`/`y_black`/`y_refl` fields of the model and ignores
`eotf`/`primaries`. To score HDR / wide-gamut inputs on the GPU
path today, convert to linear-BT.709 on the host first and use
the `score_from_linear_planes` entry point. Full GPU EOTF +
primaries dispatch is queued for the next tick.

## Cached-reference usage

Two ways to amortise the reference-side cost across many distorted
candidates:

- `Cvvdp::set_reference` + `Cvvdp::score_with_reference` stashes the
  raw sRGB bytes and re-runs the full pipeline per candidate. Simple
  and matches `score(ref, dist)` bit-for-bit.
- `Cvvdp::warm_reference` + `Cvvdp::compute_dkl_jod_with_warm_ref`
  materialises the REF Weber-contrast pyramid on the GPU once and
  skips that half of the pipeline per candidate — ~1.8× faster per
  DIST at 12 MP (~17-22 ns/px warm-ref on an RTX 5070 post the
  T4.L+M upload-pinning + T1.B/T1.C/T4.K LDS work, vs ~62 ns/px
  on the pre-perf-push baseline; see `src/lib.rs`'s "How we compare
  to the canonical reference" section).

```rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let reference: Vec<u8> = vec![128u8; (w * h * 3) as usize];
cvvdp.warm_reference(&reference)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let jod = cvvdp.compute_dkl_jod_with_warm_ref(&candidate, ppd)?;
    // ... use jod ...
}
# Ok::<(), cvvdp_gpu::Error>(())
```

The warm-state invalidation contract (which helpers reset the cache
vs preserve it) is pinned by the regression tests
`warm_state_invalidates_after_each_documented_dispatcher`,
`set_reference_does_not_invalidate_warm_state`, and
`gauss_chain_helpers_do_not_invalidate_warm_state` — call any
documented REF-dispatching method between `warm_reference` and the
warm-ref read and the cache is dropped; cold helpers and
`set_reference` leave it intact.

## Score interpretation

Output is a JOD (Just-Objectionable-Difference) value on the
ColorVideoVDP 0–10 scale:

- **10.0** = byte-identical (or perceptually indistinguishable)
- **8–10** = high quality, distortions rarely noticed
- **6–8**  = noticeable but acceptable distortion
- **<6**   = clearly visible degradation

Higher = better, matching the pycvvdp Python reference convention.
JOD is the canonical ColorVideoVDP output; `score` returns it as
`f64` and `compute_dkl_jod*` variants as `f32`.

## CPU backend (`cubecl-cpu`)

`Cvvdp::compute_dkl_jod` uses `Atomic<f32>::fetch_add` in
`pool_band_3ch_kernel` (the fused 3-channel pool kernel that
production dispatches, one launch per pyramid band), which
`cubecl-cpu` 0.10 doesn't support. For cpu-runtime use, switch
to `Cvvdp::compute_dkl_jod_host_pool` (or its warm-ref variant
`compute_dkl_jod_host_pool_with_warm_ref`) — they read the
per-band D arrays back to the host and reduce with
`host_scalar::lp_norm_mean`, producing the same JOD at f32
precision. See `tests/cpu_backend.rs` for direct
cpu-vs-pycvvdp parity coverage on the 73×91 odd-dim fixture.

The same `compute_dkl_jod_host_pool` path also works on Metal
(`wgpu` feature) — Metal's `Atomic<f32>::fetch_add` silently no-ops
on the f32 add, so the GPU pool produces zero. The host-pool path
sidesteps the gotcha.

## Parity vs. perf — `PerfMode`

`CvvdpParams` exposes a `perf_mode: PerfMode` field with two
variants:

- **`PerfMode::Strict`** (default) — matches pycvvdp v0.5.4
  bit-for-bit within f32 noise. Every parity test in `tests/`
  is calibrated against this mode.
- **`PerfMode::Fast`** — opt-in entry point for future
  stage-level relaxations that trade measurable per-call cost
  for a bounded JOD drift vs. Strict.

Opt in by overriding the field on `CvvdpParams::PLACEHOLDER`:

```rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::{Cvvdp, PerfMode};
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let mut cvvdp = Cvvdp::<CudaRuntime>::new(
    client, 256, 256,
    CvvdpParams { perf_mode: PerfMode::Fast, ..CvvdpParams::PLACEHOLDER },
)?;
# Ok::<(), cvvdp_gpu::Error>(())
```

Today `Fast` is a no-op (no fast-path has landed yet) — the
variant exists so callers can wire the opt-in once and future
per-stage optimizations gate on `params.perf_mode == Fast`
without forcing a breaking change. Each Fast-mode optimization
documents its drift budget in `CHANGELOG.md`. The
`perf_mode_fast_matches_strict_today` regression test pins the
current bit-pattern-equality contract; when a real Fast-mode
optimization lands the test should be relaxed (not deleted) to
that optimization's drift budget.

## Features

| Feature | Default | Effect |
|---|---|---|
| `cuda` | yes | Compile the cubecl-cuda backend. |
| `wgpu` | yes | Compile the cubecl-wgpu backend. |
| `cpu`  | yes | Compile the cubecl-cpu backend (host-pool only — see above). |
| `hip`  | no  | Compile the cubecl-hip backend. |
| `parity-goldens` | no | Compile `tests/parity.rs`, which fetches the pycvvdp v0.5.4 goldens manifest from R2 and checks JOD parity. Off by default so `cargo test` stays offline. |

## GPU memory budgeting — concurrency cap for batch sweeps

When running many `Cvvdp` instances against a shared GPU (one
process per image-size group in a sweep, multiple worker threads
per box, etc.), the crate exposes two helpers that let callers
size `PARALLEL` against the device's free memory without
reinventing the buffer-accounting math.

- `cvvdp_gpu::estimate_gpu_memory_bytes(width, height) -> Option<usize>`
  — static-analysis predictor that sums every persistent buffer
  `Cvvdp::new` allocates: the three pyramids (`gauss_ref`,
  `bands_ref`, `bands_dis`), per-level `d_scratch` (6 plane types
  × 3 channels), `weber_scratch` (non-baseband levels only),
  source byte buffers, `partials`, `srgb_lut`, `logs_row`. Uses
  ceil-div halving so it matches the actual allocator layout.
  Returns `None` below the `PYRAMID_MIN_DIM × 2 = 8×8` threshold
  (same precondition as `Cvvdp::new`).

- `cvvdp_gpu::recommend_parallel(free_gpu_bytes, width, height) -> u32`
  — bundles the predictor with `PARALLEL_SAFETY_FACTOR = 1.5` so
  callers don't maintain the constant themselves. Always returns
  at least 1 if image dims are valid (a single instance gets to
  attempt scoring; OOM after that is a back-off signal, not
  "no work").

Worked example at standard 4K geometry:

| Size            | `estimate_gpu_memory_bytes` | 8 GB GPU PARALLEL | 24 GB GPU PARALLEL |
|-----------------|-----------------------------|-------------------|--------------------|
|     64×64       | 0.8 MB                      | many              | many               |
|    256×256      | 13 MB                       | 409               | 1228               |
|    512×512      | 52 MB                       | 102               | 307                |
|   1024×1024     | 208 MB                      | 25                | 76                 |
|   2048×2048     | 833 MB                      | 6                 | 19                 |
| 4096×3072 (12MP)| 2.5 GB                      | 2                 | 6                  |

For a typical sweep worker:

```rust,no_run
# fn free_gpu_bytes() -> u64 { 0 } // cudarc / wgpu query goes here
use cvvdp_gpu::recommend_parallel;
let parallel = recommend_parallel(free_gpu_bytes(), 1024, 1024);
println!("running {parallel} concurrent Cvvdp instances on this GPU");
```

`examples/cvvdp_mem_table.rs` prints the full size-vs-budget
table — useful for choosing `PARALLEL` for a specific GPU SKU.

The 1.5× safety factor in `PARALLEL_SAFETY_FACTOR` covers per-call
transient uploads (the per-DIST `srgb` byte buffer), cubecl runtime
metadata + page alignment, NVRTC PTX cache, and driver-side
reservations. Tighten to ~1.2 when batching with `warm_reference`
(no per-DIST allocator churn); loosen to ~2.0 if the same process
also runs CPU-side decode/encode in the same memory namespace —
call `estimate_gpu_memory_bytes` directly and divide yourself.

## Sweep tooling — `CVVDP_COLUMN_NAME`

The crate exports a stable column-name constant
`cvvdp_gpu::CVVDP_COLUMN_NAME` for landing scores in parquet
sidecars without colliding with other cvvdp variants (the
canonical pycvvdp reference goes under `cvvdp_pycvvdp_v054`,
future alternative implementations would get their own tags,
etc.). Default form is `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`;
CI can override the entire string at build time via the
`CVVDP_IMPL_TAG` env var to bake in a git short hash when
iterating within the same crate version.

The full sidecar schema (identity tuple, score-column type
contract, manifest format, producer / consumer protocols) lives
in [`docs/CVVDP_SIDECAR_SCHEMA.md`](docs/CVVDP_SIDECAR_SCHEMA.md).
A Burn-based port was investigated and abandoned in tick 324
after a perf spike measured 4.32× regression vs. the
hand-written separable kernel; see
[`docs/BURN_PORT_PLAN.md`](docs/BURN_PORT_PLAN.md)'s "Status:
ABANDONED" banner and `crates/burn-conv-spike/README.md`.

## Status

Still-image score matches pycvvdp v0.5.4 within **0.005 JOD** across
q=1–90 fixtures on the v1 R2 goldens manifest. The full pipeline —
display model, DKL color, Weber pyramid, CSF, mult-mutual masking,
3-stage Minkowski pool — runs on GPU; only the final pool fold
and `met2jod` mapping happen host-side, on a ~144-byte partials
buffer. See [`docs/PORT_STATUS.md`](docs/PORT_STATUS.md) for the
per-kernel breakdown and validated parity numbers, and
[`docs/CHROMA_DRIFT_INVESTIGATION.md`](docs/CHROMA_DRIFT_INVESTIGATION.md)
for the history behind the ticks 191–207 chroma_shift hunt.

## Build

CUDA SDK version requirements depend on the target GPU
architecture, not on cubecl 0.10. cubecl's `cudarc 0.19.4`
dependency emits dlsym entries gated on a `cuda-<MMmmpp>` cargo
feature that's auto-selected from the CUDA SDK detected at build
time (`cuda-version-from-build-system` feature). The selected
feature must match symbols the host's libcuda actually exports:

- **RTX 50-series (Blackwell, sm_120)** — CUDA 13 SDK required;
  Blackwell sm_120 isn't supported below CUDA 13.
- **RTX 20/30/40, A2000, A4000, etc. (Turing through Ada)** —
  CUDA 12.6 SDK works and is recommended for production
  deployment. The vast.ai backfill fleet (tick 384+) runs the
  binary built against CUDA 12.6; verified end-to-end on RTX
  2060 SUPER, RTX 3060, RTX A2000 hosts under driver 535+.

Building against CUDA 13 produces a binary that calls
`cuCoredumpDeregisterCompleteCallback`, a symbol gated behind
`cuda-13020` in cudarc but absent from every released NVIDIA
libcuda; that binary panics at first kernel dispatch on every
host. Pick the SDK version that matches the GPU you target.

The cubecl-cuda runtime dynamically loads CUDA via `dlopen`, so
the build itself succeeds without nvcc on PATH; only the
matching libcuda is needed at runtime:

```bash
# RTX 50-series target:
CUDA_PATH=/usr/local/cuda-13 cargo build -p cvvdp-gpu --release

# RTX 20/30/40, A2000, etc. target (recommended for fleet deployment):
CUDA_PATH=/usr/local/cuda-12.6 cargo build -p cvvdp-gpu --release

# Run tests against whichever CUDA + driver is present:
CUDA_PATH=/usr/local/cuda cargo test --release -p cvvdp-gpu
```

NVRTC headers (`cuda-cudart-dev-<MMmm>` on Debian/Ubuntu) are
required at **runtime** on every worker, not just at build:
cubecl emits kernels with `#include <cuda_runtime.h>` and NVRTC
compiles them at first launch. Missing headers manifest as
`Cvvdp::score: image too small for the configured pyramid`
errors (the dual-purpose `InvalidImageSize` variant masks the
NVRTC compile failure). See
`scripts/sweep/onstart_cvvdp_backfill_imazen.sh` for the apt
install sequence the fleet uses.

For Metal (Apple) or non-NVIDIA GPUs, build wgpu-only:

```bash
cargo build -p cvvdp-gpu --no-default-features --features wgpu
```

## Memory modes

cvvdp-gpu exposes the workspace's unified `MemoryMode` enum but has
**no Strip implementation** — the spatial-frequency decomposition is
full-image by construction and architecturally blocked at 24 MP
square (see `docs/STRIP_PROCESSING.md`). `MemoryMode::Auto` (the
default) resolves to Full; `Strip` / `Tile` return
`Error::ModeUnsupported`. When the image's working set exceeds the
VRAM cap (`ZENMETRICS_VRAM_CAP_BYTES` env var, default 8 GB), Auto
surfaces `Error::TooBigForFull` with the gap.

The pre-existing `recommend_parallel(free_gpu_bytes, w, h)` helper
already handles per-instance VRAM budgeting for parallel sweeps and
is unaffected by the new mode surface.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial.
