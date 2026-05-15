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

## Cached-reference usage

Two ways to amortise the reference-side cost across many distorted
candidates:

- `Cvvdp::set_reference` + `Cvvdp::score_with_reference` stashes the
  raw sRGB bytes and re-runs the full pipeline per candidate. Simple
  and matches `score(ref, dist)` bit-for-bit.
- `Cvvdp::warm_reference` + `Cvvdp::compute_dkl_jod_with_warm_ref`
  materialises the REF Weber-contrast pyramid on the GPU once and
  skips that half of the pipeline per candidate — ~1.8× faster per
  DIST at 12 MP (~62 → ~34 ns/px on an RTX 5070; see `src/lib.rs`'s
  "How we compare to the canonical reference" section).

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

## Sweep tooling — `CVVDP_COLUMN_NAME`

The crate exports a stable column-name constant
`cvvdp_gpu::CVVDP_COLUMN_NAME` for landing scores in parquet
sidecars without colliding with other cvvdp variants (the
canonical pycvvdp reference goes under `cvvdp_pycvvdp_v054`, a
future Burn-based port would get its own tag, etc.). Default form
is `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`; CI can override the
entire string at build time via the `CVVDP_IMPL_TAG` env var to
bake in a git short hash when iterating within the same crate
version.

The full sidecar schema (identity tuple, score-column type
contract, manifest format, producer / consumer protocols) lives
in [`docs/CVVDP_SIDECAR_SCHEMA.md`](docs/CVVDP_SIDECAR_SCHEMA.md).
The Burn-based port that would land alongside as
`cvvdp_burn_v*` is scoped in
[`docs/BURN_PORT_PLAN.md`](docs/BURN_PORT_PLAN.md).

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

CUDA 13.2 required for cubecl 0.10's CUDA backend. RTX 50-series
(Blackwell, sm_120) needs CUDA 13 anyway. The cubecl-cuda runtime
dynamically loads CUDA via `dlopen`, so the build itself succeeds
without nvcc on PATH:

```bash
CUDA_PATH=/usr/local/cuda cargo build -p cvvdp-gpu
CUDA_PATH=/usr/local/cuda cargo test --release -p cvvdp-gpu
```

For Metal (Apple) or non-NVIDIA GPUs, build wgpu-only:

```bash
cargo build -p cvvdp-gpu --no-default-features --features wgpu
```

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial.
