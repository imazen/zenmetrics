# dssim-gpu

Multi-vendor GPU implementation of the DSSIM (Structural Dissimilarity)
perceptual image quality metric, built on
[CubeCL](https://github.com/tracel-ai/cubecl). One Rust kernel source,
runs on:

- **CUDA** (NVIDIA) via `cubecl-cuda`
- **WGPU** (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- **HIP** (AMD ROCm) via `cubecl-hip` (`hip` feature)
- **CPU** fallback via `cubecl-cpu` for build-checking only — the
  reduction kernels need atomics + `CUBE_COUNT`, which `cubecl-cpu` 0.10
  doesn't yet support, so the CPU backend is **not** a runtime parity
  target. Use the published `dssim-core` crate as the CPU reference
  instead (that's what the integration tests do).

Algorithmic parity target is `dssim-core` v3.4 (the canonical Rust
DSSIM crate). At the pyramid level this also matches the existing
`dssim-cuda` crate — same five scales, same custom-Lab conversion,
same fixed 3×3 Gaussian, same per-scale MAD score. The CUDA-specific
crate stays in the workspace; this one extends reach to AMD / Intel /
Apple / WebGPU without giving anything up on NVIDIA.

## Single-image usage

```rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use dssim_gpu::Dssim;

let client = WgpuRuntime::client(&Default::default());
let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;

let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
let dist_srgb: Vec<u8> = vec![0; 256 * 256 * 3];

let result = d.compute(&ref_srgb, &dist_srgb)?;
println!("DSSIM = {:.6}", result.score);
# Ok::<(), dssim_gpu::Error>(())
```

## Cached-reference usage

When scoring many distorted candidates against one reference (encoder
rate-distortion search), call `set_reference` once and then
`compute_with_reference` per candidate. Skips the reference-side
pyramid + Lab + reference-blur work each call.

```rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use dssim_gpu::Dssim;

let client = WgpuRuntime::client(&Default::default());
let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;

let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
d.set_reference(&ref_srgb)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let r = d.compute_with_reference(&candidate)?;
    // ... use r.score ...
}
# Ok::<(), dssim_gpu::Error>(())
```

## Score interpretation

Output is the standard DSSIM scalar: 0 = identical, larger = more
distortion. Mirrors the `f64` returned by `dssim_core::Dssim::compare`.

## Features

| Feature | Default | Effect |
|---|---|---|
| `cuda` | yes | Compile the cubecl-cuda backend. |
| `wgpu` | yes | Compile the cubecl-wgpu backend. |
| `cpu`  | yes | Compile the cubecl-cpu backend (build-only — see above). |
| `hip`  | no  | Compile the cubecl-hip backend. |
| `fast-reduction` | **no** (since 2026-05-27, Phase 8e.4) | Use `Atomic<f32>::fetch_add` for the per-scale Σ reduction. Verified correct on CUDA / Windows DX12 / HIP; **BROKEN on Metal** (cubecl-wgpu's Metal backend silently no-ops the atomic — every reduction returns zero, every score collapses to the default value). Off by default since 2026-05-27 so default builds work on Metal out of the box. Opt in for CUDA-only deployments where the ~2-3× reduction-step speedup matters more than reproducibility. Root cause + upstream patch in [`../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md`](../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md). |

### Metal status

dssim-gpu works correctly on Metal **out of the box** as of 2026-05-27.
The default feature set drops `fast-reduction`, so the portable
per-thread-partials + finalize reduction is what ships. Build:

```bash
# Default — works on Metal, CUDA, DX12, Vulkan, ROCm
cargo build -p dssim-gpu

# CUDA-only with the atomic-add fast path (~2-3× faster reduction step
# but non-deterministic and broken on Metal)
cargo build -p dssim-gpu --no-default-features --features cuda,fast-reduction
```

When the upstream `feat/metal-atomic-fix` lands (CAS-loop lowering for
WGSL `atomicAdd<f32>` on Metal — tracked in
[`CUBECL_METAL_ATOMIC_FIX.md`](../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md)),
`fast-reduction` will work correctly on Metal too; the default-off
state stays as a determinism guard.

## Memory modes

dssim-gpu implements both whole-image and strip processing. `MemoryMode::Auto`
(default via `Dssim::new`) picks **Full whenever it fits the VRAM cap** —
strip mode is 2-5× slower on this crate, so it's only engaged when Full
exceeds the cap. Cap policy: `ZENMETRICS_VRAM_CAP_BYTES` env var, else
8 GB default.

```rust
use dssim_gpu::{DssimOpaque, MemoryMode};

// Force whole-image regardless of cap.
let scorer = DssimOpaque::new_with_memory_mode(
    backend, w, h, params, MemoryMode::Full,
)?;

// Pin an explicit strip body. h_body must be a positive multiple of 16
// (matches `Dssim::new_strip`'s pyramid-alignment contract).
let scorer = DssimOpaque::new_with_memory_mode(
    backend, w, h, params, MemoryMode::Strip { h_body: Some(128) },
)?;
```

See the workspace README for the cross-crate matrix.

## Status

Initial port from `dssim-cuda`. See `PORT_STATUS.md` for the
per-kernel breakdown, runtime coverage, and validated parity numbers.
