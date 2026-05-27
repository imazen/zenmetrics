# butteraugli-gpu

Multi-vendor GPU implementation of the butteraugli perceptual image
quality metric, built on [CubeCL](https://github.com/tracel-ai/cubecl).
One Rust kernel source dispatches across CUDA, WGPU (Vulkan / Metal /
DX12 / WebGPU), HIP, and a CPU SIMD reference.

Algorithmic parity with `butteraugli` v0.9.2: `score` is the max-norm,
`pnorm_3` is the libjxl 3-norm aggregation, both produced in one fused
on-device reduction.

## Memory modes

`butteraugli-gpu` exposes three memory modes via [`MemoryMode`] and the
typed constructors below. Pick based on image size, available VRAM, and
whether you need the multi-resolution (half-res sibling) aggregation
that the CPU reference uses by default.

| Mode | Constructor | Use case |
|---|---|---|
| Whole, single-res | `Butteraugli::new` | Reference-quality single-resolution. No half-res sibling. Cheapest path for small images. |
| Whole, multi-res | `Butteraugli::new_multires` | CPU-reference parity (includes half-res supersample-add). Allocates a full-res instance + a quarter-sized half-res sibling. |
| Strip, single-res | `Butteraugli::new_strip` | Constant-VRAM walker for large images. Holds one `body_h`-row slab + halo at a time, stitches reductions host-side. |
| Strip, multi-res | `Butteraugli::new_multires_strip` | Constant-VRAM multi-res — full-res strip walks paired with synchronized half-res strips. Matches `new_multires` output within 1e-4 rel. |

The opaque `ButteraugliOpaque::new_with_memory_mode(... MemoryMode::Auto)`
resolver picks Strip vs Whole automatically using
[`memory_mode::resolve_auto`] against `vram_cap_bytes()`.

## Multi-resolution strip walker

`new_multires_strip` is the constant-VRAM analog of the CPU reference's
default multi-resolution path. It walks the full-res image in
`body_h`-tall strips like `new_strip`, but each strip pass also drives
a half-res sibling whose body covers the half-res image rows
`[body_top_full / 2, body_end_full.div_ceil(2))`. The two pipelines run
back-to-back per strip:

1. Upload `(ref, dist)` sRGB strip planes to the full-res instance.
2. Downsample the full-res linear-RGB strip planes into the half-res
   sibling (the half-res image isn't decoded separately — its content
   comes from the 2× downsample of the full-res linear-RGB).
3. Run the full-res strip pipeline (opsin → freq split → masking → diff).
4. Run the half-res strip pipeline on the populated linear-RGB.
5. Supersample-add the half-res strip diffmap into the full-res strip
   diffmap.
6. Reduce the full-res strip's body rows into the running partials.

The constructor requires `body_h` even — that keeps every full-res
`body_top` even and the half-res strip's body aligns to half-res image
rows without sub-pixel drift. For images whose `image_h` isn't a
multiple of `body_h`, the last strip's half-res counterpart uses
`body_end_full.div_ceil(2) - body_top_full/2` rows so the half-res
image's last row is covered.

Numerical tolerance vs `new_multires`: per-strip host-side max + p3 / p6
/ p12 reduction order differs from the single fused on-device reduce,
so a small drift is expected. The in-tree tests
(`tests/multires_strip.rs`) enforce `1e-4` relative tolerance across the
256² → 4000×3000 grid, including non-square aspect ratios and uneven
body sizes.

## Benchmark — multires strip vs multires whole

CUDA, RTX 5070, `bench_multires_strip_vs_whole_cuda` example, body=256:

| Size | Whole median | Strip median | Speedup | Whole alloc | Strip alloc |
|---:|---:|---:|---:|---:|---:|
| 4 MP (2000×2000) | 76.8 ms | 143.8 ms | 0.53× | 954 MB | 160 MB |
| 12 MP (4000×3000) | 1736.6 ms | 153.6 ms | 11.31× | 2.79 GB | 320 MB |
| 24 MP (6144×4096) | (OOM) | 275.9 ms | — | 5.86 GB | 492 MB |

At 4 MP the strip walker's per-strip kernel-launch overhead dominates
its locality win. At 12 MP and above the whole-image multires path
spills past on-chip caches and L2; the strip path stays inside L2 and
runs an order of magnitude faster. At 24 MP the whole-image path
exceeds the 12 GB consumer VRAM budget; strip is the only viable mode.

CSV: `benchmarks/butter_multires_strip_<date>.csv` (schema mirrors the
single-res `butter_strip_vs_whole_<date>.csv` file).

## Metal status

butteraugli-gpu works correctly on Metal **out of the box** as of
2026-05-27 (Phase 8e.4). The default feature set drops
`fast-reduction`, so the portable per-thread-partials + finalize
reduction is what ships. Without that flip, the per-octave reduction
relied on `Atomic<f32>::fetch_add`, which cubecl-wgpu's Metal backend
silently no-ops — every reduction returned zero and every score
collapsed to the default. Root cause + upstream patch in
[`../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md`](../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md).

Build configurations:

```bash
# Default — works on Metal, CUDA, DX12, Vulkan, ROCm
cargo build -p butteraugli-gpu

# CUDA-only with the atomic-add fast path (~2-3× faster reduction step
# but non-deterministic and broken on Metal)
cargo build -p butteraugli-gpu --no-default-features --features cuda,fast-reduction
```

When the upstream `feat/metal-atomic-fix` lands, `fast-reduction` will
work correctly on Metal too; the default-off state stays as a
determinism guard (atomic-add commit order is launch-dependent and
breaks bit-reproducibility).

## See also

- `bench_strip_vs_whole_cuda` — single-resolution strip vs whole bench.
- `PORT_STATUS.md` — per-module port status from the cuda reference.
- `tests/strip_parity.rs` — single-resolution strip parity (19 tests).
- `tests/multires_strip.rs` — multi-resolution strip parity (11 tests).
- `tests/opaque_strip_parity.rs` — strip-mode routing through the
  opaque shim (10 tests).
