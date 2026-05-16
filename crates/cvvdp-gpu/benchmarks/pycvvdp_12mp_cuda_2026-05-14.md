# pycvvdp vs cvvdp-gpu — 12 MP CUDA benchmark (2026-05-14)

Honest head-to-head: cvvdp-gpu's `compute_dkl_jod` (cold) and
`compute_dkl_jod_with_warm_ref` (cached) measured against
pycvvdp v0.5.4's `cvvdp.predict()` on the same RTX-class CUDA
host. Both run a 4000×3000 synthetic RGB pair with the
`standard_4k` display geometry.

The pycvvdp comparison is the one that matters — it's the
canonical reference at `gfxdisp/ColorVideoVDP`. Prior snapshots
in this directory compare against `fcvvdp` (a C+Zig fork at
`halidecx/fcvvdp`), which is a separate implementation with
different perf characteristics. **fcvvdp is NOT pycvvdp.**

## Environment

- Host: lilith's water-cooled 7950X / 128 GB
- GPU: NVIDIA GeForce RTX 5070 (CUDA capability 12.0)
- PyTorch: 2.12.0+cu130
- pycvvdp: v0.5.4 (ColorVideoVDP, gfxdisp)
- cvvdp-gpu commit: `59b89911` (tick 172)

## Results

### pycvvdp 12 MP CUDA (`bench_12mp_cuda.py`)

| run | warm-up | iters median | per-pixel | JOD |
| --- | ------ | ------------ | --------- | --- |
| 1   | 986 ms | 176.0 ms     | 14.7 ns/px | 9.4580 |
| 2   | 12.8 s | 159.1 ms     | 13.3 ns/px | 9.4580 |

Steady-state median: **~14 ns/px**. The warm-up cost
(PyTorch graph compile + kernel JIT) is large (1–13 s) but
one-time per process.

### cvvdp-gpu 12 MP CUDA (`time_12mp` tick 171, same host)

| metric                          | per-pixel  |
| -----                           | ----       |
| `compute_dkl_jod` (cold)        | 36.1 ns/px |
| `compute_dkl_jod_with_warm_ref` | 20.6 ns/px |

### Speed ratios (per-pixel, steady-state)

| comparison                                          | ratio |
| -----                                               | ----  |
| pycvvdp                                             | 14 ns/px (baseline)     |
| cvvdp-gpu cold        vs pycvvdp                    | **2.58× slower**        |
| cvvdp-gpu warm-ref    vs pycvvdp                    | **1.47× slower**        |
| pycvvdp warm-up (one-time, second run)              | ~12.8 s                 |
| cvvdp-gpu warm-up (one-time, cubecl JIT)            | ~1 s                    |

## Quality

Both pycvvdp runs returned identical JOD (9.4580). cvvdp-gpu's
host-scalar path is parity-locked vs pycvvdp at ≤0.006 JOD on
the v1 manifest corpus (`shadow_jod` test). The GPU composition
path drifts ~0.4 JOD at q=1 (`shadow_jod_gpu` anchor) due to
cumulative f32 noise through `met2jod`'s steep slope, but
matches host scalar within f32 precision at q≥20.

## What we lose, what we keep

**pycvvdp wins on raw throughput** — its PyTorch path benefits
from cuDNN-optimized separable convolutions, asynchronous CUDA
streams, and PyTorch's mature memory allocator. Our cubecl
kernels are hand-written and don't reach cuDNN-level
optimization on the downscale + upscale pyramid stages.

**cvvdp-gpu wins on integration & portability:**

| dimension                | pycvvdp                       | cvvdp-gpu                                  |
| ---                      | ---                           | ---                                        |
| Backends                 | CUDA only (via PyTorch)       | CUDA + WGPU + HIP + (CPU pending)         |
| Static binary size       | ~3 GB runtime (PyTorch deps)  | ~50 MB statically linked                  |
| FFI overhead in Rust     | Heavy (PyO3 / torch::cuda C++) | Native                                    |
| Cold-start time          | 1–13 s (graph compile)        | ~1 s (cubecl JIT, smaller graph)          |
| Manifest-precise scoring | Yes                           | Via host_scalar (`Cvvdp::score`)          |
| Warm-ref batch path      | Implicit (PyTorch graph reuse) | Explicit (`warm_reference`)               |

## Earlier claims in this repo

Prior snapshots' headline numbers like **"2.06× faster than
fcvvdp 8-thread"** (tick 166) and **"4.17× faster than fcvvdp
8t per DIST"** (tick 171) are accurate vs **fcvvdp**, but
**fcvvdp is not the canonical reference**. The reader should
understand:

- vs **pycvvdp** (canonical): cvvdp-gpu is **slower** at 12 MP.
- vs **fcvvdp** (a CPU C+Zig fork at 360p): cvvdp-gpu is faster.

CHANGELOG / PORT_STATUS / lib.rs entries are being updated to
make the distinction visible.

## Reproducing this benchmark

```bash
cd scripts/cvvdp_goldens
uv venv .venv --python python3.10
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python 'cvvdp==0.5.4' 'pillow>=10' 'numpy>=1.26'
.venv/bin/python bench_12mp_cuda.py
```

(The Torch CUDA install snaps to the user's installed CUDA;
`cu124` triggers a 13.0 nvJitLink build on this host.)

## Open work

- cvvdp-gpu doesn't currently hit pycvvdp's perf because the
  downscale + upscale pyramid kernels are hand-written 5-tap
  separable convolutions; pycvvdp's PyTorch path uses cuDNN's
  optimized depthwise convolution kernels. Closing this gap
  requires either: (a) shared-memory tiled downscale/upscale,
  (b) cuDNN bindings via FFI (defeats the multi-vendor purpose),
  or (c) batched separable conv across the whole pyramid.
- The 1.47× warm-ref gap is much closer; with the right
  downscale rewrite + further dispatch flattening it could
  close.
- Multi-vendor is the moat. WGPU/HIP backends still work,
  pycvvdp doesn't.
