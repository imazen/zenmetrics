# Burn port plan — alternative to hand-tuning cubecl kernels

> **Status (tick 324, 2026-05-15): ABANDONED.** A perf spike at
> `crates/burn-conv-spike/` (commit `e101c895`) measured the
> proposed `cubek::conv2d(5×1) + conv2d(1×5)` separable path
> against our hand-written `downscale_kernel` at 4000×3000 f32 on
> an RTX 5070 sm_120. Result: **4.32× slower** even with the best
> algorithm choice (`SimpleSyncCyclic + Mma`, 1.46 ms/op vs
> 0.34 ms/op for the hand-written direct stencil). Other cubek
> algorithm choices (CMMA variants, strided, tilewise) all landed
> 4.98–5.03× slower. The Mma path's tensor-core 16×16×16 tiles
> waste 15/16 of the work when `in_channels = out_channels = 1`,
> and im2col → GEMM doubles memory traffic vs. a direct stencil.
> The "recover cuDNN-class perf via Burn" pitch doesn't hold for
> our use case. ABANDON. The remaining content below is preserved
> as the design context that surfaced the gap; future ticks that
> want to close the 2.4× gap to pycvvdp should look at
> shared-memory tiling / register tiling of the existing direct
> stencil kernel instead.

Captured tick 190 (2026-05-15) after the user asked: "why not have an
agent try to port pycvvdp from pytorch to burn as an alt path that
might be easier?". Agent investigation confirmed: yes, this is the
right path.

## Why Burn

Our current `cvvdp-gpu` hand-writes every cubecl kernel (downscale,
upscale, weber, CSF, masking, pool). The 5-tap separable downscale
+ upscale dominate perf at 12 MP — we're ~2.4× slower than pycvvdp
because PyTorch routes those convs through cuDNN's auto-tuned
per-shape kernels.

[Burn](https://burn.dev) is "PyTorch in Rust" with cubecl as the
backend (the same runtime we already use). Its high-level tensor
ops route through [cubek](https://github.com/tracel-ai/cubek)
(cubecl-kernels), which contains the auto-tuned conv/pool/reduce
implementations that match cuDNN-class performance. Burn 0.21
released 2026-05.

The deal: replace our hand-written downscale/upscale with
`burn::tensor::module::conv2d` and we inherit cuDNN-equivalent
optimisation for free.

## Coverage assessment

Burn covers **~85% of pycvvdp's pipeline natively**:

| Stage                    | Pycvvdp op                  | Burn equivalent                              | Notes |
| ----                     | ----                        | ----                                         | ---- |
| sRGB → DKL color         | per-pixel arithmetic         | `Tensor::matmul` (3×3) + element-wise         | trivial |
| Gauss pyramid (5-tap)    | `F.conv2d`                   | `burn::tensor::module::conv2d`               | cuDNN-class path |
| Pyramid upscale          | `F.conv_transpose2d`         | `burn::tensor::module::conv_transpose2d`     | same |
| Weber-contrast           | divide + clamp               | `Tensor` arithmetic + `clamp`                | trivial |
| **CSF 32×32 LUT interp** | `interp1` custom Python      | **needs custom CubeCL** (no `grid_sample`)   | ~50 LOC |
| PU blur (13-tap, σ=3)    | `F.conv2d`                   | `conv2d` again                               | cuDNN-class path |
| min_abs / mult_mutual    | element-wise + reduce        | `Tensor` ops, fused via cubek                | maybe custom fused kernel for ~40 LOC perf win |
| Spatial pool L_p         | `safe_pow + sum_dim`         | `powf + sum_dim`                             | direct |
| 3-stage Minkowski        | nested L_p folds             | `Tensor` ops                                 | direct |
| `met2jod` piecewise      | mask + arithmetic            | `mask_where` chain                           | direct |

What needs custom kernels:
1. **CSF LUT bilinear** — Burn has no `grid_sample`. Keep our
   `csf_apply_*_kernel` cubecl code; Burn tensors are
   handle-compatible with cubecl (same runtime), so the custom
   kernel slots cleanly into a Burn graph.
2. **Mult-mutual cross-channel fused masking** — expressible in
   pure Burn ops but a custom fused kernel might help at 12 MP.
   Profile-driven decision after the port.

## Effort estimate

- **Burn port: 3-5 days.** Pipeline rewrite is mostly 1:1 from
  pycvvdp to Burn (the Python uses PyTorch idioms that map directly
  to Burn). Plus 1-2 small CubeCL kernels we already have working
  code for.
- **Continued hand-tuning of cubecl kernels: 1-2 weeks minimum**
  to write a fused 5-tap matching cuDNN (im2col + GEMM, register
  tiling, async double-buffer). High risk of still missing
  pycvvdp's perf because cuDNN auto-tunes per shape.

## Plan

1. Create a sibling crate `crates/cvvdp-gpu-burn/` with the same
   public API (`Cvvdp::compute_dkl_jod`, `score`, `warm_reference`,
   `compute_dkl_jod_with_warm_ref`).
2. Wire the goldens manifest from `scripts/cvvdp_goldens/` so the
   new crate's parity tests use the same pycvvdp golden values.
3. Burn the pipeline stages over one at a time:
   a. Color transform (smallest, trivial parity check)
   b. Gauss pyramid (the perf-critical piece)
   c. Weber contrast + log_l_bkg
   d. CSF (keep cubecl kernel, wire it into a Burn graph)
   e. Masking (Burn ops first; fuse later if needed)
   f. Pool + finalize (Burn ops)
4. Benchmark vs pycvvdp at 12 MP + 256² fixtures.
5. Decide which path to maintain. The cvvdp-gpu crate stays
   working until the Burn port matches or beats it.

## Open questions

- Burn's CubeCL CUDA backend currently uses cubecl 0.10. Our crate
  uses the same pin (`0.10.0-pre.4`). No version conflict expected,
  but verify Burn 0.21's exact cubecl dep before starting.
- Can a Burn tensor share a `cubecl::server::Handle` with our
  existing cubecl-direct kernels (zero-copy)? If yes, the
  hybrid-pipeline model is clean. If not, the boundary needs a
  small copy per stage.
- Burn's mixed precision support (f16/bf16) is mature. Worth
  exploring once parity holds, since cvvdp's CSF doesn't need
  f32 precision everywhere.

## Why not start now

This is a multi-day rewrite. The current cvvdp-gpu crate has 73
green parity tests, ≤0.0017 JOD agreement with pycvvdp across 4
distortion types at 256², and 0.0003 at 12 MP. Better to land
this as a focused PR than dribble it across loop ticks.

Future ticks can extend the goldens manifest (more fixture sizes,
more distortion types) and tighten parity guards — those
investments transfer 1:1 to the Burn crate once it lands. Treat
this doc as the design contract for that work.
