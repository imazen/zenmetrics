# Why burn ran VGG 100× slower on Vulkan than CUDA (2026-10-15)

Follow-up to [`nn_runtime_shootout_2026-10-14.md`](nn_runtime_shootout_2026-10-14.md),
where burn 0.22's Vulkan backend ran the LPIPS VGG16 trunk at 256² in 146 ms against
3.1 ms on CUDA, but ResNet50 only ~4× slower. Host i265, RTX 4080, driver 595.99.02,
burn 0.22.0 (`cuda`, `vulkan`, `fusion`, `autotune`), cubecl-wgpu 0.11.0. Run through
`run-heavy` (16 G cap).
Harness: [`burn_wgpu_conv_2026-10-15/`](burn_wgpu_conv_2026-10-15/) (`src/main.rs`,
`run.log`): one 3×3 conv, stride 1, pad 1, batch 1, at each VGG16 layer shape,
median of 20 after 3 warmups, synced by a scalar readback.

## Root cause

For a dense 3×3 conv, burn's autotune set (`burn-cubecl` 0.22
`kernel/conv/forward/tune.rs`) holds `conv_direct`, the depthwise kernels (which
decline dense convs), `conv_im2col_1x1` (1×1 only) and implicit-GEMM kernels that
need CMMA or MMA tile support. cubecl-wgpu registers CMMA on Vulkan from
`VK_KHR_cooperative_matrix`, which the 4080 exposes, but Vulkan cooperative
matrices take f16/bf16 (or int8) inputs, not f32. So **an f32 3×3 conv on Vulkan
can only run `conv_direct`**, which reaches 160–1170 GFLOP/s here. ResNet50 is
mostly 1×1 convs, which take the im2col + matmul path, which is why it suffered
less.

| layer (cin→cout @ size) | CUDA f32 | Vulkan f32 | Vulkan f16 |
|---|---|---|---|
| 3→64 @ 256 | 0.32 ms | 0.99 ms | 0.78 ms |
| 64→64 @ 256 | 0.52 ms (9.3 TFLOP/s) | 4.14 ms (1.2 TFLOP/s) | 0.66 ms (7.3 TFLOP/s) |
| 128→128 @ 128 | 0.32 ms | 7.74 ms | 0.82 ms |
| 256→256 @ 64 | 0.32 ms (15.0 TFLOP/s) | 20.1 ms (0.24 TFLOP/s) | 0.74 ms (6.6 TFLOP/s) |
| 512→512 @ 32 | 0.50 ms | 28.9 ms (0.17 TFLOP/s) | 0.56 ms (8.7 TFLOP/s) |
| 512→512 @ 16 | 0.51 ms | 7.41 ms | 1.02 ms |

Summing the Vulkan f32 rows over VGG16's 13 convs at 256² gives ~170 ms, which
matches the 146 ms trunk timing, so the conv kernel explains the whole gap.

An explicit `unfold4d` (im2col) + `matmul` workaround is slower still on both
backends (Vulkan f32 65→12 GFLOP/s, CUDA f32 1.2 TFLOP/s→29 GFLOP/s as channels
grow), so it is not a fix.

## Consequences for the plan

- **burn on WebGPU/Vulkan is only fast in f16.** The f16 CMMA path runs
  5.9–8.7 TFLOP/s on these shapes. Whether f16 activations hold LPIPS parity has
  to be measured (PyTorch fp16 moved LPIPS-vgg by up to 2.7e-4,
  [`nn_lpips_phase0_2026-10-15.md`](nn_lpips_phase0_2026-10-15.md)).
- **An f32 WebGPU route needs a better f32 conv kernel** than `conv_direct`: a
  tiled, register-blocked implicit GEMM without tensor cores. Not written.
- **Whether burn's CUDA f32 path uses TF32 is not determined.** 9–15 TFLOP/s
  is well under the 4080's non-tensor-core f32 peak (76 SMs × 128 f32 lanes ×
  2 FLOP = 19.5 TFLOP/s per GHz; nvidia-smi reports a 3105 MHz max SM clock), so
  the rate alone doesn't say. PyTorch's TF32 moved LPIPS-vgg by 1.5e-5, so the CUDA
  engine needs its own parity check before its f32 scores are trusted at 1e-6.
- WebGPU route 2 (rten on wasm SIMD) is unaffected.
