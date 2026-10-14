# NN inference runtime shootout — 2026-10-14

Which Rust runtime can run the pyiqa deep-metric backbones (VGG16 LPIPS trunk,
ResNet50) at PyTorch speed? Input to
[`docs/PYIQA_PORT_PLAN.md`](../docs/PYIQA_PORT_PLAN.md).

- **Host:** `i265` — Core Ultra 7 265K (8P+12E, 20 threads, AVX2, **no AVX-512**),
  RTX 4080 (driver 595.99). **Not measured on dev (Zen 5) or mac.**
- **Models:** random-init torchvision graphs exported to ONNX opset 17, batch 1
  (`nn_runtime_shootout_2026-10-14/scripts/export.py`). `vgg16lpips` =
  `vgg16.features[:30]` returning the five LPIPS taps; `resnet50` = classifier removed.
- **Timing only.** Outputs were not compared across engines.
- **Sizes are two points per model** (256²/512², 224²/384²). That is enough to pick a
  runtime, not to fit α+β·MP; the size-swept measurement is phase 0 of the plan.
- Every run went through `run-heavy`; peak RSS ≤ 3.6 GiB (builds), ≤ 2.5 GiB (benches).
- Raw logs, scripts and the two Rust bench crates: `nn_runtime_shootout_2026-10-14/`.
  The scripts were run from `~/tmp/nnrt-research` and still carry that path.

## CPU (ms, min; ratio = engine ÷ PyTorch)

| model | threads | PyTorch 2.14.1+cpu | ORT 1.31 | rten 0.27.0 | tract 0.23.8 | burn-flex 0.22.0 |
|---|---|---|---|---|---|---|
| vgg 256² | 1 | 292.3 | 247.7 (0.85) | 292.4 (1.00) | 358.0 (1.22) | 432.5 (1.48) |
| vgg 512² | 1 | 1212.2 | 1015.5 (0.84) | 1213.1 (1.00) | 1527.5 (1.26) | 1739.1 (1.43) |
| r50 224² | 1 | 65.9 | 51.6 (0.78) | 64.0 (0.97) | 74.9 (1.14) | 107.0 (1.62) |
| r50 384² | 1 | 181.8 | 150.2 (0.83) | 181.9 (1.00) | 217.4 (1.20) | 263.4 (1.45) |
| vgg 256² | 8 | 45.9 | 46.2 | 45.7 (1.00) | 99.1 (2.16) | 150.8 (3.29) |
| vgg 512² | 8 | 188.6 | 145.7 | 190.2 (1.01) | 367.8 (1.95) | 501.1 (2.66) |
| r50 224² | 8 | 16.2 | 12.3 | 20.7 (1.28) | 30.2 (1.86) | 46.6 (2.88) |
| r50 384² | 8 | 33.3 | 33.6 | 48.2 (1.45) | 66.3 (1.99) | 120.3 (3.61) |
| vgg 256² | 16 | 37.0 | 26.2 | 38.4 (1.04) | 61.7 (1.67) | 129.3 (3.49) |
| vgg 512² | 16 | 156.9 | 98.4 | 143.1 (0.91) | 269.5 (1.72) | 387.8 (2.47) |
| r50 224² | 16 | 14.1 | 9.0 | 18.3 (1.30) | 18.9 (1.34) | 47.5 (3.37) |
| r50 384² | 16 | 27.9 | 18.9 | 37.4 (1.34) | 47.7 (1.71) | 98.6 (3.53) |

- PyTorch and ORT ran interleaved in one process (`scripts/bench.py`); rten
  (`rten-cli -k`), tract (`tractbench/`) and burn-flex (`burnbench/`) ran as separate
  processes, back to back.
- **`bench_t16.log` is invalid**: ORT's spinning intra-op pool starved PyTorch's OpenMP
  threads (480–1160 ms). The table uses `bench_t{8,16}_nospin.log`.
- tract multithreads only with `tract-linalg/multithread-mm` plus an executor;
  zentract does not enable it.

## GPU (RTX 4080, batch 1, ms median)

Timer = forward + device-side `sum()` of every output + one scalar readback (the shape
of an LPIPS score); fresh input per iteration, synced before the timer starts.

| model | torch fp32 (TF32 off) | torch fp16 channels_last | burn-cuda f32 | burn-vulkan f32 |
|---|---|---|---|---|
| vgg 256² | 1.45 | 0.87 | 3.10 (2.1×) | 146.2 (101×) |
| vgg 512² | 4.87 | 2.83 | 7.59 (1.56×) | 584.9 (120×) |
| r50 224² | 2.26 | 1.26 | 2.80 (1.24×) | 12.4 (5.5×) |
| r50 384² | 2.61 | 1.30 | 3.12 (1.20×) | 29.0 (11×) |

- PyTorch 2.11.0+cu128, cuDNN 9.19, `cudnn.benchmark = True`. burn with `fusion` +
  `autotune`. ORT CUDA EP (default TF32) measured r50 1.24 / 1.78 ms; its VGG numbers
  include a full-output readback and are not comparable.
- **burn-vulkan on VGG is slower than this host's CPU.** Not root-caused.
- **burn 0.22 timing trap:** `Tensor::into_data()` returns before the GPU finishes; the
  cost lands in `TensorData::to_vec`. First-pass burn GPU numbers (0.6 ms for VGG
  512²) were enqueue-only and were discarded; the table's numbers were cross-checked
  against an external wall clock. Stop any burn GPU timer after a scalar readback.

## Takeaways

- **rten** is the only pure-Rust CPU engine at PyTorch parity (VGG at every thread
  count; ResNet50 1.28–1.45× behind at 8–16 threads).
- **burn** is the only candidate with one model source for CPU, CUDA, Vulkan/Metal and
  WebGPU, and is behind PyTorch everywhere measured.
- **ORT** beats PyTorch on CPU but links C++: oracle only.
- The `burn-conv-spike` verdict (cubek conv 4.32× slower than a hand stencil) was a
  1-channel separable blur and does not transfer to 64–512-channel CNN layers.

## burn + zenforks-cubecl in one binary

Probe: `nn_runtime_shootout_2026-10-14/coexist-probe/` (own `[workspace]`, path dep on
`crates/ssim2-gpu`; built at master `b1d63847` under `run-heavy`, 160 s, peak RSS
3.47 GiB). It links burn 0.22.0 (`flex,cuda,vulkan,fusion,autotune`; upstream cubecl
0.11.0, wgpu 30.0.1) and `ssim2-gpu` (zenforks-cubecl 0.10.2 @ `90842401`, wgpu 29.0.4)
and, in one process, alternates a burn 512² matmul-sum with an ssim2-gpu 256² score,
five rounds, on CUDA and then on wgpu (Vulkan), each mode run twice (`run.log`).

- Builds, links, runs: yes, on both backends.
- ssim2 score: 8.939322 (CUDA) and 8.938630 (wgpu) on every round of both runs.
- `cargo tree -d`: cubecl-{common,ir,macros,zspace} 0.10 + 0.11 and wgpu 29 + 30 side
  by side; **one shared `cudarc` 0.19.10**, feature set identical with and without burn
  (`cuda-version-from-build-system, driver, fallback-dynamic-loading, fallback-latest,
  nccl, nccl-02030, nvrtc, std`).
- Not measured: VRAM overhead of two pools, binary size delta, Metal.
