# zenmetrics CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.

## Local CUDA toolkit (for building/running GPU metrics)

The water-cooled 7950X workstation has CUDA 13.2.1 SDK installed at the
default location, but **nvcc is not on PATH by default**. CUDA layout:

    /usr/local/cuda            → /usr/local/cuda-13.2  (current symlink)
    /usr/local/cuda-13.2/bin/nvcc
    /usr/local/cuda-13.2/lib64/  (libcudart.so etc.)

Other versions also installed: 12.6, 13. Use `/usr/local/cuda` (the
symlink) unless you have a reason for a specific version.

To compile a `cargo` invocation that needs nvcc, prepend:

    PATH=/usr/local/cuda/bin:$PATH
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

But note: **cubecl-cuda dynamically loads CUDA at runtime** via dlopen,
so building `--features sweep,gpu,gpu-cuda` succeeds even with nvcc off
PATH. The runtime fallback is sufficient for `zen-metrics` builds. Set
PATH explicitly only when shelling out to nvcc directly.

GPU info: `nvidia-smi` driver 596.21 / CUDA capability runtime 13.2.

## Sweep build cheat sheet

- **Default CPU+GPU build (development)**:
  `cargo build --release -p zen-metrics-cli`
  → includes both `cpu-metrics` (default) and `sweep` codecs. ~2 min cold,
  seconds incremental.

- **Forced GPU-only sweep build (production worker)**:
  `cargo build --release -p zen-metrics-cli --no-default-features --features sweep,png,gpu,gpu-cuda`
  → drops cpu-metrics so CPU butteraugli/zensim/ssim2 are *unavailable*;
  any chunk specifying a CPU metric will fail loudly. Use this for vast.ai
  workers so they can't silently fall back to slow CPU scoring. ~4 min cold.

- **WGPU variant (broader GPU compatibility, no CUDA SDK required)**:
  `cargo build --release -p zen-metrics-cli --no-default-features --features sweep,png,gpu,gpu-wgpu`
  → uses Vulkan/Metal/DX12 via wgpu. Use when targeting AMD/Intel GPUs
  on vast.ai. CUDA NVIDIA GPUs work but CUDA backend is faster.

## Sweep runner discipline

- **GPU metrics only on production workers.** Mixing CPU/GPU scores
  across a sweep produces inconsistent training data — pickers/trainers
  expect a single metric backend. The forced-GPU build above prevents
  accidental fall-back.
- **Pre-uploaded binary lives at**
  `s3://coefficient/binaries/zen-metrics-<version>-linux-x86_64`
  (R2 endpoint: `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`). Workers
  fetch via `SWEEP_BIN_OVERRIDE` env var.
- **Onstart script**: `scripts/sweep/onstart_v3.sh`. Fans out N parallel
  zen-metrics processes per box (one per CPU core) sharing the GPU for
  scoring; each claims its own chunk from `chunks.jsonl` on R2.

## CHANGELOG.md

Maintained in repo root.
