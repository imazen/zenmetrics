# ssim2 GPU backend A/B — CUDA vs Vulkan, and native Linux vs WSL2

**Date:** 2026-08-07
**Harness:** `crates/ssim2-gpu/examples/backend_ab.rs` (landed `2d4100cc`)
**Raw data:** `ssim2_backend_ab_2026-08-07_{lianli,lianli_cuda,wsl_cuda}.csv`

Two questions, one harness:

1. How fast is cubecl on **Vulkan** vs its **CUDA** backend, same kernel, same card?
2. What does **WSL2** cost versus **native Linux** for GPU scoring?

## Machines

| | lianli | wsl |
|---|---|---|
| OS | Ubuntu 26.04 (bare metal) | WSL2 on Windows (Ubuntu 22.04) |
| GPU | RTX 2080, 8 GB (Turing) | RTX 5070, 12 GB (Blackwell) |
| Driver | 595.84 | 596.21 |
| CUDA | 12.6 (lifted from the fleet worker image) | 13.2 (system toolkit) |
| Vulkan | NVIDIA ICD present → real GPU | **none** — only `llvmpipe` (CPU) |

`rustc 1.97.1` on both; `[profile.release]` mirrored from the workspace root.

## Method

Identical typed `Ssim2<R>` pipeline on each runtime — only `R` differs. Measures the
**warm-reference** path (`set_reference` once, then `compute_with_reference` per
candidate), which is the production scoring shape. Pipeline construction is timed
separately as the per-cell fixed cost. Rounds alternate **ABBA** so monotonic
clock/thermal drift cancels; one live pipeline at a time (two 12 MP ssim2 pipelines do
not fit 8 GB). 5 rounds × 5 reps + 2 warmup. Medians reported.

Two guards, both of which earned their keep:

- **Vulkan adapter is asserted `DiscreteGpu`.** lianli exposes *two* Vulkan devices —
  the RTX 2080 and `llvmpipe`. Silently selecting llvmpipe would have benchmarked a CPU
  software rasterizer and labelled it "Vulkan".
- **Preflight proves both backends compute the real kernel.** With `CUDA_PATH` unset,
  cubecl-cuda panics on a *worker* thread — invisible to any `Result` on the calling
  thread — and the run prints a full column of plausible-looking garbage. This happened
  before the guard existed.

## Q1 — Vulkan vs CUDA (lianli, same RTX 2080)

Warm-reference median, ms:

| size | CUDA | Vulkan | Vulkan/CUDA | construction CUDA → Vulkan |
|---|---|---|---|---|
| 64×64 | 0.587 | 1.570 | **2.67×** | 1.0 → 1.2 ms |
| 256×256 | 1.306 | 3.151 | **2.41×** | 4.0 → 4.1 ms |
| 1024×1024 | 4.817 | 8.994 | **1.87×** | 87.1 → 64.0 ms |
| 2048×2048 | 13.965 | 23.301 | **1.67×** | 434.7 → 569.9 ms |

**Vulkan is 1.7–2.7× slower than CUDA for the same kernel on the same card**, and the
gap *narrows monotonically with size* — the signature of a fixed per-dispatch cost being
amortised over more work. Two-point fit over the largest sizes (1024²→2048²):

- CUDA: ≈ **1.99 ms + 2.73 ms/MP**
- Vulkan: ≈ **4.23 ms + 4.55 ms/MP**

So it is not purely launch overhead: Vulkan pays **both** a ~2.2 ms larger fixed cost
**and** ~1.6× more per megapixel. The per-MP component points at codegen — this build
goes WGSL→naga→SPIR-V, since cubecl-wgpu's `spirv` feature (native SPIR-V codegen via
`cubecl-spirv`) is **not enabled anywhere in this workspace**. That is the obvious next
experiment before concluding anything about Vulkan-the-API.

**Correctness: no divergence.** Scores agree to ≤2.5e-3 absolute (≤6e-6 relative):

| size | CUDA score | Vulkan score | Δ |
|---|---|---|---|
| 64×64 | −61.854164 | −61.854223 | 5.9e-5 |
| 256×256 | −194.745017 | −194.746924 | 1.9e-3 |
| 1024×1024 | −409.574777 | −409.577296 | 2.5e-3 |
| 2048×2048 | −435.631671 | −435.631157 | 5.1e-4 |

That is float reduction-order noise, not a bug. The gap is speed, not correctness.

## Q2 — native Linux vs WSL2

**The GPUs differ, so per-pixel throughput is NOT a WSL-vs-native measurement.** What
*is* comparable is behaviour at sizes where compute is negligible, and pipeline
construction — both dominated by driver round-trips rather than FLOPs.

Warm-reference median, CUDA only, ms:

| size | lianli (2080, native) | wsl (5070) | wsl/native |
|---|---|---|---|
| 64×64 | 0.585 | 2.084 | **3.56×** |
| 256×256 | 1.191 | 2.684 | **2.25×** |
| 1024×1024 | 4.852 | 4.250 | 0.88× |
| 2048×2048 | 13.448 | 9.553 | 0.71× |

Pipeline construction, ms:

| size | lianli | wsl | wsl/native |
|---|---|---|---|
| 64×64 | 0.8 | 3.1 | **3.9×** |
| 256×256 | 3.9 | 13.6 | **3.5×** |
| 1024×1024 | 83.5 | 217.6 | **2.6×** |
| 2048×2048 | 462.1 | 1169.2 | **2.5×** |

Three findings:

1. **WSL2 imposes a large fixed per-operation tax.** At 64×64 there is almost no compute,
   so the time is nearly all submission overhead — and WSL is 3.6× slower *despite
   running two GPU generations newer silicon*. Because the 5070 should be **faster**, the
   raw ratio is a **lower bound** on the virtualization penalty, not an estimate of it.
2. **Raw compute is not taxed.** At 2048² the 5070 delivers its expected generational win
   (1.41× faster than the 2080). The overhead is per-operation, not per-FLOP.
3. **WSL latency is far jitterier.** median/min spread at 64×64 is **1.73× on WSL vs
   1.02× native** — the native box is metronomic, WSL is not. For a scoring fleet this
   matters as much as the mean: per-cell overhead compounds across hundreds of thousands
   of cells.

**Throughput implication:** on WSL, small and medium images pay a heavy, jittery fixed
cost, so batching and reference-reuse are worth far more there than on native Linux.
Construction alone is 1.17 s at 2048² on WSL — any scoring path that rebuilds a pipeline
per cell is paying more in setup than in scoring.

## Caveats

- **Different GPUs** (2080 vs 5070). Per-pixel numbers are not cross-machine comparable;
  only the overhead-dominated and construction numbers are argued above.
- **Different CUDA** (12.6 vs 13.2) and driver (595.84 vs 596.21). NVRTC version differs,
  which plausibly affects construction (JIT) timing.
- **Concurrent tenants.** lianli ran the avifgen fleet worker throughout (144–2448 MiB);
  the WSL box hosts three other agent lanes. Medians are robust but the 2048² CUDA p90 on
  lianli hit 71.5 ms vs a 14.0 ms median — contention spikes are real. **Neither run had
  an exclusive GPU.**
- **Repeatability floor ≈ 4–9%**, measured by running the CUDA arm twice on lianli
  (both-backend vs cuda-only mode). All the effects claimed above are 1.7–3.9×, well clear
  of that floor.
- **Synthetic LCG noise images**, not real content. ssim2's `Faster` skip-map may behave
  differently on real photographs, which could shift absolute times (not obviously the
  cross-backend ratio).
- Vulkan measured via **WGSL→naga**, not cubecl's native SPIR-V path.

## Reproduce

```bash
export CUDA_PATH=/usr/local/cuda   # must contain include/ — NVRTC needs the headers
BENCH_BACKENDS=both BENCH_SIZES=64x64,256x256,1024x1024,2048x2048 \
BENCH_REPS=5 BENCH_ROUNDS=5 ./backend_ab
```

On a box with only the driver, the CUDA runtime can be lifted out of the fleet image:

```bash
cid=$(docker create ghcr.io/imazen/zenfleet-worker:exec-gpu)
docker cp "$cid":/usr/local/cuda/. ~/opt/cuda/
```
