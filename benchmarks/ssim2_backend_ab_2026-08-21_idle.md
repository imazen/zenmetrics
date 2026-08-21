# ssim2 GPU backend A/B on an IDLE GPU — WGSL vs native SPIR-V

**Date:** 2026-08-21
**Harness:** `crates/ssim2-gpu/examples/backend_ab.rs`
**Raw data:** `ssim2_backend_ab_2026-08-21_idle_{wgsl,spirv}.csv`
**Box:** `r7900x` (was "lianli"), Ubuntu 26.04 bare metal, 24c/29 GB,
**NVIDIA GTX 1060 6 GB**, NVIDIA Vulkan ICD present.
**GPU was completely idle** — no fleet worker, no other tenants.

This is the idle-GPU re-measure that imazen/zenmetrics#44 asked for, and it
**materially corrects** the headline in `ssim2_backend_ab_2026-08-07.md`.

## 1. Native SPIR-V vs WGSL→naga (the #44 question)

Same card, same session, two binaries from one source, run back-to-back.
Vulkan-arm median, ms:

| size | WGSL→naga | native SPIR-V | gain | noise floor | above noise? |
|---|---|---|---|---|---|
| 64×64 | 1.933 | 1.048 | **1.85×** | 15.2% | yes |
| 256×256 | 3.482 | 2.279 | **1.53×** | 2.3% | yes |
| 1024×1024 | 11.518 | 10.583 | **1.09×** | 0.3% | yes |
| 2048×2048 | 37.110 | 35.186 | **1.05×** | 0.2% | yes |

**Noise floor is measured, not assumed:** the CUDA arm is identical code in both
binaries, so its run-to-run spread bounds the noise — 15.2% at 64×64 (latency
-dominated and jittery), 2.3% at 256², and **under 0.5% at 1024²/2048²**. Every
SPIR-V gain clears its own size's floor.

The gain is largest at small sizes and shrinks with area — the signature of a
fixed per-dispatch codegen/setup cost, not a per-pixel win. Scores are unchanged
across codegen paths, so this is pure codegen.

## 2. Vulkan vs CUDA — the 2026-08-07 headline was inflated by contention

Vulkan/CUDA ratio (>1 = Vulkan slower):

| size | 08-07 RTX 2080, **contended**, WGSL | 08-21 GTX 1060, **idle**, WGSL | 08-21 GTX 1060, **idle**, SPIR-V |
|---|---|---|---|
| 64×64 | 2.67× | 1.19× | **0.76×** |
| 256×256 | 2.41× | 1.07× | **0.71×** |
| 1024×1024 | 1.87× | 1.11× | 1.02× |
| 2048×2048 | 1.67× | 1.52× | 1.44× |

Two things follow:

- **"Vulkan is 1.67–2.67× slower than CUDA" does not survive an idle GPU.** On an
  uncontended card the WGSL gap is 1.07–1.52×. The earlier run shared an 8 GB card
  with the avifgen fleet worker (144 MiB–2.4 GiB resident, and a 2048² CUDA p90 of
  71.5 ms against a 14.0 ms median) — that contention inflated the ratio.
- **With native SPIR-V, Vulkan BEATS CUDA at small sizes** (0.76× and 0.71×, i.e.
  ~1.3–1.4× faster) and reaches parity at 1024² (1.02×). CUDA retains a clear
  advantage only at 2048² (1.44×).

**Caveat, stated plainly:** the two dated runs used *different cards* (RTX 2080 vs
GTX 1060) as well as different contention, so this table cannot separate
"contention" from "card" for the CUDA-vs-Vulkan delta. What it does establish is
that the earlier ratio is not reproducible on an idle GPU and should not be quoted
as the cost of Vulkan. The SPIR-V-vs-WGSL comparison in §1 has no such confound —
same card, same session, back-to-back.

## 3. What this means for `gpu-wgpu-spirv`

Native SPIR-V is a strict win on this hardware — faster at every size, above the
noise floor at every size, identical scores. Nothing here argues against enabling
it for wgpu-backed builds.

**Still not done before flipping the default:** this is still *one metric* (ssim2)
on *one card*. #44 asked for "across metrics", and butteraugli / zensim / cvvdp
have no backend-parameterized harness yet. Note also that cubecl-wgpu documents
`msl` and `spirv` as mutually exclusive, so a Metal build cannot take this path.

## Reproduce

```bash
export CUDA_PATH=/usr/local/cuda   # must contain include/ — NVRTC needs the headers
BENCH_SIZES=64x64,256x256,1024x1024,2048x2048 BENCH_REPS=5 BENCH_ROUNDS=5 ./backend_ab
# SPIR-V variant: build the driver with `spirv-native = ["cubecl/vulkan"]`
```
