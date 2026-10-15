# LPIPS-vgg phase 0: PyTorch baseline and the first rten slice (2026-10-15)

Host i265: Intel Core Ultra 7 265K (20 cores, AVX2, no AVX-512), 30 GiB, RTX 4080
16 GB, driver 595.99.02. Every run went through `run-heavy` (16 G cap, 16 jobs).
PyTorch 2.14.1+cpu and 2.11.0+cu128, `lpips` 0.1.4, `DISTS_pytorch` 0.1. rten
0.27.0. Crate code: `crates/zenmetrics-nn` at the commit that adds this file.
Data and scripts: [`nn_lpips_phase0_2026-10-15/`](nn_lpips_phase0_2026-10-15/).

## Inputs

Three imazen-26 sources (`ladder_sources.txt`: a photo, a texture, a render),
centre-cropped square and Lanczos-downscaled to 64–4096 px (never upscaled).
Distortion: Gaussian blur σ 1.0 plus Gaussian noise σ 6 with a fixed seed per
image and size (`scripts/ladder.py`; `ladder_manifest.tsv` has every sha256).
Timing uses image 1415; parity uses all three.

Modes, identical in `scripts/bar.py` (PyTorch) and `examples/lpips_bench.rs`
(rten): `compute` = decoded pixels in memory → score; `e2e` = read and decode
both PNGs → score; `warm` (rten only) = reference features cached, one candidate
scored. Even-count medians take the upper middle value in rten and the mean of
the two middle values in PyTorch. Median of up to 50 runs within a 4 s budget, after one warmup.

## PyTorch reference (Bar A / Bar B)

LPIPS-vgg, median ms (`bar_gpu_i265.tsv`, `bar_cpu_i265.tsv`):

| size | CPU fp32, 16 thr, compute | CPU e2e | 4080 fp32 compute | 4080 fp32 e2e | 4080 fp16 compute |
|---|---|---|---|---|---|
| 64 | 8.0 | 8.7 | 1.09 | 1.47 | 0.97 |
| 256 | 75.1 | 79.6 | 3.29 | 5.55 | 2.12 |
| 512 | 402 | 411 | 12.7 | 21.1 | 6.7 |
| 1024 | 1815 | 1872 | 61.5 | 92.1 | 32.4 |
| 2048 | 7251 | 7400 | 252 | 393 | 140 |
| 4096 | not run (RAM cap) | | OOM | OOM | OOM |

Findings:

- **4096² LPIPS-vgg and DISTS do not fit in 16 GB** on the 4080 in any precision
  for LPIPS-vgg, nor in fp32 for DISTS (DISTS fp16 runs: 597 ms).
- **fp16 is not safe by default:** upstream LPIPS-alex returns NaN at 2048² in
  fp16 on this image, and fp16 moves LPIPS-vgg scores by up to 2.7e-4 (256²).
- **TF32 (PyTorch's default for cuDNN convolutions) moves scores by up to
  1.5e-5**; with it off, 4080 scores match CPU to ~1e-7 (`fp32strict` rows).
- **At large sizes PIL's PNG decode dominates e2e**: LPIPS-alex at 4096² is
  62 ms compute and 634 ms e2e on the 4080.

## rten slice

Parity: 18 pairs (3 images × 64–2048), worst |Δ| **4.5e-8** against upstream
`lpips` on CPU fp32 (`lpips_vgg_goldens.tsv`; test `tests/parity.rs`, proposed
tolerance 1e-6).

LPIPS-vgg, 16 threads, median ms:

| size | PyTorch compute | rten compute, first cut | rten compute, paired | rten e2e | rten warm | PyTorch e2e |
|---|---|---|---|---|---|---|
| 64 | 8.0 | 14.4 | 8.75 | 8.5 | 7.5 | 8.7 |
| 128 | 22.2 | 38.8 | 25.1 | 25.5 | 17.0 | 23.2 |
| 256 | 75.1 | 128 | 87.4 | 88.2 | 53.2 | 79.6 |
| 512 | 402 | 438 | 355 | 360 | 211 | 411 |
| 1024 | 1815 | 1692 | 1408 | 1423 | 855 | 1872 |
| 2048 | 7251 | 6557 | 5495 | 5573 | 3305 | 7400 |

- **Paired runs** (`rten_pair_i265.tsv`): reference and distorted trunks run
  concurrently on two 8-thread pools. One run stops scaling past ~8 threads
  (`rten_threads_i265.tsv`: at 64² warm, 8 threads 7.5 ms, 16 threads 7.1 ms),
  so two half-width runs at once beat two full-width runs in a row. rten is now
  1.13–1.32× faster than PyTorch from 512² up and 9–16 % slower at 64–256².
- **Warm reference**: one trunk run per candidate, 1.17× faster than a cold
  pair at 64² and 1.5–1.7× from 128² up. Upstream `lpips` has no equivalent.
- **Weight prepacking** (`rten_cpu_prepack_i265.tsv`): no change (64² 15.8 vs
  14.4 ms, 1024² 1693 vs 1692 ms); left off.
- 95 % of a 64² run is convolution (`RTEN_TIMING`), so the remaining small-size
  gap is rten's conv kernel and its threading, not our head or preprocessing.

Not measured: dev (Zen 5, AVX-512), other images for timing, GPU engines.
