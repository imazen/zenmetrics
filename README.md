# zenmetrics

Multi-vendor GPU implementations of the perceptual image quality
metrics Imazen runs in production, plus a unified CLI.

Built on [CubeCL](https://github.com/tracel-ai/cubecl) — a single
`#[cube]`-annotated Rust kernel source dispatches across CUDA (NVIDIA),
WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a
build-time CPU fallback.

## Crates

| Crate | Metric | Range / shape | Parity reference |
|---|---|---|---|
| [`butteraugli-gpu`](crates/butteraugli-gpu/) | Butteraugli | distance, max-norm + 3-norm | [`butteraugli`](https://crates.io/crates/butteraugli) v0.9 |
| [`ssim2-gpu`](crates/ssim2-gpu/) | SSIMULACRA2 | 0–100, higher better | [`ssimulacra2`](https://crates.io/crates/ssimulacra2) v0.5 |
| [`dssim-gpu`](crates/dssim-gpu/) | DSSIM | distance, 0 = identical | [`dssim-core`](https://crates.io/crates/dssim-core) v3.4 |
| [`zensim-gpu`](crates/zensim-gpu/) | zensim feature extractor | 228-feature vector + scalar score 0–100 | [`zensim`](https://github.com/imazen/zensim) v0.2.8 |
| [`zen-metrics-cli`](crates/zen-metrics-cli/) | CLI front-end | — | uses the four metrics above |
| [`zenmetrics-corpus`](crates/zenmetrics-corpus/) | shared test images | — | (test infra) |

## SRCC sanity table

Spearman rank correlation coefficient against published still-image
MOS datasets (numbers from Cloudinary's SSIMULACRA2 benchmark, sign
normalized so higher = better):

| Metric | TID2013 | KADID-10k | CID22 |
|---|---|---|---|
| `dssim-gpu` (= DSSIM) | 0.871 | 0.856 | 0.872 |
| `ssim2-gpu` (= SSIMULACRA2) | 0.819 | 0.785 | 0.885 |
| `zensim-gpu` (= zensim) | (Imazen-internal benchmark) | | |
| `butteraugli-gpu` (3-norm) | 0.664 | 0.543 | 0.794 |

## Documentation

- [`docs/CUBECL_PORTING_GUIDE.md`](docs/CUBECL_PORTING_GUIDE.md) — patterns
  for porting more CUDA / scalar metrics to multi-vendor CubeCL.
- [`docs/CUBECL_GOTCHAS.md`](docs/CUBECL_GOTCHAS.md) — 30-entry catalogue
  of cubecl-0.10-era traps with symptoms / fixes / examples.
- [`docs/SSIMULACRA2_PORTING_PLAN.md`](docs/SSIMULACRA2_PORTING_PLAN.md),
  [`docs/SSIM2_GPU_HANDOFF.md`](docs/SSIM2_GPU_HANDOFF.md) — the per-crate
  porting playbooks.

## License

Dual-licensed: AGPL-3.0-only (see [`LICENSE-AGPL3`](LICENSE-AGPL3)) or
Imazen commercial (see [`COMMERCIAL.md`](COMMERCIAL.md)). `dssim-gpu`'s
commercial track threads through Pornel's upstream DSSIM licensing —
see `COMMERCIAL.md`.
