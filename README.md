# zenmetrics [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenmetrics/actions/workflows/ci.yml) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

zenmetrics is a pure-Rust collection of full-reference image-quality metrics,
with one CLI that scores any `(reference, distorted)` pair with any of them.

It covers the perceptual metrics codec work leans on (SSIMULACRA2,
Butteraugli, ColorVideoVDP, DSSIM, VMAF, HDR-VDP), the classical IQA set
(SSIM, MS-SSIM, IW-SSIM, GMSD, FSIM, VSI, HaarPSI, VIF, MAD, NLPD, PSNR-HVS),
and the exact variants the JPEG AIC-4 dataset publishes. Each metric is
checked against its reference implementation; the measured residuals are in
[DIVERGENCES.md](DIVERGENCES.md). Six of them also run on the GPU (CUDA,
Vulkan, Metal, DX12) through CubeCL. `#![forbid(unsafe_code)]` throughout.

## Quick start

Nothing here is on crates.io yet (every crate is `publish = false`). The
workspace path-depends on sibling Imazen repos, so clone those next to it
first, exactly as CI does:

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh            # clones the pinned siblings into ../
cargo build --release -p zenmetrics-cli      # binary: target/release/zenmetrics
```

Score one pair:

```sh
zenmetrics score --metric ssim2 --reference ref.png --distorted out.jpg
```

Score several encodes of one reference with several metrics, decoding each
image once:

```sh
zenmetrics compare --reference ref.png \
  --variant q60.jpg --variant q80.jpg --variant out.avif \
  --metric ssim2 --metric butteraugli --metric dssim --output tsv
```

Score a TSV of pairs (`ref_path`, `dist_path` columns plus a header row;
extra columns pass through to the output TSV):

```sh
zenmetrics batch --metric cvvdp --display-model standard_fhd \
  --pairs pairs.tsv --output scores.tsv --jobs 8
```

`zenmetrics list-metrics` prints what your build enabled and which metrics
need a GPU; `zenmetrics list-formats` prints the decoders.

## Metrics

All of these take sRGB 8-bit input through `--metric` in the default build,
except where noted. "Matches" names the implementation each port is
validated against; [docs/METRIC_PROVENANCE.md](docs/METRIC_PROVENANCE.md) has
the papers, oracle commits and gated tolerances.

**Perceptual**

| `--metric` | What it is | Scale | Matches |
|---|---|---|---|
| `ssim2` | SSIMULACRA2 (via [`fast-ssim2`](https://github.com/imazen/fast-ssim2)) | 0–100, higher is better | Cloudinary `ssimulacra2` |
| `butteraugli` | Butteraugli; emits max and libjxl 3-norm | distance, 0 = identical | libjxl butteraugli |
| `cvvdp` | ColorVideoVDP, still images (`--display-model` required) | JOD 0–10, 10 = no visible difference | pycvvdp 0.5.7 |
| `dssim` | DSSIM (via `dssim-core`) | distance, 0 = identical | dssim-core |
| `zensim` | Imazen's trained metric (via [`zensim`](https://github.com/imazen/zensim)) | 0–100 | in-house |
| `hdrvdp` | HDR-VDP 2.2.2 — absolute-luminance input only, via `--hdr` | Q, higher is better | official MATLAB 2.2.2 |
| `hdrvdp3` | HDR-VDP 3.0.7 — `--hdr` plus explicit viewing conditions (`--hdrvdp3-ppd`) | JOD 0–10 | official MATLAB 3.0.7 |

**Video-codec metrics**

| `--metric` | What it is | Scale | Matches |
|---|---|---|---|
| `vmaf` | VMAF v0.6.1 | 0–100 | libvmaf 3.2.1 |
| `vmaf-neg` | VMAF v0.6.1 NEG (no enhancement gain) | 0–100 | libvmaf 3.2.1 |
| `vmaf-4k` | VMAF v0.6.1 4K model | 0–100 | libvmaf 3.2.1 |
| `vmaf-v1` | VMAF v1.0.16 (3d0h) | 0–100 | libvmaf 3.2.1 |
| `ssim-libvmaf` | libvmaf `float_ssim` | 0–1 | libvmaf (FFI oracle) |
| `msssim-libvmaf` | libvmaf `float_ms_ssim` | 0–1 | libvmaf (FFI oracle) |
| `psnrhvs-daala` | Daala/Xiph integer PSNR-HVS | dB | libvmaf `psnr_hvs` (FFI oracle) |

**Structural and statistical**

| `--metric` | What it is | Scale | Matches |
|---|---|---|---|
| `ssim` | single-scale SSIM, 11-tap Gaussian, per RGB channel | 0–1 | — |
| `msssim` | MS-SSIM (Wang et al. 2003), luma | 0–1 | authors' `msssim.m` |
| `iwssim` | IW-SSIM (Wang & Li 2011) | 0–1 | Python-IW-SSIM f9de37c |
| `iwssim-piq` | IW-SSIM on unrounded luma (the AIC-4 column) | 0–1 | jpeg-ai-qaf `IW_SSIM` |
| `vif` | VIFp, multi-scale pixel domain | ≥0, ~1 = identical | authors' `vifp_mscale.m` |
| `vifvec` | VIF, steerable-pyramid vector GSM (a different algorithm) | ≥0, ~1 = identical | authors' `vifvec.m` |
| `mad` | MAD (Larson & Chandler 2010) | distance, 0 = identical | official MATLAB release |
| `nlpd` | Normalized Laplacian Pyramid Distance, RGB (Laparra et al.) | distance, 0 = identical | authors' PyTorch reference |
| `nlpd-iqa` | NLPD, `IQA_pytorch` single-channel configuration (the AIC-4 column) | distance, 0 = identical | `IQA_pytorch` |

**Gradient, phase and saliency**

| `--metric` | What it is | Scale | Matches |
|---|---|---|---|
| `gmsd` | GMSD (Xue et al. 2014) | distance, 0 = identical | libgmsd |
| `ms-gmsd` | multi-scale GMSD | distance, 0 = identical | paper |
| `ms-gmsdc` | multi-scale GMSD with colour | distance, 0 = identical | paper |
| `mdsi` | MDSI (Nafchi et al. 2016) | distance, 0 = identical | paper |
| `fsim`, `fsim-y` | FSIMc / FSIM on luma | 0–1 | authors' `FR_FSIMc.m` |
| `vsi` | VSI (Zhang et al. 2014) | 0–1 | authors' `VSI.m` |
| `haarpsi`, `haarpsi-y` | HaarPSI / on luma | 0–1 | authors' `haarpsi.m` |

**PSNR family**

| `--metric` | What it is | Scale |
|---|---|---|
| `psnr` | PSNR over RGB8 | dB |
| `psnr-y` | PSNR on full-range BT.709 luma | dB |
| `psnr-y601` | PSNR on full-range BT.601 luma (the MATLAB `rgb2gray` convention) | dB |
| `psnr-y-studio601` | PSNR on studio-swing BT.601 luma (the JPEG / AIC-4 `PSNR-Y` column) | dB |
| `psnr-y-libvmaf` | PSNR on studio-swing BT.709 luma (what libvmaf's `psnr` feature reports) | dB |
| `psnrhvs`, `psnrhvs-y` | PSNR-HVS and PSNR-HVS-M (Ponomarenko, `psnrhvsm.m`); `-y` on luma | dB |
| `mdctpsnr` | mDCT-PSNR (Richter 2009), matches the compiled `thorfdbg/mDCTpsnr` | dB |

**GPU twins** (`--features gpu-<metric>`; CUDA or wgpu): `ssim2-gpu`,
`butteraugli-gpu`, `dssim-gpu`, `iwssim-gpu`, `zensim-gpu`, `cvvdp-gpu`. Each
is parity-tested against its CPU twin; tolerances are in
[docs/GPU_METRIC_PARITY.md](docs/GPU_METRIC_PARITY.md).

### Flags that change scores

- `--display-model <preset>` is required for `cvvdp`: there is no default
  display. The score column names it (`…_standard_fhd`). Presets come from
  pycvvdp's `display_models.json` (`standard_fhd`, `standard_4k`,
  `standard_phone`, …).
- `--luma-ingress yuv601-studio` makes the luma-only metrics read the
  studio-swing BT.601 plane libvmaf and JPEG AIC build, which reproduces the
  published AIC-4 luma columns. The default, `house`, is each metric's own
  documented RGB-to-luma.
- `--hdr` decodes HDR sources (EXR, Ultra HDR JPEG, gain-map HEIC) to absolute
  luminance and feeds each metric its HDR path. See
  [docs/HDR_COMMON_PRIMARIES_2026-09-15.md](docs/HDR_COMMON_PRIMARIES_2026-09-15.md).

## What does it cost?

Single-threaded, on one 3355×2516 (8.4 MP) photo pair, in the default CLI
build. Each metric ran in its own process; the figure is the second (warm)
call. Ryzen 9 7900X. Full table, 1 MP figures, peak memory and caveats:
[benchmarks/cpu_metrics_1t8t_2026-10-09.md](benchmarks/cpu_metrics_1t8t_2026-10-09.md).

| Time per pair | Metrics |
|---|---|
| under 20 ms | `psnr`, `psnr-y`, `gmsd`, `mdsi` |
| 0.1–0.5 s | `ssim-libvmaf`, `psnrhvs`, `haarpsi`, `vmaf`¹, `ms-gmsd`, `ms-gmsdc`, `nlpd-iqa`, `psnrhvs-daala`, `msssim`, `nlpd`, `zensim` |
| 0.6–1.8 s | `vsi`, `iwssim`, `iwssim-piq`, `fsim`, `msssim-libvmaf`, `ssim2`, `vif`, `butteraugli`, `ssim`, `cvvdp`, `dssim` |
| 3.8 s | `vifvec` |
| about 1 minute | `mad` (59.7 s), `mdctpsnr` (65.8 s) |

<sub>¹ at 3354×2516; `vmaf` refuses odd widths.</sub>

`mad` grows about 3.8× faster than pixel count between 1 MP and 8.4 MP; the
cause is not yet known. Only `zensim`, `dssim` and `butteraugli` use more than
one thread in the default build (see Limitations).

## Other subcommands

| Subcommand | Does |
|---|---|
| `score-video` | cvvdp's temporal path over two directories of frames |
| `sweep` | encode an image set across a quality × knob grid and score every variant (`--features sweep`) |
| `score-pairs` | score a TSV of pairs into a parquet sidecar (`--features sweep`) |
| `assemble` | join score sidecars onto feature tables for training corpora (`--features assemble`) |
| `jobexec` | run one job for the [zenfleet](docs/RUNNING_JOBS.md) job system (`--features jobexec`) |
| `fleet-plan` | size a fleet (RAM, cores, VRAM, box count) for a sweep |
| `size-invariance` | check that scores stay stable under downsample-and-rescore |
| `capabilities` | print the features this binary was built with; `--probe` tests the GPU |

## Using it as a library

- [`zenmetrics-api`](crates/zenmetrics-api/README.md): one `MetricKind`
  enum and one `Metric` type over every per-crate scorer, CPU or GPU.
- [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/README.md):
  scores many pairs in one process, picks CPU or GPU per metric from a
  benchmark cache, and falls back on out-of-memory.
- Each metric crate under `crates/` also works on its own; see its README.

## Limitations

- Build from source only, with the sibling repos checked out (see Quick
  start). There are no crates.io releases yet.
- Most metrics run on one thread in the default build. The CLI builds eleven
  in-tree metric crates without their `parallel` (rayon) feature, and
  `fast-ssim2` without `rayon`. Turning them on leaves scores bit-identical
  and, at 8 threads on an 8.4 MP pair, speeds those metrics up 1.4–2.4×
  (cvvdp 2.4×, mad 1.8×); see the threading section of the benchmark doc.
  `cargo build --release -p cpu-profile --features metrics-parallel` builds
  the benchmark harness that way.
- `vmaf` rejects odd image dimensions. Crop to even sizes first.
- `hdrvdp` and `hdrvdp3` need absolute-luminance input (`--hdr`); they refuse
  sRGB pairs.
- Deep-learning metrics (LPIPS, DISTS, TOPIQ and similar) are out of scope:
  everything here is classical or trained-but-small Rust.
- `ms-gmsd`, `ms-gmsdc` and `mdsi` are paper-derived; there is no author
  software to reproduce, so their parity claims are weaker than the others'.

## Documentation

- [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md): the detailed
  reference this README used to be — GPU memory modes, the per-mode API, the
  orchestrator, CPU/GPU performance profiles, GPU CI.
- [DIVERGENCES.md](DIVERGENCES.md): every known deviation from each reference.
- [docs/METRIC_PROVENANCE.md](docs/METRIC_PROVENANCE.md): papers, oracles,
  tolerances, and the JPEG AIC-4 reproduction matrix.
- [docs/AIC2026_METRICS_AND_FITTING.md](docs/AIC2026_METRICS_AND_FITTING.md):
  which AIC-4 columns we reproduce, which we can't, and JND normalization.
- [docs/RUNNING_JOBS.md](docs/RUNNING_JOBS.md): the zenfleet job system for
  distributed sweeps and backfills.
- [crates/margarine/README.md](crates/margarine/README.md): Margarine, an
  unpublished research metric (not in `--metric` yet).
- [docs/CUBECL_PORTING_GUIDE.md](docs/CUBECL_PORTING_GUIDE.md),
  [docs/CUBECL_GOTCHAS.md](docs/CUBECL_GOTCHAS.md): porting metrics to CubeCL.

## License

Dual-licensed: AGPL-3.0-only (see [`LICENSE-AGPL3`](https://github.com/imazen/zenmetrics/blob/master/LICENSE-AGPL3))
or Imazen commercial (see [`COMMERCIAL.md`](https://github.com/imazen/zenmetrics/blob/master/COMMERCIAL.md)).
`dssim-gpu`'s commercial track requires Kornel's upstream DSSIM licensing — see
[`COMMERCIAL.md`](https://github.com/imazen/zenmetrics/blob/master/COMMERCIAL.md); this crate is
neither maintained nor warrantied by him.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · **zenmetrics** · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
