# zenmetrics [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenmetrics/actions/workflows/ci.yml) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

zenmetrics is a pure-Rust collection of full-reference image-quality metrics,
with one CLI that scores any `(reference, distorted)` pair with any of them.

Most metrics are ports of the authors' published method, checked against a
reference implementation; the table below says which, and by how much, and
[DIVERGENCES.md](DIVERGENCES.md) explains each difference. Where a dataset or tool computed
a metric with its own conventions (the scores published with the JPEG AIC2026
dataset, libvmaf), that variant is available too, so published numbers can be reproduced. Six metrics
also run on the GPU (CUDA, Vulkan, Metal, DX12) through CubeCL.
`#![forbid(unsafe_code)]` throughout.

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

Listed by name. "Validated against" is the implementation each port is
checked against; "Worst difference" is the largest deviation we have measured
from it, or the test's bound where marked ≤ ([DIVERGENCES.md](DIVERGENCES.md)
has the conditions for each).
[docs/METRIC_PROVENANCE.md](docs/METRIC_PROVENANCE.md) records oracle commits
and tolerances. If you are an author and something here misdescribes your
work, please open an issue.

| Metric | Authors | `--metric` | Validated against | Worst difference |
|---|---|---|---|---|
| Butteraugli | Jyrki Alakuijala, Google ([google/butteraugli](https://github.com/google/butteraugli)) | `butteraugli` (max-norm and libjxl 3-norm), `butteraugli-gpu` | libjxl's implementation, via the [`butteraugli`](https://github.com/imazen/butteraugli) crate | GPU vs CPU 1e-4 relative |
| ColorVideoVDP | Rafał K. Mantiuk, Param Hanji, Maliha Ashraf, Yuta Asano, Alexandre Chapiro. ACM TOG 43(4), 2024 ([doi](https://doi.org/10.1145/3658144)) | `cvvdp` (requires `--display-model`), `cvvdp-gpu`; the `score-video` subcommand for video | pycvvdp 0.5.7 | 1.2e-5 JOD images, 2e-6 JOD video |
| DSSIM | Kornel Lesiński ([kornelski/dssim](https://github.com/kornelski/dssim)) | `dssim`, `dssim-gpu` | `dssim-core` (called directly) | GPU twin tested against `dssim-core` |
| FSIM | Lin Zhang, Lei Zhang, Xuanqin Mou, David Zhang. IEEE TIP 20(8), 2011 ([doi](https://doi.org/10.1109/TIP.2011.2109730)) | `fsim` (FSIMc), `fsim-y` (FSIM on luma) | authors' `FR_FSIMc.m`, GNU Octave | 5e-8 |
| GMSD | Wufeng Xue, Lei Zhang, Xuanqin Mou, Alan C. Bovik. IEEE TIP 23(2), 2014 ([doi](https://doi.org/10.1109/TIP.2013.2293423)) | `gmsd` | [libgmsd](https://github.com/clunietp/libgmsd), Tom Clunie's C port of the authors' `GMSD.m` | similarity map bit-identical on even sizes |
| HaarPSI | Rafael Reisenhofer, Sebastian Bosse, Gitta Kutyniok, Thomas Wiegand. Signal Processing: Image Communication 61, 2018 ([doi](https://doi.org/10.1016/j.image.2017.11.001)) | `haarpsi`, `haarpsi-y` (luma) | authors' `HaarPSI.m`, GNU Octave | 5e-5 |
| HDR-VDP-2 | Rafał K. Mantiuk, Kil Joong Kim, Allan G. Rempel, Wolfgang Heidrich. ACM TOG 30(4), 2011 ([doi](https://doi.org/10.1145/1964921.1964935)); 2.2 recalibration: Manish Narwaria, Rafał K. Mantiuk, Matthieu Perreira Da Silva, Patrick Le Callet. JEI 24(1), 2015 ([doi](https://doi.org/10.1117/1.JEI.24.1.010501)) | `hdrvdp` (absolute-luminance input, via `--hdr`) | official HDR-VDP 2.2.2 MATLAB release, GNU Octave | Q 7.8e-4 |
| HDR-VDP-3 | Rafał K. Mantiuk, Dounia Hammou, Param Hanji. arXiv:2304.13625, 2023 ([arXiv](https://arxiv.org/abs/2304.13625)) | `hdrvdp3` (`--hdr` plus explicit viewing conditions) | official HDR-VDP 3.0.7 MATLAB release | identical at print precision on 3 image pairs |
| IW-SSIM | Zhou Wang, Qiang Li. IEEE TIP 20(5), 2011 ([doi](https://doi.org/10.1109/TIP.2010.2092435)) | `iwssim`, `iwssim-gpu`; `iwssim-piq` (jpeg-ai-qaf's unrounded-luma convention, used for the AIC2026 published scores) | [Python-IW-SSIM](https://github.com/Jack-guo-xy/Python-IW-SSIM) @ `f9de37c`, the Python port linked from the authors' page; not yet compared with their MATLAB release | 1e-5 identical inputs, 5e-3 distorted |
| MAD | Eric C. Larson, Damon M. Chandler. JEI 19(1), 2010 ([doi](https://doi.org/10.1117/1.3267105)) | `mad` (also reports the two strategy indices) | authors' MATLAB and C release, GNU Octave | 1.3e-6 relative |
| mDCT-PSNR | Thomas Richter. QoMEX 2009 ([doi](https://doi.org/10.1109/QOMEX.2009.5246978)) | `mdctpsnr` | [mDCTpsnr](https://github.com/thorfdbg/mDCTpsnr) by Thomas Richter, with a contribution from Jon Sneyers, as compiled | 4e-6 dB |
| MDSI | Hossein Ziaei Nafchi, Atena Shahkolaei, Rachid Hedjam, Mohamed Cheriet. IEEE Access 4, 2016 ([doi](https://doi.org/10.1109/ACCESS.2016.2604042)) | `mdsi` | implemented from the paper; not yet compared with the authors' code | — |
| MS-GMSD | Bo Zhang, Pedro V. Sander, Amine Bermak. ICASSP 2017 ([doi](https://doi.org/10.1109/ICASSP.2017.7952357)) | `ms-gmsd`, `ms-gmsdc` (with chroma) | implemented from the paper | — |
| MS-SSIM | Zhou Wang, Eero P. Simoncelli, Alan C. Bovik. Asilomar 2003 ([doi](https://doi.org/10.1109/ACSSC.2003.1292216)) | `msssim`; `msssim-libvmaf` (libvmaf's `float_ms_ssim` convention) | authors' `msssim.m`, GNU Octave; libvmaf | 8.8e-6; libvmaf ≤2e-4 |
| NLPD | Valero Laparra, Johannes Ballé, Alexander Berardino, Eero P. Simoncelli. HVEI 2016 ([doi](https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103)) | `nlpd`; `nlpd-iqa` (IQA_pytorch's single-channel configuration, used for the AIC2026 published scores) | authors' PyTorch implementation; IQA_pytorch | 1e-5 |
| PSNR | — | `psnr` (RGB); `psnr-y` (BT.709 luma), `psnr-y601` (BT.601), `psnr-y-studio601` (studio-swing BT.601, the JPEG and AIC2026 convention), `psnr-y-libvmaf` (studio-swing BT.709, as libvmaf) | — | — |
| PSNR-HVS, PSNR-HVS-M | Karen Egiazarian, Jaakko Astola, Nikolay Ponomarenko, Vladimir Lukin, Federica Battisti, Marco Carli. VPQM 2006; Nikolay Ponomarenko, Flavia Silvestri, Karen Egiazarian, Marco Carli, Jaakko Astola, Vladimir Lukin. VPQM 2007 | `psnrhvs`, `psnrhvs-y` (luma); `psnrhvs-daala` (the Daala/Xiph integer-DCT version in libvmaf) | authors' `psnrhvsm.m`, GNU Octave; libvmaf | 4e-4 dB; libvmaf ≤2e-4 dB |
| SSIM | Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. IEEE TIP 13(4), 2004 ([doi](https://doi.org/10.1109/TIP.2003.819861)) | `ssim` (Gaussian window, mean of R, G, B); `ssim-libvmaf` (libvmaf's `float_ssim` convention) | `ssim`: not yet compared with the authors' code; `ssim-libvmaf`: libvmaf | libvmaf ≤2e-4 |
| SSIMULACRA 2 | Jon Sneyers, Cloudinary ([cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2); SSIMULACRA 2.1 in [libjxl](https://github.com/libjxl/libjxl/blob/main/tools/ssimulacra2.cc)) | `ssim2`, `ssim2-gpu` | the C++ reference, via the [`fast-ssim2`](https://github.com/imazen/fast-ssim2) crate | tracked in `fast-ssim2` |
| VIF | Hamid R. Sheikh, Alan C. Bovik. IEEE TIP 15(2), 2006 ([doi](https://doi.org/10.1109/TIP.2005.859378)) | `vifvec` (wavelet-domain vector GSM, as in the paper); `vif` (the authors' pixel-domain multi-scale VIFp release) | authors' `vifvec.m` with matlabPyrTools; `vifp_mscale.m`; GNU Octave | 1e-9; 5e-13 |
| VMAF | Zhi Li et al., Netflix, 2016 ([Netflix/vmaf](https://github.com/Netflix/vmaf)) | `vmaf` (v0.6.1), `vmaf-neg`, `vmaf-4k`, `vmaf-v1` (v1.0.16) | libvmaf 3.2.1 | features ≤1e-4, score ≤0.02 |
| VSI | Lin Zhang, Ying Shen, Hongyu Li. IEEE TIP 23(10), 2014 ([doi](https://doi.org/10.1109/TIP.2014.2346028)) | `vsi` | authors' `VSI.m`, GNU Octave | 1e-4 |
| zensim | Imazen ([imazen/zensim](https://github.com/imazen/zensim)) | `zensim`, `zensim-gpu` | our own metric; the crate defines it | GPU vs CPU 2e-4 |

The `-gpu` variants need `--features gpu-<metric>` and run on CUDA or wgpu;
[docs/GPU_METRIC_PARITY.md](docs/GPU_METRIC_PARITY.md) has their tolerances.

### Flags that change scores

- `--display-model <preset>` is required for `cvvdp`: there is no default
  display. The JPEG AIC Common Test Conditions (WG1 N101156; Dietmar Saupe,
  Jon Sneyers, Shima Mohammadi, João Ascenso, 2025) use `standard_fhd`. The score column names it (`…_standard_fhd`). Presets come from
  pycvvdp's `display_models.json` (`standard_fhd`, `standard_4k`,
  `standard_phone`, …).
- `--luma-ingress yuv601-studio` makes the luma-only metrics read the
  studio-swing BT.601 plane libvmaf and JPEG AIC build, which reproduces the
  luma-based scores published with the JPEG AIC2026 dataset. The default, `house`, is each metric's own
  documented RGB-to-luma.
- `--hdr` decodes HDR sources (EXR, Ultra HDR JPEG, gain-map HEIC) to absolute
  luminance and feeds each metric its HDR path. See
  [docs/HDR_COMMON_PRIMARIES_2026-09-15.md](docs/HDR_COMMON_PRIMARIES_2026-09-15.md).

## Datasets we validate against

With thanks to their authors:

- **JPEG AIC2026**: Mohsen Jenadeleh, Jon Sneyers, João Ascenso, Thomas Richter,
  Alexander Karabutov, Panqi Jia, Elena Alshina, Osamu Watanabe, António
  Pinheiro, Touradj Ebrahimi, Dietmar Saupe. "JPEG AIC2026: A Large-Scale
  Dataset for Fine-Grained Assessment of Image Coding", 2026
  ([arXiv:2607.22783](https://arxiv.org/abs/2607.22783); data
  [doi:10.18419/DARUS-6156](https://doi.org/10.18419/DARUS-6156), CC BY-SA 4.0).
  Its published metric scores are what the AIC2026 variants above reproduce;
  [docs/AIC2026_METRICS_AND_FITTING.md](docs/AIC2026_METRICS_AND_FITTING.md)
  has the per-column results.
- **CID22**: Jon Sneyers, Elad Ben Baruch, Yaron Vaxman (Cloudinary), JPEG AIC-3
  contribution.
- **TID2013**: Nikolay Ponomarenko, Lina Jin, Oleg Ieremeiev, Vladimir Lukin,
  Karen Egiazarian, Jaakko Astola. Signal Processing: Image Communication 30,
  2015 ([doi](https://doi.org/10.1016/j.image.2014.10.009)).
- **KADID-10k**: Hanhe Lin, Vlad Hosu, Dietmar Saupe. QoMEX 2019
  ([database](https://database.mmsp-kn.de/kadid-10k-database.html)).

## How fast are these implementations?

These are timings of our ports, not of the metrics: several ports are not
optimized yet. On one 3355×2516 (8.4 MP) photo pair, single-threaded on a Ryzen
9 7900X, most take 0.1–2 s; `psnr`, `gmsd` and `mdsi` take under 20 ms;
our `mad` and `mdctpsnr` ports take about a minute each, and our `mad` grows
faster than pixel count. Per-metric figures, peak memory, threading and
caveats: [benchmarks/cpu_metrics_1t8t_2026-10-09.md](benchmarks/cpu_metrics_1t8t_2026-10-09.md).

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
- Metrics that need a neural-network runtime (LPIPS, DISTS, TOPIQ and
  similar) are not included.
- `ms-gmsd`, `ms-gmsdc` and `mdsi` are implemented from their papers and
  have not yet been compared with the authors' code; `ssim` has not been
  compared with the authors' `ssim_index.m`.

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
