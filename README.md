# zenmetrics [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenmetrics/actions/workflows/ci.yml) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

[Reproducible CPU benchmarks and per-OS installation](docs/BENCHMARKING.md).
zenmetrics is a pure-Rust collection of full-reference image-quality metrics,
with one CLI that scores any `(reference, distorted)` pair with any of them.

Most metrics are ports of the authors' published method, checked against a
reference implementation; the table below says which, and by how much, and
[DIVERGENCES.md](DIVERGENCES.md) explains each difference. Where a dataset or tool computed
a metric with its own conventions (the scores published with the JPEG AIC2026
dataset, libvmaf), that variant is available too, so published numbers can be reproduced. Six metrics
also run on the GPU (CUDA, Vulkan, Metal, DX12) through CubeCL.
`unsafe_code = "forbid"` is the workspace default; some crates override it.

[Interactive browser viewer](https://metrics.imazen.org/) ·
[Checkpoint downloads](https://github.com/imazen/zenmetrics/releases)

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

[Checkpoint binaries](docs/CHECKPOINT_BINARIES.md) package the CLI, IQA comparison
drivers and fleet executors with pinned build identities.

## Metrics

[Models & metrics explorer](https://imazen.github.io/zenmetrics/#models) · [Full cards, papers and terms](docs/MODEL_CARDS.md)

Every CLI name is listed below. FR/NR indicates reference requirements; NN marks neural models. Code and model terms, paper citations and full reference identities remain in the linked cards.

**Rust/C** is Rust time divided by canonical time (lower is faster). Approximate local comparisons are descriptive, with their input size, thread budget and source pins in the [comparison ledger](benchmarks/canonical_recheck_2026-10-19/README.md). A dash means no comparable measurement. “Prior” means older recorded evidence; “gate” means a declared bound, not a measured pass.

| Metric | Type | Canonical | Check | Rust/C |
| --- | --- | --- | --- | ---: |
| [`afine`](docs/MODEL_CARDS.md#afine) | FR/NN | ChrisDud0257/AFINE | prior | — |
| [`ahiq`](docs/MODEL_CARDS.md#ahiq) | FR/NN | IIGROUP/AHIQ | prior | — |
| [`ahiq-pyiqa`](docs/MODEL_CARDS.md#ahiq-pyiqa) | FR/NN | pyiqa 0.1.16 create_metric('a… | prior | — |
| [`arniqa`](docs/MODEL_CARDS.md#arniqa) | NR/NN | miccunifi/ARNIQA | prior | — |
| [`brisque`](docs/MODEL_CARDS.md#brisque) | NR | PIQ 0.8.0 | pass | ≈0.12× |
| [`butteraugli`](docs/MODEL_CARDS.md#butteraugli) | FR | libjxl butteraugli v0.9.2 / 0… | unchecked | — |
| [`butteraugli-gpu`](docs/MODEL_CARDS.md#butteraugli-gpu) | FR | libjxl butteraugli v0.9.2 / 0… | unchecked | — |
| [`butteraugli-original`](docs/MODEL_CARDS.md#butteraugli-original) | FR | unrecorded | unchecked | — |
| [`cambi`](docs/MODEL_CARDS.md#cambi) | NR | unrecorded | unchecked | — |
| [`cambi-aic2026`](docs/MODEL_CARDS.md#cambi-aic2026) | NR | unrecorded | unchecked | — |
| [`ciede2000`](docs/MODEL_CARDS.md#ciede2000) | FR | unrecorded | unchecked | — |
| [`ckdn`](docs/MODEL_CARDS.md#ckdn) | FR/NN | researchmm/CKDN | prior | — |
| [`cvvdp`](docs/MODEL_CARDS.md#cvvdp) | FR | pycvvdp v0.5.7 (PyPI) | prior | — |
| [`cvvdp-gpu`](docs/MODEL_CARDS.md#cvvdp-gpu) | FR | pycvvdp v0.5.7 + CPU twin | unchecked | — |
| [`cw-ssim`](docs/MODEL_CARDS.md#cw-ssim) | FR | unrecorded | unchecked | — |
| [`cw-ssim-pyiqa`](docs/MODEL_CARDS.md#cw-ssim-pyiqa) | FR | pyiqa 0.1.16 create_metric('c… | prior | — |
| [`deepdc`](docs/MODEL_CARDS.md#deepdc) | FR/NN | h4nwei/DeepDC | prior | — |
| [`dists`](docs/MODEL_CARDS.md#dists) | FR/NN | dingkeyan93/DISTS | prior | — |
| [`dmm`](docs/MODEL_CARDS.md#dmm) | FR/NN | Baoliang93/DMM | partial | — |
| [`dmm-pyiqa`](docs/MODEL_CARDS.md#dmm-pyiqa) | FR/NN | pyiqa 0.1.16 create_metric('d… | prior | — |
| [`dreamsim`](docs/MODEL_CARDS.md#dreamsim) | FR/NN | ssundaram21/dreamsim | prior | — |
| [`dssim`](docs/MODEL_CARDS.md#dssim) | FR | dssim-core itself (the canoni… | prior | — |
| [`dssim-gpu`](docs/MODEL_CARDS.md#dssim-gpu) | FR | dssim-core | unchecked | — |
| [`flip`](docs/MODEL_CARDS.md#flip) | FR | unrecorded | unchecked | — |
| [`flip-v17`](docs/MODEL_CARDS.md#flip-v17) | FR | NVIDIA FLIP 1.7 | Δ 0 | ≈0.34× |
| [`fsim`](docs/MODEL_CARDS.md#fsim) | FR | authors' FR_FSIMc.m | prior | — |
| [`fsim-k2`](docs/MODEL_CARDS.md#fsim-k2) | FR | unrecorded | unchecked | — |
| [`fsim-k3`](docs/MODEL_CARDS.md#fsim-k3) | FR | unrecorded | unchecked | — |
| [`fsim-k4`](docs/MODEL_CARDS.md#fsim-k4) | FR | unrecorded | unchecked | — |
| [`fsim-nodownscale`](docs/MODEL_CARDS.md#fsim-nodownscale) | FR | unrecorded | unchecked | — |
| [`fsim-y`](docs/MODEL_CARDS.md#fsim-y) | FR | authors' FR_FSIMc.m | unchecked | — |
| [`gmsd`](docs/MODEL_CARDS.md#gmsd) | FR | libgmsd de646c9a (C) | prior | — |
| [`haarpsi`](docs/MODEL_CARDS.md#haarpsi) | FR | authors' haarpsi.m | prior | — |
| [`haarpsi-y`](docs/MODEL_CARDS.md#haarpsi-y) | FR | authors' haarpsi.m | unchecked | — |
| [`hdrvdp`](docs/MODEL_CARDS.md#hdrvdp) | FR | official HDR-VDP 2.2.2 MATLAB… | prior | — |
| [`hdrvdp-aic2026`](docs/MODEL_CARDS.md#hdrvdp-aic2026) | FR | unrecorded | unchecked | — |
| [`hdrvdp3`](docs/MODEL_CARDS.md#hdrvdp3) | FR | official HDR-VDP 3.0.7 MATLAB… | prior | — |
| [`hdrvdp3-aic2026`](docs/MODEL_CARDS.md#hdrvdp3-aic2026) | FR | unrecorded | unchecked | — |
| [`hyperiqa`](docs/MODEL_CARDS.md#hyperiqa) | NR/NN | SSL92/hyperIQA | prior | — |
| [`iwssim`](docs/MODEL_CARDS.md#iwssim) | FR | Python-IW-SSIM f9de37c (the P… | unchecked | — |
| [`iwssim-gpu`](docs/MODEL_CARDS.md#iwssim-gpu) | FR | CPU twin | unchecked | — |
| [`iwssim-piq`](docs/MODEL_CARDS.md#iwssim-piq) | FR | jpeg-ai-qaf IW_SSIM (torch po… | prior | — |
| [`liqe`](docs/MODEL_CARDS.md#liqe) | NR/NN | zwx8981/LIQE | prior | — |
| [`lpips-alex`](docs/MODEL_CARDS.md#lpips-alex) | FR/NN | richzhang/PerceptualSimilarity | prior | — |
| [`lpips-squeeze`](docs/MODEL_CARDS.md#lpips-squeeze) | FR/NN | richzhang/PerceptualSimilarity | prior | — |
| [`lpips-vgg`](docs/MODEL_CARDS.md#lpips-vgg) | FR/NN | richzhang/PerceptualSimilarity | prior | — |
| [`mad`](docs/MODEL_CARDS.md#mad) | FR | official MATLAB release (lars… | unchecked | — |
| [`maniqa`](docs/MODEL_CARDS.md#maniqa) | NR/NN | IIGROUP/MANIQA | prior | — |
| [`mdctpsnr`](docs/MODEL_CARDS.md#mdctpsnr) | FR | author's official C++ thorfdb… | prior | — |
| [`mdsi`](docs/MODEL_CARDS.md#mdsi) | FR | unrecorded | unchecked | — |
| [`ms-gmsd`](docs/MODEL_CARDS.md#ms-gmsd) | FR | unrecorded | unchecked | — |
| [`ms-gmsdc`](docs/MODEL_CARDS.md#ms-gmsdc) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa`](docs/MODEL_CARDS.md#ms-ssim-pyiqa) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k2) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k3) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k4) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k5) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k6) | FR | unrecorded | unchecked | — |
| [`msssim`](docs/MODEL_CARDS.md#msssim) | FR | authors' ssim_mscale_new.m | prior | — |
| [`msssim-libvmaf`](docs/MODEL_CARDS.md#msssim-libvmaf) | FR | libvmaf float_ms_ssim @ f85a8… | prior | — |
| [`musiq`](docs/MODEL_CARDS.md#musiq) | NR/NN | google-research/google-research | prior | — |
| [`niqe`](docs/MODEL_CARDS.md#niqe) | NR | BasicSR 1.4.2 | pass | ≈0.18× |
| [`nlpd`](docs/MODEL_CARDS.md#nlpd) | FR | Valerolaparra/NLPD_Pytorch (t… | prior | — |
| [`nlpd-iqa`](docs/MODEL_CARDS.md#nlpd-iqa) | FR | IQA_pytorch source mirrored a… | prior | — |
| [`paq2piq`](docs/MODEL_CARDS.md#paq2piq) | NR/NN | baidut/paq2piq | prior | — |
| [`pieapp`](docs/MODEL_CARDS.md#pieapp) | FR/NN | Official v0.1 public compute_… | prior | — |
| [`pieapp-pyiqa`](docs/MODEL_CARDS.md#pieapp-pyiqa) | FR/NN | pyiqa 0.1.16 create_metric('p… | prior | — |
| [`piqe`](docs/MODEL_CARDS.md#piqe) | NR | pypiqe 1.2 | pass | ≈0.1× |
| [`proposal-butteraugli`](docs/MODEL_CARDS.md#proposal-butteraugli) | FR | unrecorded | unchecked | — |
| [`psnr`](docs/MODEL_CARDS.md#psnr) | FR | unrecorded | unchecked | — |
| [`psnr-cb`](docs/MODEL_CARDS.md#psnr-cb) | FR | unrecorded | unchecked | — |
| [`psnr-cr`](docs/MODEL_CARDS.md#psnr-cr) | FR | unrecorded | unchecked | — |
| [`psnr-y`](docs/MODEL_CARDS.md#psnr-y) | FR | unrecorded | unchecked | — |
| [`psnr-y-libvmaf`](docs/MODEL_CARDS.md#psnr-y-libvmaf) | FR | unrecorded | unchecked | — |
| [`psnr-y-studio601`](docs/MODEL_CARDS.md#psnr-y-studio601) | FR | unrecorded | unchecked | — |
| [`psnr-y601`](docs/MODEL_CARDS.md#psnr-y601) | FR | unrecorded | unchecked | — |
| [`psnr-ycbcr611`](docs/MODEL_CARDS.md#psnr-ycbcr611) | FR | unrecorded | unchecked | — |
| [`psnrhvs`](docs/MODEL_CARDS.md#psnrhvs) | FR | authors' psnrhvsm.m | prior | — |
| [`psnrhvs-daala`](docs/MODEL_CARDS.md#psnrhvs-daala) | FR | libvmaf psnr_hvs feature @ f8… | prior | — |
| [`psnrhvs-y`](docs/MODEL_CARDS.md#psnrhvs-y) | FR | authors' psnrhvsm.m | unchecked | — |
| [`ssim`](docs/MODEL_CARDS.md#ssim) | FR | unrecorded | unchecked | — |
| [`ssim-libvmaf`](docs/MODEL_CARDS.md#ssim-libvmaf) | FR | libvmaf float_ssim @ f85a8536… | prior | — |
| [`ssim-pyiqa`](docs/MODEL_CARDS.md#ssim-pyiqa) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ssim-pyiqa-f32) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k2) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k3) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k4) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k5) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k6) | FR | unrecorded | unchecked | — |
| [`ssim2`](docs/MODEL_CARDS.md#ssim2) | FR | primary SSIMULACRA2 v2.1 | Δ 0 | ≈0.58× |
| [`ssim2-gpu`](docs/MODEL_CARDS.md#ssim2-gpu) | FR | fast-ssim2 (CPU) + third-part… | unchecked | — |
| [`ssimc`](docs/MODEL_CARDS.md#ssimc) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k2) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k3) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k4) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k5) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k6) | FR | unrecorded | unchecked | — |
| [`ssimulacra1`](docs/MODEL_CARDS.md#ssimulacra1) | FR | original SSIMULACRA1 / OpenCV | Δ 3.8e-09 | — |
| [`stlpips-alex`](docs/MODEL_CARDS.md#stlpips-alex) | FR/NN | abhijay9/ShiftTolerant-LPIPS | prior | — |
| [`stlpips-vgg`](docs/MODEL_CARDS.md#stlpips-vgg) | FR/NN | abhijay9/ShiftTolerant-LPIPS | prior | — |
| [`topiq-fr`](docs/MODEL_CARDS.md#topiq-fr) | FR/NN | chaofengc/IQA-PyTorch | gate | — |
| [`topiq-fr-pipal`](docs/MODEL_CARDS.md#topiq-fr-pipal) | FR/NN | chaofengc/IQA-PyTorch | gate | — |
| [`vif`](docs/MODEL_CARDS.md#vif) | FR | authors' vifp_mscale.m (MATLAB) | prior | — |
| [`vifvec`](docs/MODEL_CARDS.md#vifvec) | FR | authors' vifvec.m + matlabPyr… | prior | — |
| [`vmaf`](docs/MODEL_CARDS.md#vmaf) | FR | libvmaf 3.2.1 via FFI oracle | prior | — |
| [`vmaf-4k`](docs/MODEL_CARDS.md#vmaf-4k) | FR | unrecorded | unchecked | — |
| [`vmaf-neg`](docs/MODEL_CARDS.md#vmaf-neg) | FR | unrecorded | unchecked | — |
| [`vmaf-neg-studio601`](docs/MODEL_CARDS.md#vmaf-neg-studio601) | FR | unrecorded | unchecked | — |
| [`vmaf-studio601`](docs/MODEL_CARDS.md#vmaf-studio601) | FR | unrecorded | unchecked | — |
| [`vmaf-v1`](docs/MODEL_CARDS.md#vmaf-v1) | FR | unrecorded | unchecked | — |
| [`vsi`](docs/MODEL_CARDS.md#vsi) | FR | authors' VSI.m | prior | — |
| [`vsi-k2`](docs/MODEL_CARDS.md#vsi-k2) | FR | unrecorded | unchecked | — |
| [`vsi-k3`](docs/MODEL_CARDS.md#vsi-k3) | FR | unrecorded | unchecked | — |
| [`vsi-k4`](docs/MODEL_CARDS.md#vsi-k4) | FR | unrecorded | unchecked | — |
| [`vsi-nodownscale`](docs/MODEL_CARDS.md#vsi-nodownscale) | FR | unrecorded | unchecked | — |
| [`wadiqam-fr-live`](docs/MODEL_CARDS.md#wadiqam-fr-live) | FR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-fr-tid`](docs/MODEL_CARDS.md#wadiqam-fr-tid) | FR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-nr-live`](docs/MODEL_CARDS.md#wadiqam-nr-live) | NR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-nr-tid`](docs/MODEL_CARDS.md#wadiqam-nr-tid) | NR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-pyiqa`](docs/MODEL_CARDS.md#wadiqam-pyiqa) | FR/NN | pyiqa 0.1.16 create_metric('w… | prior | — |
| [`wd`](docs/MODEL_CARDS.md#wd) | FR/NN | Orange-OpenSource/Cool-Chic | prior | — |
| [`ws-hv`](docs/MODEL_CARDS.md#ws-hv) | FR | unrecorded | unchecked | — |
| [`zensim`](docs/MODEL_CARDS.md#zensim) | FR | unrecorded | unchecked | — |
| [`zensim-gpu`](docs/MODEL_CARDS.md#zensim-gpu) | FR | unrecorded | unchecked | — |

## Datasets

Datasets this repo validates or benchmarks against, with thanks to their
authors:

- **AIC-HDR2025**: Mohsen Jenadeleh, Jon Sneyers, Davi Lazzarotto, Shima
  Mohammadi, Dominik Keller, Atanas Boev, Rakesh Rao Ramachandra Rao, António
  Pinheiro. "Fine-Grained HDR Image Quality Assessment From Noticeably Distorted
  to Very High Fidelity", QoMEX 2025
  ([doi](https://doi.org/10.1109/QoMEX65720.2025.11219975),
  [arXiv:2506.12505](https://arxiv.org/abs/2506.12505)).
- **JPEG AIC2026**: Mohsen Jenadeleh, Jon Sneyers, João Ascenso, Thomas Richter,
  Alexander Karabutov, Panqi Jia, Elena Alshina, Osamu Watanabe, António
  Pinheiro, Touradj Ebrahimi, Dietmar Saupe. "JPEG AIC2026: A Large-Scale
  Dataset for Fine-Grained Assessment of Image Coding", 2026
  ([arXiv:2607.22783](https://arxiv.org/abs/2607.22783); data
  [doi:10.18419/DARUS-6156](https://doi.org/10.18419/DARUS-6156), CC BY-SA 4.0).
  Its published metric scores are what the AIC2026 variants above reproduce;
  [docs/AIC2026_METRICS_AND_FITTING.md](docs/AIC2026_METRICS_AND_FITTING.md)
  has the per-column results.
- **JPEG AIC-3**: Michela Testolina, Mohsen Jenadeleh, Shima Mohammadi, Shaolin
  Su, João Ascenso, Touradj Ebrahimi, Jon Sneyers, Dietmar Saupe. "Fine-Grained
  Subjective Visual Quality Assessment for High-Fidelity Compressed Images",
  DCC 2025 ([doi](https://doi.org/10.1109/DCC62719.2025.00020),
  [arXiv:2410.09501](https://arxiv.org/abs/2410.09501)).
- **CID22**: Jon Sneyers, Elad Ben Baruch, Yaron Vaxman (Cloudinary), JPEG AIC-3
  contribution.
- **KADID-10k**: Hanhe Lin, Vlad Hosu, Dietmar Saupe. QoMEX 2019
  ([database](https://database.mmsp-kn.de/kadid-10k-database.html)).
- **KADIS-700k**: Hanhe Lin, Vlad Hosu, Dietmar Saupe. "DeepFL-IQA: Weak
  Supervision for Deep IQA Feature Learning", 2020
  ([arXiv:2001.08113](https://arxiv.org/abs/2001.08113)).
- **KonJND-1k**: Hanhe Lin, Guangan Chen, Mohsen Jenadeleh, Vlad Hosu,
  Ulf-Dietrich Reips, Raouf Hamzaoui, Dietmar Saupe. "Large-scale crowdsourced
  subjective assessment of picturewise just noticeable difference", IEEE TCSVT
  32(9), 2022 ([doi](https://doi.org/10.1109/TCSVT.2022.3163860)).
- **TID2013**: Nikolay Ponomarenko, Lina Jin, Oleg Ieremeiev, Vladimir Lukin,
  Karen Egiazarian, Jaakko Astola. Signal Processing: Image Communication 30,
  2015 ([doi](https://doi.org/10.1016/j.image.2014.10.009)).
- **UPIQ**: Aliaksei Mikhailiuk, María Pérez-Ortiz, Dingcheng Yue, Wilson Suen,
  Rafał K. Mantiuk. "Consolidated Dataset and Metrics for High-Dynamic-Range
  Image Quality", 2021 ([arXiv:2012.10758](https://arxiv.org/abs/2012.10758)).

The JPEG AIC test methodology we follow is the group's Common Test Conditions
on objective quality assessment: WG1 N101156 (Dietmar Saupe, Jon Sneyers, Shima
Mohammadi, João Ascenso) and v2.0, WG1 N101246 (Dietmar Saupe, Mohsen Jenadeleh,
Jon Sneyers). For how established metrics behave in this high-fidelity range,
see Shima Mohammadi, Mohsen Jenadeleh, Jon Sneyers, Dietmar Saupe, João Ascenso,
"Evaluation of Objective Image Quality Metrics for High-Fidelity Image
Compression", IEEE Access, 2026
([doi](https://doi.org/10.1109/ACCESS.2026.3669417),
[arXiv:2509.13150](https://arxiv.org/abs/2509.13150)).

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

- Source builds require the pinned sibling repos (see Quick start).
  [Checkpoint packages](docs/CHECKPOINT_BINARIES.md) include target-specific
  executables; GPU drivers and platform runtime dependencies remain required.
  There are no crates.io releases yet.
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
- Most metrics proposed to JPEG AIC-4 are not reproduced here yet: DVIFM, the
  Distortion Visibility Image Fidelity Measure by Maciej Pedzisz (Nokia); IDFIQA
  by Amirreza Khakpour, Sina Yademellat and Azadeh Mansouri, which needs a
  pretrained CNN; and the proposal variant of Butteraugli. The
  `proposal-mDCTPSNR` column is mDCT-PSNR in JND units, from the reference
  tool's `-jnd` option, which the `mdctpsnr` crate reproduces as
  `mdct_psnr_jnd`.
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
- [docs/MULTIDIMENSIONAL_BENCHMARKING.md](docs/MULTIDIMENSIONAL_BENCHMARKING.md):
  proposed metric cost studies across image size, hardware and OS.
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
