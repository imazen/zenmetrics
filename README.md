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

[Models & metrics explorer](https://imazen.github.io/zenmetrics/#models) · [Full cards and BibTeX](docs/MODEL_CARDS.md)

Source-linked records for each CLI metric; rows group variants only when the displayed paper, reference, terms and parity status match. Reported parity is scoped to its oracle and fixtures; a declared gate is not a measured result. Missing bibliographic fields are labelled **citation needed**. Code and checkpoint terms are separate.

| Metric | Family | Paper | Reference implementation | Licence | Parity status |
| --- | --- | --- | --- | --- | --- |
| [`afine`](docs/MODEL_CARDS.md#afine) | FR · Learned | [Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption](https://arxiv.org/abs/2503.11221) | https://github.com/ChrisDud0257/AFINE | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: noncommercial | Reported results |
| [`ahiq`](docs/MODEL_CARDS.md#ahiq) | FR · Learned | citation needed | no evidence | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`arniqa`](docs/MODEL_CARDS.md#arniqa) | NR · Learned | [ARNIQA: Learning Distortion Manifold for Image Quality Assessment](https://arxiv.org/abs/2310.14918) | https://github.com/miccunifi/ARNIQA | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`brisque`](docs/MODEL_CARDS.md#brisque) | NR · Learned | [No-Reference Image Quality Assessment in the Spatial Domain](https://doi.org/10.1109/TIP.2012.2214050) | no evidence | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not recorded | Not documented |
| [`butteraugli`](docs/MODEL_CARDS.md#butteraugli) | FR · Classical | citation needed | libjxl butteraugli v0.9.2 / 0.9.4 | Rust: no evidence; weights: not applicable | Not documented |
| [`butteraugli-gpu`](docs/MODEL_CARDS.md#butteraugli-gpu) | FR · Classical | citation needed | libjxl butteraugli v0.9.2 / 0.9.4 | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`butteraugli-original`](docs/MODEL_CARDS.md#butteraugli-original) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`cambi`](docs/MODEL_CARDS.md#cambi) | NR · Classical | [CAMBI: Contrast-aware Multiscale Banding Index](https://arxiv.org/abs/2102.00079) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`ciede2000`](docs/MODEL_CARDS.md#ciede2000) | FR · Classical | [The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations](https://doi.org/10.1002/col.20070) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`ckdn`](docs/MODEL_CARDS.md#ckdn) | FR · Learned | [Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment](https://arxiv.org/abs/2108.07948) | https://github.com/researchmm/CKDN | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: unknown | Reported results |
| [`cvvdp`](docs/MODEL_CARDS.md#cvvdp) | FR · Classical | [ColorVideoVDP: A visual difference predictor for image, video and display distortions](https://doi.org/10.1145/3658144) | pycvvdp v0.5.7 (PyPI) | Rust: AGPL-3.0-or-later; weights: not applicable | Reported results |
| [`cvvdp-gpu`](docs/MODEL_CARDS.md#cvvdp-gpu) | FR · Classical | [ColorVideoVDP: A visual difference predictor for image, video and display distortions](https://doi.org/10.1145/3658144) | pycvvdp v0.5.7 + CPU twin | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`cw-ssim`](docs/MODEL_CARDS.md#cw-ssim) | FR · Classical | [Complex Wavelet Structural Similarity: A New Image Similarity Index](https://doi.org/10.1109/TIP.2009.2025923) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`deepdc`](docs/MODEL_CARDS.md#deepdc) | FR · Learned | [DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator](https://arxiv.org/abs/2211.04927) | https://github.com/h4nwei/DeepDC | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`dists`](docs/MODEL_CARDS.md#dists) | FR · Learned | [Image Quality Assessment: Unifying Structure and Texture Similarity](https://arxiv.org/abs/2004.07728) | https://github.com/dingkeyan93/DISTS | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`dmm`](docs/MODEL_CARDS.md#dmm) | FR · Learned | citation needed | no evidence | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`dreamsim`](docs/MODEL_CARDS.md#dreamsim) | FR · Learned | [DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data](https://arxiv.org/abs/2306.09344) | https://github.com/ssundaram21/dreamsim | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: unknown | Reported results |
| [`dssim`](docs/MODEL_CARDS.md#dssim) | FR · Classical | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) | dssim-core itself (the canonical impl) | Rust: no evidence; weights: not applicable | Reported results |
| [`dssim-gpu`](docs/MODEL_CARDS.md#dssim-gpu) | FR · Classical | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) | dssim-core | Rust: AGPL-3.0-only OR LicenseRef-Imazen-DSSIM-Commercial; weights: not applicable | Not documented |
| [`flip`](docs/MODEL_CARDS.md#flip), [`flip-v17`](docs/MODEL_CARDS.md#flip-v17) | FR · Classical | [FLIP: A Difference Evaluator for Alternating Images](https://doi.org/10.1145/3406183) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`fsim`](docs/MODEL_CARDS.md#fsim) | FR · Classical | [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730) | authors' FR_FSIMc.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`fsim-k2`](docs/MODEL_CARDS.md#fsim-k2), [`fsim-k3`](docs/MODEL_CARDS.md#fsim-k3), [`fsim-k4`](docs/MODEL_CARDS.md#fsim-k4), [`fsim-nodownscale`](docs/MODEL_CARDS.md#fsim-nodownscale) | FR · Classical | [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`fsim-y`](docs/MODEL_CARDS.md#fsim-y) | FR · Classical | [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730) | authors' FR_FSIMc.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`gmsd`](docs/MODEL_CARDS.md#gmsd) | FR · Classical | [Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index](https://doi.org/10.1109/TIP.2013.2293423) | libgmsd de646c9a (C) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`haarpsi`](docs/MODEL_CARDS.md#haarpsi) | FR · Classical | [A Haar wavelet-based perceptual similarity index for image quality assessment](https://doi.org/10.1016/j.image.2017.11.001) | authors' haarpsi.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`haarpsi-y`](docs/MODEL_CARDS.md#haarpsi-y) | FR · Classical | [A Haar wavelet-based perceptual similarity index for image quality assessment](https://doi.org/10.1016/j.image.2017.11.001) | authors' haarpsi.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`hdrvdp`](docs/MODEL_CARDS.md#hdrvdp) | FR · Classical | [HDR-VDP-2: a calibrated visual metric for visibility and quality predictions in all luminance conditions](https://doi.org/10.1145/1964921.1964935) | official HDR-VDP 2.2.2 MATLAB release (ISC-licensed) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`hdrvdp3`](docs/MODEL_CARDS.md#hdrvdp3) | FR · Classical | [HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content](https://arxiv.org/abs/2304.13625) | official HDR-VDP 3.0.7 MATLAB release + jpeg-ai-qaf VDP3/ numpy port @ 0628a6b (verified equal — see DIVERGENCES) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`hyperiqa`](docs/MODEL_CARDS.md#hyperiqa) | NR · Learned | [Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network](https://doi.org/10.1109/CVPR42600.2020.00372) | https://github.com/SSL92/hyperIQA | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: unknown | Reported results |
| [`iwssim`](docs/MODEL_CARDS.md#iwssim) | FR · Classical | [Information Content Weighting for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2010.2092435) | Python-IW-SSIM f9de37c (the Python port linked from the authors' IW-SSIM page; not yet compared directly with the authors' MATLAB release) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not compared |
| [`iwssim-gpu`](docs/MODEL_CARDS.md#iwssim-gpu) | FR · Classical | [Information Content Weighting for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2010.2092435) | CPU twin | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`iwssim-piq`](docs/MODEL_CARDS.md#iwssim-piq) | FR · Classical | [Information Content Weighting for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2010.2092435) | jpeg-ai-qaf IW_SSIM (torch port of the same Jack-guo ref) on the unrounded 0–255 Y — the AIC-4 IW-SSIM column; upstream mirror [local path] @ 0628a6b | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`liqe`](docs/MODEL_CARDS.md#liqe) | NR · Learned | [Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective](https://arxiv.org/abs/2303.14968) | https://github.com/zwx8981/LIQE | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: unknown | Reported results |
| [`lpips-alex`](docs/MODEL_CARDS.md#lpips-alex), [`lpips-squeeze`](docs/MODEL_CARDS.md#lpips-squeeze), [`lpips-vgg`](docs/MODEL_CARDS.md#lpips-vgg) | FR · Learned | [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) | https://github.com/richzhang/PerceptualSimilarity | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`mad`](docs/MODEL_CARDS.md#mad) | FR · Classical | [Most apparent distortion: full-reference image quality assessment and the role of strategy](https://doi.org/10.1117/1.3267105) | official MATLAB release (larschandler.com) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`maniqa`](docs/MODEL_CARDS.md#maniqa) | NR · Learned | [MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](https://arxiv.org/abs/2204.08958) | https://github.com/IIGROUP/MANIQA | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`mdctpsnr`](docs/MODEL_CARDS.md#mdctpsnr) | FR · Classical | [On the mDCT-PSNR image quality index](https://doi.org/10.1109/QOMEX.2009.5246978) | author's official C++ thorfdbg/mDCTpsnr (Thomas Richter; contributor Jon Sneyers) (zlib-style license), built GCC -O3 -ffast-math AVX2 + glibc libmvec | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`mdsi`](docs/MODEL_CARDS.md#mdsi) | FR · Classical | [Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image Quality Evaluator](https://doi.org/10.1109/ACCESS.2016.2604042) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`ms-gmsd`](docs/MODEL_CARDS.md#ms-gmsd), [`ms-gmsdc`](docs/MODEL_CARDS.md#ms-gmsdc) | FR · Classical | [Gradient magnitude similarity deviation on multiple scales for color image quality assessment](https://doi.org/10.1109/ICASSP.2017.7952357) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`ms-ssim-pyiqa`](docs/MODEL_CARDS.md#ms-ssim-pyiqa), [`ms-ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32), [`ms-ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k2), [`ms-ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k3), [`ms-ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k4), [`ms-ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k5), [`ms-ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k6) | FR · Classical | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`msssim`](docs/MODEL_CARDS.md#msssim) | FR · Classical | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) | authors' ssim_mscale_new.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`msssim-libvmaf`](docs/MODEL_CARDS.md#msssim-libvmaf) | FR · Classical | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) | libvmaf float_ms_ssim @ f85a8536 — FFI oracle | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`musiq`](docs/MODEL_CARDS.md#musiq) | NR · Learned | [MUSIQ: Multi-scale Image Quality Transformer](https://arxiv.org/abs/2108.05997) | https://github.com/google-research/google-research/tree/master/musiq | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`niqe`](docs/MODEL_CARDS.md#niqe) | NR · Classical | [Making a “Completely Blind” Image Quality Analyzer](https://doi.org/10.1109/LSP.2012.2227726) | no evidence | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not recorded | Not documented |
| [`nlpd`](docs/MODEL_CARDS.md#nlpd) | FR · Classical | [Perceptual image quality assessment using a normalized Laplacian pyramid](https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103) | Valerolaparra/NLPD_Pytorch (the authors' PyTorch reference; PyTorch 2.5.1 oracle) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`nlpd-iqa`](docs/MODEL_CARDS.md#nlpd-iqa) | FR · Classical | [Perceptual image quality assessment using a normalized Laplacian pyramid](https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103) | IQA_pytorch source mirrored at [local path] semantics replicated to med \|Δ\| 0.0026 / max 0.0067 vs the AIC-4 NLPD column (n=53) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`paq2piq`](docs/MODEL_CARDS.md#paq2piq) | NR · Learned | [From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality](https://arxiv.org/abs/1912.10088) | https://github.com/baidut/paq2piq | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: noncommercial | Reported results |
| [`pieapp`](docs/MODEL_CARDS.md#pieapp) | FR · Learned | [PieAPP: Perceptual Image-Error Assessment through Pairwise Preference](https://arxiv.org/abs/1806.02067) | Official v0.1 public compute_score, release 0513613d1f575136ea52418e35f2cebc5a49f783 | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: Modified BSD-2 for Non-Commercial Use Only, UC Regents | Reported results |
| [`piqe`](docs/MODEL_CARDS.md#piqe) | NR · Classical | [Blind image quality evaluation using perception based features](https://doi.org/10.1109/NCC.2015.7084843) | no evidence | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`psnr`](docs/MODEL_CARDS.md#psnr) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-cb`](docs/MODEL_CARDS.md#psnr-cb) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-cr`](docs/MODEL_CARDS.md#psnr-cr) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-y`](docs/MODEL_CARDS.md#psnr-y) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-y-libvmaf`](docs/MODEL_CARDS.md#psnr-y-libvmaf) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-y-studio601`](docs/MODEL_CARDS.md#psnr-y-studio601) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-y601`](docs/MODEL_CARDS.md#psnr-y601) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnr-ycbcr611`](docs/MODEL_CARDS.md#psnr-ycbcr611) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`psnrhvs`](docs/MODEL_CARDS.md#psnrhvs) | FR · Classical | citation needed | authors' psnrhvsm.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`psnrhvs-daala`](docs/MODEL_CARDS.md#psnrhvs-daala) | FR · Classical | citation needed | libvmaf psnr_hvs feature @ f85a8536 — FFI oracle | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`psnrhvs-y`](docs/MODEL_CARDS.md#psnrhvs-y) | FR · Classical | citation needed | authors' psnrhvsm.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`ssim`](docs/MODEL_CARDS.md#ssim), [`ssim-pyiqa`](docs/MODEL_CARDS.md#ssim-pyiqa), [`ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ssim-pyiqa-f32), [`ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k2), [`ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k3), [`ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k4), [`ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k5), [`ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k6), [`ssimc`](docs/MODEL_CARDS.md#ssimc), [`ssimc-pyiqa-f32`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32), [`ssimc-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k2), [`ssimc-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k3), [`ssimc-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k4), [`ssimc-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k5), [`ssimc-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k6) | FR · Classical | [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`ssim-libvmaf`](docs/MODEL_CARDS.md#ssim-libvmaf) | FR · Classical | [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861) | libvmaf float_ssim @ f85a8536 — FFI oracle (vmaf-head-sys dev-dep) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`ssim2`](docs/MODEL_CARDS.md#ssim2) | FR · Classical | citation needed | Cloudinary ssimulacra2 C++ (v2 lineage) | Rust: no evidence; weights: not applicable | Reported results |
| [`ssim2-gpu`](docs/MODEL_CARDS.md#ssim2-gpu) | FR · Classical | citation needed | fast-ssim2 (CPU) + third-party crates.io ssimulacra2 for parity | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |
| [`ssimulacra1`](docs/MODEL_CARDS.md#ssimulacra1) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`stlpips-alex`](docs/MODEL_CARDS.md#stlpips-alex), [`stlpips-vgg`](docs/MODEL_CARDS.md#stlpips-vgg) | FR · Learned | [Shift-tolerant Perceptual Similarity Metric](https://arxiv.org/abs/2207.13686) | https://github.com/abhijay9/ShiftTolerant-LPIPS | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`vif`](docs/MODEL_CARDS.md#vif) | FR · Classical | [Image information and visual quality](https://doi.org/10.1109/TIP.2005.859378) | authors' vifp_mscale.m (MATLAB) | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`vifvec`](docs/MODEL_CARDS.md#vifvec) | FR · Classical | [Image information and visual quality](https://doi.org/10.1109/TIP.2005.859378) | authors' vifvec.m + matlabPyrTools sp5Filters/buildSpyr/corrDn/vifsub_est_M/ind2wtree — the implementation behind pyiqa/IQA_pytorch VIFs and the AIC-4 VIF column | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`vmaf`](docs/MODEL_CARDS.md#vmaf) | FR · Classical | citation needed | libvmaf 3.2.1 via FFI oracle | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`vmaf-4k`](docs/MODEL_CARDS.md#vmaf-4k) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`vmaf-neg`](docs/MODEL_CARDS.md#vmaf-neg) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`vmaf-neg-studio601`](docs/MODEL_CARDS.md#vmaf-neg-studio601) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`vmaf-studio601`](docs/MODEL_CARDS.md#vmaf-studio601) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`vmaf-v1`](docs/MODEL_CARDS.md#vmaf-v1) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`vsi`](docs/MODEL_CARDS.md#vsi) | FR · Classical | [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028) | authors' VSI.m | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Reported results |
| [`vsi-k2`](docs/MODEL_CARDS.md#vsi-k2), [`vsi-k3`](docs/MODEL_CARDS.md#vsi-k3), [`vsi-k4`](docs/MODEL_CARDS.md#vsi-k4), [`vsi-nodownscale`](docs/MODEL_CARDS.md#vsi-nodownscale) | FR · Classical | [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028) | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`wadiqam-fr-live`](docs/MODEL_CARDS.md#wadiqam-fr-live), [`wadiqam-fr-tid`](docs/MODEL_CARDS.md#wadiqam-fr-tid) | FR · Learned | [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697) | https://github.com/dmaniry/deepIQA | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`wadiqam-nr-live`](docs/MODEL_CARDS.md#wadiqam-nr-live), [`wadiqam-nr-tid`](docs/MODEL_CARDS.md#wadiqam-nr-tid) | NR · Learned | [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697) | https://github.com/dmaniry/deepIQA | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`wd`](docs/MODEL_CARDS.md#wd) | FR · Learned | [Wasserstein Distortion: Unifying Fidelity and Realism](https://arxiv.org/abs/2310.03629) | https://github.com/Orange-OpenSource/Cool-Chic/blob/main/coolchic/training/metrics/wasserstein.py | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: permissive | Reported results |
| [`ws-hv`](docs/MODEL_CARDS.md#ws-hv) | FR · Classical | citation needed | no evidence | Rust: no evidence; weights: not applicable | Not documented |
| [`zensim`](docs/MODEL_CARDS.md#zensim) | FR · Classical | citation needed | — | Rust: no evidence; weights: not applicable | Not documented |
| [`zensim-gpu`](docs/MODEL_CARDS.md#zensim-gpu) | FR · Classical | citation needed | — | Rust: AGPL-3.0-only OR LicenseRef-Imazen-Commercial; weights: not applicable | Not documented |

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
