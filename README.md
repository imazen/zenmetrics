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

**Rust/C** is Rust time divided by canonical time (lower is faster). Approximate local comparisons are descriptive, with their input size, thread budget and source pins in the [comparison ledger](benchmarks/canonical_recheck_2026-10-19/README.md). A dash means no comparable measurement. “Prior” means older recorded evidence; “gate” means a declared bound, not a measured pass. ! flags an observed ratio above 1.08; see the ledger for confirmation status.

| Metric | Type | Canonical | Check | Rust/C |
| --- | --- | --- | --- | ---: |
| [`afine`](docs/MODEL_CARDS.md#afine) | FR/NN | ChrisDud0257/AFINE | prior | — |
| [`ahiq`](docs/MODEL_CARDS.md#ahiq) | FR/NN | IIGROUP/AHIQ | prior | — |
| [`ahiq-pyiqa`](docs/MODEL_CARDS.md#ahiq-pyiqa) | FR/NN | pyiqa 0.1.16 create_m… | prior | — |
| [`arniqa`](docs/MODEL_CARDS.md#arniqa) | NR/NN | miccunifi/ARNIQA | prior | — |
| [`brisque`](docs/MODEL_CARDS.md#brisque) | NR | PIQ 0.8.0 | 170/170 pass | ≈0.12× |
| [`butteraugli`](docs/MODEL_CARDS.md#butteraugli) | FR | libjxl butteraugli v0… | unchecked | — |
| [`butteraugli-gpu`](docs/MODEL_CARDS.md#butteraugli-gpu) | FR | libjxl butteraugli v0… | unchecked | — |
| [`butteraugli-original`](docs/MODEL_CARDS.md#butteraugli-original) | FR | unrecorded | unchecked | — |
| [`cambi`](docs/MODEL_CARDS.md#cambi) | NR | libvmaf 3.2.0 / named… | Δ 0 | ≈1× |
| [`cambi-aic2026`](docs/MODEL_CARDS.md#cambi-aic2026) | NR | libvmaf 3.2.0 / named… | Δ 0 | ≈0.68× |
| [`ciede2000`](docs/MODEL_CARDS.md#ciede2000) | FR | unrecorded | unchecked | — |
| [`ckdn`](docs/MODEL_CARDS.md#ckdn) | FR/NN | researchmm/CKDN | prior | — |
| [`cvvdp`](docs/MODEL_CARDS.md#cvvdp) | FR | pycvvdp v0.5.7 (PyPI) | prior | — |
| [`cvvdp-gpu`](docs/MODEL_CARDS.md#cvvdp-gpu) | FR | pycvvdp v0.5.7 + CPU… | unchecked | — |
| [`cw-ssim`](docs/MODEL_CARDS.md#cw-ssim) | FR | unrecorded | unchecked | — |
| [`cw-ssim-pyiqa`](docs/MODEL_CARDS.md#cw-ssim-pyiqa) | FR | pyiqa 0.1.16 create_m… | prior | — |
| [`deepdc`](docs/MODEL_CARDS.md#deepdc) | FR/NN | h4nwei/DeepDC | prior | — |
| [`dists`](docs/MODEL_CARDS.md#dists) | FR/NN | dingkeyan93/DISTS | prior | — |
| [`dmm`](docs/MODEL_CARDS.md#dmm) | FR/NN | Baoliang93/DMM | partial | — |
| [`dmm-pyiqa`](docs/MODEL_CARDS.md#dmm-pyiqa) | FR/NN | pyiqa 0.1.16 create_m… | prior | — |
| [`dreamsim`](docs/MODEL_CARDS.md#dreamsim) | FR/NN | ssundaram21/dreamsim | prior | — |
| [`dssim`](docs/MODEL_CARDS.md#dssim) | FR | original dssim-core e… | 1 process errors | — |
| [`dssim-gpu`](docs/MODEL_CARDS.md#dssim-gpu) | FR | dssim-core | unchecked | — |
| [`flip`](docs/MODEL_CARDS.md#flip) | FR | NVIDIA FLIP 1.7 / f64… | Δ 3.6e-14 | ≈0.33× |
| [`flip-v17`](docs/MODEL_CARDS.md#flip-v17) | FR | NVIDIA FLIP 1.7 | Δ 0 | ≈0.34× |
| [`fsim`](docs/MODEL_CARDS.md#fsim) | FR | original FeatureSIM /… | 6/6 pass | ≈0.22× |
| [`fsim-k2`](docs/MODEL_CARDS.md#fsim-k2) | FR | original FeatureSIM /… | 6/6 pass | ≈0.18× |
| [`fsim-k3`](docs/MODEL_CARDS.md#fsim-k3) | FR | original FeatureSIM /… | 6/6 pass | ≈0.16× |
| [`fsim-k4`](docs/MODEL_CARDS.md#fsim-k4) | FR | original FeatureSIM /… | 6/6 pass | ≈0.14× |
| [`fsim-nodownscale`](docs/MODEL_CARDS.md#fsim-nodownscale) | FR | original FeatureSIM /… | 6/6 pass | ≈0.21× |
| [`fsim-y`](docs/MODEL_CARDS.md#fsim-y) | FR | authors' FR_FSIMc.m | unchecked | — |
| [`gmsd`](docs/MODEL_CARDS.md#gmsd) | FR | libgmsd de646c9a (C) | prior | — |
| [`haarpsi`](docs/MODEL_CARDS.md#haarpsi) | FR | original HaarPSI / Oc… | 6/6 pass | ≈0.12× |
| [`haarpsi-y`](docs/MODEL_CARDS.md#haarpsi-y) | FR | original HaarPSI / Oc… | 6/6 pass | ≈0.13× |
| [`hdrvdp`](docs/MODEL_CARDS.md#hdrvdp) | FR | official HDR-VDP 2.2.… | prior | — |
| [`hdrvdp-aic2026`](docs/MODEL_CARDS.md#hdrvdp-aic2026) | FR | unrecorded | unchecked | — |
| [`hdrvdp3`](docs/MODEL_CARDS.md#hdrvdp3) | FR | official HDR-VDP 3.0.… | prior | — |
| [`hdrvdp3-aic2026`](docs/MODEL_CARDS.md#hdrvdp3-aic2026) | FR | unrecorded | unchecked | — |
| [`hyperiqa`](docs/MODEL_CARDS.md#hyperiqa) | NR/NN | SSL92/hyperIQA | prior | — |
| [`iwssim`](docs/MODEL_CARDS.md#iwssim) | FR | Python-IW-SSIM f9de37… | unchecked | — |
| [`iwssim-gpu`](docs/MODEL_CARDS.md#iwssim-gpu) | FR | CPU twin | unchecked | — |
| [`iwssim-piq`](docs/MODEL_CARDS.md#iwssim-piq) | FR | jpeg-ai-qaf IW_SSIM (… | prior | — |
| [`liqe`](docs/MODEL_CARDS.md#liqe) | NR/NN | zwx8981/LIQE | prior | — |
| [`lpips-alex`](docs/MODEL_CARDS.md#lpips-alex) | FR/NN | richzhang/PerceptualS… | prior | — |
| [`lpips-squeeze`](docs/MODEL_CARDS.md#lpips-squeeze) | FR/NN | richzhang/PerceptualS… | prior | — |
| [`lpips-vgg`](docs/MODEL_CARDS.md#lpips-vgg) | FR/NN | richzhang/PerceptualS… | prior | — |
| [`mad`](docs/MODEL_CARDS.md#mad) | FR | official MATLAB relea… | unchecked | — |
| [`maniqa`](docs/MODEL_CARDS.md#maniqa) | NR/NN | IIGROUP/MANIQA | prior | — |
| [`mdctpsnr`](docs/MODEL_CARDS.md#mdctpsnr) | FR | author's official C++… | prior | — |
| [`mdsi`](docs/MODEL_CARDS.md#mdsi) | FR | original MDSI / Octav… | 6/6 pass | ≈0.21× |
| [`ms-gmsd`](docs/MODEL_CARDS.md#ms-gmsd) | FR | unrecorded | unchecked | — |
| [`ms-gmsdc`](docs/MODEL_CARDS.md#ms-gmsdc) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa`](docs/MODEL_CARDS.md#ms-ssim-pyiqa) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.014× |
| [`ms-ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k2) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k3) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k4) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k5) | FR | unrecorded | unchecked | — |
| [`ms-ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k6) | FR | unrecorded | unchecked | — |
| [`msssim`](docs/MODEL_CARDS.md#msssim) | FR | original msssim / Oct… | 6/6 pass | ≈0.028× |
| [`msssim-libvmaf`](docs/MODEL_CARDS.md#msssim-libvmaf) | FR | libvmaf 3.2.0 / named… | Δ 1.5e-07 | ≈0.35× |
| [`musiq`](docs/MODEL_CARDS.md#musiq) | NR/NN | google-research/googl… | prior | — |
| [`niqe`](docs/MODEL_CARDS.md#niqe) | NR | BasicSR 1.4.2 | 170/170 pass | ≈0.18× |
| [`nlpd`](docs/MODEL_CARDS.md#nlpd) | FR | Valerolaparra/NLPD_Py… | prior | — |
| [`nlpd-iqa`](docs/MODEL_CARDS.md#nlpd-iqa) | FR | IQA_pytorch source mi… | prior | — |
| [`paq2piq`](docs/MODEL_CARDS.md#paq2piq) | NR/NN | baidut/paq2piq | prior | — |
| [`pieapp`](docs/MODEL_CARDS.md#pieapp) | FR/NN | Official v0.1 public… | prior | — |
| [`pieapp-pyiqa`](docs/MODEL_CARDS.md#pieapp-pyiqa) | FR/NN | pyiqa 0.1.16 create_m… | prior | — |
| [`piqe`](docs/MODEL_CARDS.md#piqe) | NR | pypiqe 1.2 | 170/170 pass | ≈0.1× |
| [`proposal-butteraugli`](docs/MODEL_CARDS.md#proposal-butteraugli) | FR | unrecorded | unchecked | — |
| [`psnr`](docs/MODEL_CARDS.md#psnr) | FR | scikit-image 0.26 / d… | Δ 0 | ≈0.21× |
| [`psnr-cb`](docs/MODEL_CARDS.md#psnr-cb) | FR | unrecorded | unchecked | — |
| [`psnr-cr`](docs/MODEL_CARDS.md#psnr-cr) | FR | unrecorded | unchecked | — |
| [`psnr-y`](docs/MODEL_CARDS.md#psnr-y) | FR | scikit-image 0.26 / d… | Δ 2e-12 | ≈0.14× |
| [`psnr-y-libvmaf`](docs/MODEL_CARDS.md#psnr-y-libvmaf) | FR | libvmaf 3.2.0 / named… | Δ 0 | ≈0.76× |
| [`psnr-y-studio601`](docs/MODEL_CARDS.md#psnr-y-studio601) | FR | unrecorded | unchecked | — |
| [`psnr-y601`](docs/MODEL_CARDS.md#psnr-y601) | FR | scikit-image 0.26 / d… | Δ 2.4e-12 | ≈0.14× |
| [`psnr-ycbcr611`](docs/MODEL_CARDS.md#psnr-ycbcr611) | FR | unrecorded | unchecked | — |
| [`psnrhvs`](docs/MODEL_CARDS.md#psnrhvs) | FR | authors' psnrhvsm.m | prior | — |
| [`psnrhvs-daala`](docs/MODEL_CARDS.md#psnrhvs-daala) | FR | libvmaf psnr_hvs feat… | prior | — |
| [`psnrhvs-y`](docs/MODEL_CARDS.md#psnrhvs-y) | FR | authors' psnrhvsm.m | unchecked | — |
| [`ssim`](docs/MODEL_CARDS.md#ssim) | FR | unrecorded | unchecked | — |
| [`ssim-libvmaf`](docs/MODEL_CARDS.md#ssim-libvmaf) | FR | libvmaf 3.2.0 / named… | Δ 6e-08 | ≈0.36× |
| [`ssim-pyiqa`](docs/MODEL_CARDS.md#ssim-pyiqa) | FR | unrecorded | unchecked | — |
| [`ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ssim-pyiqa-f32) | FR | PyIQA 0.1.16 / named… | 2/6 pass | ≈0.0096× |
| [`ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k2) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.025× |
| [`ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k3) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.041× |
| [`ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k4) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.067× |
| [`ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k5) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.16× |
| [`ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k6) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.28× |
| [`ssim2`](docs/MODEL_CARDS.md#ssim2) | FR | primary SSIMULACRA2 v2.1 | Δ 0 | ≈0.58× |
| [`ssim2-gpu`](docs/MODEL_CARDS.md#ssim2-gpu) | FR | fast-ssim2 (CPU) + th… | unchecked | — |
| [`ssimc`](docs/MODEL_CARDS.md#ssimc) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32) | FR | PyIQA 0.1.16 / named… | 1/3 pass | ≈0.013× |
| [`ssimc-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k2) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k3) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k4) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k5) | FR | unrecorded | unchecked | — |
| [`ssimc-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k6) | FR | unrecorded | unchecked | — |
| [`ssimulacra1`](docs/MODEL_CARDS.md#ssimulacra1) | FR | original SSIMULACRA1… | Δ 3.8e-09 | ≈1.1× ! |
| [`stlpips-alex`](docs/MODEL_CARDS.md#stlpips-alex) | FR/NN | abhijay9/ShiftToleran… | prior | — |
| [`stlpips-vgg`](docs/MODEL_CARDS.md#stlpips-vgg) | FR/NN | abhijay9/ShiftToleran… | prior | — |
| [`topiq-fr`](docs/MODEL_CARDS.md#topiq-fr) | FR/NN | chaofengc/IQA-PyTorch | gate | — |
| [`topiq-fr-pipal`](docs/MODEL_CARDS.md#topiq-fr-pipal) | FR/NN | chaofengc/IQA-PyTorch | gate | — |
| [`vif`](docs/MODEL_CARDS.md#vif) | FR | original vifp_mscale… | 6/6 pass | ≈0.076× |
| [`vifvec`](docs/MODEL_CARDS.md#vifvec) | FR | original vifvec / Oct… | 6/6 pass | ≈0.096× |
| [`vmaf`](docs/MODEL_CARDS.md#vmaf) | FR | libvmaf 3.2.0 / named… | 2/2 pass | ≈0.89× |
| [`vmaf-4k`](docs/MODEL_CARDS.md#vmaf-4k) | FR | libvmaf 3.2.0 / named… | Δ 0 | ≈0.89× |
| [`vmaf-neg`](docs/MODEL_CARDS.md#vmaf-neg) | FR | libvmaf 3.2.0 / named… | 2/2 pass | ≈0.87× |
| [`vmaf-neg-studio601`](docs/MODEL_CARDS.md#vmaf-neg-studio601) | FR | libvmaf 3.2.0 / named… | 2/2 pass | ≈0.95× |
| [`vmaf-studio601`](docs/MODEL_CARDS.md#vmaf-studio601) | FR | libvmaf 3.2.0 / named… | 2/2 pass | ≈0.91× |
| [`vmaf-v1`](docs/MODEL_CARDS.md#vmaf-v1) | FR | libvmaf 3.2.0 / named… | Δ 0.00023 | ≈0.74× |
| [`vsi`](docs/MODEL_CARDS.md#vsi) | FR | original VSI / Octave… | 6/6 pass | ≈0.078× |
| [`vsi-k2`](docs/MODEL_CARDS.md#vsi-k2) | FR | original VSI / Octave… | 6/6 pass | ≈0.068× |
| [`vsi-k3`](docs/MODEL_CARDS.md#vsi-k3) | FR | original VSI / Octave… | 6/6 pass | ≈0.063× |
| [`vsi-k4`](docs/MODEL_CARDS.md#vsi-k4) | FR | original VSI / Octave… | 5/6 pass | ≈0.062× |
| [`vsi-nodownscale`](docs/MODEL_CARDS.md#vsi-nodownscale) | FR | original VSI / Octave… | 6/6 pass | ≈0.078× |
| [`wadiqam-fr-live`](docs/MODEL_CARDS.md#wadiqam-fr-live) | FR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-fr-tid`](docs/MODEL_CARDS.md#wadiqam-fr-tid) | FR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-nr-live`](docs/MODEL_CARDS.md#wadiqam-nr-live) | NR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-nr-tid`](docs/MODEL_CARDS.md#wadiqam-nr-tid) | NR/NN | dmaniry/deepIQA | prior | — |
| [`wadiqam-pyiqa`](docs/MODEL_CARDS.md#wadiqam-pyiqa) | FR/NN | pyiqa 0.1.16 create_m… | prior | — |
| [`wd`](docs/MODEL_CARDS.md#wd) | FR/NN | Orange-OpenSource/Coo… | prior | — |
| [`ws-hv`](docs/MODEL_CARDS.md#ws-hv) | FR | unrecorded | unchecked | — |
| [`zensim`](docs/MODEL_CARDS.md#zensim) | FR | unrecorded | unchecked | — |
| [`zensim-gpu`](docs/MODEL_CARDS.md#zensim-gpu) | FR | unrecorded | unchecked | — |

### Executable, versions, output differences and terms

Use each exact metric name below with `target/release/zenmetrics score --metric NAME --reference ref.png --distorted dist.png`. The linked card supplies its build features, model export, consent and display arguments; the command is not a substitute for those requirements.

**Licensing is not uniformly cleared.** Our Rust license, upstream reference-code terms and weight terms are separate. See the [pinned upstream audit](docs/PYIQA_UPSTREAM_LICENSES_2026-10-15.md) for research-only clauses and open questions affecting FSIM, VSI, PSNR-HVS, HDR-VDP, unlicensed reference sources and pretrained backbones. A permissive reference-code license does not grant permission for separately licensed weights. “Not recorded” means unresolved, not unrestricted.

**Deltas are absolute raw-score differences**, written as decimals (for example `0.00003`). Maxima below come only from the named [local canonical comparison cohorts](benchmarks/canonical_recheck_2026-10-19/README.md), with their exact Rust revision, input dimensions and thread configurations. They are not global error bounds or AIC published-column agreement. A median is shown only when retained as a median of paired score differences; it is never inferred from a maximum or a pass threshold. Missing local deltas do not erase older parity evidence in the cards.

<details>
<summary>afine — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`afine`](docs/MODEL_CARDS.md#afine) | git clone of ChrisDud0257/AFINE (no pip package; QuickInference/) | not recorded | not recorded | Apache-2.0 / noncommercial |

</details>

<details>
<summary>ahiq — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ahiq`](docs/MODEL_CARDS.md#ahiq) | IIGROUP/AHIQ epoch-32 checkpoint, CPU, fp32, upstream-path nine-crop ensemble | not recorded | not recorded | MIT / unknown |

</details>

<details>
<summary>ahiq-pyiqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ahiq-pyiqa`](docs/MODEL_CARDS.md#ahiq-pyiqa) | pyiqa 0.1.16 create_metric('ahiq'), CPU fp32, metric(dist, ref) | not recorded | not recorded | PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab) / nc |

</details>

<details>
<summary>arniqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`arniqa`](docs/MODEL_CARDS.md#arniqa) | git clone of miccunifi/ARNIQA (no pip package), pinned 66d16eb0ff1e1655872d32c0c233614a3922aaad | not recorded | not recorded | Apache-2.0 / permissive |

</details>

<details>
<summary>brisque — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`brisque`](docs/MODEL_CARDS.md#brisque) | PIQ 0.8.0 | 0.0025634765625 | not recorded | not recorded / not recorded |

</details>

<details>
<summary>butteraugli — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`butteraugli`](docs/MODEL_CARDS.md#butteraugli) | libjxl butteraugli v0.9.2 / 0.9.4 | not recorded | not recorded | not recorded / not applicable |
| [`butteraugli-gpu`](docs/MODEL_CARDS.md#butteraugli-gpu) | libjxl butteraugli v0.9.2 / 0.9.4 | not recorded | not recorded | not recorded / not applicable |
| [`butteraugli-original`](docs/MODEL_CARDS.md#butteraugli-original) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>cambi — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`cambi`](docs/MODEL_CARDS.md#cambi) | libvmaf 3.2.0 / named RGB ingress | 0.0 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>cambi-aic2026 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`cambi-aic2026`](docs/MODEL_CARDS.md#cambi-aic2026) | libvmaf 3.2.0 / named RGB ingress | 0.0 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ciede2000 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ciede2000`](docs/MODEL_CARDS.md#ciede2000) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ckdn — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ckdn`](docs/MODEL_CARDS.md#ckdn) | git clone of researchmm/CKDN (no pip package; commit tip 2026-10-15) | not recorded | not recorded | MIT / unknown |

</details>

<details>
<summary>cvvdp — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`cvvdp`](docs/MODEL_CARDS.md#cvvdp) | pycvvdp v0.5.7 (PyPI) | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>cvvdp-gpu — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`cvvdp-gpu`](docs/MODEL_CARDS.md#cvvdp-gpu) | pycvvdp v0.5.7 + CPU twin | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>cw-ssim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`cw-ssim`](docs/MODEL_CARDS.md#cw-ssim) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`cw-ssim-pyiqa`](docs/MODEL_CARDS.md#cw-ssim-pyiqa) | pyiqa 0.1.16 create_metric('cw_ssim'), CPU, metric(dist, ref) | not recorded | not recorded | PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab) / none |

</details>

<details>
<summary>deepdc — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`deepdc`](docs/MODEL_CARDS.md#deepdc) | DeepDC_PyTorch==0.2 | not recorded | not recorded | MIT / permissive |

</details>

<details>
<summary>dists — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`dists`](docs/MODEL_CARDS.md#dists) | DISTS_pytorch==0.1 | not recorded | not recorded | MIT / permissive |

</details>

<details>
<summary>dmm — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`dmm`](docs/MODEL_CARDS.md#dmm) | pip DMM_PyTorch (DMM_PyTorch/DMM.py) | not recorded | not recorded | MIT (Baoliang Chen 2024) / permissive |

</details>

<details>
<summary>dmm-pyiqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`dmm-pyiqa`](docs/MODEL_CARDS.md#dmm-pyiqa) | pyiqa 0.1.16 create_metric('dmm'), CPU fp32 (torch.svd, MKL), metric(dist, ref) | not recorded | not recorded | PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); LAPACK routines BSD-3-Clause (../../LICENSE-lapack) / permissive |

</details>

<details>
<summary>dreamsim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`dreamsim`](docs/MODEL_CARDS.md#dreamsim) | pip dreamsim 0.21.2 (identical to the repo's dreamsim/ tree) | not recorded | not recorded | MIT / unknown |

</details>

<details>
<summary>dssim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`dssim`](docs/MODEL_CARDS.md#dssim) | original dssim-core eec33f55 | 0.00000003772523560208185 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>dssim-gpu — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`dssim-gpu`](docs/MODEL_CARDS.md#dssim-gpu) | dssim-core | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>flip — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`flip`](docs/MODEL_CARDS.md#flip) | NVIDIA FLIP 1.7 / f64 map mean | 0.00000000000003552713678800501 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>flip-v17 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`flip-v17`](docs/MODEL_CARDS.md#flip-v17) | NVIDIA FLIP 1.7 | 0.0 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>fsim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`fsim`](docs/MODEL_CARDS.md#fsim) | original FeatureSIM / Octave 6.4 | 0.000001329927450832713 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`fsim-k2`](docs/MODEL_CARDS.md#fsim-k2) | original FeatureSIM / Octave 6.4 | 0.0000000888441795687811 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`fsim-k3`](docs/MODEL_CARDS.md#fsim-k3) | original FeatureSIM / Octave 6.4 | 0.00000018109096577845207 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`fsim-k4`](docs/MODEL_CARDS.md#fsim-k4) | original FeatureSIM / Octave 6.4 | 0.0000000138497604496024 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`fsim-nodownscale`](docs/MODEL_CARDS.md#fsim-nodownscale) | original FeatureSIM / Octave 6.4 | 0.000001329927450832713 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`fsim-y`](docs/MODEL_CARDS.md#fsim-y) | authors' FR_FSIMc.m | not recorded | not recorded | not recorded / not applicable; open terms: see pinned audit |

</details>

<details>
<summary>gmsd — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`gmsd`](docs/MODEL_CARDS.md#gmsd) | libgmsd de646c9a (C) | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>haarpsi — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`haarpsi`](docs/MODEL_CARDS.md#haarpsi) | original HaarPSI / Octave 6.4 | 0.000000011793922838343462 | not recorded | not recorded / not applicable |
| [`haarpsi-y`](docs/MODEL_CARDS.md#haarpsi-y) | original HaarPSI / Octave 6.4 | 0.000000016168560379448138 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>hdrvdp — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`hdrvdp`](docs/MODEL_CARDS.md#hdrvdp) | official HDR-VDP 2.2.2 MATLAB release (ISC-licensed) | not recorded | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`hdrvdp-aic2026`](docs/MODEL_CARDS.md#hdrvdp-aic2026) | no evidence | not recorded | not recorded | not recorded / not applicable; open terms: see pinned audit |

</details>

<details>
<summary>hdrvdp3 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`hdrvdp3`](docs/MODEL_CARDS.md#hdrvdp3) | official HDR-VDP 3.0.7 MATLAB release + jpeg-ai-qaf VDP3/ numpy port @ 0628a6b (verified equal — see DIVERGENCES) | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>hdrvdp3-aic2026 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`hdrvdp3-aic2026`](docs/MODEL_CARDS.md#hdrvdp3-aic2026) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>hyperiqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`hyperiqa`](docs/MODEL_CARDS.md#hyperiqa) | git clone of SSL92/hyperIQA, pinned c42e7279717e7dcb693a24b891fc14a4189a45ee (the paper's official repo — NOT 'SSL-Group/hyperIQA', which 404s) | not recorded | not recorded | MIT / unknown |

</details>

<details>
<summary>iwssim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`iwssim`](docs/MODEL_CARDS.md#iwssim) | Python-IW-SSIM f9de37c (the Python port linked from the authors' IW-SSIM page; not yet compared directly with the authors' MATLAB release) | not recorded | not recorded | not recorded / not applicable |
| [`iwssim-gpu`](docs/MODEL_CARDS.md#iwssim-gpu) | CPU twin | not recorded | not recorded | not recorded / not applicable |
| [`iwssim-piq`](docs/MODEL_CARDS.md#iwssim-piq) | jpeg-ai-qaf IW_SSIM (torch port of the same Jack-guo ref) on the unrounded 0–255 Y — the AIC-4 IW-SSIM column; upstream mirror [local path] @ 0628a6b | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>liqe — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`liqe`](docs/MODEL_CARDS.md#liqe) | git clone of zwx8981/LIQE (no pip package), pinned a64655f7b84b9543b513394472851884cde5b257 | not recorded | not recorded | MIT / unknown |

</details>

<details>
<summary>lpips — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`lpips-alex`](docs/MODEL_CARDS.md#lpips-alex) | lpips==0.1.4 | not recorded | not recorded | BSD-2-Clause / permissive |
| [`lpips-squeeze`](docs/MODEL_CARDS.md#lpips-squeeze) | lpips==0.1.4 | not recorded | not recorded | BSD-2-Clause / permissive |
| [`lpips-vgg`](docs/MODEL_CARDS.md#lpips-vgg) | lpips==0.1.4 | not recorded | not recorded | BSD-2-Clause / permissive |

</details>

<details>
<summary>mad — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`mad`](docs/MODEL_CARDS.md#mad) | official MATLAB release (larschandler.com) | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>maniqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`maniqa`](docs/MODEL_CARDS.md#maniqa) | git clone of IIGROUP/MANIQA (no pip package; vendored timm 0.5.5 in-repo), pinned f573d862401243c45641951a0303b8059c1c8577 | not recorded | not recorded | Apache-2.0 / permissive |

</details>

<details>
<summary>mdctpsnr — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`mdctpsnr`](docs/MODEL_CARDS.md#mdctpsnr) | author's official C++ thorfdbg/mDCTpsnr (Thomas Richter; contributor Jon Sneyers) (zlib-style license), built GCC -O3 -ffast-math AVX2 + glibc libmvec | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>mdsi — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`mdsi`](docs/MODEL_CARDS.md#mdsi) | original MDSI / Octave 6.4 | 0.000000000000010685896612017132 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ms-gmsd — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ms-gmsd`](docs/MODEL_CARDS.md#ms-gmsd) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ms-gmsdc — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ms-gmsdc`](docs/MODEL_CARDS.md#ms-gmsdc) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ms-ssim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ms-ssim-pyiqa`](docs/MODEL_CARDS.md#ms-ssim-pyiqa) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ms-ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32) | PyIQA 0.1.16 / named convention | 0.00000003040175311497251 | not recorded | not recorded / not applicable |
| [`ms-ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k2) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ms-ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k3) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ms-ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k4) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ms-ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k5) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ms-ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ms-ssim-pyiqa-f32-k6) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>msssim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`msssim`](docs/MODEL_CARDS.md#msssim) | original msssim / Octave 6.4 | 0.0000017640623451997683 | not recorded | not recorded / not applicable |
| [`msssim-libvmaf`](docs/MODEL_CARDS.md#msssim-libvmaf) | libvmaf 3.2.0 / named RGB ingress | 0.0000001537393887751648 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>musiq — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`musiq`](docs/MODEL_CARDS.md#musiq) | git clone of google-research (sparse musiq/), pinned 3ef5211; legacy stack jax 0.3.25 + flax 0.3.3 (flax.nn) + tensorflow CPU on python 3.9 — Kaggle/TF-Hub google/musiq instances are conversions of the same npz checkpoints (docs/pyiqa_licenses_2026-10-15/deep_origins.tsv) | not recorded | not recorded | Apache-2.0 / permissive |

</details>

<details>
<summary>niqe — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`niqe`](docs/MODEL_CARDS.md#niqe) | BasicSR 1.4.2 | 0.00009065496319360378 | not recorded | not recorded / not recorded |

</details>

<details>
<summary>nlpd — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`nlpd`](docs/MODEL_CARDS.md#nlpd) | Valerolaparra/NLPD_Pytorch (the authors' PyTorch reference; PyTorch 2.5.1 oracle) | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>nlpd-iqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`nlpd-iqa`](docs/MODEL_CARDS.md#nlpd-iqa) | IQA_pytorch source mirrored at [local path] semantics replicated to med \|Δ\| 0.0026 / max 0.0067 vs the AIC-4 NLPD column (n=53) | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>paq2piq — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`paq2piq`](docs/MODEL_CARDS.md#paq2piq) | git clone of baidut/paq2piq revision 48c91e844e9f7a768f6ddcfc744dddfdd1160fea (the first author's official inference repo; LICENSE.md added 2024-10-25) | not recorded | not recorded | MIT / noncommercial |

</details>

<details>
<summary>pieapp — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`pieapp`](docs/MODEL_CARDS.md#pieapp) | Official v0.1 public compute_score, release 0513613d1f575136ea52418e35f2cebc5a49f783 | not recorded | not recorded | not recorded / Modified BSD-2 for Non-Commercial Use Only, UC Regents |

</details>

<details>
<summary>pieapp-pyiqa — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`pieapp-pyiqa`](docs/MODEL_CARDS.md#pieapp-pyiqa) | pyiqa 0.1.16 create_metric('pieapp'), CPU fp32, metric(dist, ref) | not recorded | not recorded | PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab) / nc |

</details>

<details>
<summary>piqe — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`piqe`](docs/MODEL_CARDS.md#piqe) | pypiqe 1.2 | 0.00000000000004618527782440651 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>proposal-butteraugli — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`proposal-butteraugli`](docs/MODEL_CARDS.md#proposal-butteraugli) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>psnr — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`psnr`](docs/MODEL_CARDS.md#psnr) | scikit-image 0.26 / declared RGB ingress | 0.0 | not recorded | not recorded / not applicable |
| [`psnr-cb`](docs/MODEL_CARDS.md#psnr-cb) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`psnr-cr`](docs/MODEL_CARDS.md#psnr-cr) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`psnr-y`](docs/MODEL_CARDS.md#psnr-y) | scikit-image 0.26 / declared RGB ingress | 0.0000000000020463630789890885 | not recorded | not recorded / not applicable |
| [`psnr-y-libvmaf`](docs/MODEL_CARDS.md#psnr-y-libvmaf) | libvmaf 3.2.0 / named RGB ingress | 0.0 | not recorded | not recorded / not applicable |
| [`psnr-y-studio601`](docs/MODEL_CARDS.md#psnr-y-studio601) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`psnr-y601`](docs/MODEL_CARDS.md#psnr-y601) | scikit-image 0.26 / declared RGB ingress | 0.0000000000024300561562995426 | not recorded | not recorded / not applicable |
| [`psnr-ycbcr611`](docs/MODEL_CARDS.md#psnr-ycbcr611) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>psnrhvs — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`psnrhvs`](docs/MODEL_CARDS.md#psnrhvs) | authors' psnrhvsm.m | not recorded | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`psnrhvs-daala`](docs/MODEL_CARDS.md#psnrhvs-daala) | libvmaf psnr_hvs feature @ f85a8536 — FFI oracle | not recorded | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`psnrhvs-y`](docs/MODEL_CARDS.md#psnrhvs-y) | authors' psnrhvsm.m | not recorded | not recorded | not recorded / not applicable; open terms: see pinned audit |

</details>

<details>
<summary>ssim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ssim`](docs/MODEL_CARDS.md#ssim) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssim-libvmaf`](docs/MODEL_CARDS.md#ssim-libvmaf) | libvmaf 3.2.0 / named RGB ingress | 0.00000005960464477539063 | not recorded | not recorded / not applicable |
| [`ssim-pyiqa`](docs/MODEL_CARDS.md#ssim-pyiqa) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssim-pyiqa-f32`](docs/MODEL_CARDS.md#ssim-pyiqa-f32) | PyIQA 0.1.16 / named convention | 0.00000011643768749181049 | not recorded | not recorded / not applicable |
| [`ssim-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k2) | PyIQA 0.1.16 / named convention | 0.000000055390560693702184 | not recorded | not recorded / not applicable |
| [`ssim-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k3) | PyIQA 0.1.16 / named convention | 0.00000003263846981571561 | not recorded | not recorded / not applicable |
| [`ssim-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k4) | PyIQA 0.1.16 / named convention | 0.000000022443646052749955 | not recorded | not recorded / not applicable |
| [`ssim-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k5) | PyIQA 0.1.16 / named convention | 0.00000001689128470427903 | not recorded | not recorded / not applicable |
| [`ssim-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssim-pyiqa-f32-k6) | PyIQA 0.1.16 / named convention | 0.000000013229299988815058 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ssim2 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ssim2`](docs/MODEL_CARDS.md#ssim2) | primary SSIMULACRA2 v2.1 | 0.0 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ssim2-gpu — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ssim2-gpu`](docs/MODEL_CARDS.md#ssim2-gpu) | fast-ssim2 (CPU) + third-party crates.io ssimulacra2 for parity | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ssimc — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ssimc`](docs/MODEL_CARDS.md#ssimc) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssimc-pyiqa-f32`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32) | PyIQA 0.1.16 / named convention | 0.00000014727947605930325 | not recorded | not recorded / not applicable |
| [`ssimc-pyiqa-f32-k2`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k2) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssimc-pyiqa-f32-k3`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k3) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssimc-pyiqa-f32-k4`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k4) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssimc-pyiqa-f32-k5`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k5) | no evidence | not recorded | not recorded | not recorded / not applicable |
| [`ssimc-pyiqa-f32-k6`](docs/MODEL_CARDS.md#ssimc-pyiqa-f32-k6) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>ssimulacra1 — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ssimulacra1`](docs/MODEL_CARDS.md#ssimulacra1) | original SSIMULACRA1 / OpenCV | 0.000000003826276126668582 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>stlpips — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`stlpips-alex`](docs/MODEL_CARDS.md#stlpips-alex) | stlpips_pytorch==0.0.2 | not recorded | not recorded | BSD-2-Clause / permissive |
| [`stlpips-vgg`](docs/MODEL_CARDS.md#stlpips-vgg) | stlpips_pytorch==0.0.2 | not recorded | not recorded | BSD-2-Clause / permissive |

</details>

<details>
<summary>topiq — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`topiq-fr`](docs/MODEL_CARDS.md#topiq-fr) | pyiqa 0.1.16 (torch 2.14.1+cpu) | not recorded | not recorded | PolyForm-Noncommercial-1.0.0 + S-Lab-1.0 (LICENSE-pyiqa, LICENSE-S-Lab) / https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE |
| [`topiq-fr-pipal`](docs/MODEL_CARDS.md#topiq-fr-pipal) | pyiqa 0.1.16 (torch 2.14.1+cpu) | not recorded | not recorded | PolyForm-Noncommercial-1.0.0 + S-Lab-1.0 (LICENSE-pyiqa, LICENSE-S-Lab) / https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE |

</details>

<details>
<summary>vif — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`vif`](docs/MODEL_CARDS.md#vif) | original vifp_mscale / Octave 6.4 | 0.0000000000005379585665821196 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>vifvec — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`vifvec`](docs/MODEL_CARDS.md#vifvec) | original vifvec / Octave 6.4 | 0.00000000000024030777368011513 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>vmaf — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`vmaf`](docs/MODEL_CARDS.md#vmaf) | libvmaf 3.2.0 / named RGB ingress | 0.00028193174530599663 | not recorded | not recorded / not applicable |
| [`vmaf-4k`](docs/MODEL_CARDS.md#vmaf-4k) | libvmaf 3.2.0 / named RGB ingress | 0.0 | not recorded | not recorded / not applicable |
| [`vmaf-neg`](docs/MODEL_CARDS.md#vmaf-neg) | libvmaf 3.2.0 / named RGB ingress | 0.0002482166810153785 | not recorded | not recorded / not applicable |
| [`vmaf-neg-studio601`](docs/MODEL_CARDS.md#vmaf-neg-studio601) | libvmaf 3.2.0 / named RGB ingress | 0.000007238667720343983 | not recorded | not recorded / not applicable |
| [`vmaf-studio601`](docs/MODEL_CARDS.md#vmaf-studio601) | libvmaf 3.2.0 / named RGB ingress | 0.000007264956991548388 | not recorded | not recorded / not applicable |
| [`vmaf-v1`](docs/MODEL_CARDS.md#vmaf-v1) | libvmaf 3.2.0 / named RGB ingress | 0.00022515701154190992 | not recorded | not recorded / not applicable |

</details>

<details>
<summary>vsi — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`vsi`](docs/MODEL_CARDS.md#vsi) | original VSI / Octave 6.4 | 0.0000000050279395091834544 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`vsi-k2`](docs/MODEL_CARDS.md#vsi-k2) | original VSI / Octave 6.4 | 0.000000006297474430816408 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`vsi-k3`](docs/MODEL_CARDS.md#vsi-k3) | original VSI / Octave 6.4 | 0.000000006385277306897308 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`vsi-k4`](docs/MODEL_CARDS.md#vsi-k4) | original VSI / Octave 6.4 | 0.000000004950456933272562 | not recorded | not recorded / not applicable; open terms: see pinned audit |
| [`vsi-nodownscale`](docs/MODEL_CARDS.md#vsi-nodownscale) | original VSI / Octave 6.4 | 0.0000000050279395091834544 | not recorded | not recorded / not applicable; open terms: see pinned audit |

</details>

<details>
<summary>wadiqam — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`wadiqam-fr-live`](docs/MODEL_CARDS.md#wadiqam-fr-live) | git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link) | not recorded | not recorded | MIT / permissive |
| [`wadiqam-fr-tid`](docs/MODEL_CARDS.md#wadiqam-fr-tid) | git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link) | not recorded | not recorded | MIT / permissive |
| [`wadiqam-nr-live`](docs/MODEL_CARDS.md#wadiqam-nr-live) | git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link) | not recorded | not recorded | MIT / permissive |
| [`wadiqam-nr-tid`](docs/MODEL_CARDS.md#wadiqam-nr-tid) | git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link) | not recorded | not recorded | MIT / permissive |
| [`wadiqam-pyiqa`](docs/MODEL_CARDS.md#wadiqam-pyiqa) | pyiqa 0.1.16 create_metric('wadiqam_fr'), CPU fp32, metric(dist, ref) | not recorded | not recorded | PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab) / nc |

</details>

<details>
<summary>wd — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`wd`](docs/MODEL_CARDS.md#wd) | ed9de16b8819146afeecc20b4d2d0f385f711a7fd9740288bfdbacd3e93e7b43 | not recorded | not recorded | BSD-3-Clause; source and license notices in LICENSE-wd / permissive |

</details>

<details>
<summary>ws-hv — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`ws-hv`](docs/MODEL_CARDS.md#ws-hv) | no evidence | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>zensim — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`zensim`](docs/MODEL_CARDS.md#zensim) | — | not recorded | not recorded | not recorded / not applicable |

</details>

<details>
<summary>zensim-gpu — variants and configurations</summary>

| Exact metric / command NAME | Canonical identity | Max absolute delta | Median absolute delta | Reference code / weights |
| --- | --- | ---: | ---: | --- |
| [`zensim-gpu`](docs/MODEL_CARDS.md#zensim-gpu) | — | not recorded | not recorded | not recorded / not applicable |

</details>

DVIFMISH is a separate, unpublished benchmark dependency: `publish = false`; publication requires the original author’s consent. Its six benchmark configurations are `dvifmish-port-min-v0.2`, `dvifmish-port-2d-v0.2`, `dvifmish-port-lap-v0.2`, `dvifmish-port-lap-v0.3`, `dvifmish-talk-faithful-luma-float` and `dvifmish-talk-faithful-luma-int`. These are not public `zenmetrics score` names, and benchmark repeatability does not establish native-author oracle parity. Private adapters, models and weights are not distributed by these instructions.


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
