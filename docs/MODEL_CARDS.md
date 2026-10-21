# Models and metrics

Generated from model cards, the CLI catalog and provenance ledger. Bibliographic metadata comes from the linked original publisher/arXiv record. A missing field remains missing. Reference parity, published-dataset reproduction and performance are separate evidence.

Regenerate with `python3 scripts/dashboard/model_cards.py`; verify currency with `python3 scripts/dashboard/model_cards.py --check`. The dashboard workflow regenerates before checking.

The Learned family includes neural metrics and BRISQUE’s trained regressor. Classical includes analytical and natural-scene-statistics methods. Backend is the CLI route; optional learned engines are listed separately.

## afine

FR · Learned · CPU · Reported results

**Paper:** [Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption](https://arxiv.org/abs/2503.11221). Chen, Du; Wu, Tianhe; Ma, Kede; Zhang, Lei. CVPR 2025. 2025 (arXiv submission year; publication venue may be later).

```bibtex
@misc{afine2025,
  title = {Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption},
  author = {Chen, Du and Wu, Tianhe and Ma, Kede and Zhang, Lei},
  year = {2025},
  note = {CVPR 2025},
  url = {https://arxiv.org/abs/2503.11221},
  eprint = {2503.11221},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Du et al. A-FINE: Adaptive Fidelity-Naturalness Evaluation for AI-Generated Image Quality Assessment. IEEE TMM 2024

**Reference implementation:** https://github.com/ChrisDud0257/AFINE

**Reference identity:** git clone of ChrisDud0257/AFINE (no pip package; QuickInference/)

**Reference code license:** Apache-2.0

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** noncommercial

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-better

**Cli contract:** A-FINE — learned FR metric (upstream ChrisDud0257/AFINE, Apache-2.0;
CLIP ViT-B/32 taps + naturalness/fidelity heads, raw afine_all,
lower is better). Needs feature nn; its weights carry
noncommercial terms (trained on DiffIQA data), so
ZENMETRICS_NN_ACCEPT_WEIGHTS=afine as for ckdn.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/afine/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric afine --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=afine. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "noncommercial",
  "checkpoint": "https://drive.google.com/drive/folders/1SgcMmv-9yejHYTT8F8hGN_5Vv8hfGMmR (afine.pth); plus OpenAI ViT-B-32.pt"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "0..1 then CLIP stats (0.48145466, 0.4578275, 0.40821073 / 0.26862954, 0.26130258, 0.27577711); the heads de-normalize internally (xstd+mean)",
  "resize": "upstream afine.py: F.interpolate(mode='bicubic', align_corners=False) UPSAMPLE to the next multiple of 32 on each axis that needs it (a=-0.75, 4 taps, edge-clamp, no renorm — resize.rs::resize_bicubic, impulse-verified vs torch)",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "upstream afine.pth + ViT-B-32.pt, CPU, fp32, upstream preprocessing",
  "tolerance_jnd": 0.001,
  "jnd": 0.507388,
  "results": "rten worst |d| 7.228e-6 (1.42e-5 JND); ort-cuda worst |d| 9.865e-6 (1.94e-5 JND); 15 goldens"
}
```

**Conventions:**

```json
{
  "aic4": {
    "column": "A-FINE",
    "results": "NOT reproduced: native afine_all at native resolution differs from the published column by median |d| 7.5e-1, max 1.4e0 over 61 pairs (SRCC 0.96). Closest single term is fr_scale (median |d| 0.088, SRCC 0.92). No upstream convention tried — transforms, term combinations, default-parameter ablations — reproduces the column; documented as a gap rather than redefining the metric.",
    "finding": "The published column's range (-1.95..-0.66 on the crops) identifies it as a raw afine-family adapter output, not afine_all_scale (0..100), but it is provably not upstream afine_all under any adapter weight (implied naturalness weight is non-positive). Whatever preprocessing/variant produced it is unidentified."
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/afine/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/afine/card.toml) · [crates/zenmetrics-nn/LICENSE-afine](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-afine)

**Known divergences:** [afine — A-FINE (zenmetrics-nn, 2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#afine--a-fine-zenmetrics-nn-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=afine) · [Speed](https://metric-brief.pages.dev/#speed&q=afine)

## ahiq

FR · Learned · CPU · Reported results

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** https://github.com/IIGROUP/AHIQ

**Reference identity:** IIGROUP/AHIQ epoch-32 checkpoint, CPU, fp32, upstream-path nine-crop ensemble

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** unknown

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AHIQ — learned FR quality metric (upstream IIGROUP/AHIQ, MIT
code; ViT-B/8 + ResNet-50 taps, deformable fusion, mean of nine
224×224 crops, higher is better). Needs feature nn; its
weights state no licence, so ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/ahiq/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric ahiq --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "unknown",
  "checkpoint": "Google Drive file 1Nk-IpjnDNXbWacoh3T69wkSYhYYWip2W (epoch 32, patch_size 8) linked from https://github.com/IIGROUP/AHIQ; ImageNet backbones via timm pretrained — vit_base_patch8_224.augreg_in21k_ft_in1k (the augreg B_8-i21k checkpoint timm 0.5.4's bare name resolves; NOT the augreg2 variant timm 1.x defaults to) and resnet50.a1_in1k, both Apache-2.0 via timm",
  "trained_on": "PIPAL (upstream default: --name ahiq_pipal, --train_list PIPAL.txt)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "cv2 BGR->RGB, v/255, (x-0.5)/0.5 -> [-1, 1]; NO resizing anywhere",
  "resize": "none — the port scores nine deterministic 224x224 crops (five_point_crop idx 0-8: four corners, centre, four edge centres) and averages. Upstream test.py's default is n_ensemble=20 seeded random crops (seed 1919); the <=9 branch is this nine-crop protocol, so the port is a deterministic deviation from upstream's default",
  "min_size": 224
}
```

**Parity:**

```json
{
  "oracle": "IIGROUP/AHIQ epoch-32 checkpoint, CPU, fp32, upstream-path nine-crop ensemble",
  "tolerance_jnd": 0.001,
  "jnd": 0.024882,
  "results": "rten worst |d| 5.295e-8 (2.1e-6 JND); ort-cuda worst |d| 2.582e-7 (1.0e-5 JND); 17 goldens incl. 320x240+240x320 non-square rows; warm ref == cold on both engines; logs [local path] on i265"
}
```

**Conventions:**

```json
{
  "aic4": {
    "convention": "NOT REPRODUCED by the upstream protocol. Under upstream's own inference path with the port's deterministic nine-crop convention (s9), all 61 pairs carry a systematic positive offset vs the published column; the column follows a different convention than upstream test.py scoring. The port implements upstream semantics and documents the gap rather than chasing it.",
    "results": "native s9 on all 61 pairs: median |d| 0.1405 (5.6 JND), max |d| 0.2125, SRCC 0.76. Table: benchmarks/ahiq_aic_probe/aic_ahiq.tsv; log [local path] on i265."
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/ahiq/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/ahiq/card.toml) · [crates/zenmetrics-nn/LICENSE-ahiq](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-ahiq)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ahiq) · [Speed](https://metric-brief.pages.dev/#speed&q=ahiq)

## ahiq-pyiqa

FR · Learned · CPU · Reported results

**Paper: citation needed.**

**Recorded paper:** Lao, Gong, Shi, Yang, Wu, Wang, Xia, Yang. Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network. CVPRW 2022

**Reference implementation:** pyiqa 0.1.16 create_metric('ahiq'), CPU fp32, metric(dist, ref)

**Reference identity:** pyiqa 0.1.16 create_metric('ahiq'), CPU fp32, metric(dist, ref)

**Reference code license:** PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab)

**Rust license:** code: PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); weights: nc

**Weights license:** nc

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** AHIQ as pyiqa scores it (20 uniform crops, timm 1.0.30 backbones,
epoch-33 head; the AIC2026 AHIQ column, higher is better). Needs
feature pyiqa-ports and ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq_pyiqa.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports,nn
# export the model once (see crates/zenmetrics-pyiqa-ports/models/ahiq-pyiqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric ahiq-pyiqa --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq_pyiqa. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "nc",
  "checkpoint": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/AHIQ_vit_p8_epoch33-da3ea303.pth",
  "sha256": "da3ea303b365ef71f4bbe41eba56f0ab3371c697d9c959649904dc9dd0a08b85",
  "backbones": "timm vit_base_patch8_224.augreg2_in21k_ft_in1k and resnet50.a1_in1k (Apache-2.0 via timm)",
  "consent": "AhiqPyiqa::WEIGHTS; ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq_pyiqa"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "v / 255, ImageNet mean/std inside the trunk graph",
  "crops": "uniform_crop(224, 20): rows i < 4, cols j < 5 of a 5x5 grid, origin (istep_h, jstep_w)",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 create_metric('ahiq'), CPU fp32, metric(dist, ref)",
  "goldens": "goldens_ladder.tsv (common/ladder.py, 15 pairs 64x64..2048x1536)",
  "jnd": 0.043353,
  "tolerance_jnd": 0.001,
  "measured_ladder": "rten 5.1e-8 (1.2e-6 JND), ort-cuda 4.0e-7 (9.3e-6 JND); ort-cpu on 3 of the 15 pairs (64x64, 840x944, 2048x1536: AHIQ costs 20 crops x 2 images whatever the size, and ort-cpu runs > 65 s per score here, so the full ladder does not fit one heavy-lock hold) 4.0e-7 (9.1e-6 JND)",
  "measured_aic2026_61": "ort-cuda 5.2e-7 (1.2e-5 JND); pyiqa vs the published column: median 2.5e-6, max 1.0e-5",
  "warm_reference": "caches the 20 crops' deform offsets and reduced reference features (~8.5 MB); bit-identical to cold",
  "recorded_source_comments": "1 JND: pooled median |score(pair) - score(ref, ref)| over the 1,008 KonJND-1k pairs, scored by this\nport on ort-cuda (pyiqa needs ~80 s per pair on CPU): JPEG 0.037184, BPG 0.050104.\nperf_deltas.py, i265 (Core Ultra 7 265K), rten, 1024x1024 ladder pair, cold evaluate per call;\nbenchmarks/iqa_complexity_2026-10-17/measured/pyiqa_ports_i265_1024.{tsv,meta}."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/ahiq-pyiqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/ahiq-pyiqa/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ahiq-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=ahiq-pyiqa)

## arniqa

NR · Learned · CPU · Reported results

**Paper:** [ARNIQA: Learning Distortion Manifold for Image Quality Assessment](https://arxiv.org/abs/2310.14918). Agnolucci, Lorenzo; Galteri, Leonardo; Bertini, Marco; Del Bimbo, Alberto. WACV 2024. 2023 (arXiv submission year; publication venue may be later).

```bibtex
@misc{arniqa2023,
  title = {ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
  author = {Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  year = {2023},
  note = {WACV 2024},
  url = {https://arxiv.org/abs/2310.14918},
  eprint = {2310.14918},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Agnolucci, Galteri, Bertini, Del Bimbo. ARNIQA: Learning Distortion Manifold for No-Reference Image Quality Assessment. WACV 2024

**Reference implementation:** https://github.com/miccunifi/ARNIQA

**Reference identity:** git clone of miccunifi/ARNIQA (no pip package), pinned 66d16eb0ff1e1655872d32c0c233614a3922aaad

**Reference code license:** Apache-2.0

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** ARNIQA — learned no-reference quality score (upstream
miccunifi/ARNIQA, Apache-2.0; ResNet-50 SimCLR embeddings at full +
PIL-bilinear half scale, kadid10k Ridge regressor); higher is
better, [0, 1]. Scored through PairedNoReference: emits arniqa
(distorted) and arniqa_ref (reference). As lpips-alex for
ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/arniqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric arniqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "https://github.com/miccunifi/ARNIQA/releases/tag/weights — ARNIQA.pth sha256 ad2022e5…, regressor_kadid10k.pth sha256 4315bf47…"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "ToTensor u8 -> f32 /255, then ImageNet mean/std inside the graphs",
  "resize": "single_image_inference.py: a second view at torchvision Resize((h//2, w//2)) — on a PIL image that is PIL BILINEAR fixed-point u8 (resize.rs::resize_bilinear_pil_u8); NOT the dataset path's bare img.resize (PIL's default BICUBIC — the standalone script is the scored convention)",
  "crop": "five 224x224 center_corners_crop views per scale — centre, TL, BL, TR, BR — TF.crop, zero-padded past the edges",
  "min_size": 2
}
```

**Parity:**

```json
{
  "oracle": "miccunifi/ARNIQA revision 66d16eb, single_image_inference.py, CPU, fp32 (its autocast is a no-op off CUDA), kadid10k regressor",
  "tolerance_jnd": 0.001,
  "jnd": 0.053838,
  "results": "rten vs upstream: worst |d| 3.58e-7 = 6.65e-6 JND on 15 ladder goldens (harness-measured; an earlier hand-sweep rounded 7.81e-7 before the full TOL gate — the harness number is authoritative). ort-cuda identical at print precision (CLI smoke 1415_256: 0.444375/0.639651). Goldens committed at models/arniqa/arniqa_goldens.tsv"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/arniqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/arniqa/card.toml) · [crates/zenmetrics-nn/LICENSE-arniqa](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-arniqa)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=arniqa) · [Speed](https://metric-brief.pages.dev/#speed&q=arniqa)

## brisque

NR · Learned · CPU · Not documented

**Paper:** [No-Reference Image Quality Assessment in the Spatial Domain](https://doi.org/10.1109/TIP.2012.2214050). A. Mittal; A. K. Moorthy; A. C. Bovik. IEEE Transactions on Image Processing. 2012.

```bibtex
@misc{BRISQUE2012,
  title = {No-Reference Image Quality Assessment in the Spatial Domain},
  author = {A. Mittal and A. K. Moorthy and A. C. Bovik},
  year = {2012},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2012.2214050},
  doi = {10.1109/TIP.2012.2214050}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not recorded

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** BRISQUE — no-reference blind/referenceless image spatial quality
evaluator (in-tree brisque crate, port of piq 0.8.0 brisque()
with LIVE's released SVM model). SVR distance; higher is worse.
Scored through PairedNoReference: emits brisque (distorted)
and brisque_ref (reference). Needs nr.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nr
target/release/zenmetrics score --metric brisque --reference ref.png --distorted dist.png
```

No-reference: the score describes the distorted image; the CLI still takes both paths.

**Weights:**

```json
{
  "notice": "Static trained/statistical parameters; consult the linked crate README and licence notices. Exact checkpoint terms are not normalized in this card."
}
```

**Inputs:**

```json
{
  "cli_contract": "BRISQUE — no-reference blind/referenceless image spatial quality\nevaluator (in-tree brisque crate, port of piq 0.8.0 brisque()\nwith LIVE's released SVM model). SVR distance; higher is worse.\nScored through PairedNoReference: emits brisque (distorted)\nand brisque_ref (reference). Needs nr."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/brisque/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/brisque/README.md)

# brisque

Pure-Rust CPU port of BRISQUE (Blind/Referenceless Image Spatial
Quality Evaluator), a classical no-reference image-quality metric, as a
zenmetrics-abstractions::NoReferenceMetric.

 Origin: piq/brisque.py (https://github.com/photosynthesis-team/piq)
  (piq 0.8.0, Apache-2.0 — photosynthesis-team; notice in
  LICENSE-piq (LICENSE-piq)), a PyTorch port of the official MATLAB
  BRISQUE_release.
 Model data: src/params.bin stores the ε-SVR support vectors and
  coefficients (774 × 36), rho,
  the 7×7 Gaussian kernel, and the gam-grid lookup tables
  (r_table/agg ratios and the per-α eta factor). piq's
  brisque_svm_weights.pt is bit-identical to allmodel in LIVE's
  BRISQUE_release.zip (http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip)
  (verified; hashes in tests/gen_params.py). The LIVE notice — which
  permits use/copy/modification/distribution provided it appears in all
  copies — is kept verbatim in
  LICENSE-LIVE-brisque (LICENSE-LIVE-brisque). params.bin, its sha256 and the aligned, safe
  params.rs decoder are regenerated by tests/gen_params.py; provenance
  hashes are in the decoder header. The feature-scaling ranges remain in
  lib.rs.
The LIVE grant names “code (the source files)” and documentation; allmodel
is model data. Its notice and readme attribution request are retained without
claiming a broader grant for that data.

 Paper: A. Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image
  Quality Assessment in the Spatial Domain", IEEE Transactions on Image
  Processing, vol. 21, no. 12, pp. 4695–4708, Dec. 2012.
  <https://doi.org/10.1109/TIP.2012.2214050>

## Conventions (matching the oracle)

 RGB → gray is piq's rgb2yiq Y channel rounded ties-to-even — the
  MATLAB rgb2gray full-range BT.601 luma (0.299/0.587/0.114):
  Y = round(fma(B, .114, .299·R + .587·G)), verified bitwise over all
  2²⁴ u8 triples. GRAY8 input is taken verbatim.
 Two scales: MSCN coefficients (7×7 Gaussian, σ = 7/6) → GGD (α, σ²) +
  four shifted AGGD products (α, η, left/right σ²) = 18 features per
  scale. The second scale is the MATLAB-compatible reflect-padded
  bicubic imresize(0.5) — piq's imresize port, not a plain
  half-sample.
 Features scale to [-1,1] via LIVE's allrange, then ε-SVR predict
  (γ = 0.05 RBF over LIVE's allmodel; score − rho, rho =
  -153.591). Higher = worse (ScoreKind::Distortion); scalar output
  only (map requests are refused); typical range ~0–70 (negative scores
  are possible — very clean images).
 min(w,h)<8 (unsupported two-scale geometry) reports ErrorKind::TooSmall.
  Content the oracle rejects — flat GGD σ ≈ 0, single-sign AGGD product
  maps, non-finite scaled features — reports ErrorKind::Degenerate
  instead of raising.

## Usage

rust
use brisque::Brisque;
use zenmetrics_abstractions::NoReferenceMetric;

let mut metric = Brisque::new();


Output key: brisque. Through PairedNoReference, pairs additionally
emit brisque_ref.

## Parity

The fixed gate remains JND/1000 (JND = 15.138428). Integer squares and cubes
use multiplication. A 400,000-case synthetic GGD/AGGD sweep matches torch's ratios
and gamma-grid argmins exactly. The final implementation uses libm::expf,
deterministic f64 fixed-lane accumulators for the map reductions, and lazy
balanced tap trees for the resize dots. Maximum score error is 0.00830078125
(0.0005483252 JND) on the 170 goldens against the pinned oracle environment.
The reduction rework is measured end-to-end: all 2016 KonJND images drift
≤ 0.01171875 vs the previous pairwise-f32 implementation (argmin grid flips
from ~1e-8 feature ulp changes), and against a freshly installed piq-0.8.0
oracle on all 2058 images the two implementations' error envelopes coincide
(max 0.0943 vs 0.0947 — the offset is oracle-environment drift, not port
error). No tolerance was relaxed; final scores are not bit-exact.

Computing gamma tables with f64 libm::lgamma/exp exceeded the same image gate
with both reducers (final golden maximum 0.0615234375, or 0.004064057212545451
JND), so the fixed oracle tables remain in params.bin.

The oracle is piq 0.8.0 / torch 2.14.1, CPU AVX2, one thread. Full traces and
runtime metadata are recorded by the review drivers in
benchmarks/brisque_review_2026-10-15 (../../benchmarks/brisque_review_2026-10-15).
The PyTorch-derived reducer and its notice were removed at coordinator direction;
summation.rs independently specifies the binary addition tree.
Paths in the golden panel resolve relative to ZENMETRICS_NR_GOLDEN_ROOT
(default [local path]).

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/brisque/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/brisque/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/brisque/LICENSE-LIVE-brisque](https://github.com/imazen/zenmetrics/blob/master/crates/brisque/LICENSE-LIVE-brisque) · [crates/brisque/LICENSE-piq](https://github.com/imazen/zenmetrics/blob/master/crates/brisque/LICENSE-piq)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=brisque) · [Speed](https://metric-brief.pages.dev/#speed&q=brisque)

## butteraugli

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** Jyrki Alakuijala, Google — no paper; google/butteraugli (https://github.com/google/butteraugli)

**Reference implementation:** libjxl butteraugli v0.9.2 / 0.9.4

**Reference identity:** libjxl butteraugli v0.9.2 / 0.9.4

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** distance; emits _max + _pnorm3

**Direction:** not recorded

**Cli contract:** Butteraugli — CPU implementation via the butteraugli crate.
Emits two columns per cell: butteraugli_max
(ButteraugliResult::score, the per-block maximum) and
butteraugli_pnorm3 (ButteraugliResult::pnorm_3, the libjxl-style
3-norm aggregation matching butteraugli_main --pnorm and the
CID22 paper by Sneyers, Ben Baruch & Vaxman, Cloudinary). One compute() call yields both numbers,
so emitting both is free.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric butteraugli --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "libjxl butteraugli v0.9.2 / 0.9.4"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=butteraugli) · [Speed](https://metric-brief.pages.dev/#speed&q=butteraugli)

## butteraugli-gpu

FR · Classical · GPU · Not documented

**Paper: citation needed.**

**Recorded paper:** Jyrki Alakuijala, Google — no paper; google/butteraugli (https://github.com/google/butteraugli)

**Reference implementation:** libjxl butteraugli v0.9.2 / 0.9.4

**Reference identity:** libjxl butteraugli v0.9.2 / 0.9.4

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** distance; emits _max + _pnorm3

**Direction:** not recorded

**Cli contract:** Butteraugli — GPU implementation via the butteraugli-gpu crate.
Emits two columns: butteraugli_max_gpu and butteraugli_pnorm3_gpu.
Same single-compute() rationale as the CPU variant.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features gpu-butteraugli
target/release/zenmetrics score --metric butteraugli-gpu --reference ref.png --distorted dist.png
```

Requires a GPU backend at run time.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "libjxl butteraugli v0.9.2 / 0.9.4"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/butteraugli-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/butteraugli-gpu/README.md)

# butteraugli-gpu

Multi-vendor GPU implementation of the butteraugli perceptual image
quality metric, built on CubeCL (https://github.com/tracel-ai/cubecl).
One Rust kernel source dispatches across CUDA, WGPU (Vulkan / Metal /
DX12 / WebGPU), HIP, and a CPU SIMD reference.

Algorithmic parity with butteraugli v0.9.2: score is the max-norm,
pnorm_3 is the libjxl 3-norm aggregation, both produced in one fused
on-device reduction.

## Memory modes

butteraugli-gpu exposes three memory modes via [MemoryMode] and the
typed constructors below. Pick based on image size, available VRAM, and
whether you need the multi-resolution (half-res sibling) aggregation
that the CPU reference uses by default.

| Mode | Constructor | Use case |
|---|---|---|
| Whole, single-res | Butteraugli::new | Reference-quality single-resolution. No half-res sibling. Cheapest path for small images. |
| Whole, multi-res | Butteraugli::new_multires | CPU-reference parity (includes half-res supersample-add). Allocates a full-res instance + a quarter-sized half-res sibling. |
| Strip, single-res | Butteraugli::new_strip | Constant-VRAM walker for large images. Holds one body_h-row slab + halo at a time, stitches reductions host-side. |
| Strip, multi-res | Butteraugli::new_multires_strip | Constant-VRAM multi-res — full-res strip walks paired with synchronized half-res strips. Matches new_multires output within 1e-4 rel. |

The opaque ButteraugliOpaque::new_with_memory_mode(... MemoryMode::Auto)
resolver picks Strip vs Whole automatically using
[memory_mode::resolve_auto] against vram_cap_bytes().

## Multi-resolution strip walker

new_multires_strip is the constant-VRAM analog of the CPU reference's
default multi-resolution path. It walks the full-res image in
body_h-tall strips like new_strip, but each strip pass also drives
a half-res sibling whose body covers the half-res image rows
body_top_full / 2, body_end_full.div_ceil(2)). The two pipelines run
back-to-back per strip:

1. Upload (ref, dist) sRGB strip planes to the full-res instance.
2. Downsample the full-res linear-RGB strip planes into the half-res
   sibling (the half-res image isn't decoded separately — its content
   comes from the 2× downsample of the full-res linear-RGB).
3. Run the full-res strip pipeline (opsin → freq split → masking → diff).
4. Run the half-res strip pipeline on the populated linear-RGB.
5. Supersample-add the half-res strip diffmap into the full-res strip
   diffmap.
6. Reduce the full-res strip's body rows into the running partials.

The constructor requires body_h even — that keeps every full-res
body_top even and the half-res strip's body aligns to half-res image
rows without sub-pixel drift. For images whose image_h isn't a
multiple of body_h, the last strip's half-res counterpart uses
body_end_full.div_ceil(2) - body_top_full/2 rows so the half-res
image's last row is covered.

Numerical tolerance vs new_multires: per-strip host-side max + p3 / p6
/ p12 reduction order differs from the single fused on-device reduce,
so a small drift is expected. The in-tree tests
(crates/butteraugli-gpu/tests/it/multires_strip.rs) enforce 1e-4 relative tolerance across the
256² → 4000×3000 grid, including non-square aspect ratios and uneven
body sizes.

## Benchmark — multires strip vs multires whole

CUDA, RTX 5070, bench_multires_strip_vs_whole_cuda example, body=256:

| Size | Whole median | Strip median | Speedup | Whole alloc | Strip alloc |
|---:|---:|---:|---:|---:|---:|
| 4 MP (2000×2000) | 76.8 ms | 143.8 ms | 0.53× | 954 MB | 160 MB |
| 12 MP (4000×3000) | 1736.6 ms | 153.6 ms | 11.31× | 2.79 GB | 320 MB |
| 24 MP (6144×4096) | (OOM) | 275.9 ms | — | 5.86 GB | 492 MB |

At 4 MP the strip walker's per-strip kernel-launch overhead dominates
its locality win. At 12 MP and above the whole-image multires path
spills past on-chip caches and L2; the strip path stays inside L2 and
runs an order of magnitude faster. At 24 MP the whole-image path
exceeds the 12 GB consumer VRAM budget; strip is the only viable mode.

CSV: benchmarks/butter_multires_strip_<date>.csv (schema mirrors the
single-res butter_strip_vs_whole_<date>.csv file).

## Metal status

butteraugli-gpu works correctly on Metal out of the box as of
2026-05-27 (Phase 8e.4). The default feature set drops
fast-reduction, so the portable per-thread-partials + finalize
reduction is what ships. Without that flip, the per-octave reduction
relied on Atomic<f32>::fetch_add, which cubecl-wgpu's Metal backend
silently no-ops — every reduction returned zero and every score
collapsed to the default. Root cause + upstream patch in
[../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md (../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md).

Build configurations:

bash
# Default — works on Metal, CUDA, DX12, Vulkan, ROCm
cargo build -p butteraugli-gpu

# CUDA-only with the atomic-add fast path (~2-3× faster reduction step
# but non-deterministic and broken on Metal)
cargo build -p butteraugli-gpu --no-default-features --features cuda,fast-reduction


When the upstream feat/metal-atomic-fix lands, fast-reduction will
work correctly on Metal too; the default-off state stays as a
determinism guard (atomic-add commit order is launch-dependent and
breaks bit-reproducibility).

## See also

- bench_strip_vs_whole_cuda — single-resolution strip vs whole bench.
- PORT_STATUS.md — per-module port status from the cuda reference.
- tests/strip_parity.rs — single-resolution strip parity (19 tests).
- crates/butteraugli-gpu/tests/it/multires_strip.rs — multi-resolution strip parity (11 tests).
- crates/butteraugli-gpu/tests/it/opaque_strip_parity.rs — strip-mode routing through the
  opaque shim (10 tests).

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/butteraugli-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/butteraugli-gpu/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/butteraugli-gpu/LICENSE-libjxl](https://github.com/imazen/zenmetrics/blob/master/crates/butteraugli-gpu/LICENSE-libjxl)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=butteraugli-gpu) · [Speed](https://metric-brief.pages.dev/#speed&q=butteraugli-gpu)

## butteraugli-original

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** distance; emits _max + _pnorm3

**Direction:** not recorded

**Cli contract:** Original Google Butteraugli (revision 2019-03-19); lower is better.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric butteraugli-original --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=butteraugli-original) · [Speed](https://metric-brief.pages.dev/#speed&q=butteraugli-original)

## cambi

NR · Classical · CPU · Not documented

**Paper:** [CAMBI: Contrast-aware Multiscale Banding Index](https://arxiv.org/abs/2102.00079). Tandon, Pulkit; Afonso, Mariana; Sole, Joel; Krasula, Lukáš. arXiv preprint. 2021.

```bibtex
@misc{CAMBI2021,
  title = {CAMBI: Contrast-aware Multiscale Banding Index},
  author = {Tandon, Pulkit and Afonso, Mariana and Sole, Joel and Krasula, Lukáš},
  year = {2021},
  note = {arXiv preprint},
  url = {https://arxiv.org/abs/2102.00079},
  eprint = {2102.00079},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Contrast-aware multiscale banding index, studio-swing BT.601 ingress.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric cambi --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Contrast-aware multiscale banding index, studio-swing BT.601 ingress."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=cambi) · [Speed](https://metric-brief.pages.dev/#speed&q=cambi)

## cambi-aic2026

NR · Classical · CPU · Not documented

**Paper:** [CAMBI: Contrast-aware Multiscale Banding Index](https://arxiv.org/abs/2102.00079). Tandon, Pulkit; Afonso, Mariana; Sole, Joel; Krasula, Lukáš. arXiv preprint. 2021.

```bibtex
@misc{CAMBI2021,
  title = {CAMBI: Contrast-aware Multiscale Banding Index},
  author = {Tandon, Pulkit and Afonso, Mariana and Sole, Joel and Krasula, Lukáš},
  year = {2021},
  note = {arXiv preprint},
  url = {https://arxiv.org/abs/2102.00079},
  eprint = {2102.00079},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** CAMBI with the AIC2026 swscale-compatible integer luma ingress.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric cambi-aic2026 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "format": "packed sRGB8; distorted image only",
  "resize": "crop to even width and height",
  "luma": "t=(8414R+16519G+3208B+(32<<14)+(1<<8))>>9; Y=(t+32)>>6",
  "kernel": "CAMBI 8-bit luma; existing defaults",
  "source": "crates/zenmetrics-cli/src/metrics/vmaf.rs"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-cli/src/metrics/vmaf.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/vmaf.rs) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=cambi-aic2026) · [Speed](https://metric-brief.pages.dev/#speed&q=cambi-aic2026)

## ciede2000

FR · Classical · CPU · Not documented

**Paper:** [The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations](https://doi.org/10.1002/col.20070). Gaurav Sharma; Wencheng Wu; Edul N. Dalal. Color Research &amp; Application. 2004.

```bibtex
@misc{CIEDE20002004,
  title = {The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations},
  author = {Gaurav Sharma and Wencheng Wu and Edul N. Dalal},
  year = {2004},
  note = {Color Research &amp; Application},
  url = {https://doi.org/10.1002/col.20070},
  doi = {10.1002/col.20070}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Mean CIEDE2000 colour difference (ΔE00) — sRGB → linear → XYZ D65
→ CIELAB → ΔE00 per pixel → mean. Matches JPEG AIC-4 CIEDE2000
column. 61-pair median|Δ|=3.0e-5, max|Δ|=2.96e-4 (target ≤ 1e-3).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ciede2000 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Mean CIEDE2000 colour difference (ΔE00) — sRGB → linear → XYZ D65\n→ CIELAB → ΔE00 per pixel → mean. Matches JPEG AIC-4 CIEDE2000\ncolumn. 61-pair median|Δ|=3.0e-5, max|Δ|=2.96e-4 (target ≤ 1e-3)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ciede2000) · [Speed](https://metric-brief.pages.dev/#speed&q=ciede2000)

## ckdn

FR · Learned · CPU · Reported results

**Paper:** [Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment](https://arxiv.org/abs/2108.07948). Zheng, Heliang; Yang, Huan; Fu, Jianlong; Zha, Zheng-Jun; Luo, Jiebo. ICCV 2021. 2021 (arXiv submission year; publication venue may be later).

```bibtex
@misc{ckdn2021,
  title = {Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment},
  author = {Zheng, Heliang and Yang, Huan and Fu, Jianlong and Zha, Zheng-Jun and Luo, Jiebo},
  year = {2021},
  note = {ICCV 2021},
  url = {https://arxiv.org/abs/2108.07948},
  eprint = {2108.07948},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Zheng, Fu, Zeng, Zha, Luo. Learning Conditional Knowledge Distillation for Degraded-Reference Image Quality Assessment. ICCV 2021

**Reference implementation:** https://github.com/researchmm/CKDN

**Reference identity:** git clone of researchmm/CKDN (no pip package; commit tip 2026-10-15)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** unknown

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-closer

**Cli contract:** CKDN (upstream researchmm/CKDN checkpoint, fed as
cat(dist, ref, dist, ref) → dr; upstream eval transform:
short side →329 then centre-crop 288, bicubic for the distorted
image, nearest for the reference); higher is better —
it is a sigmoid quality score, not a distance. As lpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/ckdn/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric ckdn --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "unknown",
  "checkpoint": "https://drive.google.com/file/d/13HzxPsRWdPC_CAsbvgMgU5IG_TKPVbNe/view"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "u8 as f32 (0..255), then (x - mean255) / (std255) inside the graph — upstream's PrefetchLoader does exactly this on the 12-channel input (ImageNet mean/std repeated per 3-channel group)",
  "resize": "eval transform always applied: short side -> 329 (long side floored, torchvision Resize(int) semantics) then CentreCrop(288); bicubic for the distorted image, NEAREST for the reference. This is upstream's own convention: predict_score.py's timm loader runs transforms_imagenet_eval(img_size=288, crop_pct=0.875), whose paired_transforms_tv04.Resize applies interpolation (bicubic per resolve_data_config) to the image and interpolation_tg=Image.NEAREST to the reference target. The AIC-4 published column uses the same convention (matched to median |d| 4e-5 over 61 pairs).",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "researchmm/CKDN checkpoint epoch 14, CPU, fp32",
  "tolerance_jnd": 0.001,
  "jnd": 0.005276,
  "results": "rten worst |d| 2.981e-7 (5.7e-5 JND); ort-cuda worst |d| 2.683e-7 (5.1e-5 JND); 15 goldens under the eval transform"
}
```

**Conventions:**

```json
{
  "aic4": {
    "column": "CKDN",
    "results": "matches: median |d| 4.0e-05, max |d| 1.4e-04 on all 61 crop pairs (this port, ort-cuda). Native feed without the transform was median 3.5e-02 / max 3.7e-01 — the transform IS the published convention",
    "finding": "The published column is the upstream epoch-14 checkpoint under upstream's own eval convention: paired_transforms_tv04.Resize(329, bicubic on the image, interpolation_tg=NEAREST on the reference target) + CentreCrop(288), then cat(d',r',d',r') -> dr as predict_score.py does. torch's antialiased bicubic is Catmull-Rom a=-0.5 (verified vs F.interpolate impulse weights), torch NEAREST is floor(iin/out); both re-implemented in src/resize.rs."
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/ckdn/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/ckdn/card.toml) · [crates/zenmetrics-nn/LICENSE-ckdn](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-ckdn)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ckdn) · [Speed](https://metric-brief.pages.dev/#speed&q=ckdn)

## cvvdp

FR · Classical · CPU · Reported results

**Paper:** [ColorVideoVDP: A visual difference predictor for image, video and display distortions](https://doi.org/10.1145/3658144). Rafal K. Mantiuk; Param Hanji; Maliha Ashraf; Yuta Asano; Alexandre Chapiro. ACM Transactions on Graphics. 2024.

```bibtex
@misc{ColorVideoVDP2024,
  title = {ColorVideoVDP: A visual difference predictor for image, video and display distortions},
  author = {Rafal K. Mantiuk and Param Hanji and Maliha Ashraf and Yuta Asano and Alexandre Chapiro},
  year = {2024},
  note = {ACM Transactions on Graphics},
  url = {https://doi.org/10.1145/3658144},
  doi = {10.1145/3658144}
}
```

**Recorded paper:** Mantiuk et al., ColorVideoVDP, ACM SIGGRAPH 2024

**Reference implementation:** pycvvdp v0.5.7 (PyPI)

**Reference identity:** pycvvdp v0.5.7 (PyPI)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-or-later

**Weights license:** not applicable

**Display:** Required: consult explicit viewing contract

**Output range:** JOD 0–10

**Direction:** not recorded

**Cli contract:** ColorVideoVDP (still-image, JOD scale 0–10, 10 = imperceptible) —
CPU implementation via the in-tree cvvdp crate (native SIMD port,
dispatched through zenmetrics_api::cpu_dispatch / Backend::Cpu).
This is the unsuffixed default, matching the ssim2/dssim/zensim
convention (unsuffixed = CPU); use cvvdp-gpu for the GPU backend.
Emits the CPU column name (cvvdp_cpu_imazen_v), distinct from the
GPU column so sidecars don't collide.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric cvvdp --reference ref.png --distorted dist.png --display-model standard_fhd
```

Choose the display model that matches your viewing condition; standard_fhd is an example, not a default.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 + display preset (default standard_4k)"
}
```

**Parity:**

```json
{
  "reported": "reported gate: ≤ 1e-3 JOD; observed: ~1e-4 rel / 1e-3 JOD",
  "oracle": "pycvvdp v0.5.7 (PyPI)"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/cvvdp/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/cvvdp/README.md)

# cvvdp ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/cvvdp.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/cvvdp?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/cvvdp?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/cvvdp?style=flat-square) ![License](https://img.shields.io/crates/l/cvvdp?style=flat-square)

Pure-Rust CPU port of ColorVideoVDP (https://github.com/gfxdisp/ColorVideoVDP)
(still-image and video scoring). Built on top of the canonical pycvvdp v0.5.7
algorithm (still-image path identical to v0.5.4), designed as a drop-in perceptual metric for the JPEG XL
encoder's iterative quantization loop where the GPU backend's
host-to-device upload latency exceeds CPU compute time.

Companion to cvvdp-gpu (../cvvdp-gpu/) — both crates produce
scalar JOD values within ≤ 1e-3 of each other and of the pycvvdp
v0.5.7 reference.

## What it does

rust
use cvvdp::{Cvvdp, CvvdpParams};

let (w, h) = (256, 256);
let ref_srgb: Vec<u8> = vec![/ wh3 bytes /];
let dist_srgb: Vec<u8> = vec![/ wh3 bytes /];

let mut scorer = Cvvdp::new(w as u32, h as u32, CvvdpParams::default())?;
let jod: f32 = scorer.score(&ref_srgb, &dist_srgb)?;
// JOD ∈ [0, 10]; 10 = imperceptible difference.


Buttloop-style hot path (same reference, many distorted candidates):

rust
scorer.warm_reference(&ref_srgb)?;
for candidate in candidates {
    let jod = scorer.score_with_warm_ref(&candidate)?;
    // ...
}


Per-pixel diffmap for spatial error localization (JPEG XL
quantization refinement):

rust
let mut diffmap = Vec::new();
let jod = scorer.score_with_diffmap(&ref_srgb, &dist_srgb, &mut diffmap)?;
// diffmap.len() == w  h, row-major, contiguous.
// Non-negative; zero where ref == dist; concentrates spatially
// on the distorted region.


Video scoring (streaming; holds only the temporal-filter window of
frames, not the whole clip):

rust
use cvvdp::{DisplayGeometry, VideoScorer};

let mut v = VideoScorer::new(1920, 1080, 30.0, CvvdpParams::default(),
                           DisplayGeometry::STANDARD_FHD)?;
for (ref_frame, dist_frame) in ref_frames.iter().zip(dist_frames.iter()) {
    v.push_frame(ref_frame, dist_frame)?;   // &[u8] sRGB-8, wh3 each
}
let jod = v.finish()?;


A whole-clip convenience cvvdp::score_video(&ref_frames,
&dist_frames, w, h, fps, params, geometry) wraps the same
push/finish loop. cvvdp::score_video_with_stats additionally
accepts a FrameLayout (Interleaved/Planar, the dim_order
analog) and returns VideoStats — the (Q_jod, stats) pair
pycvvdp's predict returns, including loss() = 10 − JOD.
VideoScorer::with_options takes VideoScorerOptions
(layout, temp_padding — Replicate, Symmetric, plus Valid
(upstream lists it but errors at runtime — a deliberate extension:
only complete causal windows emit) — temp_filter (the pycvvdp
temp_filter parameter: Default 5 Hz Gaussian band-pass plus the
HpTrans high-pass and GradTrans two-frame-gradient alternates),
temp_resample (resample q_per_ch onto a nominal fps grid before
pooling — upstream's implementation is dead code; this implements the
intended frame-axis semantics), heatmap (per-output-frame
stats['heatmap']-equivalent planes, drainable via pop_heatmap),
and low_memory, a source-sample ring window
that cuts peak RSS by ~⅓ at 1080p for a few percent CPU; scores
bit-identical).

Stills and video accept three display-encoded sample types, matching
pycvvdp's video_source_array dtypes: u8 (v/255), u16
(v/65535 — score_u16, push_frame_u16, score_video_u16) and
f32 ([0,1] as-is — score_f32, push_frame_f32,
score_video_f32). Under an HDR display model the u16/f32 path is
genuine nit-domain scoring, not an 8-bit upscale — verified on a
committed real HDR10/PQ clip (video_parity u16 cells: max
|Δ| = 2e-6 JOD video, 1.2e-5 stills vs pycvvdp v0.5.7). Mixing
sample types within one scorer errors (Error::MixedSampleTypes).

A 1-frame clip routes through the still path, bit-identical to
Cvvdp::score. See docs/VIDEO.md for the port
design, measured parity, and benchmarks — including the
fast-ssim2-per-frame comparison (wall, user+sys CPU, peak RSS, 1t vs
8t; ~160 ms/frame at 1080p/8t where fast-ssim2 0.9.0 runs ~135 —
roughly at parity through 720p; ~1.0 GB peak RSS, ~0.7 GB with
low_memory; the streaming bound is the temporal-filter window, not
a small footprint).

## Scope

- Still-image and video scoring (temporal channels, transient
  achromatic channel, causal replicate padding — pycvvdp's
  defaults).
- DKLd65 opponent + Weber-contrast pyramid + castleCSF + mult-mutual
  masking + Minkowski pooling (spatial β=2, per-band/channel β=4,
  frames β=2 for video).
- Matches pycvvdp v0.5.7 within ≤ 1e-3 JOD: stills on synthetic
  fixtures 16²–512² (crates/cvvdp/tests/it/parity_against_host_scalar.rs); video on
  44 cells (11 situations × 4 displays) at 24/30/60 fps
  (cvvdp-conformance::video_parity, measured max |Δ| = 3e-6 JOD).

## Why a CPU port

cvvdp-gpu running on CUDA is ~14 ns/px on an RTX 5070. Per-iteration
host-to-device upload at 12 MP is ~12 MB × 2 ≈ 24 MB, which on
PCIe 4.0 at ~20 GB/s is ~1.2 ms even pinned. For JPEG XL's buttloop
at 1024×1024, the GPU compute time is ~30 ms; CPU compute time is
roughly comparable on a 7950X. When compute and upload are at parity
the CPU path wins by eliminating the upload entirely (and the cold
cubecl JIT compile on first call).

This crate is the CPU twin so we can drop the GPU dependency in
contexts (CI, container deploys, embedded) where shipping a CUDA
runtime is impractical.

## Features

| flag         | default | effect |
|--------------|---------|--------|
| std        | on      | enable std-dependent paths |
| alloc      | on      | use the alloc crate (always required) |
| parallel   | on      | per-band rayon parallelism (requires std) |
| pixels     | off     | zenpixels::PixelSlice integration |

no_std + alloc builds work; parallel implies std.

## License

Dual-licensed under either:

- AGPL-3.0-or-later (LICENSE-AGPL3 at the workspace root)
- A commercial license (contact <support@imazen.io>)

## Parity reference

Pinned against pycvvdp v0.5.7 (https://github.com/gfxdisp/ColorVideoVDP/releases/tag/v0.5.7)
(v0.5.4 until 2026-09-23; the still-image code path is numerically identical).
The pinned reference version constant is re-exported as
cvvdp::PYCVVDP_REFERENCE_VERSION.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/cvvdp/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/cvvdp/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=cvvdp) · [Speed](https://metric-brief.pages.dev/#speed&q=cvvdp)

## cvvdp-gpu

FR · Classical · GPU · Not documented

**Paper:** [ColorVideoVDP: A visual difference predictor for image, video and display distortions](https://doi.org/10.1145/3658144). Rafal K. Mantiuk; Param Hanji; Maliha Ashraf; Yuta Asano; Alexandre Chapiro. ACM Transactions on Graphics. 2024.

```bibtex
@misc{ColorVideoVDP2024,
  title = {ColorVideoVDP: A visual difference predictor for image, video and display distortions},
  author = {Rafal K. Mantiuk and Param Hanji and Maliha Ashraf and Yuta Asano and Alexandre Chapiro},
  year = {2024},
  note = {ACM Transactions on Graphics},
  url = {https://doi.org/10.1145/3658144},
  doi = {10.1145/3658144}
}
```

**Recorded paper:** Mantiuk et al., ColorVideoVDP, ACM SIGGRAPH 2024

**Reference implementation:** pycvvdp v0.5.7 + CPU twin

**Reference identity:** pycvvdp v0.5.7 + CPU twin

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** Required: consult explicit viewing contract

**Output range:** JOD 0–10

**Direction:** not recorded

**Cli contract:** ColorVideoVDP (still-image, JOD scale 0–10, 10 = imperceptible) —
GPU implementation via the cvvdp-gpu crate (dispatched through the
umbrella by default; score-pairs uses the typed CvvdpBatchScorer
for instance reuse across pairs). Emits the versioned GPU column
(cvvdp_imazen_v).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features gpu-cvvdp
target/release/zenmetrics score --metric cvvdp-gpu --reference ref.png --distorted dist.png --display-model standard_fhd
```

Requires a GPU backend at run time.
Choose the display model that matches your viewing condition; standard_fhd is an example, not a default.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 + display preset (default standard_4k)"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "pycvvdp v0.5.7 + CPU twin"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/cvvdp-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/cvvdp-gpu/README.md)

# cvvdp-gpu

Multi-vendor GPU implementation of ColorVideoVDP (https://github.com/gfxdisp/ColorVideoVDP)
(still-image mode), built on CubeCL (https://github.com/tracel-ai/cubecl).
One Rust kernel source, runs on:

- CUDA (NVIDIA) via cubecl-cuda
- WGPU (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- HIP (AMD ROCm) via cubecl-hip (hip feature)
- CPU via cubecl-cpu through Cvvdp::compute_dkl_jod_host_pool
  (host-side Minkowski fold) — the GPU pool kernel uses
  Atomic<f32>::fetch_add, which cubecl-cpu 0.10 doesn't support.

Algorithmic parity target is the published
ColorVideoVDP (https://github.com/gfxdisp/ColorVideoVDP) Python
reference v0.5.7 (pinned 2026-09-23; v0.5.4 before that — the
still-image code path is numerically identical between the two), still-image
code path only. Video / temporal
channels (sustained + transient) are intentionally out of scope for
v0 — defer until still-mode parity is locked.

## Single-image usage

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let ref_srgb: Vec<u8> = vec![128u8; (w  h  3) as usize];
let dis_srgb: Vec<u8> = vec![128u8; (w  h  3) as usize];
let jod = cvvdp.score(&ref_srgb, &dis_srgb)?;
println!("JOD = {jod:.4}");
# Ok::<(), cvvdp_gpu::Error>(())


Swap cubecl::cuda::CudaRuntime for cubecl::wgpu::WgpuRuntime to
target Metal / Vulkan / DX12 / WebGPU, or cubecl::hip::HipRuntime
for AMD ROCm. On cubecl::cpu::CpuRuntime use
Cvvdp::compute_dkl_jod_host_pool instead of Cvvdp::score — see
the CPU backend section below.

## Per-pixel diffmap

Cvvdp::score_with_diffmap returns the same JOD scalar as
Cvvdp::score AND fills a caller-owned Vec<f32> with a
per-pixel error signal (row-major, width  height non-negative
f32 values, large where the masked error is large in any DKL
channel or pyramid band). The diffmap is the per-pixel input the
jxl-encoder CVVDP-loop fork uses for its per-block 8×8 median + MAD
heuristic. See src/kernels/diffmap.rs for the recipe and
docs/DIFFMAP_DIVERGENCES.md for the relationship between the
diffmap and the scalar JOD.

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let ref_srgb = vec![128u8; (w  h  3) as usize];
let dis_srgb = vec![100u8; (w  h  3) as usize];

// Pre-allocate once; reuse across many score calls.
let mut diffmap: Vec<f32> = Vec::with_capacity((w  h) as usize);
let jod = cvvdp.score_with_diffmap(&ref_srgb, &dis_srgb, &mut diffmap)?;
assert_eq!(diffmap.len(), (w  h) as usize);
println!(
    "JOD = {jod:.4}, peak pixel error = {:.4}",
    diffmap.iter().cloned().fold(0.0_f32, f32::max)
);
# Ok::<(), cvvdp_gpu::Error>(())


The diffmap is also available from the warm-ref + linear-planes
families (score_with_warm_ref_diffmap,
score_from_linear_planes_with_diffmap,
score_from_linear_planes_with_warm_ref_diffmap). Identical
inputs (ref ≡ dist) produce an all-zero diffmap to 1e-7 absolute
— the buttloop consumer relies on this invariant.

## Linear-RGB planes entry points

For callers that already have linear-light sRGB primaries in planar
f32 (e.g. JPEG XL encoder buttloops),
Cvvdp::score_from_linear_planes (and the warm-ref / diffmap
variants) skips the host-side sRGB pack + sRGB→linear LUT lookup.
Mirrors butteraugli-gpu's
compute_with_reference_from_linear_planes (W44-PHASE3-B4 pattern):

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let n_pix = (w  h) as usize;
let ref_r = vec![0.5_f32; n_pix];
let ref_g = vec![0.5_f32; n_pix];
let ref_b = vec![0.5_f32; n_pix];
let dis_r = vec![0.4_f32; n_pix];
let dis_g = vec![0.5_f32; n_pix];
let dis_b = vec![0.5_f32; n_pix];

let jod = cvvdp.score_from_linear_planes(
    &ref_r, &ref_g, &ref_b, &dis_r, &dis_g, &dis_b,
)?;
# Ok::<(), cvvdp_gpu::Error>(())


The display model (y_peak, y_black, y_refl) and DKL matrix
still apply on GPU — the caller is responsible for the sRGB EOTF
only.

## Display models

DisplayModel carries the per-display photometric + colorimetric
configuration that drives the EOTF, primaries, and ambient-reflection
math:

- y_peak, y_black, y_refl — historical fields; preserved for
  back-compat with v1 callers and the canonical sRGB parity tests.
- eotf — sRGB / PQ / HLG / Linear / BT.1886 / Gamma(g). Defaults
  to Eotf::Srgb.
- primaries — BT.709 / BT.2020 / Display P3 / DCI-P3. Defaults
  to Primaries::Bt709.
- e_ambient_lux, k_refl — ambient illuminance and screen
  reflectivity; the host derives y_refl = e_ambient_lux / π 
  k_refl per cvvdp's vvdp_display_photo_eotf.

Construct from upstream-style parameters:

rust
use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};

let standard_4k = DisplayModel::new(
    200.0,                 // y_peak (cd/m²)
    1000.0,                // contrast (y_black = y_peak / contrast)
    250.0,                 // e_ambient_lux
    0.005,                 // k_refl
    Eotf::Srgb,
    Primaries::Bt709,
);
assert_eq!(standard_4k.y_peak, DisplayModel::STANDARD_4K.y_peak);


Or load a named preset from the vendored display_models.json:

rust
use cvvdp_gpu::params::{DisplayModel, DisplayGeometry, Eotf, Primaries};

let hdr_pq = DisplayModel::by_name("standard_hdr_pq").unwrap();
assert_eq!(hdr_pq.y_peak, 1500.0);
assert_eq!(hdr_pq.eotf, Eotf::Pq);
assert_eq!(hdr_pq.primaries, Primaries::Bt2020);

let geo = DisplayGeometry::by_name("standard_4k").unwrap();
assert_eq!(geo.resolution_w, 3840);


The registry mirrors every preset shipped in pycvvdp v0.5.7 main
(standard_4k, standard_hdr_pq, standard_hdr_hlg,
standard_hdr_linear, standard_fhd, iphone_,
macbook_pro_16, lg_oled_, 65inch_hdr_pq_, …). See
docs/DISPLAY_SPECS.md (docs/DISPLAY_SPECS.md) for the full list
and what each preset routes to.

Note: as of this release the EOTF + primaries dispatch is wired
through the host-scalar pipeline (display_byte_to_dkl_scalar,
display_linear_rgb_to_dkl_scalar in kernels::color). The GPU
fast path (Cvvdp::score, compute_dkl_jod, etc.) still assumes
sRGB + BT.709 for the kernel uploads — it reads the
y_peak/y_black/y_refl fields of the model and ignores
eotf/primaries. To score HDR / wide-gamut inputs on the GPU
path today, convert to linear-BT.709 on the host first and use
the score_from_linear_planes entry point. Full GPU EOTF +
primaries dispatch is queued for the next tick.

## Cached-reference usage

Two ways to amortise the reference-side cost across many distorted
candidates:

- Cvvdp::set_reference + Cvvdp::score_with_reference stashes the
  raw sRGB bytes and re-runs the full pipeline per candidate. Simple
  and matches score(ref, dist) bit-for-bit.
- Cvvdp::warm_reference + Cvvdp::compute_dkl_jod_with_warm_ref
  materialises the REF Weber-contrast pyramid on the GPU once and
  skips that half of the pipeline per candidate — ~1.8× faster per
  DIST at 12 MP (~17-22 ns/px warm-ref on an RTX 5070 post the
  T4.L+M upload-pinning + T1.B/T1.C/T4.K LDS work, vs ~62 ns/px
  on the pre-perf-push baseline; see src/lib.rs's "How we compare
  to the canonical reference" section).

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

let client = CudaRuntime::client(&Default::default());
let (w, h) = (256u32, 256u32);
let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)?;

let reference: Vec<u8> = vec![128u8; (w  h  3) as usize];
cvvdp.warm_reference(&reference)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let jod = cvvdp.compute_dkl_jod_with_warm_ref(&candidate, ppd)?;
    // ... use jod ...
}
# Ok::<(), cvvdp_gpu::Error>(())


The warm-state invalidation contract (which helpers reset the cache
vs preserve it) is pinned by the regression tests
warm_state_invalidates_after_each_documented_dispatcher,
set_reference_does_not_invalidate_warm_state, and
gauss_chain_helpers_do_not_invalidate_warm_state — call any
documented REF-dispatching method between warm_reference and the
warm-ref read and the cache is dropped; cold helpers and
set_reference leave it intact.

## Score interpretation

Output is a JOD (Just-Objectionable-Difference) value on the
ColorVideoVDP 0–10 scale:

- 10.0 = byte-identical (or perceptually indistinguishable)
- 8–10 = high quality, distortions rarely noticed
- 6–8  = noticeable but acceptable distortion
- <6   = clearly visible degradation

Higher = better, matching the pycvvdp Python reference convention.
JOD is the canonical ColorVideoVDP output; score returns it as
f64 and compute_dkl_jod variants as f32.

## CPU backend (cubecl-cpu)

Cvvdp::compute_dkl_jod uses Atomic<f32>::fetch_add in
pool_band_3ch_kernel (the fused 3-channel pool kernel that
production dispatches, one launch per pyramid band), which
cubecl-cpu 0.10 doesn't support. For cpu-runtime use, switch
to Cvvdp::compute_dkl_jod_host_pool (or its warm-ref variant
compute_dkl_jod_host_pool_with_warm_ref) — they read the
per-band D arrays back to the host and reduce with
host_scalar::lp_norm_mean, producing the same JOD at f32
precision. See tests/cpu_backend.rs for direct
cpu-vs-pycvvdp parity coverage on the 73×91 odd-dim fixture.

The same compute_dkl_jod_host_pool path also works on Metal
(wgpu feature) — Metal's Atomic<f32>::fetch_add silently no-ops
on the f32 add, so the GPU pool produces zero. The host-pool path
sidesteps the gotcha.

## Parity vs. perf — PerfMode

CvvdpParams exposes a perf_mode: PerfMode field with two
variants:

- PerfMode::Strict (default) — matches pycvvdp v0.5.4
  bit-for-bit within f32 noise. Every parity test in tests/
  is calibrated against this mode.
- PerfMode::Fast — opt-in entry point for future
  stage-level relaxations that trade measurable per-call cost
  for a bounded JOD drift vs. Strict.

Opt in by overriding the field on CvvdpParams::PLACEHOLDER:

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::{Cvvdp, PerfMode};
use cvvdp_gpu::params::CvvdpParams;

let client = CudaRuntime::client(&Default::default());
let mut cvvdp = Cvvdp::<CudaRuntime>::new(
    client, 256, 256,
    CvvdpParams { perf_mode: PerfMode::Fast, ..CvvdpParams::PLACEHOLDER },
)?;
# Ok::<(), cvvdp_gpu::Error>(())


Today Fast is a no-op (no fast-path has landed yet) — the
variant exists so callers can wire the opt-in once and future
per-stage optimizations gate on params.perf_mode == Fast
without forcing a breaking change. Each Fast-mode optimization
documents its drift budget in CHANGELOG.md. The
perf_mode_fast_matches_strict_today regression test pins the
current bit-pattern-equality contract; when a real Fast-mode
optimization lands the test should be relaxed (not deleted) to
that optimization's drift budget.

## Features

| Feature | Default | Effect |
|---|---|---|
| cuda | yes | Compile the cubecl-cuda backend. |
| wgpu | yes | Compile the cubecl-wgpu backend. |
| cpu  | yes | Compile the cubecl-cpu backend (host-pool only — see above). |
| hip  | no  | Compile the cubecl-hip backend. |
| parity-goldens | no | Compile crates/cvvdp-gpu/tests/it/parity.rs, which fetches the pycvvdp v0.5.4 goldens manifest from R2 and checks JOD parity. Off by default so cargo test stays offline. |

## GPU memory budgeting — concurrency cap for batch sweeps

When running many Cvvdp instances against a shared GPU (one
process per image-size group in a sweep, multiple worker threads
per box, etc.), the crate exposes two helpers that let callers
size PARALLEL against the device's free memory without
reinventing the buffer-accounting math.

- cvvdp_gpu::estimate_gpu_memory_bytes(width, height) -> Option<usize>
  — static-analysis predictor that sums every persistent buffer
  Cvvdp::new allocates: the three pyramids (gauss_ref,
  bands_ref, bands_dis), per-level d_scratch (6 plane types
  × 3 channels), weber_scratch (non-baseband levels only),
  source byte buffers, partials, srgb_lut, logs_row. Uses
  ceil-div halving so it matches the actual allocator layout.
  Returns None below the PYRAMID_MIN_DIM × 2 = 8×8 threshold
  (same precondition as Cvvdp::new).

- cvvdp_gpu::recommend_parallel(free_gpu_bytes, width, height) -> u32
  — bundles the predictor with PARALLEL_SAFETY_FACTOR = 1.5 so
  callers don't maintain the constant themselves. Always returns
  at least 1 if image dims are valid (a single instance gets to
  attempt scoring; OOM after that is a back-off signal, not
  "no work").

Worked example at standard 4K geometry:

| Size            | estimate_gpu_memory_bytes | 8 GB GPU PARALLEL | 24 GB GPU PARALLEL |
|-----------------|-----------------------------|-------------------|--------------------|
|     64×64       | 0.8 MB                      | many              | many               |
|    256×256      | 13 MB                       | 409               | 1228               |
|    512×512      | 52 MB                       | 102               | 307                |
|   1024×1024     | 208 MB                      | 25                | 76                 |
|   2048×2048     | 833 MB                      | 6                 | 19                 |
| 4096×3072 (12MP)| 2.5 GB                      | 2                 | 6                  |

For a typical sweep worker:

rust,no_run
# fn free_gpu_bytes() -> u64 { 0 } // cudarc / wgpu query goes here
use cvvdp_gpu::recommend_parallel;
let parallel = recommend_parallel(free_gpu_bytes(), 1024, 1024);
println!("running {parallel} concurrent Cvvdp instances on this GPU");


examples/cvvdp_mem_table.rs prints the full size-vs-budget
table — useful for choosing PARALLEL for a specific GPU SKU.

The 1.5× safety factor in PARALLEL_SAFETY_FACTOR covers per-call
transient uploads (the per-DIST srgb byte buffer), cubecl runtime
metadata + page alignment, NVRTC PTX cache, and driver-side
reservations. Tighten to ~1.2 when batching with warm_reference
(no per-DIST allocator churn); loosen to ~2.0 if the same process
also runs CPU-side decode/encode in the same memory namespace —
call estimate_gpu_memory_bytes directly and divide yourself.

## Sweep tooling — CVVDP_COLUMN_NAME

The crate exports a stable column-name constant
cvvdp_gpu::CVVDP_COLUMN_NAME for landing scores in parquet
sidecars without colliding with other cvvdp variants (the
canonical pycvvdp reference goes under cvvdp_pycvvdp_v054,
future alternative implementations would get their own tags,
etc.). Default form is cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>;
CI can override the entire string at build time via the
CVVDP_IMPL_TAG env var to bake in a git short hash when
iterating within the same crate version.

The full sidecar schema (identity tuple, score-column type
contract, manifest format, producer / consumer protocols) lives
in docs/CVVDP_SIDECAR_SCHEMA.md (docs/CVVDP_SIDECAR_SCHEMA.md).
A Burn-based port was investigated and abandoned in tick 324
after a perf spike measured 4.32× regression vs. the
hand-written separable kernel; see
docs/BURN_PORT_PLAN.md (docs/BURN_PORT_PLAN.md)'s "Status:
ABANDONED" banner and crates/burn-conv-spike/README.md.

## Status

Still-image score matches pycvvdp v0.5.7 within 0.005 JOD across
q=1–90 fixtures on the v2 R2 goldens manifest (the v1 / v0.5.4 manifest
agrees with v2 to 1.4e-6 JOD). The full pipeline —
display model, DKL color, Weber pyramid, CSF, mult-mutual masking,
3-stage Minkowski pool — runs on GPU; only the final pool fold
and met2jod mapping happen host-side, on a ~144-byte partials
buffer. See docs/PORT_STATUS.md (docs/PORT_STATUS.md) for the
per-kernel breakdown and validated parity numbers, and
docs/CHROMA_DRIFT_INVESTIGATION.md (docs/CHROMA_DRIFT_INVESTIGATION.md)
for the history behind the ticks 191–207 chroma_shift hunt.

## Build

CUDA SDK version requirements depend on the target GPU
architecture, not on cubecl 0.10. cubecl's cudarc 0.19.4
dependency emits dlsym entries gated on a cuda-<MMmmpp> cargo
feature that's auto-selected from the CUDA SDK detected at build
time (cuda-version-from-build-system feature). The selected
feature must match symbols the host's libcuda actually exports:

- RTX 50-series (Blackwell, sm_120) — CUDA 13 SDK required;
  Blackwell sm_120 isn't supported below CUDA 13.
- RTX 20/30/40, A2000, A4000, etc. (Turing through Ada) —
  CUDA 12.6 SDK works and is recommended for production
  deployment. The vast.ai backfill fleet (tick 384+) runs the
  binary built against CUDA 12.6; verified end-to-end on RTX
  2060 SUPER, RTX 3060, RTX A2000 hosts under driver 535+.

Building against CUDA 13 produces a binary that calls
cuCoredumpDeregisterCompleteCallback, a symbol gated behind
cuda-13020 in cudarc but absent from every released NVIDIA
libcuda; that binary panics at first kernel dispatch on every
host. Pick the SDK version that matches the GPU you target.

The cubecl-cuda runtime dynamically loads CUDA via dlopen, so
the build itself succeeds without nvcc on PATH; only the
matching libcuda is needed at runtime:

bash
# RTX 50-series target:
CUDA_PATH=/usr/local/cuda-13 cargo build -p cvvdp-gpu --release

# RTX 20/30/40, A2000, etc. target (recommended for fleet deployment):
CUDA_PATH=/usr/local/cuda-12.6 cargo build -p cvvdp-gpu --release

# Run tests against whichever CUDA + driver is present:
CUDA_PATH=/usr/local/cuda cargo test --release -p cvvdp-gpu


NVRTC headers (cuda-cudart-dev-<MMmm> on Debian/Ubuntu) are
required at runtime on every worker, not just at build:
cubecl emits kernels with #include <cuda_runtime.h> and NVRTC
compiles them at first launch. Missing headers manifest as
Cvvdp::score: image too small for the configured pyramid
errors (the dual-purpose InvalidImageSize variant masks the
NVRTC compile failure). See
scripts/sweep/onstart_cvvdp_backfill_imazen.sh for the apt
install sequence the fleet uses.

For Metal (Apple) or non-NVIDIA GPUs, build wgpu-only:

bash
cargo build -p cvvdp-gpu --no-default-features --features wgpu


### Metal status (2026-05-27, Phase 8e.4)

cvvdp-gpu's production compute_dkl_jod path uses
pool_band_3ch_lds_kernel which commits per-workgroup pool sums via
Atomic<f32>::fetch_add. cubecl-wgpu's Metal backend silently no-ops
that atomic — every band reduction returns zero, the JOD score
collapses to the identity-pair default (~10.0). Metal users MUST
use compute_dkl_jod_host_pool until the upstream
feat/metal-atomic-fix lands. The host-pool variant reads D bands
back to host before pooling with the scalar lp_norm_mean and is
unaffected by the atomic bug. It is the same path used for the
cubecl-cpu backend (which has the same atomic-f32 unsupported
limitation). Root cause + upstream patch in
../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md (../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md).

cvvdp-gpu is deliberately excluded from the metal-tests CI job
(.github/workflows/ci.yml) until the upstream fix lands. Re-add
once the workspace cubecl pin bumps to a fork rev with
feat/metal-atomic-fix.

## Memory modes

cvvdp-gpu exposes the workspace's unified MemoryMode enum but has
no Strip implementation — the spatial-frequency decomposition is
full-image by construction and architecturally blocked at 24 MP
square (see docs/STRIP_PROCESSING.md). MemoryMode::Auto (the
default) resolves to Full; Strip / Tile return
Error::ModeUnsupported. When the image's working set exceeds the
VRAM cap (ZENMETRICS_VRAM_CAP_BYTES env var, default 8 GB), Auto
surfaces Error::TooBigForFull with the gap.

The pre-existing recommend_parallel(free_gpu_bytes, w, h) helper
already handles per-instance VRAM budgeting for parallel sweeps and
is unaffected by the new mode surface.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/cvvdp-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/cvvdp-gpu/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=cvvdp-gpu) · [Speed](https://metric-brief.pages.dev/#speed&q=cvvdp-gpu)

## cw-ssim

FR · Classical · CPU · Not documented

**Paper:** [Complex Wavelet Structural Similarity: A New Image Similarity Index](https://doi.org/10.1109/TIP.2009.2025923). M.P. Sampat; Zhou Wang; S. Gupta; A.C. Bovik; M.K. Markey. IEEE Transactions on Image Processing. 2009.

```bibtex
@misc{CWSSIM2009,
  title = {Complex Wavelet Structural Similarity: A New Image Similarity Index},
  author = {M.P. Sampat and Zhou Wang and S. Gupta and A.C. Bovik and M.K. Markey},
  year = {2009},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2009.2025923},
  doi = {10.1109/TIP.2009.2025923}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Complex-wavelet SSIM: four levels, eight orientations, coarse 7x7 windows.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric cw-ssim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Complex-wavelet SSIM: four levels, eight orientations, coarse 7x7 windows."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=cw-ssim) · [Speed](https://metric-brief.pages.dev/#speed&q=cw-ssim)

## cw-ssim-pyiqa

FR · Classical · CPU · Reported results

**Paper:** [Complex Wavelet Structural Similarity: A New Image Similarity Index](https://doi.org/10.1109/TIP.2009.2025923). M.P. Sampat; Zhou Wang; S. Gupta; A.C. Bovik; M.K. Markey. IEEE Transactions on Image Processing. 2009.

```bibtex
@misc{CWSSIM2009,
  title = {Complex Wavelet Structural Similarity: A New Image Similarity Index},
  author = {M.P. Sampat and Zhou Wang and S. Gupta and A.C. Bovik and M.K. Markey},
  year = {2009},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2009.2025923},
  doi = {10.1109/TIP.2009.2025923}
}
```

**Recorded paper:** Sampat, Wang, Gupta, Bovik, Markey. Complex Wavelet Structural Similarity: A New Image Similarity Index. IEEE TIP 2009

**Reference implementation:** pyiqa 0.1.16 create_metric('cw_ssim'), CPU, metric(dist, ref)

**Reference identity:** pyiqa 0.1.16 create_metric('cw_ssim'), CPU, metric(dist, ref)

**Reference code license:** PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab)

**Rust license:** code: PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); weights: none

**Weights license:** none

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** CW-SSIM as pyiqa computes it (coarsest of 4 levels, 8 orientations,
f32 masks; the AIC2026 CW-SSIM column, higher is better, ≥ 64²).
Needs feature pyiqa-ports; no model files.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports
target/release/zenmetrics score --metric cw-ssim-pyiqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "none"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "luma": "YIQ Y with f32 weights (0.299, 0.587, 0.114) on f32 v/255 widened to f64, x255, round half to even",
  "pyramid": "SCFpyr height 4, 8 orientations; only the coarsest oriented level is scored, every FFT mask rounded to f32 as pyiqa does",
  "min_size": 64
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 create_metric('cw_ssim'), CPU, metric(dist, ref)",
  "goldens": "goldens_ladder.tsv (common/ladder.py, 15 pairs 64x64..2048x1536)",
  "jnd": 0.004347,
  "tolerance_jnd": 0.001,
  "measured_ladder": "1.3e-15 (3.1e-13 JND)",
  "measured_aic2026_61": "2.3e-15 (5.4e-13 JND); pyiqa vs the published column: max 2.2e-16",
  "warm_reference": "caches the size plan (masks, Gaussian) and the reference's bands; bit-identical to cold",
  "recorded_source_comments": "1 JND: pooled median |1 - score| over the 1,008 KonJND-1k pairs, pyiqa CPU (JPEG 0.003649, BPG 0.005758).\nperf_deltas.py, i265 (Core Ultra 7 265K), rten, 1024x1024 ladder pair, cold evaluate per call;\nbenchmarks/iqa_complexity_2026-10-17/measured/pyiqa_ports_i265_1024.{tsv,meta}."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/cw-ssim-pyiqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/cw-ssim-pyiqa/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=cw-ssim-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=cw-ssim-pyiqa)

## deepdc

FR · Learned · CPU · Reported results

**Paper:** [DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator](https://arxiv.org/abs/2211.04927). Zhu, Hanwei; Chen, Baoliang; Zhu, Lingyu; Wang, Shiqi; Lin, Weisi. arXiv preprint. 2022 (arXiv submission year; publication venue may be later).

```bibtex
@misc{deepdc2022,
  title = {DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator},
  author = {Zhu, Hanwei and Chen, Baoliang and Zhu, Lingyu and Wang, Shiqi and Lin, Weisi},
  year = {2022},
  note = {arXiv preprint},
  url = {https://arxiv.org/abs/2211.04927},
  eprint = {2211.04927},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Zhu, Chen, Zhu, Wang, Lin. DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator. arXiv:2211.04927

**Reference implementation:** https://github.com/h4nwei/DeepDC

**Reference identity:** DeepDC_PyTorch==0.2

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** DeepDC (upstream DeepDC_PyTorch 0.2 with its ImageNet input
normalisation on, as the JPEG AIC-4 DeepDC column; no resize); lower
is better. As lpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/deepdc/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric deepdc --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "trunk": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "v / 255, then ImageNet mean/std inside the graph: upstream's use_input_norm=True (its source: \"Correlation: use_norm = True\"; the package default is off; the JPEG AIC-4 DeepDC column needs it)",
  "resize": "none (upstream's demo prepare_image resizes the short side to 224; the model takes any size)",
  "min_size": 16
}
```

**Parity:**

```json
{
  "oracle": "DeepDC_PyTorch 0.2 with use_input_norm=True, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs: rten 1.2e-7, ort-cuda 1.2e-7. Against the\nJPEG AIC-4 cropped-set DeepDC column, 61 pairs: max 3.0e-5."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/deepdc/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/deepdc/card.toml) · [crates/zenmetrics-nn/LICENSE-deepdc](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-deepdc)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=deepdc) · [Speed](https://metric-brief.pages.dev/#speed&q=deepdc)

## dists

FR · Learned · CPU · Reported results

**Paper:** [Image Quality Assessment: Unifying Structure and Texture Similarity](https://arxiv.org/abs/2004.07728). Ding, Keyan; Ma, Kede; Wang, Shiqi; Simoncelli, Eero P.. IEEE TPAMI; preprint 2020. 2020 (arXiv submission year; publication venue may be later).

```bibtex
@misc{dists2020,
  title = {Image Quality Assessment: Unifying Structure and Texture Similarity},
  author = {Ding, Keyan and Ma, Kede and Wang, Shiqi and Simoncelli, Eero P.},
  year = {2020},
  note = {IEEE TPAMI; preprint 2020},
  url = {https://arxiv.org/abs/2004.07728},
  eprint = {2004.07728},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Ding, Ma, Wang, Simoncelli. Image Quality Assessment: Unifying Structure and Texture Similarity. IEEE TPAMI 2020. arXiv:2004.07728

**Reference implementation:** https://github.com/dingkeyan93/DISTS

**Reference identity:** DISTS_pytorch==0.1

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** DISTS (upstream DISTS_pytorch 0.1, no resize); lower is better.
As lpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/dists/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric dists --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "alpha_beta": "DISTS_pytorch 0.1: weights.pt",
  "trunk": "https://download.pytorch.org/models/vgg16-397923af.pth"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "v / 255 (ToTensor); ImageNet mean/std inside the graph",
  "resize": "none (the package's prepare_image default; its resize=True mode is not ported)",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "DISTS_pytorch 0.1, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs (3 images, 64-2048 px), i265:\nrten 1.8e-7, ort-cuda 1.8e-7. Identical pair: rten 0.0 (as upstream),\nort-cuda 1.19e-7 (one f32 step at 1.0)."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/dists/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/dists/card.toml) · [crates/zenmetrics-nn/LICENSE-dists](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-dists)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=dists) · [Speed](https://metric-brief.pages.dev/#speed&q=dists)

## dmm

FR · Learned · CPU · Partial canonical parity

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** https://github.com/Baoliang93/DMM

**Reference identity:** pip DMM_PyTorch (DMM_PyTorch/DMM.py)

**Reference code license:** MIT (Baoliang Chen 2024)

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** DMM — learned FR distance (upstream Baoliang93/DMM, MIT code;
VGG16 L2-pooled taps, per-patch SVD head, lower is better).
Needs feature nn; its only weights are torchvision VGG16
(permissive), so no consent gate.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/dmm/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric dmm --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "torchvision vgg16 IMAGENET1K_V1 (fetched by torchvision)",
  "trunk_url": "https://download.pytorch.org/models/vgg16-397923af.pth",
  "trained_on": "n/a — no fitted parameters"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "v/255 -> [0,1]; upstream's input_norm defaults to off",
  "resize": "prepare_image_adt: if min(h,w) > 128, PIL-bilinear resize short side to max(floor(min/48)32, 128), long side int(newlong/short); <=128 images pass through. The port reproduces Pillow's fixed-point BILINEAR bit-exactly (resize_bilinear_pil_u8 — zenresize has no Pillow-exactness knob: i16@2^14 largest-remainder vs Pillow's 2^22 last-tap fold)",
  "max_side": 4096,
  "max_pixels_64bit": 16777216,
  "max_pixels_32bit": 1048576,
  "max_aspect_ratio": 32,
  "min_size": 128
}
```

**Parity:**

```json
{
  "regression_floor_policy": "Regenerating per-row residual floors requires owner approval. Never increase an existing tolerance or move a jnd row to pinned without explicit approval. Candidate generation: just nn-dmm-netlib-goldens; goldens in this directory are reviewed source artifacts.",
  "oracle": "Frozen per-engine SciPy SGESDD scores on the engine's own ONNX taps; upstream DMM() CPU fp32 MKL scores are recorded separately",
  "tolerance_jnd": 0.001,
  "jnd": "0.191420 (KonJND-1k 1008 pairs, ort-cuda, 2026-10-16; JPEG 0.167535 / BPG 0.218804 — benchmarks/nn_konjnd_scale_2026-10-15.md)",
  "results": "The active score-revision-1 solver passes all 48 frozen row limits. Canonical JND coverage is rten 6/24 and ort-cuda 7/24; the remaining 18/17 rows use owner-approved recorded-residual +5% floors. Those pinned rows are regression controls, not a parity claim. Full-precision native comparisons against upstream MKL: rten median 0.001673475654 / max 0.05674371081; ort-cuda median 0.0007338384395 / max 0.1764253188. No original limit or classification changes. Arithmetic hypotheses are measured individually and qualify only if every frozen limit passes and no row residual increases; all eight candidates are rejected, with complete 48-row comparisons and bit-identical 24-row CUDA repeats. The original numerical section, SVD assertions and frozen goldens are unchanged. Source IDs, per-change results, historical rounded receipts and rejected combined experiments: benchmarks/nn_port_2026-10-15/dmm_fix_2026-10-17.md and dmm_bisect_results_2026-10-17.md."
}
```

**Conventions:**

```json
{
  "aic4": {
    "convention": "AIC-4 2026 publishes a DMM column (metrics_subset.csv); the column is not reproduced here and its evaluation pipeline remains undetermined.",
    "results": "Historical score-revision-1 comparison, not reproduced by the upstream convention — native port vs published on the 61-pair crop set: median |d| 7.47e-2, max 0.417 (benchmarks/dmm_aic_probe/aic_dmm.tsv)."
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/dmm/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/dmm/card.toml) · [crates/zenmetrics-nn/LICENSE-dmm](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-dmm)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=dmm) · [Speed](https://metric-brief.pages.dev/#speed&q=dmm)

## dmm-pyiqa

FR · Learned · CPU · Partial canonical parity

**Paper: citation needed.**

**Recorded paper:** Chen, Zhu, Zhu, Wang, Pan, Wang. Debiased Mapping for Full-Reference Image Quality Assessment. IEEE TMM 2025

**Reference implementation:** pyiqa 0.1.16 create_metric('dmm'), CPU fp32 (torch.svd, MKL), metric(dist, ref)

**Reference identity:** pyiqa 0.1.16 create_metric('dmm'), CPU fp32 (torch.svd, MKL), metric(dist, ref)

**Reference code license:** PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); LAPACK routines BSD-3-Clause (../../LICENSE-lapack)

**Rust license:** code: PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); LAPACK routines BSD-3-Clause (../../LICENSE-lapack); weights: permissive

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-better

**Cli contract:** DMM as pyiqa scores it (square bilinear resize; close to the AIC2026
DMM column, lower is better, ≥ 128²). Needs feature
pyiqa-ports; torchvision VGG16 weights, no consent gate.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports,nn
# export the model once (see crates/zenmetrics-pyiqa-ports/models/dmm-pyiqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric dmm-pyiqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "https://download.pytorch.org/models/vgg16-397923af.pth"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "resize": "min(h,w) > 128: F.interpolate bilinear (align_corners False, no antialias) to a SQUARE tar x tar, tar = max(floor(min/48)32, 128). zenmetrics-nn's dmm keeps the aspect ratio (PIL short-side resize): that is the convention difference behind the AIC2026 column.",
  "min_size": 128
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 create_metric('dmm'), CPU fp32 (torch.svd, MKL), metric(dist, ref)",
  "goldens": "goldens_ladder.tsv (common/ladder.py; pyiqa refuses the three 64x64 pairs: nan)",
  "jnd": 0.178438,
  "tolerance_jnd": 0.001,
  "measured_ladder_rten": "12 pairs scored, 3 refused (64x64, as pyiqa); worst 1.26e-2 (0.071 JND) at 2207_256x256; 256x256 pairs 4.5e-3..1.26e-2, larger sizes 3e-5..1.9e-3: does not meet 1e-3 JND",
  "measured_aic2026_61_rten": "median 2.17e-4, p90 1.36e-3, max 1.94e-3 (0.011 JND): does not meet 1e-3 JND",
  "measured_vs_published_column": "port median 8.77e-3 / max 0.115, Spearman 0.9975; pyiqa CPU itself median 9.05e-3 / max 0.114; zenmetrics-nn dmm median 0.075",
  "blocker": "SVD solver basis. models/dmm-pyiqa/solver_spread.py, 6 AIC2026 pairs, pyiqa's own taps: torch.svd (MKL) vs SciPy gesdd f32 differ by up to 2.4e-4 (1.3e-3 JND), vs gesvd f32 by up to 5.7e-3 (0.032 JND), vs torch f64 by up to 1.5e-4. On the three 256x256 ladder pairs (160x160 after the resize, 4 patches per relu4_3 channel) the spread is larger: gesdd vs MKL up to 4.3e-3, f64 vs MKL up to 2.4e-2, gesvd up to 0.37; the port deltas there (4.5e-3, 1.26e-2, 7.2e-3) lie inside the f64-vs-MKL spread. The port's residual on those pairs (<= 2.1e-4) lies inside the gesdd-vs-MKL spread. wt consumes the singular basis inside degenerate clusters (see zenmetrics-nn models/dmm/card.toml).",
  "warm_reference": "caches every reference patch SVD; the cold path streams them; bit-identical",
  "recorded_source_comments": "1 JND: pooled median score over the 1,008 KonJND-1k pairs, pyiqa CPU (JPEG 0.145006, BPG 0.225096).\nperf_deltas.py, i265 (Core Ultra 7 265K), rten, 1024x1024 ladder pair, cold evaluate per call;\nbenchmarks/iqa_complexity_2026-10-17/measured/pyiqa_ports_i265_1024.{tsv,meta}."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/dmm-pyiqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/dmm-pyiqa/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=dmm-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=dmm-pyiqa)

## dreamsim

FR · Learned · CPU · Reported results

**Paper:** [DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data](https://arxiv.org/abs/2306.09344). Fu, Stephanie; Tamir, Netanel; Sundaram, Shobhita; Chai, Lucy; Zhang, Richard; Dekel, Tali; Isola, Phillip. NeurIPS 2023. 2023 (arXiv submission year; publication venue may be later).

```bibtex
@misc{dreamsim2023,
  title = {DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data},
  author = {Fu, Stephanie and Tamir, Netanel and Sundaram, Shobhita and Chai, Lucy and Zhang, Richard and Dekel, Tali and Isola, Phillip},
  year = {2023},
  note = {NeurIPS 2023},
  url = {https://arxiv.org/abs/2306.09344},
  eprint = {2306.09344},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Steiner, Sundaram et al. DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data. NeurIPS 2023

**Reference implementation:** https://github.com/ssundaram21/dreamsim

**Reference identity:** pip dreamsim 0.21.2 (identical to the repo's dreamsim/ tree)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** unknown

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-better

**Cli contract:** DreamSim ensemble — learned perceptual similarity distance
(upstream ssundaram21/dreamsim, MIT; DINO + CLIP + OpenCLIP
ViT-B/16 ensemble, 1 - cosine, lower is better). Needs
feature nn; its weights state no licence, so
ZENMETRICS_NN_ACCEPT_WEIGHTS=dreamsim.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/dreamsim/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric dreamsim --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=dreamsim. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "unknown",
  "checkpoint": "https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip; plus https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "0..1, then per-extractor stats inside the graph (ImageNet 0.485/0.456/0.406, 0.229/0.224/0.225 for DINO; CLIP 0.48145466/0.4578275/0.40821073, 0.26862954/0.26130258/0.27577711 for both CLIP nets)",
  "resize": "upstream preprocess: PIL/torchvision antialiased bicubic Resize to EXACTLY 224x224 on both axes (anamorphic, aspect ignored — resize.rs::resize_bicubic_aa), then ToTensor",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "pip dreamsim 0.21.2 ensemble, CPU, fp32, upstream preprocess",
  "tolerance_jnd": 0.001,
  "jnd": 0.017641,
  "results": "rten worst |d| 2.064e-7 (1.17e-5 JND); ort-cuda worst |d| 3.577e-7 (2.03e-5 JND); 21 goldens; identical pairs score exactly 0 on both engines"
}
```

**Conventions:**

```json
{
  "aic4": {
    "column": "DreamSim",
    "results": "matches: median |d| 2.9e-07, max |d| 6.1e-07 on all 61 crop pairs (this port, ort-cuda), SRCC 1.0000",
    "finding": "The published column is the raw 1-cosine ensemble distance on the anamorphic 224x224 PIL-bicubic resize; no eval transform. The port's fixed-point Pillow resize (resize.rs::resize_bicubic_pil_u8) is byte-exact vs PIL."
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/dreamsim/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/dreamsim/card.toml) · [crates/zenmetrics-nn/LICENSE-dreamsim](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-dreamsim)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=dreamsim) · [Speed](https://metric-brief.pages.dev/#speed&q=dreamsim)

## dssim

FR · Classical · CPU · Reported results

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** Wang et al. MS-SSIM 2003 (porneL's variant)

**Reference implementation:** dssim-core itself (the canonical impl)

**Reference identity:** dssim-core itself (the canonical impl)

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** 0 best, unbounded

**Direction:** not recorded

**Cli contract:** DSSIM — CPU implementation via the dssim-core crate. Distance metric: 0 = identical.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric dssim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "reported gate: upstream; observed: —",
  "oracle": "dssim-core itself (the canonical impl)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=dssim) · [Speed](https://metric-brief.pages.dev/#speed&q=dssim)

## dssim-gpu

FR · Classical · GPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** Wang et al. MS-SSIM 2003 (porneL's variant)

**Reference implementation:** dssim-core

**Reference identity:** dssim-core

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-DSSIM-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** 0 best, unbounded

**Direction:** not recorded

**Cli contract:** DSSIM — GPU implementation via the dssim-gpu crate
(dispatched through the zenmetrics-api umbrella).
Distance metric: 0 = identical.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features gpu-dssim
target/release/zenmetrics score --metric dssim-gpu --reference ref.png --distorted dist.png
```

Requires a GPU backend at run time.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "dssim-core"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/dssim-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/dssim-gpu/README.md)

# dssim-gpu

Multi-vendor GPU implementation of the DSSIM (Structural Dissimilarity)
perceptual image quality metric, built on
CubeCL (https://github.com/tracel-ai/cubecl). One Rust kernel source,
runs on:

- CUDA (NVIDIA) via cubecl-cuda
- WGPU (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- HIP (AMD ROCm) via cubecl-hip (hip feature)
- CPU fallback via cubecl-cpu for build-checking only — the
  reduction kernels need atomics + CUBE_COUNT, which cubecl-cpu 0.10
  doesn't yet support, so the CPU backend is not a runtime parity
  target. Use the published dssim-core crate as the CPU reference
  instead (that's what the integration tests do).

Algorithmic parity target is dssim-core v3.4 (the canonical Rust
DSSIM crate). At the pyramid level this also matches the existing
dssim-cuda crate — same five scales, same custom-Lab conversion,
same fixed 3×3 Gaussian, same per-scale MAD score. The CUDA-specific
crate stays in the workspace; this one extends reach to AMD / Intel /
Apple / WebGPU without giving anything up on NVIDIA.

## Single-image usage

rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use dssim_gpu::Dssim;

let client = WgpuRuntime::client(&Default::default());
let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;

let ref_srgb: Vec<u8> = vec![0; 256  256  3];
let dist_srgb: Vec<u8> = vec![0; 256  256  3];

let result = d.compute(&ref_srgb, &dist_srgb)?;
println!("DSSIM = {:.6}", result.score);
# Ok::<(), dssim_gpu::Error>(())


## Cached-reference usage

When scoring many distorted candidates against one reference (encoder
rate-distortion search), call set_reference once and then
compute_with_reference per candidate. Skips the reference-side
pyramid + Lab + reference-blur work each call.

rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use dssim_gpu::Dssim;

let client = WgpuRuntime::client(&Default::default());
let mut d = Dssim::<WgpuRuntime>::new(client, 256, 256)?;

let ref_srgb: Vec<u8> = vec![0; 256  256  3];
d.set_reference(&ref_srgb)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let r = d.compute_with_reference(&candidate)?;
    // ... use r.score ...
}
# Ok::<(), dssim_gpu::Error>(())


## Score interpretation

Output is the standard DSSIM scalar: 0 = identical, larger = more
distortion. Mirrors the f64 returned by dssim_core::Dssim::compare.

## Features

| Feature | Default | Effect |
|---|---|---|
| cuda | yes | Compile the cubecl-cuda backend. |
| wgpu | yes | Compile the cubecl-wgpu backend. |
| cpu  | yes | Compile the cubecl-cpu backend (build-only — see above). |
| hip  | no  | Compile the cubecl-hip backend. |
| fast-reduction | no (since 2026-05-27, Phase 8e.4) | Use Atomic<f32>::fetch_add for the per-scale Σ reduction. Verified correct on CUDA / Windows DX12 / HIP; BROKEN on Metal (cubecl-wgpu's Metal backend silently no-ops the atomic — every reduction returns zero, every score collapses to the default value). Off by default since 2026-05-27 so default builds work on Metal out of the box. Opt in for CUDA-only deployments where the ~2-3× reduction-step speedup matters more than reproducibility. Root cause + upstream patch in ../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md (../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md). |

### Metal status

dssim-gpu works correctly on Metal out of the box as of 2026-05-27.
The default feature set drops fast-reduction, so the portable
per-thread-partials + finalize reduction is what ships. Build:

bash
# Default — works on Metal, CUDA, DX12, Vulkan, ROCm
cargo build -p dssim-gpu

# CUDA-only with the atomic-add fast path (~2-3× faster reduction step
# but non-deterministic and broken on Metal)
cargo build -p dssim-gpu --no-default-features --features cuda,fast-reduction


When the upstream feat/metal-atomic-fix lands (CAS-loop lowering for
WGSL atomicAdd<f32> on Metal — tracked in
CUBECL_METAL_ATOMIC_FIX.md (../zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md)),
fast-reduction will work correctly on Metal too; the default-off
state stays as a determinism guard.

## Memory modes

dssim-gpu implements both whole-image and strip processing. MemoryMode::Auto
(default via Dssim::new) picks Full whenever it fits the VRAM cap —
strip mode is 2-5× slower on this crate, so it's only engaged when Full
exceeds the cap. Cap policy: ZENMETRICS_VRAM_CAP_BYTES env var, else
8 GB default.

rust
use dssim_gpu::{DssimOpaque, MemoryMode};

// Force whole-image regardless of cap.
let scorer = DssimOpaque::new_with_memory_mode(
    backend, w, h, params, MemoryMode::Full,
)?;

// Pin an explicit strip body. h_body must be a positive multiple of 16
// (matches Dssim::new_strip's pyramid-alignment contract).
let scorer = DssimOpaque::new_with_memory_mode(
    backend, w, h, params, MemoryMode::Strip { h_body: Some(128) },
)?;


See the workspace README for the cross-crate matrix.

## Status

Initial port from dssim-cuda. See PORT_STATUS.md for the
per-kernel breakdown, runtime coverage, and validated parity numbers.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/dssim-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/dssim-gpu/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/dssim-gpu/LICENSE-dssim-core](https://github.com/imazen/zenmetrics/blob/master/crates/dssim-gpu/LICENSE-dssim-core)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=dssim-gpu) · [Speed](https://metric-brief.pages.dev/#speed&q=dssim-gpu)

## flip

FR · Classical · CPU · Not documented

**Paper:** [FLIP: A Difference Evaluator for Alternating Images](https://doi.org/10.1145/3406183). Pontus Andersson; Jim Nilsson; Tomas Akenine-Möller; Magnus Oskarsson; Kalle Åström; Mark D. Fairchild. Proceedings of the ACM on Computer Graphics and Interactive Techniques. 2020.

```bibtex
@misc{FLIP2020,
  title = {FLIP: A Difference Evaluator for Alternating Images},
  author = {Pontus Andersson and Jim Nilsson and Tomas Akenine-Möller and Magnus Oskarsson and Kalle Åström and Mark D. Fairchild},
  year = {2020},
  note = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
  url = {https://doi.org/10.1145/3406183},
  doi = {10.1145/3406183}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** NVIDIA LDR-FLIP with default viewing conditions (lower is better);
the error map's mean summed in f64, as NVlabs FLIP computes it.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric flip --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "NVIDIA LDR-FLIP with default viewing conditions (lower is better);\nthe error map's mean summed in f64, as NVlabs FLIP computes it."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=flip) · [Speed](https://metric-brief.pages.dev/#speed&q=flip)

## flip-v17

FR · Classical · CPU · Not documented

**Paper:** [FLIP: A Difference Evaluator for Alternating Images](https://doi.org/10.1145/3406183). Pontus Andersson; Jim Nilsson; Tomas Akenine-Möller; Magnus Oskarsson; Kalle Åström; Mark D. Fairchild. Proceedings of the ACM on Computer Graphics and Interactive Techniques. 2020.

```bibtex
@misc{FLIP2020,
  title = {FLIP: A Difference Evaluator for Alternating Images},
  author = {Pontus Andersson and Jim Nilsson and Tomas Akenine-Möller and Magnus Oskarsson and Kalle Åström and Mark D. Fairchild},
  year = {2020},
  note = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
  url = {https://doi.org/10.1145/3406183},
  doi = {10.1145/3406183}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** LDR-FLIP pooled as flip-evaluator 1.7 does (row-major f32 sum), which
reproduces the JPEG AIC-4 FLIP column exactly. Reads low against
flip on large images (about 0.1% at 2048², 1.7% at 4096²).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric flip-v17 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "LDR-FLIP pooled as flip-evaluator 1.7 does (row-major f32 sum), which\nreproduces the JPEG AIC-4 FLIP column exactly. Reads low against\nflip on large images (about 0.1% at 2048², 1.7% at 4096²)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=flip-v17) · [Speed](https://metric-brief.pages.dev/#speed&q=flip-v17)

## fsim

FR · Classical · CPU · Reported results

**Paper:** [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730). Lin Zhang; Lei Zhang; Xuanqin Mou; D. Zhang. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{FSIM2011,
  title = {FSIM: A Feature Similarity Index for Image Quality Assessment},
  author = {Lin Zhang and Lei Zhang and Xuanqin Mou and D. Zhang},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2011.2109730},
  doi = {10.1109/TIP.2011.2109730}
}
```

**Recorded paper:** Zhang et al., IEEE TIP 20(8) 2011, doi:10.1109/TIP.2011.2109730

**Reference implementation:** authors' FR_FSIMc.m

**Reference identity:** authors' FR_FSIMc.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** FSIM / FSIMc (Zhang et al., IEEE TIP 20(8), 2011) — CPU
implementation via the in-tree fsim crate (a port of the
authors' FR_FSIMc.m: log-Gabor phase-congruency + Scharr
gradient similarity, I/Q chroma term). Similarity on ~[0, 1]:
1 = identical (degenerate constant inputs return NaN, like the
reference). One call emits both fsim_imazen_v and
fsimc_imazen_v. Not routed through the umbrella or the
orchestrator (CPU-only, no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric fsim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (FSIMc chroma terms) / luma (-y)"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 5e-5; observed: within gate",
  "oracle": "authors' FR_FSIMc.m"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/fsim/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/fsim/README.md)

# fsim

Pure-Rust CPU implementation of FSIM (feature similarity index)
and FSIMc (chromatic variant), ported for behavioral fidelity to
the authors' reference FR_FSIMc.m (Zhang, Zhang, Mou & Zhang, IEEE
TIP 20(8), 2011) and validated against it under GNU Octave.

- fsim_plane_f32 / fsim_luma8 — grayscale FSIM.
- fsim_rgb8 — returns Scores { fsim, fsimc }: FSIM on the luma
  plane plus the FSIMc chroma term (YIQ, same matrix as the reference).

Pipeline, matching the reference: optional box-filter subsampling
(factor F = max(1, round(min/256)), single shot — F=2 from min-dim
384), Scharr gradients, phasecong2-style log-Gabor
phase-congruency maps (4 scales × 6 orientations, 2D-FFT filterbank),
PC + gradient similarity maps weighted by max(PC1, PC2), FSIMc chroma
term |ISim·QSim|^0.03. Constant inputs return NaN, as the
reference's phase-congruency maps degenerate to 0/0.

The 2D FFT is a self-contained radix-2 + Bluestein implementation
vendored from this repository's crates/hdrvdp/src/fft.rs (same
provenance; HDR-specific helpers omitted).

## Validation

src/tests.rs checks 18 Octave goldens (grayscale, RGB/YIQ, luma;
even/odd dims; subsampled sizes; identical and degenerate inputs) —
see validation/ for the generator and provenance. Scalar and SIMD
tiers are bit-identical by construction (fixed-order f32 elementwise
math, fixed-order f64 pooling).

## Features

- std (default) — std + alloc; without it the crate is no_std
  + alloc.
- avx512 — enable the x86-64 AVX-512 tier at runtime dispatch.
- parallel — rayon row-parallelism.
- _dev — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2-FMA / AVX-512 / NEON / wasm128) via
archmage/magetypes, dispatched on first call.

## Fixed downscale variants

Fsim::with_downscale(Downscale::Factor(NonZeroU32::new(2).unwrap()))
selects a fixed factor for score_rgb8 and generic Metric::evaluate (including
maps). Default and the free scoring functions retain automatic
decimation. Factor 1 disables decimation. All factors use the same reference box
filter and sampling anchor; map geometry records the selected factor.

### Fixed-factor luma-only scoring

Fsim::with_downscale(Downscale::Factor(factor)).score_luma8(...) applies the
same reference box-filter decimation to unrounded BT.601 luma, without computing
chroma planes. It accepts strided RGB8 rows. Downscale::Auto matches the existing
fsim_luma8 function; score_rgb8 continues to return both FSIM and FSIMc.

The fixed_luma_reps example accepts prepared RGB pairs from the complexity
fixture exporter, so counter-delta measurements exclude decoding and tiling.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/fsim/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/fsim/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/fsim/LICENSE-kovesi](https://github.com/imazen/zenmetrics/blob/master/crates/fsim/LICENSE-kovesi)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=fsim) · [Speed](https://metric-brief.pages.dev/#speed&q=fsim)

## fsim-k2

FR · Classical · CPU · Not documented

**Paper:** [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730). Lin Zhang; Lei Zhang; Xuanqin Mou; D. Zhang. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{FSIM2011,
  title = {FSIM: A Feature Similarity Index for Image Quality Assessment},
  author = {Lin Zhang and Lei Zhang and Xuanqin Mou and D. Zhang},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2011.2109730},
  doi = {10.1109/TIP.2011.2109730}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 2. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric fsim-k2 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (FSIMc chroma terms) / luma (-y)",
  "fixed_factor_contract": {
    "kernel_size": 2,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 2.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=fsim-k2) · [Speed](https://metric-brief.pages.dev/#speed&q=fsim-k2)

## fsim-k3

FR · Classical · CPU · Not documented

**Paper:** [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730). Lin Zhang; Lei Zhang; Xuanqin Mou; D. Zhang. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{FSIM2011,
  title = {FSIM: A Feature Similarity Index for Image Quality Assessment},
  author = {Lin Zhang and Lei Zhang and Xuanqin Mou and D. Zhang},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2011.2109730},
  doi = {10.1109/TIP.2011.2109730}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 3. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric fsim-k3 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (FSIMc chroma terms) / luma (-y)",
  "fixed_factor_contract": {
    "kernel_size": 3,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 3.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=fsim-k3) · [Speed](https://metric-brief.pages.dev/#speed&q=fsim-k3)

## fsim-k4

FR · Classical · CPU · Not documented

**Paper:** [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730). Lin Zhang; Lei Zhang; Xuanqin Mou; D. Zhang. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{FSIM2011,
  title = {FSIM: A Feature Similarity Index for Image Quality Assessment},
  author = {Lin Zhang and Lei Zhang and Xuanqin Mou and D. Zhang},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2011.2109730},
  doi = {10.1109/TIP.2011.2109730}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 4. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric fsim-k4 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (FSIMc chroma terms) / luma (-y)",
  "fixed_factor_contract": {
    "kernel_size": 4,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 4.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=fsim-k4) · [Speed](https://metric-brief.pages.dev/#speed&q=fsim-k4)

## fsim-nodownscale

FR · Classical · CPU · Not documented

**Paper:** [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730). Lin Zhang; Lei Zhang; Xuanqin Mou; D. Zhang. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{FSIM2011,
  title = {FSIM: A Feature Similarity Index for Image Quality Assessment},
  author = {Lin Zhang and Lei Zhang and Xuanqin Mou and D. Zhang},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2011.2109730},
  doi = {10.1109/TIP.2011.2109730}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 1. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric fsim-nodownscale --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (FSIMc chroma terms) / luma (-y)",
  "fixed_factor_contract": {
    "kernel_size": 1,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 1.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=fsim-nodownscale) · [Speed](https://metric-brief.pages.dev/#speed&q=fsim-nodownscale)

## fsim-y

FR · Classical · CPU · Not documented

**Paper:** [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730). Lin Zhang; Lei Zhang; Xuanqin Mou; D. Zhang. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{FSIM2011,
  title = {FSIM: A Feature Similarity Index for Image Quality Assessment},
  author = {Lin Zhang and Lei Zhang and Xuanqin Mou and D. Zhang},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2011.2109730},
  doi = {10.1109/TIP.2011.2109730}
}
```

**Recorded paper:** Zhang et al., IEEE TIP 20(8) 2011, doi:10.1109/TIP.2011.2109730

**Reference implementation:** authors' FR_FSIMc.m

**Reference identity:** authors' FR_FSIMc.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** FSIM on unrounded BT.601 luma only (the grayscale path — no I/Q
chroma term). Emits fsimy_imazen_v.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric fsim-y --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (FSIMc chroma terms) / luma (-y)"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "authors' FR_FSIMc.m"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=fsim-y) · [Speed](https://metric-brief.pages.dev/#speed&q=fsim-y)

## gmsd

FR · Classical · CPU · Reported results

**Paper:** [Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index](https://doi.org/10.1109/TIP.2013.2293423). Wufeng Xue; Lei Zhang; Xuanqin Mou; Alan C. Bovik. IEEE Transactions on Image Processing. 2014.

```bibtex
@misc{GMSD2014,
  title = {Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index},
  author = {Wufeng Xue and Lei Zhang and Xuanqin Mou and Alan C. Bovik},
  year = {2014},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2013.2293423},
  doi = {10.1109/TIP.2013.2293423}
}
```

**Recorded paper:** Xue et al., IEEE TIP 23(2) 2014

**Reference implementation:** libgmsd de646c9a (C)

**Reference identity:** libgmsd de646c9a (C)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** 0 best, unbounded

**Direction:** not recorded

**Cli contract:** GMSD (Gradient Magnitude Similarity Deviation, Xue et al. 2014) —
CPU implementation via the in-tree gmsd crate (a port of
libgmsd). Distance: 0 = identical, larger = worse; typical values
0..0.35. Scored on BT.601 8-bit luma at half resolution, as the
reference does. Emits gmsd_cpu_imazen_v. Not routed through the
umbrella or the orchestrator (CPU-only, no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric gmsd --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → luma"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 1e-12 rel (map), 1e-14 mean; observed: ~1e-12",
  "oracle": "libgmsd de646c9a (C)"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/gmsd/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/gmsd/README.md)

# gmsd

Pure-Rust CPU port of GMSD — Gradient Magnitude Similarity Deviation, the
full-reference image quality index by Wufeng Xue, Lei Zhang, Xuanqin Mou and
Alan C. Bovik ("Gradient Magnitude Similarity Deviation: A Highly Efficient
Perceptual Image Quality Index", IEEE Transactions on Image Processing
23(2), 2014).

The port follows libgmsd by Tom Clunie
(<https://github.com/clunietp/libgmsd>, commit de646c9a), a portable C
implementation of the authors' GMSD.m. libgmsd is MIT-licensed; its
copyright and licence notice are kept in LICENSE-libgmsd (LICENSE-libgmsd)
and at the head of src/kernel.rs (src/kernel.rs), which carries the ported
arithmetic.

## Algorithm

1. 2×2 average, keep every second sample (half resolution).
2. Prewitt gradient magnitude on both images ([1 1 1; 0 0 0; −1 −1 −1]/3
   and its transpose, zero-padded conv2 'same').
3. Per-sample similarity GMS = (2·m_r·m_d + c) / (m_r² + m_d² + c).
4. Deviation pooling: the score is the sample standard deviation of the
   GMS map. 0 = identical; larger = worse.

There are no trained parameters. The only constant is c.

### Which c

c = 170 on the 0..255 intensity scale. This is what the authors' GMSD.m
and libgmsd use. The paper states c = 0.0026 for intensities normalised to
0..1; 0.0026 · 255² = 169.065, so the two are the same constant to 0.55 %.
The port follows the reference implementations (170) so that its output is
comparable with published GMSD numbers and bit-comparable with libgmsd.

## API

rust
let r = gmsd::GrayImage::new(&ref_plane, width, height, stride)?; // f32, 0..255
let d = gmsd::GrayImage::new(&dist_plane, width, height, stride)?;
let score = gmsd::gmsd(r, d)?;                 // .gmsd (deviation), .mean_gms
let mut map = vec![0.0f32; (width / 2)  (height / 2)];
let score = gmsd::gmsd_with_map(r, d, &mut map)?; // + the GMS map


- Input is any strided f32 gray plane. gmsd_rgb8 / rgb8_to_gray convert
  sRGB8 the way libgmsd's command-line tool does:
  round(0.299·R + 0.587·G + 0.114·B).
- Feature pixels: gmsd_pixels scores any zenpixels PixelSlice, converted
  row by row to sRGB8 by zenpixels-convert and then to the same 8-bit luma.
- The GMS map is a spatial map at half resolution (map_dims(w, h)).
- GmsdScore::mean_gms is the map mean (the paper's GMSM), free alongside.
- Feature parallel: rayon over 64-row bands. The score is bit-identical at
  every thread count (per-row sums are combined in row order).

## Parity with libgmsd

On even-sized input the GMS map is bit-identical to libgmsd's: every
expression reproduces libgmsd's float evaluation order with no FMA, and the
unit tests pin the fast path to a straight-line transcription of it. The score
differs only by f64 rounding (one-pass shifted accumulation instead of
libgmsd's two-pass mean/variance). The measured record over real image pairs
is benchmarks/gmsd_parity_2026-09-22.md.

Deliberate difference: on odd width or height the half-resolution grid is
⌊w/2⌋ × ⌊h/2⌋ — the size libgmsd documents — and the trailing row/column is
dropped. (libgmsd's downsample_2x2 writes one element past that allocation
on odd input; GMSD.m instead keeps an extra half-zero-padded sample.)

## SIMD

One #[arcane] entry per tier (v4 AVX-512 behind crate feature avx512,
v3 AVX2, neon, wasm128, scalar) through archmage; the per-row helpers
(sRGB8 → gray + 2×2 decimation; gradient + GMS + pooling sums) are magetypes
#[rite] variants of the same tier, inlined into it. Every tier is
bit-identical to the scalar tier (tested).

The sRGB8 conversion is integer SIMD: for S = 299·R + 587·G + 114·B,
(S + 499) / 1000 equals round(0.299·R + 0.587·G + 0.114·B) everywhere
except S ≡ 500 (mod 1000), where the f64 sum lands either side of the
half-integer — those rare lanes are recomputed in f64, so the result is
bit-identical to libgmsd (proved exhaustively over all 2^24 triplets). The
2×2 decimation then sums the four integer luma values and multiplies by 0.25
once — bit-identical to libgmsd's ordered f32 chain because every partial sum
is a multiple of 0.25 ≤ 255.

gmsd_rgb8 never materialises full-size gray planes: each band converts and
decimates only the input rows it reads, in one fused pass, in its own tier
and thread. Published speed numbers come from zensim's ssim2_speed_bar
owner, not from examples/split_timing.rs diagnostics.

## MDSI (quarantined development addition)

mdsi_rgb8(reference, distorted, width, height, stride_bytes) returns the
default summation MDSI distance as Result<f64>. Inputs are interleaved
gamma-encoded RGB8 with a byte stride; the scorer performs no colour
management, orientation or alpha processing. The caller supplies RGB in
the intended common colour space. The CLI exposes --metric mdsi under
the existing cpu-gmsd feature for score and score-pairs.

MDSI is implemented from the paper (Nafchi, Shahkolaei, Hedjam and Cheriet,
"Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image
Quality Evaluator", IEEE Access 4, 2016) and validated against scores computed
by the authors' reference software, which is not included here or used at run
time. Where the paper is silent, docs/MDSI_CHOICES.md (docs/MDSI_CHOICES.md)
records the reading taken and why. On 116 image pairs (odd sizes, sub-64 px,
1 px, and averaging factors 1 to 4) the score agrees with the authors' scores
to a relative difference of 4.84e-10 at most, inside the 1e-9 gate; a wrong
constant fails 109 of the 116 (the other 7 are exact-zero pairs no constant
moves). The model uses unrounded f64 luminance and two opponent channels,
size-dependent box filtering with zero-padded odd edges, a fused-luminance
gradient term, chromaticity similarity with C3=550, and principal complex
fourth roots for negative combined similarities in the pooling.

The box average is the only full-resolution work (the averaged planes are about
256 samples on the short side), and it runs in exact integer arithmetic. The
score is bit-identical to the straight-line evaluation of the paper's equations
kept in src/mdsi.rs (reference) at every archmage tier and every thread
count. The 116-pair gate is an ordinary test that the caller selects with
GMSD_MDSI_GATE=require|skip and GMSD_MDSI_TARGETS=<table> (see the
workspace justfile).

ms_gmsd_rgb8 and ms_gmsdc_rgb8 are quarantined paper-derived variants
of Zhang et al.'s four-scale masked GMSD and colour extension. They use
unrounded f64 YIQ, zero-padded Prewitt with c=170, replicated odd-edge 2x2
averaging, population deviation, and scale-3 I/Q RMSE. These conventions
are explicit choices where the paper is underspecified. Independent
NumPy qualification is pending; no author-software parity is claimed.

## License

AGPL-3.0-only OR the Imazen commercial license, like the rest of zenmetrics.
The ported libgmsd arithmetic remains available under its MIT licence
(LICENSE-libgmsd).
MDSI is written from the paper; see the attribution above.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/gmsd/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/gmsd/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/gmsd/LICENSE-libgmsd](https://github.com/imazen/zenmetrics/blob/master/crates/gmsd/LICENSE-libgmsd)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=gmsd) · [Speed](https://metric-brief.pages.dev/#speed&q=gmsd)

## haarpsi

FR · Classical · CPU · Reported results

**Paper:** [A Haar wavelet-based perceptual similarity index for image quality assessment](https://doi.org/10.1016/j.image.2017.11.001). Rafael Reisenhofer; Sebastian Bosse; Gitta Kutyniok; Thomas Wiegand. Signal Processing: Image Communication. 2018.

```bibtex
@misc{HaarPSI2018,
  title = {A Haar wavelet-based perceptual similarity index for image quality assessment},
  author = {Rafael Reisenhofer and Sebastian Bosse and Gitta Kutyniok and Thomas Wiegand},
  year = {2018},
  note = {Signal Processing: Image Communication},
  url = {https://doi.org/10.1016/j.image.2017.11.001},
  doi = {10.1016/j.image.2017.11.001}
}
```

**Recorded paper:** Reisenhofer, Bosse, Kutyniok & Wiegand, Signal Processing: Image Communication 61 (2018) 33–43, doi:10.1016/j.image.2017.11.001

**Reference implementation:** authors' haarpsi.m

**Reference identity:** authors' haarpsi.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** HaarPSI (Reisenhofer et al., Signal Processing: Image
Communication 61, 2018) — CPU implementation via the in-tree
haarpsi crate (a port of the authors' MIT-licensed
HaarPSI.m). Similarity on ~[0, 1]: 1 = identical (degenerate
zero-weight inputs return NaN, like the reference). Scored on the
sRGB pair through the reference's YIQ color path with the default
subsampled preprocessing. Emits haarpsi_imazen_v. Not routed
through the umbrella or the orchestrator (CPU-only, no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric haarpsi --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "RGB (YIQ chroma) / luma (-y)"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 5e-5; observed: ~1e-5",
  "oracle": "authors' haarpsi.m"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/haarpsi/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/haarpsi/README.md)

# haarpsi

Pure-Rust CPU port of HaarPSI — the Haar wavelet perceptual
similarity index of R. Reisenhofer, S. Bosse, G. Kutyniok, T. Wiegand,
"A Haar Wavelet-Based Perceptual Similarity Measure for Images", Signal
Processing: Image Communication 61 (2018).

The port follows the authors' MIT-licensed reference implementation
HaarPSI.m (<https://github.com/rgcda/haarpsi>, redistributed under
validation/ with its license) and is verified against it under GNU
Octave (see validation/ (validation/)).

## Algorithm

1. Optional preprocessing (preprocessWithSubsampling, the reference's
   third argument, default on): conv2(X, ones(2,2)/4, 'same') then
   decimation at the odd indices — a forward-looking 2×2 box mean with
   zero-padded borders, matching MATLAB's even-kernel same anchor.
2. sRGB → YIQ for color input (Y = 0.299R + 0.587G + 0.114B, the
   reference's I and Q rows) — no quantization.
3. Three-scale Haar decomposition in horizontal and vertical
   orientations; each filter is a rank-1 box-sum × strip-difference, so
   the whole pass is separable stencil work on padded planes.
4. Luma similarity (2|a||b| + C)/(|a|² + |b|² + C) on scales 1–2
   (C = 30), averaged over the two scales and both orientations; weights
   max(|a|, |b|) from scale 3.
5. Color: the same similarity on the 2×2-smoothed |I| and |Q|
   planes, pooled with the luma orientation weights.
6. Logistic pooling Σ sigmoid(sim·α)·W / ΣW (α = 4.2), inverse
   logistic, squared. An all-zero pair has zero total weight and yields
   NaN — the reference's behavior.

## API

rust
let s = haarpsi::haarpsi_rgb8(&ref_rgb, &dist_rgb, w, h, w  3)?;
// s ≈ 1.0 for identical images (the logistic inverse is exact at sim = 1)


- haarpsi_plane_f32(r, d, w, h, stride) — strided f32 planes on the
  0..255 scale, reference default preprocessing.
- haarpsi_plane_f32_opts(.., subsample) — explicit preprocessing flag
  (the reference's third argument).
- haarpsi_rgb8(r, d, w, h, stride) — interleaved sRGB via the YIQ
  color path.
- haarpsi_luma8(r, d, w, h, stride) — sRGB → unrounded BT.601 luma
  through the grayscale path (no color channel).

## Determinism

Every map value is a fixed-order f32 expression and the final pooling
accumulates f64 in a fixed lane-grouped order, so every SIMD tier
(scalar, v3 AVX2, v4 AVX-512 behind avx512, neon, wasm128)
and every parallel thread count produces bit-identical output.
libm covers log/exp so cross-platform agreement holds.

Features: std (default), parallel (rayon row parallelism),
avx512, _dev (per-tier entry points for parity tests/disassembly).

## Verification

src/tests.rs pins 16 golden rows produced by HaarPSI.m under GNU
Octave on deterministic integer patterns: grayscale with and without
subsampled preprocessing, odd sizes, identical inputs, constant and
zero-weight degenerate inputs, the RGB/YIQ path, and the luma path.
All pass within 5e-5 — the residual is the port's f32 plane pipeline vs
the reference's f64.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/haarpsi/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/haarpsi/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=haarpsi) · [Speed](https://metric-brief.pages.dev/#speed&q=haarpsi)

## haarpsi-y

FR · Classical · CPU · Not documented

**Paper:** [A Haar wavelet-based perceptual similarity index for image quality assessment](https://doi.org/10.1016/j.image.2017.11.001). Rafael Reisenhofer; Sebastian Bosse; Gitta Kutyniok; Thomas Wiegand. Signal Processing: Image Communication. 2018.

```bibtex
@misc{HaarPSI2018,
  title = {A Haar wavelet-based perceptual similarity index for image quality assessment},
  author = {Rafael Reisenhofer and Sebastian Bosse and Gitta Kutyniok and Thomas Wiegand},
  year = {2018},
  note = {Signal Processing: Image Communication},
  url = {https://doi.org/10.1016/j.image.2017.11.001},
  doi = {10.1016/j.image.2017.11.001}
}
```

**Recorded paper:** Reisenhofer, Bosse, Kutyniok & Wiegand, Signal Processing: Image Communication 61 (2018) 33–43, doi:10.1016/j.image.2017.11.001

**Reference implementation:** authors' haarpsi.m

**Reference identity:** authors' haarpsi.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** HaarPSI on unrounded BT.601 luma only (the grayscale path — no
I/Q color similarity). Emits haarpsiy_imazen_v.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric haarpsi-y --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "RGB (YIQ chroma) / luma (-y)"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "authors' haarpsi.m"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=haarpsi-y) · [Speed](https://metric-brief.pages.dev/#speed&q=haarpsi-y)

## hdrvdp

FR · Classical · CPU · Reported results

**Paper:** [HDR-VDP-2: a calibrated visual metric for visibility and quality predictions in all luminance conditions](https://doi.org/10.1145/1964921.1964935). Rafat Mantiuk; Kil Joong Kim; Allan G. Rempel; Wolfgang Heidrich. ACM SIGGRAPH 2011 papers. 2011.

```bibtex
@misc{HDRVDP22011,
  title = {HDR-VDP-2: a calibrated visual metric for visibility and quality predictions in all luminance conditions},
  author = {Rafat Mantiuk and Kil Joong Kim and Allan G. Rempel and Wolfgang Heidrich},
  year = {2011},
  note = {ACM SIGGRAPH 2011 papers},
  url = {https://doi.org/10.1145/1964921.1964935},
  doi = {10.1145/1964921.1964935}
}
```

**Recorded paper:** Mantiuk et al., HDR-VDP-2, ACM TOG 30(4) 2011

**Reference implementation:** official HDR-VDP 2.2.2 MATLAB release (ISC-licensed)

**Reference identity:** official HDR-VDP 2.2.2 MATLAB release (ISC-licensed)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** JOD / P_det / Q

**Direction:** not recorded

**Cli contract:** HDR-VDP 2.2.2 (JOD scale 0–100, 100 = identical) — CPU implementation
via the in-tree hdrvdp crate (native SIMD port of the official
HDR-VDP-2.2.2; dispatched through zenmetrics_api::cpu_dispatch /
Backend::Cpu). 8-bit sRGB pairs use upstream's own sRGB-display
encoding, which hdrvdp.m documents for standard (LDR) colour images:
the sRGB EOTF onto a 100 cd/m² peak, 1 cd/m² black display. Absolute
luminance (cd/m², ColorEncoding::RgbBt709 inside) goes through
score-pairs --hdr / sweep --hdr. Viewing geometry defaults to
DEFAULT_PIX_PER_DEG (30 — the measured UPIQ protocol). Emits
hdrvdp_imazen_v. No GPU twin.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric hdrvdp --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "absolute luminance cd/m² + display params; 8-bit sRGB through upstream's sRGB-display encoding (sRGB EOTF onto a 100 cd/m² peak, 1 cd/m² black display) at 30 ppd, checked against the official release in crates/hdrvdp/tests/golden_srgb_display.rs"
}
```

**Parity:**

```json
{
  "reported": "reported gate: P_det 1e-4; JOD ~1e-2 cases; observed: 1.4e-5 P_det",
  "oracle": "official HDR-VDP 2.2.2 MATLAB release (ISC-licensed)"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/hdrvdp/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/hdrvdp/README.md)

# hdrvdp

A pure-Rust CPU port of HDR-VDP-2.2 (Mantiuk, Kim, Rempel &
Heidrich, SIGGRAPH 2011; quality recalibrated by Narwaria et al., JEI 2015) —
f32 planes and intermediates, f64 scalar tables and reductions — plus
HDR-VDP-3.0.7 in hdrvdp::v3 (display-adaptive revision; f64
end-to-end, see below).

HDR-VDP-2 predicts two things for a pair of images given in absolute
luminance: visibility — the per-pixel probability that a human notices a
difference — and quality — a single MOS correlate. It is the field's reference
HDR metric, and the bar every HDR feeding experiment in this repo has been
measured against:

| dataset | HDR-VDP-2 | best in this workspace |
|---|--:|--:|
| AIC-HDR2025 (HDR compression JND) | 0.936 | SSIMULACRA2 0.906 |
| UPIQ (380 HDR compression pairs, JOD) | 0.812 | this crate 0.820 |

AIC-HDR2025: Jenadeleh, Sneyers, Lazzarotto, Mohammadi, Keller, Boev, Ramachandra
Rao, Pinheiro, "Fine-Grained HDR Image Quality Assessment From Noticeably
Distorted to Very High Fidelity", QoMEX 2025 (arXiv:2506.12505). UPIQ:
Mikhailiuk, Pérez-Ortiz, Yue, Suen, Mantiuk, "Consolidated Dataset and Metrics
for High-Dynamic-Range Image Quality", 2021 (arXiv:2012.10758).

## Status — chunks 1–4 of 6, plus an optimisation pass: scores, validated against official 2.2.2 and on UPIQ

Landed in chunks (tracked in [imazen/zenmetrics#50]). hdrvdp::hdrvdp() now takes
two images in absolute luminance and returns the official res.Q (100 =
identical), the removed-upstream Q_MOS (see below), and the per-pixel
visibility maps P_map / C_map; hdrvdp::score() returns res.Q alone and
skips the visibility-map reconstruction (~40% cheaper for sweep workloads).
Runtime dependencies are archmage/magetypes for the SIMD kernels (runtime
dispatch — scalar/wasm128/neon/v3/v4x tiers keep i686 / wasm /
windows-arm portable) plus an opt-in parallel feature (rayon) that
parallelises the per-image pipelines, per-level orientations, FFT row/column
passes and masking planes — deterministic at any thread count.

Measured at 1024×1024 (stage_probe, luminance, 32-core box):

| path | default | parallel |
|---|---:|---:|
| hdrvdp() (full HdrVdpResult) | ~780 ms | ~330 ms |
| score() (res.Q only) | ~520 ms | ~185 ms |

From a scalar-f64 1.85 s/pair at the same size — ~10× on the score path.

| stage | module |
|---|---|
| the five colour encodings → cd/m², plus the "you passed relative values" detector | display |
| optical MTF, neural CSF, rod/cone contrast-versus-intensity | csf + fft |
| cone fundamentals, scotopic V′, display emission, the LMSR mixing matrix | spectral |
| luminance → JND-space photoreceptor response | photoreceptor |
| MTF → LMSR → adapting luminance → non-linearity → cone/rod DC removal | pathway |
| sp3 steerable pyramid (4 orientations), build + reconstruct | spyr |
| the [1, 4×H, 1] band list + base-band CSF filtering | bands |
| MATLAB-compatible imresize (bicubic, antialiased) | resize |
| mutual masking, the three masking terms, the transducer, per-band D | masking |
| S_map → P_map / P_det / C_max, and Q → res.Q / Q_MOS | pool |
| the end-to-end entry point | metric |

Official validation (2026-09-24/25): run against goldens generated by the
official hdrvdp-2.2.2 MATLAB release under Octave 11.1 (24 synthetic
luminance cases, two sizes × two pixels_per_degree). Worst-case deltas on
the current f32 pipeline: P_det 1.4e-5, C_max 1.0e-4 relative,
P_map 1.7e-2 absolute, res.Q 7.8e-4 — diffuse quantisation noise,
far below the metric's JND resolution (the earlier f64 build agreed to
~1e-11 / res.Q 9.9e-7 before the precision switch).
See docs/VALIDATION.md (docs/VALIDATION.md) for the corpus, the tolerances,
and the upstream Octave is_mex bug the reference run had to work around.

UPIQ real-corpus validation (2026-09-24, chunk 4): all 380 HDR pairs
(upiq_score example, luminance encoding, full resolution, fixed
pix_per_deg=30 — the protocol that reproduces the released score
column; the subjective CSV's own pix_per_deg lands ~12 points low).
Ours vs the official per-condition HDRVDP2_2 column: SROCC 0.9962 /
PLCC 0.9984, delta +0.046±0.363 — korshunov is essentially exact
(median |delta| 0.013); the narwaria residual concentrates in one n-i07
reference family (~+2, a presumed upstream protocol detail). Ours vs JOD:
SROCC 0.8203 vs the official column's 0.8117 on the same join —
clears the published 0.812 bar. Full record:
benchmarks/hdrvdp_upiq_2026-09-24.md (../../benchmarks/hdrvdp_upiq_2026-09-24.md).

Since chunk 4a (a11bcad4, first end-to-end scores on real pixels) the crate
has taken an optimisation pass — FFT plans with hoisted/shared twiddles and
chirp, corr_dn reflect-pad-once with dense blocked correlation, up_conv
pad-once with tap lists and 8-wide output blocking, box3x3 interior fast path,
imresize tap slicing, pair-shared MTF filter and a memoised LUT range
(5c156c2e, 2447406a, 1223a583, 5cc2c63d, 624615e8, 2443d88c). That
pass was fenced by a byte-exact output lock taken before any of it
(f4a380e9) plus an FFT lock extended to 2048-point transforms (dec95f49); a
second pass moved planes/intermediates to f32 for fleet throughput — the
official-golden tolerance test is now the correctness gate and the bit lock was
retired. A third pass added the magetypes SIMD kernels (masked transducer +
psychometric reshape, squared-error reduction, LUT/log10 lookups, a batched
SoA f32x8 FFT with conj/filter/scale fused into its transposes), the
mirror-symmetric MTF filter build, and the opt-in parallel feature. A
zenbench suite
and the full perf record live in benchmarks/ (4cf99288).

Done (2026-09-24): umbrella wiring landed as MetricKind::Hdrvdp
(feature cpu-hdrvdp, in cpu-metrics) — zenmetrics-api
cpu_dispatch scores absolute-nits interleaved linear-RGB via
ColorEncoding::RgbBt709 at DEFAULT_PIX_PER_DEG, and the CLI's
--hdr sweep/batch path reaches it through HdrScorer →
Backend::Cpu (32244668). GPU is not currently planned — the
parallel CPU path is the fleet-sweep answer.

### Three things to know before reading a number

- q is official res.Q, q_mos is the score upstream removed. res.Q =
  100 − Σ(log(msre+ε)−log ε)·w_f is the 2.2.2-published correlate (100 =
  identical, lower = worse, may go negative). q_mos reproduces the logistic
  upstream commented out in 2.2.1 — 100/(1+exp(3.455·(Q_raw+0.8886))) on the
  raw accumulator — which saturates at ≈4.4 for identical pairs and ≈0 for
  anything visibly distorted. Upstream dropped it as unreliable; prefer q.
- After spatial pooling, C_max is a total, not a peak. S_map ·= sum/max
  makes the new maximum equal the old sum, so a C_max of 400 does not mean
  "400× threshold". P_map's spatial shape is unaffected.
- An identical pair scores res.Q = 100 exactly — every msre is 0 so
  every (log(0+ε) − log ε) term vanishes.

## HDR-VDP-3 (hdrvdp::v3)

v3 is the display-adaptive third revision — a different pipeline (sp0
isotropic pyramid, photoreceptor LUTs per band, per-band mutual masking,
task-dependent modelling) on f64 end-to-end, reimplemented from the
jpeg-ai-qaf VDP3/ numpy port @ 0628a6b and verified against the
official MATLAB HDR-VDP-3.0.7 release (bit-equal at print precision on
AIC-4 pairs; 21-case golden corpus worst |ΔQ_JOD| = 1.6e-13 covering
every encoding × task × emission × surround combination).

Two API differences from v2 are deliberate:

- No default viewing geometry. v3::Params requires
  ViewingConditions — pix_per_deg (there is no v3 default; the score
  moves materially with it), Surround (none/mean/absolute cd/m²)
  and observer age — plus an explicit Task (quality/side-by-side/
  flicker), InputEncoding (rgb-bt709/rgb-bt2020/rgb-native/xyz/
  luminance) and Emission (preset or custom spectra).
- Output is q_jod on [0,10] (10 = identical) — the upstream
  Q_JOD, not v2's res.Q [0,100]. The CLI column
  (hdrvdp3_jod_imazen_v) keeps them distinct.

CLI: --metric hdrvdp3 --hdr --hdrvdp3-ppd <f64> (required) with
--hdrvdp3-task quality|side-by-side|flicker,
--hdrvdp3-input rgb-bt709|rgb-bt2020|rgb-native|xyz,
--hdrvdp3-emission ccfl-lcd|crt|led-lcd|led-lcd-srgb|led-lcd-wcg|oled|d65,
--hdrvdp3-age <u8> (24) and --hdrvdp3-surround none|mean|<cd/m²> (none).
Defaults match the jpeg-ai-qaf harness configuration.

The v3 Fourier path reuses one f64 plan per axis and shares those plans
between convolution's forward and inverse transforms. Bluestein chirps,
filter transforms and scratch are reused without changing operation order.
Exact comparisons cover repeated plans, arbitrary lengths and 2-D row/column
ordering; the reference goldens retain their existing tolerances.
Measurements and commands (../../benchmarks/ranked_cpu_2026-10-12.md#hdr-vdp-3).

Caveats carried over from the reference (full record in
DIVERGENCES.md (../../DIVERGENCES.md)): the 'srgb-display' encoding
string is case-sensitive inconsistently upstream; ignore_freqs_lower_than
crashes the reference (D unbound) — ours handles it; the published
AIC-4 HDR_VDP_3 column is not reproduced under any recovered config
(we match MATLAB 3.0.7 exactly — the column is a different run config).

## Units

Everything is absolute luminance in cd/m² (nits) — the same currency
zenmetrics_api::hdr speaks. Feeding relative or normalised values scores at the
wrong adaptation level and silently produces wrong answers;
display::looks_relative catches the common form of that mistake.

## Licensing

Ported from the ISC-licensed HDR-VDP-2.2 MATLAB release. Three upstream
helpers carry no permission grant ("internal use, do not redistribute") and are
therefore not ported — they are reimplemented from their descriptions in
fft.rs and spectral.rs, with the one knowable behavioural difference
documented at the site. The steerable pyramid (chunk 2) comes from the MIT
matlabPyrTools. Full detail, including the notices that must travel with this
code: THIRD-PARTY-NOTICES.md (THIRD-PARTY-NOTICES.md).

[imazen/zenmetrics#50]: https://github.com/imazen/zenmetrics/issues/50

Canonical AIC2026 configuration research, including the complete upstream
3.0.7 crop control and the separate port comparison, is recorded in
the 2026-10-19 research note (docs/AIC2026_CANONICAL_2026-10-19.md).

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/hdrvdp/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/hdrvdp/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/hdrvdp/LICENSE-hdrvdp-3.0.7](https://github.com/imazen/zenmetrics/blob/master/crates/hdrvdp/LICENSE-hdrvdp-3.0.7) · [crates/hdrvdp/LICENSE-matlabPyrTools](https://github.com/imazen/zenmetrics/blob/master/crates/hdrvdp/LICENSE-matlabPyrTools) · [crates/hdrvdp/LICENSE-pyPyrTools](https://github.com/imazen/zenmetrics/blob/master/crates/hdrvdp/LICENSE-pyPyrTools)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=hdrvdp) · [Speed](https://metric-brief.pages.dev/#speed&q=hdrvdp)

## hdrvdp-aic2026

FR · Classical · CPU · Not documented

**Paper:** [HDR-VDP-2: a calibrated visual metric for visibility and quality predictions in all luminance conditions](https://doi.org/10.1145/1964921.1964935). Rafat Mantiuk; Kil Joong Kim; Allan G. Rempel; Wolfgang Heidrich. ACM SIGGRAPH 2011 papers. 2011.

```bibtex
@misc{HDRVDP22011,
  title = {HDR-VDP-2: a calibrated visual metric for visibility and quality predictions in all luminance conditions},
  author = {Rafat Mantiuk and Kil Joong Kim and Allan G. Rempel and Wolfgang Heidrich},
  year = {2011},
  note = {ACM SIGGRAPH 2011 papers},
  url = {https://doi.org/10.1145/1964921.1964935},
  doi = {10.1145/1964921.1964935}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** JOD / P_det / Q

**Direction:** not recorded

**Cli contract:** HDR-VDP 2 res.Q with unrounded sRGB at 200 nits and 37.84 ppd.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric hdrvdp-aic2026 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "format": "packed sRGB8; f64 conversion to absolute nits",
  "sdr_peak_nits": 200,
  "ppd": 37.84,
  "code_bits_before_srgb_eotf": 0,
  "code_rounding": "nearest, ties away from zero when enabled",
  "encoding": "RGB BT.709",
  "xyz_matrix": null,
  "task": "quality",
  "emission": "HDR-VDP 2 RGB BT.709 reference defaults",
  "age": "reference defaults",
  "surround": "reference defaults",
  "overrides": [
    "--hdrvdp-sdr-peak",
    "--hdrvdp-sdr-ppd",
    "--hdrvdp-sdr-code-bits"
  ],
  "v2_strict_arithmetic": "--hdrvdp-sdr-strict; default false",
  "outputs": [
    "hdrvdp_aic2026_q"
  ],
  "source": "crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=hdrvdp-aic2026) · [Speed](https://metric-brief.pages.dev/#speed&q=hdrvdp-aic2026)

## hdrvdp3

FR · Classical · CPU · Reported results

**Paper:** [HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content](https://arxiv.org/abs/2304.13625). Mantiuk, Rafal K.; Hammou, Dounia; Hanji, Param. arXiv preprint. 2023 (arXiv submission year; publication venue may be later).

```bibtex
@misc{HDRVDP32023,
  title = {HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content},
  author = {Mantiuk, Rafal K. and Hammou, Dounia and Hanji, Param},
  year = {2023},
  note = {arXiv preprint},
  url = {https://arxiv.org/abs/2304.13625},
  eprint = {2304.13625},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** same HDR-VDP lineage — Mantiuk et al., ACM TOG 30(4) 2011, doi:10.1145/2010324.1964935 (v3 is the display-adaptive revision; upstream asks to cite the 2011 paper + version number)

**Reference implementation:** official HDR-VDP 3.0.7 MATLAB release + jpeg-ai-qaf VDP3/ numpy port @ 0628a6b (verified equal — see DIVERGENCES)

**Reference identity:** official HDR-VDP 3.0.7 MATLAB release + jpeg-ai-qaf VDP3/ numpy port @ 0628a6b (verified equal — see DIVERGENCES)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** q_jod 0–10 (10 = identical) + P_map

**Direction:** not recorded

**Cli contract:** HDR-VDP-3.0.7 (Q_JOD scale 0–10, 10 = identical) — CPU implementation
via the in-tree hdrvdp::v3 module (native port of the
jpeg-ai-qaf VDP3 NumPy reference, itself a port of the official
HDR-VDP-3.0.7 MATLAB release). 8-bit sRGB pairs use upstream's
sRGB-display encoding (as hdrvdp), task quality, the encoding's
upstream emission (led-lcd-srgb), age 24, no surround, and the same
30 pixels per degree as v2. For absolute-luminance input (--hdr)
there is no default viewing geometry: --hdrvdp3-ppd is required, and
--hdrvdp3-task/--hdrvdp3-input/--hdrvdp3-emission/
--hdrvdp3-age/--hdrvdp3-surround select the rest of the
upstream parameter surface (defaults match the jpeg-ai-qaf
metrics.py harness: task=quality, rgb-bt.709, led-lcd-srgb,
age=24, surround=none). Emits hdrvdp3_jod_imazen_v. No GPU twin.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric hdrvdp3 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "absolute luminance cd/m², explicit required viewing conditions — ppd, task, encoding (rgb-bt709/bt2020/native/xyz/luminance), emission, surround, observer age; 8-bit sRGB through upstream's sRGB-display with task quality, led-lcd-srgb, age 24, no surround, 30 ppd"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 21-case golden corpus; observed: 1.6e-13 JOD worst |Q_JOD|",
  "oracle": "official HDR-VDP 3.0.7 MATLAB release + jpeg-ai-qaf VDP3/ numpy port @ 0628a6b (verified equal — see DIVERGENCES)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=hdrvdp3) · [Speed](https://metric-brief.pages.dev/#speed&q=hdrvdp3)

## hdrvdp3-aic2026

FR · Classical · CPU · Not documented

**Paper:** [HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content](https://arxiv.org/abs/2304.13625). Mantiuk, Rafal K.; Hammou, Dounia; Hanji, Param. arXiv preprint. 2023 (arXiv submission year; publication venue may be later).

```bibtex
@misc{HDRVDP32023,
  title = {HDR-VDP-3: A multi-metric for predicting image differences, quality and contrast distortions in high dynamic range and regular content},
  author = {Mantiuk, Rafal K. and Hammou, Dounia and Hanji, Param},
  year = {2023},
  note = {arXiv preprint},
  url = {https://arxiv.org/abs/2304.13625},
  eprint = {2304.13625},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** q_jod 0–10 (10 = identical) + P_map

**Direction:** not recorded

**Cli contract:** HDR-VDP 3 res.Q and Q_JOD with 200-nit sRGB to XYZ, OLED, 37.842505866 ppd.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric hdrvdp3-aic2026 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "format": "packed sRGB8; f64 conversion to absolute nits",
  "sdr_peak_nits": 200,
  "ppd": 37.842505866,
  "code_bits_before_srgb_eotf": 10,
  "code_rounding": "nearest, ties away from zero when enabled",
  "encoding": "D65 XYZ from linear sRGB",
  "xyz_matrix": [
    [
      0.4124564,
      0.3575761,
      0.1804375
    ],
    [
      0.2126729,
      0.7151522,
      0.072175
    ],
    [
      0.0193339,
      0.119192,
      0.9503041
    ]
  ],
  "task": "quality",
  "emission": "OLED",
  "age": 24,
  "surround": "none",
  "overrides": [
    "--hdrvdp-sdr-peak",
    "--hdrvdp-sdr-ppd",
    "--hdrvdp-sdr-code-bits"
  ],
  "v2_strict_arithmetic": "--hdrvdp-sdr-strict; default false",
  "outputs": [
    "hdrvdp3_aic2026_q",
    "hdrvdp3_aic2026_jod"
  ],
  "source": "crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=hdrvdp3-aic2026) · [Speed](https://metric-brief.pages.dev/#speed&q=hdrvdp3-aic2026)

## hyperiqa

NR · Learned · CPU · Reported results

**Paper:** [Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network](https://doi.org/10.1109/CVPR42600.2020.00372). Su, Shaolin; Yan, Qingsen; Zhu, Yu; Zhang, Cheng; Ge, Xin; Sun, Jinqiu; Zhang, Yanning. CVPR 2020. 2020.

```bibtex
@misc{hyperiqa2020,
  title = {Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network},
  author = {Su, Shaolin and Yan, Qingsen and Zhu, Yu and Zhang, Cheng and Ge, Xin and Sun, Jinqiu and Zhang, Yanning},
  year = {2020},
  note = {CVPR 2020},
  url = {https://doi.org/10.1109/CVPR42600.2020.00372},
  doi = {10.1109/CVPR42600.2020.00372}
}
```

**Recorded paper:** Su, Yan, Zhu, Zhang, Ge, Sun, Zhang. Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network. CVPR 2020

**Reference implementation:** https://github.com/SSL92/hyperIQA

**Reference identity:** git clone of SSL92/hyperIQA, pinned c42e7279717e7dcb693a24b891fc14a4189a45ee (the paper's official repo — NOT 'SSL-Group/hyperIQA', which 404s)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** unknown

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** HyperIQA — learned no-reference quality score (upstream
SSL92/hyperIQA, MIT code; weights licence unknown —
consent-gated; fine-tuned ResNet-50 hypernetwork,
koniq_pretrained.pkl); higher is better, MOS-scale
[0, 100]. Scored through PairedNoReference: emits hyperiqa
(distorted) and hyperiqa_ref (reference). As arniqa for
ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE, plus
ZENMETRICS_NN_ACCEPT_WEIGHTS=hyperiqa.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/hyperiqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric hyperiqa --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=hyperiqa. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "unknown",
  "checkpoint": "https://drive.google.com/file/d/1OOUmnbvpGea0LIGpIWEbOyxfWx6UCiiE/view (README 'pre-trained model', koniq-10k; koniq_pretrained.pkl, ~110 MiB; sha256 recorded in hyperiqa_export.meta)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "torchvision ToTensor (u8 -> f32 /255) then ImageNet mean/std (0.485,0.456,0.406)/(0.229,0.224,0.225)",
  "resize": "PIL bilinear u8 to (W,H)=(384,512) — torchvision Resize((512,384)) — bit-exact port resize_bilinear_pil_u8 (LICENSE-pillow)",
  "crop": "DEVIATION FROM UPSTREAM: upstream's test transform is RandomCrop(224) scored test_patch_num=25 times and averaged (demo.py uses 10) — nondeterministic. This port crops a deterministic 5x5 grid instead: x-origins round(i160/4)={0,40,80,120,160}, y-origins round(j288/4)={0,72,144,216,288} — 25 crops, the same count as test_patch_num, then the batch mean. The model graph and weights are upstream's; only crop placement differs. The goldens were produced by oracle.py running upstream's own network on this grid.",
  "crop_grid_measurement": "Independent upstream-network check: 8 seeds × 4 images; upstream random 25-crop means have run-to-run SD 0.35–0.85 score units (0.08–0.19 JND). The deterministic grid differs from their random-crop mean by 0.16–0.49 score units (≤0.11 JND), within that protocol noise. Upstream random crops cannot be reproduced to the 1e-3 JND gate; parity below qualifies the documented deterministic grid.",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "SSL92/hyperIQA revision c42e7279, HyperNet+TargetNet on the deterministic 5x5 grid (models.py imported verbatim), CPU fp32, koniq_pretrained.pkl",
  "tolerance_jnd": 0.001,
  "jnd": 4.461776,
  "results": "rten: worst |d| 1.526e-5 (3.42e-6 JND) on 27 ladder goldens; ort-cuda: worst |d| 1.526e-5 (3.42e-6 JND). ONNX-CPU vs upstream-model goldens (the deterministic grid) worst 1.5e-5; upstream-TargetNet vs exported graph |d| = 0.0. rten smoke 1415_256: 32.6361008 vs golden 32.636105."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/hyperiqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/hyperiqa/card.toml) · [crates/zenmetrics-nn/LICENSE-hyperiqa](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-hyperiqa)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=hyperiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=hyperiqa)

## iwssim

FR · Classical · CPU · Not compared

**Paper:** [Information Content Weighting for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2010.2092435). Zhou Wang; Qiang Li. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{IWSSIM2011,
  title = {Information Content Weighting for Perceptual Image Quality Assessment},
  author = {Zhou Wang and Qiang Li},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2010.2092435},
  doi = {10.1109/TIP.2010.2092435}
}
```

**Recorded paper:** Wang & Li, IEEE TIP 20(5) 2011, doi:10.1109/TIP.2010.2092435

**Reference implementation:** Python-IW-SSIM f9de37c (the Python port linked from the authors' IW-SSIM page; not yet compared directly with the authors' MATLAB release)

**Reference identity:** Python-IW-SSIM f9de37c (the Python port linked from the authors' IW-SSIM page; not yet compared directly with the authors' MATLAB release)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** IW-SSIM (Information-Content Weighted SSIM, Wang & Li 2011) — CPU
implementation via the in-tree iwssim crate (native SIMD port,
dispatched through zenmetrics_api::cpu_dispatch / Backend::Cpu).
Unsuffixed = CPU (the ssim2/dssim/zensim convention); use
iwssim-gpu for the GPU backend. Score in [0, 1] where 1 =
identical. Requires min(W, H) >= 176 per the paper's 5-level
pyramid + 11×11 valid-mode SSIM stats constraint; smaller images
return an error. Emits the CPU column name (iwssim_cpu_imazen_v).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric iwssim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → u8-rounded BT.601 gray (0.2989/0.5870/0.1140, LumaConvention::Bt601Rounded)"
}
```

**Parity:**

```json
{
  "reported": "not compared (ledger)",
  "oracle": "Python-IW-SSIM f9de37c (the Python port linked from the authors' IW-SSIM page; not yet compared directly with the authors' MATLAB release)"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/iwssim/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/iwssim/README.md)

# iwssim

Pure-Rust IW-SSIM (Wang & Li 2011) — and, with information weighting switched
off, a true MS-SSIM (Wang, Simoncelli & Bovik 2003). CPU implementation; the
iwssim-gpu sibling crate runs the same algorithm on CubeCL.

## Two metrics in one pipeline

The pipeline is the Wang & Li formulation: 5-scale dyadic pyramid, 11×11
Gaussian SSIM statistics per scale, cross-scale combination
Π |wmcs_j|^{β_j} with the standard exponents
β = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].

- IwssimParams { iw_flag: true } (default) — IW-SSIM: each scale's
  contrast-structure map is pooled with information-content weights from the
  GSM model (parent-band conditioned, per-pixel mutual information).
- IwssimParams { iw_flag: false } — MS-SSIM: each scale pools by plain
  mean. With weighting off, the Wang & Li construction reduces exactly to
  canonical MS-SSIM.

Both modes are locked to the reference implementation
(Python-IW-SSIM (https://github.com/Jack-guo-xy/Python-IW-SSIM), commit
f9de37c) by committed goldens: goldens/python_iwssim_2026-05-27.json
(IW) and goldens/python_msssim_2026-06-10.json (MS-SSIM), enforced by
tests/parity_python.rs on deterministic fixtures (identical pairs ≤ 1e-5,
distorted ≤ 5e-3).

## Input

score(ref_rgb, dist_rgb) takes sRGB8 and converts to gray via BT.601 on
code values (no linearization), matching the reference. score_gray is the
canonical entry — float gray planes in 0..255 scale, width × height
samples. Minimum dimension 176 (5-level pyramid + 11×11 valid blur);
allow_small tiles smaller inputs up to the floor.

The RGB→gray conversion is selectable via IwssimParams::luma
(LumaConvention):

- Bt601Rounded (default) — round(0.2989R+0.5870G+0.1140B), the
  reference's utils.rgb2gray exactly.
- YiqUnrounded — 0.299R+0.587G+0.114B with no rounding, the piq /
  jpeg-ai-qaf rgb2yiq-channel-0 convention. IwssimParams::piq_luma()
  selects it; this is the ingress the JPEG AIC-4 IW-SSIM column was
  computed under (CLI --metric iwssim-piq, column
  iwssim_piq_imazen_v; med |Δ| 3.9e-6 / max 2.6e-5 vs
  metrics_fullres.tab, n=54).

Both conventions share one pipeline: the IW weighting uses the next
Laplacian pyramid level as the GSM parent band
(pyrBand(pyro, pind, nband+1) in the MATLAB release,
imgopr[scale+1] in Python-IW-SSIM and jpeg-ai-qaf's
IW_SSIM_PyTorch.py).

The float gray entry matters for HDR: feeding PU21(bt709-luma)·255/PU21(peak)
as floats (no u8 round-trip) scored SROCC 0.808 (IW) / 0.812 (MS-SSIM)
on the UPIQ HDR subset vs 0.628 through a u8 shell — see
benchmarks/pu_integrated_upiq_2026-06-09.md and the umbrella's
HdrFeeding::PuLumaGrayF32 routing.

## Batch and streaming

- warm_reference / score_with_warm_ref{,_gray} reuse the reference-side
  pyramid + per-scale eigendecomposition across many distorted candidates.
- score_strip{,_gray} processes bounded-height strips for large images;
  strip and whole-image paths agree to ≤ 1e-6 (tests/strip_parity.rs).

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/iwssim/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/iwssim/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/iwssim/LICENSE-Python-IW-SSIM](https://github.com/imazen/zenmetrics/blob/master/crates/iwssim/LICENSE-Python-IW-SSIM)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=iwssim) · [Speed](https://metric-brief.pages.dev/#speed&q=iwssim)

## iwssim-gpu

FR · Classical · GPU · Not documented

**Paper:** [Information Content Weighting for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2010.2092435). Zhou Wang; Qiang Li. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{IWSSIM2011,
  title = {Information Content Weighting for Perceptual Image Quality Assessment},
  author = {Zhou Wang and Qiang Li},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2010.2092435},
  doi = {10.1109/TIP.2010.2092435}
}
```

**Recorded paper:** Wang & Li, IEEE TIP 20(5) 2011, doi:10.1109/TIP.2010.2092435

**Reference implementation:** CPU twin

**Reference identity:** CPU twin

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** IW-SSIM — GPU implementation via the iwssim-gpu crate
(dispatched through the zenmetrics-api umbrella).
Range [0, 1]; 1.0 = identical.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features gpu-iwssim
target/release/zenmetrics score --metric iwssim-gpu --reference ref.png --distorted dist.png
```

Requires a GPU backend at run time.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → unrounded 0.299R+0.587G+0.114B"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "CPU twin"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/iwssim-gpu/LICENSE-Python-IW-SSIM](https://github.com/imazen/zenmetrics/blob/master/crates/iwssim-gpu/LICENSE-Python-IW-SSIM)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=iwssim-gpu) · [Speed](https://metric-brief.pages.dev/#speed&q=iwssim-gpu)

## iwssim-piq

FR · Classical · CPU · Reported results

**Paper:** [Information Content Weighting for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2010.2092435). Zhou Wang; Qiang Li. IEEE Transactions on Image Processing. 2011.

```bibtex
@misc{IWSSIM2011,
  title = {Information Content Weighting for Perceptual Image Quality Assessment},
  author = {Zhou Wang and Qiang Li},
  year = {2011},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2010.2092435},
  doi = {10.1109/TIP.2010.2092435}
}
```

**Recorded paper:** Wang & Li, IEEE TIP 20(5) 2011, doi:10.1109/TIP.2010.2092435

**Reference implementation:** jpeg-ai-qaf IW_SSIM (torch port of the same Jack-guo ref) on the unrounded 0–255 Y — the AIC-4 IW-SSIM column; upstream mirror [local path] @ 0628a6b

**Reference identity:** jpeg-ai-qaf IW_SSIM (torch port of the same Jack-guo ref) on the unrounded 0–255 Y — the AIC-4 IW-SSIM column; upstream mirror [local path] @ 0628a6b

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** IW-SSIM with the piq / JPEG AIC-4 ingress — identical
in-tree iwssim algorithm, but the RGB→luma conversion is
piq's unrounded rgb2yiq channel 0 (0.299/0.587/0.114,
0–255, no u8 rounding) instead of the Python-IW-SSIM
reference's rounded BT.601. This is the convention the
published AIC-4 IW-SSIM column was computed under
(med|Δ| = 1.3e-6 vs metrics_fullres.tab, 53 pairs,
2026-09-25). Emits iwssim_piq_imazen_v so it never mixes
with iwssim's column in a join. CPU-only.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric iwssim-piq --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → unrounded 0.299R+0.587G+0.114B"
}
```

**Parity:**

```json
{
  "reported": "reported gate: — (same code path); observed: —",
  "oracle": "jpeg-ai-qaf IW_SSIM (torch port of the same Jack-guo ref) on the unrounded 0–255 Y — the AIC-4 IW-SSIM column; upstream mirror [local path] @ 0628a6b"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=iwssim-piq) · [Speed](https://metric-brief.pages.dev/#speed&q=iwssim-piq)

## liqe

NR · Learned · CPU · Reported results

**Paper:** [Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective](https://arxiv.org/abs/2303.14968). Zhang, Weixia; Zhai, Guangtao; Wei, Ying; Yang, Xiaokang; Ma, Kede. CVPR 2023. 2023 (arXiv submission year; publication venue may be later).

```bibtex
@misc{liqe2023,
  title = {Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},
  author = {Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},
  year = {2023},
  note = {CVPR 2023},
  url = {https://arxiv.org/abs/2303.14968},
  eprint = {2303.14968},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Zhang, Zhai, Wei, Yang, Ma. Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective. CVPR 2023, pp. 14071–14081

**Reference implementation:** https://github.com/zwx8981/LIQE

**Reference identity:** git clone of zwx8981/LIQE (no pip package), pinned a64655f7b84b9543b513394472851884cde5b257

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** unknown

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** LIQE — learned no-reference quality score (upstream
zwx8981/LIQE, MIT code / weights licence unknown:
consent-gated; fine-tuned CLIP ViT-B/32, LIQE.pt); higher
is better, MOS [1, 5]. Scored through PairedNoReference:
emits liqe (distorted) and liqe_ref (reference). As arniqa
for ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE, plus
ZENMETRICS_NN_ACCEPT_WEIGHTS=liqe.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/liqe/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric liqe --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=liqe. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "unknown",
  "checkpoint": "https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view (README 'Pre-trained Weights'; LIQE.pt, ~354 MiB)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "torchvision ToTensor (u8 -> f32 /255), then CLIP mean/std normalise inside LIQE.forward and so inside the graphs",
  "resize": "none",
  "crop": "unfold 224x224 patches stride 32 (partial borders dropped), then sel[i] = (batch//15)i: batch is 1 in every eval, so sel = [0]15 — the TOP-LEFT 224x224 patch is scored 15 times (upstream code, reproduced verbatim; the mean over 15 identical logits is the single patch's)",
  "min_size": 224
}
```

**Parity:**

```json
{
  "oracle": "zwx8981/LIQE revision a64655f, LIQE.forward on ToTensor input, CPU fp32, LIQE.pt",
  "tolerance_jnd": 0.001,
  "jnd": 0.393781,
  "results": "rten: worst |d| 4.93e-6 (1.25e-5 JND) on 9 ladder goldens; ort-cuda: worst |d| 1.955e-5 (4.96e-5 JND). CLI smoke 1415_256: 3.814563/4.990012 vs golden 3.81456041/4.99001217"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/liqe/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/liqe/card.toml) · [crates/zenmetrics-nn/LICENSE-liqe](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-liqe)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=liqe) · [Speed](https://metric-brief.pages.dev/#speed&q=liqe)

## lpips-alex

FR · Learned · CPU · Reported results

**Paper:** [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924). Zhang, Richard; Isola, Phillip; Efros, Alexei A.; Shechtman, Eli; Wang, Oliver. CVPR 2018. 2018 (arXiv submission year; publication venue may be later).

```bibtex
@misc{lpips2018,
  title = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  year = {2018},
  note = {CVPR 2018},
  url = {https://arxiv.org/abs/1801.03924},
  eprint = {1801.03924},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Zhang, Isola, Efros, Shechtman, Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018. arXiv:1801.03924

**Reference implementation:** https://github.com/richzhang/PerceptualSimilarity

**Reference identity:** lpips==0.1.4

**Reference code license:** BSD-2-Clause

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** LPIPS v0.1, AlexNet trunk (upstream lpips 0.1.4); lower is
better. Needs the nn feature and an exported model directory in
ZENMETRICS_NN_MODEL_DIR (see crates/zenmetrics-nn).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/lpips/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric lpips-alex --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "trunk": {
    "alex": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "vgg": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "squeeze": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth"
  }
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "(v / 255)  2 - 1, then LPIPS ScalingLayer (inside the graph)",
  "resize": "none",
  "min_size": "per net, written to lpips_<net>.meta by export.py (alex 31, vgg 16, squeeze 17)"
}
```

**Parity:**

```json
{
  "oracle": "lpips 0.1.4, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs (3 images, 64-2048 px), i265:\nrten:     alex 7.6e-8, vgg 4.5e-8, squeeze 1.1e-7\nort-cuda: alex 6.0e-8, vgg 6.0e-8, squeeze 9.0e-8"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/lpips/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/lpips/card.toml) · [crates/zenmetrics-nn/LICENSE-lpips](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-lpips)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=lpips-alex) · [Speed](https://metric-brief.pages.dev/#speed&q=lpips-alex)

## lpips-squeeze

FR · Learned · CPU · Reported results

**Paper:** [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924). Zhang, Richard; Isola, Phillip; Efros, Alexei A.; Shechtman, Eli; Wang, Oliver. CVPR 2018. 2018 (arXiv submission year; publication venue may be later).

```bibtex
@misc{lpips2018,
  title = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  year = {2018},
  note = {CVPR 2018},
  url = {https://arxiv.org/abs/1801.03924},
  eprint = {1801.03924},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Zhang, Isola, Efros, Shechtman, Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018. arXiv:1801.03924

**Reference implementation:** https://github.com/richzhang/PerceptualSimilarity

**Reference identity:** lpips==0.1.4

**Reference code license:** BSD-2-Clause

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** LPIPS v0.1, SqueezeNet 1.1 trunk; as lpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/lpips/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric lpips-squeeze --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "trunk": {
    "alex": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "vgg": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "squeeze": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth"
  }
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "(v / 255)  2 - 1, then LPIPS ScalingLayer (inside the graph)",
  "resize": "none",
  "min_size": "per net, written to lpips_<net>.meta by export.py (alex 31, vgg 16, squeeze 17)"
}
```

**Parity:**

```json
{
  "oracle": "lpips 0.1.4, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs (3 images, 64-2048 px), i265:\nrten:     alex 7.6e-8, vgg 4.5e-8, squeeze 1.1e-7\nort-cuda: alex 6.0e-8, vgg 6.0e-8, squeeze 9.0e-8"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/lpips/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/lpips/card.toml) · [crates/zenmetrics-nn/LICENSE-lpips](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-lpips)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=lpips-squeeze) · [Speed](https://metric-brief.pages.dev/#speed&q=lpips-squeeze)

## lpips-vgg

FR · Learned · CPU · Reported results

**Paper:** [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924). Zhang, Richard; Isola, Phillip; Efros, Alexei A.; Shechtman, Eli; Wang, Oliver. CVPR 2018. 2018 (arXiv submission year; publication venue may be later).

```bibtex
@misc{lpips2018,
  title = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author = {Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
  year = {2018},
  note = {CVPR 2018},
  url = {https://arxiv.org/abs/1801.03924},
  eprint = {1801.03924},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Zhang, Isola, Efros, Shechtman, Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018. arXiv:1801.03924

**Reference implementation:** https://github.com/richzhang/PerceptualSimilarity

**Reference identity:** lpips==0.1.4

**Reference code license:** BSD-2-Clause

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** LPIPS v0.1, VGG16 trunk; as lpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/lpips/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric lpips-vgg --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "trunk": {
    "alex": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    "vgg": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "squeeze": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth"
  }
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "(v / 255)  2 - 1, then LPIPS ScalingLayer (inside the graph)",
  "resize": "none",
  "min_size": "per net, written to lpips_<net>.meta by export.py (alex 31, vgg 16, squeeze 17)"
}
```

**Parity:**

```json
{
  "oracle": "lpips 0.1.4, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs (3 images, 64-2048 px), i265:\nrten:     alex 7.6e-8, vgg 4.5e-8, squeeze 1.1e-7\nort-cuda: alex 6.0e-8, vgg 6.0e-8, squeeze 9.0e-8"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/lpips/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/lpips/card.toml) · [crates/zenmetrics-nn/LICENSE-lpips](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-lpips)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=lpips-vgg) · [Speed](https://metric-brief.pages.dev/#speed&q=lpips-vgg)

## mad

FR · Classical · CPU · Not documented

**Paper:** [Most apparent distortion: full-reference image quality assessment and the role of strategy](https://doi.org/10.1117/1.3267105). Eric C. Larson; Damon M. Chandler. Journal of Electronic Imaging. 2010.

```bibtex
@misc{MAD2010,
  title = {Most apparent distortion: full-reference image quality assessment and the role of strategy},
  author = {Eric C. Larson and Damon M. Chandler},
  year = {2010},
  note = {Journal of Electronic Imaging},
  url = {https://doi.org/10.1117/1.3267105},
  doi = {10.1117/1.3267105}
}
```

**Recorded paper:** Larson & Chandler, JEI 19(1) 2010

**Reference implementation:** official MATLAB release (larschandler.com)

**Reference identity:** official MATLAB release (larschandler.com)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** distance; emits mad, mad_hi, mad_lo

**Direction:** not recorded

**Cli contract:** MAD — Most Apparent Distortion (in-tree mad-iqa crate, port
of the authors' hi_index.m/lo_index.m + ical_std/
ical_stat C-mex): DISTANCE score ≥ 0, 0 = identical,
unbounded above; min(w,h) < 34 yields NaN (the reference's
mp(17:end-17) edge kill leaves nothing to pool). Emits
mad_imazen_v plus mad_hi_imazen_v / mad_lo_imazen_v
(the two strategy indices). CPU-only — not routed through the
umbrella or the orchestrator.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric mad --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → luma"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "official MATLAB release (larschandler.com)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=mad) · [Speed](https://metric-brief.pages.dev/#speed&q=mad)

## maniqa

NR · Learned · CPU · Reported results

**Paper:** [MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment](https://arxiv.org/abs/2204.08958). Yang, Sidi; Wu, Tianhe; Shi, Shuwei; Lao, Shanshan; Gong, Yuan; Cao, Mingdeng; Wang, Jiahao; Yang, Yujiu. CVPR Workshops 2022. 2022 (arXiv submission year; publication venue may be later).

```bibtex
@misc{maniqa2022,
  title = {MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment},
  author = {Yang, Sidi and Wu, Tianhe and Shi, Shuwei and Lao, Shanshan and Gong, Yuan and Cao, Mingdeng and Wang, Jiahao and Yang, Yujiu},
  year = {2022},
  note = {CVPR Workshops 2022},
  url = {https://arxiv.org/abs/2204.08958},
  eprint = {2204.08958},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Yang, Wu, Shi, Lao, Gong, Cao, Wang, Yang. MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment. CVPR Workshops 2022. Tsinghua University IIGROUP + vivo

**Reference implementation:** https://github.com/IIGROUP/MANIQA

**Reference identity:** git clone of IIGROUP/MANIQA (no pip package; vendored timm 0.5.5 in-repo), pinned f573d862401243c45641951a0303b8059c1c8577

**Reference code license:** Apache-2.0

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** MANIQA — learned no-reference quality score (upstream
IIGROUP/MANIQA, Apache-2.0; ViT-B/8 + TABlocks + Swin stages,
ckpt_koniq10k); higher is better, normalized [0, 1]
(upstream min-max normalizes koniq10k MOS labels). Scored
through PairedNoReference: emits
maniqa (distorted) and maniqa_ref (reference). As arniqa
for ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/maniqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric maniqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "https://github.com/IIGROUP/MANIQA/releases/tag/Koniq10k — ckpt_koniq10k.pt (543 MB, sha256 recorded in maniqa_export.meta provenance)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "u8/255 then (x-0.5)/0.5 — utils/inference_process.py Normalize(0.5,0.5)+ToTensor on the cv2-decoded /255 image",
  "resize": "data/koniq10k.py eval loader: cv2.resize((224,224), INTER_CUBIC) on the u8 image — resize.rs::resize_bicubic_cv2_u8. Convention: OpenCV's own scalar u8 path (export pins OPENCV_IPP=disabled; the wheels' IPP differs on ~4% of pixels, and cv2 4.12 ≡ 5.0 bitwise with it off): f32 Horner coeffs quantised rint(c2048) -> i16, i32 horizontal mid, unfused f32 vertical chain (VResizeCubicVec_32s8u at the SSE baseline; no FMA3 in the pinned build), A=-0.75, BORDER_REPLICATE. Measured bitwise vs cv2-noIPP on probes; ~0 diffs on ladder images. OpenCV Apache-2.0 in ../../LICENSE-opencv",
  "crop": "five_point_crop(0) on the 224x224 image = the image itself (num_avg_val=1)",
  "min_size": 1
}
```

**Parity:**

```json
{
  "oracle": "IIGROUP/MANIQA revision f573d86, koniq10k eval path (train_maniqa.py validation), CPU fp32, ckpt_koniq10k.pt",
  "tolerance_jnd": 0.001,
  "jnd": 0.021396,
  "results": "rten 3.64e-6 = 1.70e-4 JND; ort-cuda 3.16e-6 = 1.48e-4 JND vs upstream (IPPs-disabled goldens), 15 ladder goldens — the exact fixed-point port + non-IPP convention opened the margin ~6x over the earlier float-model run (2.07e-5 = 9.7e-4 JND). Goldens committed at models/maniqa/maniqa_goldens.tsv; oracle env: cv2 5.0.0 OPENCV_IPP=disabled, numpy 2.5.2, Pillow 12.3.0"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/maniqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/maniqa/card.toml) · [crates/zenmetrics-nn/LICENSE-maniqa](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-maniqa)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=maniqa) · [Speed](https://metric-brief.pages.dev/#speed&q=maniqa)

## mdctpsnr

FR · Classical · CPU · Reported results

**Paper:** [On the mDCT-PSNR image quality index](https://doi.org/10.1109/QOMEX.2009.5246978). Thomas Richter. 2009 International Workshop on Quality of Multimedia Experience. 2009.

```bibtex
@misc{mDCTPSNR2009,
  title = {On the mDCT-PSNR image quality index},
  author = {Thomas Richter},
  year = {2009},
  note = {2009 International Workshop on Quality of Multimedia Experience},
  url = {https://doi.org/10.1109/QOMEX.2009.5246978},
  doi = {10.1109/QOMEX.2009.5246978}
}
```

**Recorded paper:** Richter, "On the mDCT-PSNR image quality index", QoMEX 2009, doi:10.1109/QOMEX.2009.5246978 (authored impl)

**Reference implementation:** author's official C++ thorfdbg/mDCTpsnr (Thomas Richter; contributor Jon Sneyers) (zlib-style license), built GCC -O3 -ffast-math AVX2 + glibc libmvec

**Reference identity:** author's official C++ thorfdbg/mDCTpsnr (Thomas Richter; contributor Jon Sneyers) (zlib-style license), built GCC -O3 -ffast-math AVX2 + glibc libmvec

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** dB, higher better; +inf identical

**Direction:** not recorded

**Cli contract:** mDCT-PSNR (masked-DCT PSNR, Thomas Richter / University of
Stuttgart — the JPEG AIC-4 mDCT-PSNR column) — CPU
implementation via the in-tree mdctpsnr crate, a port of the
reference dctpsnr binary verified to ~1e-5 dB on the AIC-4
53-pair subset. PSNR-like: dB scale, higher = better, +inf =
identical. Scored on the decoded sRGB pair through the
reference's linear-BT.601 YCbCr + sliding-8×8-DCT masking
pipeline. Emits mdctpsnr_imazen_v. Not routed through the
umbrella or the orchestrator (CPU-only, no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric mdctpsnr --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → linear → BT.601 YCbCr"
}
```

**Parity:**

```json
{
  "reported": "reported gate: ≤ 2e-4 dB on x86_64+GNU goldens (64×48 + all 8 (w−13)%8 classes), 1e-2 off-platform; observed: ≤ ~4e-6 dB",
  "oracle": "author's official C++ thorfdbg/mDCTpsnr (Thomas Richter; contributor Jon Sneyers) (zlib-style license), built GCC -O3 -ffast-math AVX2 + glibc libmvec"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/mdctpsnr/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/mdctpsnr/README.md)

# mdctpsnr

Pure-Rust CPU reimplementation of mDCT-PSNR — the masked-DCT perceptual
image quality metric by Thomas Richter (University of Stuttgart), "On the
mDCT-PSNR image quality index" (QoMEX 2009, doi:10.1109/QOMEX.2009.5246978;
the metric the JPEG AIC-4 benchmark publishes in its mDCT-PSNR column).

The port follows the author's reference implementation
thorfdbg/mDCTpsnr (https://github.com/thorfdbg/mDCTpsnr) (the dctpsnr
CLI), which is zlib-licensed; its notice is retained at the head of
src/lib.rs (src/lib.rs). This crate is an altered reimplementation and is
not the original work.

## Algorithm

1. sRGB-8 → sRGB-linear RGB (exact EOTF LUT) → linear BT.601 YCbCr;
   only Y is re-encoded with the sRGB gamma knee, Cb/Cr stay linear
   (signed, ±0.5).
2. Per component, a sliding 8×8 windowed DCT at every position: each block
   is windowed toward its own mean (b·w + avg·(1−w), Hamming-family
   window_table) then transformed with a two-pass scaled AAN float DCT.
3. Per (band, component) a 5×5 separable visibility mask:
   mapped = |c·vis|^expon, cosine-normalized kernel,
   mask = 0.08 / (0.08 + Σ5×5/25).
4. Per measured position, errorline[j] multiplies the per-band
   non-detection probabilities exp(−(err·max(mask)·vb)^3.5) over 195
   terms (64 bands × 3 components; the Y DC band is measured on a 4-way
   5×5 low/high extended filter split instead of its own coefficient).
5. score_dB = −20·log10(error / (w·h·comps)) with w = W−13,
   h = H−13, comps = 3 — note the denominator's H−13 vs the H−12
   measured lines; that is the reference's own normalization, replicated
   deliberately.

+inf on identical inputs, as the reference computes it. Minimum size is
14×14 (ImageTooSmall below that; the reference has unchecked UB there).

## API

rust
let score = mdctpsnr::mdct_psnr_srgb8(&ref_rgb8, &dist_rgb8, w, h)?; // f32 dB
let jnd   = mdctpsnr::mdct_psnr_jnd(score);                        // JND map


- Input is packed sRGB8 (wh3 bytes). Feature std on by default;
  no_std + alloc supported.
- mdct_psnr_jnd is the reference CLI's optional -jnd remap
  (0.000475997802·(80−dB)^2.3182, clamped at 0) — the AIC-4 column is the
  plain dB score.

## Parity with the compiled reference

The parity target is the compiled binary, not the source's apparent
semantics: built with GCC -O3 -ffast-math on x86_64 + glibc, the reference
reassociates reductions, contracts multiplies to FMAs, replaces the mask
division with vrcpps + a Newton step, and routes powf/expf/cosf
through libmvec's vectorized entry points (_ZGVdN8vv_powf,
_ZGVdN8v_expf). On x86_64 GNU/Linux this crate reproduces that codegen —
including calling the real libmvec symbols — so scores match to ~4e-6 dB
(bit-exact errorline on verified blocks/lines). Elsewhere the same
algorithm runs on scalar libm; identical algorithm, ~1ulp/step drift.

Validation (see DIVERGENCES.md for the full record):

- Synthetic goldens in the test suite, including all eight (W−13) % 8
  pooling-width residue classes (the vectorized reduction's tail guard
  rem ≥ 4 is the subtle one — rem == 3 never reads a slack lane).
- AIC-4 mDCT-PSNR column over the 53-pair stratified subset:
  median |Δ| 4.0e-7 dB, max 1.13e-5 dB.

## Not the same name, not the same metric

mDCT-PSNR (this crate — Richter's masked-DCT PSNR) is unrelated to the
AIC-4 proposal-mDCTPSNR submission column, which is a private contest
entry and cannot be reproduced by design.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial (crate). The ported algorithm
and the retained notice belong to the zlib-licensed reference; the notice
text is at the top of src/lib.rs.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/mdctpsnr/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/mdctpsnr/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/mdctpsnr/LICENSE-mDCTpsnr](https://github.com/imazen/zenmetrics/blob/master/crates/mdctpsnr/LICENSE-mDCTpsnr)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=mdctpsnr) · [Speed](https://metric-brief.pages.dev/#speed&q=mdctpsnr)

## mdsi

FR · Classical · CPU · Not documented

**Paper:** [Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image Quality Evaluator](https://doi.org/10.1109/ACCESS.2016.2604042). Hossein Ziaei Nafchi; Atena Shahkolaei; Rachid Hedjam; Mohamed Cheriet. IEEE Access. 2016.

```bibtex
@misc{MDSI2016,
  title = {Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image Quality Evaluator},
  author = {Hossein Ziaei Nafchi and Atena Shahkolaei and Rachid Hedjam and Mohamed Cheriet},
  year = {2016},
  note = {IEEE Access},
  url = {https://doi.org/10.1109/ACCESS.2016.2604042},
  doi = {10.1109/ACCESS.2016.2604042}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** MDSI default summation model (Nafchi et al. 2016), CPU, f64 L/H/M.
Distance: smaller is better. Uses the cpu-gmsd build feature.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric mdsi --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "MDSI default summation model (Nafchi et al. 2016), CPU, f64 L/H/M.\nDistance: smaller is better. Uses the cpu-gmsd build feature."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=mdsi) · [Speed](https://metric-brief.pages.dev/#speed&q=mdsi)

## ms-gmsd

FR · Classical · CPU · Not documented

**Paper:** [Gradient magnitude similarity deviation on multiple scales for color image quality assessment](https://doi.org/10.1109/ICASSP.2017.7952357). Bo Zhang; Pedro V. Sander; Amine Bermak. 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2017.

```bibtex
@misc{MSGMSD2017,
  title = {Gradient magnitude similarity deviation on multiple scales for color image quality assessment},
  author = {Bo Zhang and Pedro V. Sander and Amine Bermak},
  year = {2017},
  note = {2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  url = {https://doi.org/10.1109/ICASSP.2017.7952357},
  doi = {10.1109/ICASSP.2017.7952357}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Paper-derived four-scale masked GMSD, with declared numerical conventions.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-gmsd --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Paper-derived four-scale masked GMSD, with declared numerical conventions."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-gmsd) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-gmsd)

## ms-gmsdc

FR · Classical · CPU · Not documented

**Paper:** [Gradient magnitude similarity deviation on multiple scales for color image quality assessment](https://doi.org/10.1109/ICASSP.2017.7952357). Bo Zhang; Pedro V. Sander; Amine Bermak. 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2017.

```bibtex
@misc{MSGMSD2017,
  title = {Gradient magnitude similarity deviation on multiple scales for color image quality assessment},
  author = {Bo Zhang and Pedro V. Sander and Amine Bermak},
  year = {2017},
  note = {2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  url = {https://doi.org/10.1109/ICASSP.2017.7952357},
  doi = {10.1109/ICASSP.2017.7952357}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Paper-derived colour MS-GMSDc; not author-software parity.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-gmsdc --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Paper-derived colour MS-GMSDc; not author-software parity."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-gmsdc) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-gmsdc)

## ms-ssim-pyiqa

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AIC2026 ms-ssim-pyiqa convention.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa convention."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa)

## ms-ssim-pyiqa-f32

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AIC2026 ms-ssim-pyiqa-f32 convention.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa-f32 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa-f32 convention."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa-f32) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa-f32)

## ms-ssim-pyiqa-f32-k2

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=2. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa-f32-k2 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa-f32-k2 convention.",
  "fixed_factor_contract": {
    "kernel_size": 2,
    "definition": "Set downsample=True and kernel_size=2.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa-f32-k2) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa-f32-k2)

## ms-ssim-pyiqa-f32-k3

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=3. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa-f32-k3 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa-f32-k3 convention.",
  "fixed_factor_contract": {
    "kernel_size": 3,
    "definition": "Set downsample=True and kernel_size=3.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa-f32-k3) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa-f32-k3)

## ms-ssim-pyiqa-f32-k4

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=4. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa-f32-k4 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa-f32-k4 convention.",
  "fixed_factor_contract": {
    "kernel_size": 4,
    "definition": "Set downsample=True and kernel_size=4.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa-f32-k4) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa-f32-k4)

## ms-ssim-pyiqa-f32-k5

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=5. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa-f32-k5 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa-f32-k5 convention.",
  "fixed_factor_contract": {
    "kernel_size": 5,
    "definition": "Set downsample=True and kernel_size=5.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa-f32-k5) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa-f32-k5)

## ms-ssim-pyiqa-f32-k6

FR · Classical · CPU · Not documented

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=6. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ms-ssim-pyiqa-f32-k6 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ms-ssim-pyiqa-f32-k6 convention.",
  "fixed_factor_contract": {
    "kernel_size": 6,
    "definition": "Set downsample=True and kernel_size=6.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ms-ssim-pyiqa-f32-k6) · [Speed](https://metric-brief.pages.dev/#speed&q=ms-ssim-pyiqa-f32-k6)

## msssim

FR · Classical · CPU · Reported results

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** Wang, Simoncelli, Bovik 2003 (ACSSC)

**Reference implementation:** authors' ssim_mscale_new.m

**Reference identity:** authors' ssim_mscale_new.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** MS-SSIM — multi-scale SSIM, CPU-only in-tree msssim crate
(port of the authors' msssim.m: 5-level Gaussian-pyramid
SSIM, weighted geometric mean of per-level means). Similarity
on ~[0, 1]: 1 = identical (constant inputs also score 1.0 —
no NaN). Emits msssim_imazen_v. Not routed through the
umbrella or the orchestrator (CPU-only, no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric msssim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → BT.601 luma"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 15 Octave goldens; observed: 8.8e-6",
  "oracle": "authors' ssim_mscale_new.m"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/msssim/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/msssim/README.md)

# msssim

Pure-Rust CPU implementation of MS-SSIM (multi-scale structural
similarity), ported for behavioral fidelity to the authors' reference
msssim.m + ssim_index_new.m (Wang, Simoncelli & Bovik, IEEE
Asilomar 2003) and validated against it under GNU Octave.

- msssim_plane_f32 — MS-SSIM of two 0–255-scale planes.
- msssim_rgb8 — on the unrounded MATLAB rgb2gray luma plane
  (0.2989/0.5870/0.1140).

Pipeline, matching the reference: per level, 11×11 separable
Gaussian-weighted filter2 'valid' statistics (mu, sigma²,
sigma12), SSIM and contrast–structure maps with C1 = (0.01·255)²,
C2 = (0.03·255)², both map means; between levels a forward 2×2 box
imfilter 'symmetric'/'same' + 1:2:end decimation; final score
prod(mcs[1..L−1]^w[1..L−1]) · mssim[L]^w[L] on the canonical
[0.0448 0.2856 0.3001 0.2363 0.1333] weights.

Level count: the reference hard-errors below min(w,h) < 11·2^(L−1);
this port auto-selects level = min(5, floor(log2(min/11))+1) and
uses the first level weights — identical to calling the reference
with (level=L, weight=w(1:L)) (the goldens pin this at L = 1, 2, 3
and 5). min < 11 returns Error::TooSmall.

Negative means make the reference score complex (x.^w → |x|^w·e^{iπw});
this port returns its real part — mag·cos(π·Σw_neg) — matching what
the reference's prod yields after real(). Identical inputs score
exactly 1.0, including constant ones (no NaN — C1, C2 > 0).

## Validation

src/tests.rs checks 15 Octave goldens (even/odd dims, levels 1–5,
the 176 boundary, identical/constant/zero, negative-mean, RGB) — see
validation/ for the generator and provenance. Worst observed delta
~8.8e-6 (f32 planes vs the reference's f64). Scalar and SIMD tiers
are bit-identical by construction (fixed-order f32 elementwise math,
fixed-order f64 pooling).

## Features

- std (default) — std + alloc; without it the crate is no_std
  + alloc.
- avx512 — enable the x86-64 AVX-512 tier at runtime dispatch.
- parallel — rayon across the ref/dis filter pairs (bit-identical
  at any thread count).
- _dev — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2-FMA / AVX-512 / NEON / wasm128) via
archmage/magetypes, dispatched on first call.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/msssim/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/msssim/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/msssim/LICENSE-libvmaf](https://github.com/imazen/zenmetrics/blob/master/crates/msssim/LICENSE-libvmaf)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=msssim) · [Speed](https://metric-brief.pages.dev/#speed&q=msssim)

## msssim-libvmaf

FR · Classical · CPU · Reported results

**Paper:** [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216). Z. Wang; E.P. Simoncelli; A.C. Bovik. The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003. None.

```bibtex
@misc{MSSSIMNone,
  title = {Multiscale structural similarity for image quality assessment},
  author = {Z. Wang and E.P. Simoncelli and A.C. Bovik},
  year = {None},
  note = {The Thrity-Seventh Asilomar Conference on Signals, Systems &amp; Computers, 2003},
  url = {https://doi.org/10.1109/ACSSC.2003.1292216},
  doi = {10.1109/ACSSC.2003.1292216}
}
```

**Recorded paper:** Wang/Simoncelli/Bovik 2003 (libvmaf's 5-scale variant)

**Reference implementation:** libvmaf float_ms_ssim @ f85a8536 — FFI oracle

**Reference identity:** libvmaf float_ms_ssim @ f85a8536 — FFI oracle

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** MS-SSIM — the libvmaf float_ms_ssim feature (five-scale 9/7
low-pass pyramid, 11×11 Gaussian, β/γ weights 0.0448/0.2856/
0.3001/0.2363/0.1333 with α=l only at the coarsest scale) via
msssim::float_ms_ssim, FFI-verified against vendored libvmaf
f85a8536. Same studio-601 luma ingress as ssim-libvmaf; the
JPEG AIC-4 MS-SSIM column. NOT the Wang-MATLAB msssim
variant (different pyramid, luma, and window handling).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric msssim-libvmaf --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → studio-swing BT.601 luma — the YUV Y plane libvmaf consumes; NOT the MATLAB rgb2gray luma"
}
```

**Parity:**

```json
{
  "reported": "reported gate: ≤ 2e-4 (same harness); observed: ≪ gate",
  "oracle": "libvmaf float_ms_ssim @ f85a8536 — FFI oracle"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=msssim-libvmaf) · [Speed](https://metric-brief.pages.dev/#speed&q=msssim-libvmaf)

## musiq

NR · Learned · CPU · Reported results

**Paper:** [MUSIQ: Multi-scale Image Quality Transformer](https://arxiv.org/abs/2108.05997). Ke, Junjie; Wang, Qifei; Wang, Yilin; Milanfar, Peyman; Yang, Feng. ICCV 2021. 2021 (arXiv submission year; publication venue may be later).

```bibtex
@misc{musiq2021,
  title = {MUSIQ: Multi-scale Image Quality Transformer},
  author = {Ke, Junjie and Wang, Qifei and Wang, Yilin and Milanfar, Peyman and Yang, Feng},
  year = {2021},
  note = {ICCV 2021},
  url = {https://arxiv.org/abs/2108.05997},
  eprint = {2108.05997},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Ke, Wang, Wang, Milanfar, Yang. MUSIQ: Multi-scale Image Quality Transformer. ICCV 2021. Google Research

**Reference implementation:** https://github.com/google-research/google-research/tree/master/musiq

**Reference identity:** git clone of google-research (sparse musiq/), pinned 3ef5211; legacy stack jax 0.3.25 + flax 0.3.3 (flax.nn) + tensorflow CPU on python 3.9 — Kaggle/TF-Hub google/musiq instances are conversions of the same npz checkpoints (docs/pyiqa_licenses_2026-10-15/deep_origins.tsv)

**Reference code license:** Apache-2.0

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** MUSIQ — learned no-reference quality score (upstream
google-research/google-research/musiq, Apache-2.0 incl.
koniq_ckpt.npz; multi-scale transformer over 32x32 patches at
224/384/full resolution); higher is better, MOS-scale score.
Scored through PairedNoReference: emits musiq (distorted) and
musiq_ref (reference). As arniqa for
ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/musiq/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric musiq --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "gs://gresearch/musiq/koniq_ckpt.npz (163 MB, opt/target/ flax params; sha256 recorded in musiq_export.meta provenance)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "u8/255 then -1 + t2 (upstream normalize_value_range -> [-1,1])",
  "resize": "per scale s in {224,384}: keep aspect, longer side = s via tf.image.resize GAUSSIAN — scale_and_translate kernel_type='gaussian' antialias=False: kernel_scale=1, sigma=0.5, radius 1.5, w=exp(-2 d^2) for |d|<1.5, reciprocal-multiply normalisation, vertical-then-horizontal. LOCAL PORT in src/tfresize.rs (LICENSE-tensorflow), f64 Gaussian exponential followed by f32 coefficients and accumulation — TEMPORARY: moves to zenresize when its gaussian mode lands: zenresize PR #18's gaussian_tf() + .with_antialias(false) is the replacement once merged (kernel_scale=1, sigma=0.5, radius 1.5)",
  "patches": "32x32 stride-32 SAME zero-pad per scale; row = [3072 hwc pixels, spatial hash (tf v1 NEAREST of arange(10) over the patch grid), scale id, mask]; rescaled groups pad to ceil(s/32)^2 rows (49+144), full-res group uncapped -> [1, L, 3075], L = 193 + ceil(h/32)ceil(w/32)",
  "min_size": 1,
  "max_rows": 8192
}
```

**Parity:**

```json
{
  "oracle": "google-research/musiq revision 3ef5211, run_predict_image.py preprocess+Model.call, CPU fp32, koniq_ckpt.npz",
  "tolerance_jnd": 0.001,
  "jnd": 1.968925,
  "results": "rten worst |d| 2.37e-4 = 1.20e-4 JND; ort-cuda worst |d| 1.37e-4 = 6.98e-5 JND against upstream JAX goldens over 24 ladder pairs, including non-square images and the 2048 rung. Goldens: models/musiq/musiq_goldens.tsv; source correction and measurements: ff792750. Residual floating accumulation differences remain within the 1e-3 JND gate."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/musiq/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/musiq/card.toml) · [crates/zenmetrics-nn/LICENSE-musiq](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-musiq)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=musiq) · [Speed](https://metric-brief.pages.dev/#speed&q=musiq)

## niqe

NR · Classical · CPU · Not documented

**Paper:** [Making a “Completely Blind” Image Quality Analyzer](https://doi.org/10.1109/LSP.2012.2227726). A. Mittal; R. Soundararajan; A. C. Bovik. IEEE Signal Processing Letters. 2013.

```bibtex
@misc{NIQE2013,
  title = {Making a “Completely Blind” Image Quality Analyzer},
  author = {A. Mittal and R. Soundararajan and A. C. Bovik},
  year = {2013},
  note = {IEEE Signal Processing Letters},
  url = {https://doi.org/10.1109/LSP.2012.2227726},
  doi = {10.1109/LSP.2012.2227726}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not recorded

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** NIQE — no-reference natural-image-quality evaluator (in-tree niqe
crate, port of BasicSR niqe.py with LIVE's pristine-model params).
Mahalanobis distance to the pristine MVG; higher is worse. Scored
through PairedNoReference: emits niqe (distorted) and niqe_ref
(reference). Needs nr.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nr
target/release/zenmetrics score --metric niqe --reference ref.png --distorted dist.png
```

No-reference: the score describes the distorted image; the CLI still takes both paths.

**Weights:**

```json
{
  "notice": "Static trained/statistical parameters; consult the linked crate README and licence notices. Exact checkpoint terms are not normalized in this card."
}
```

**Inputs:**

```json
{
  "cli_contract": "NIQE — no-reference natural-image-quality evaluator (in-tree niqe\ncrate, port of BasicSR niqe.py with LIVE's pristine-model params).\nMahalanobis distance to the pristine MVG; higher is worse. Scored\nthrough PairedNoReference: emits niqe (distorted) and niqe_ref\n(reference). Needs nr."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/niqe/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/niqe/README.md)

# niqe

Pure-Rust CPU port of NIQE (Natural Image Quality Evaluator), a
classical no-reference image-quality metric, as a
zenmetrics-abstractions::NoReferenceMetric.

 Origin: basicsr/metrics/niqe.py (https://github.com/XPixelGroup/BasicSR)
  (BasicSR 1.4.2, Apache-2.0 — XPixel Group; notice in
  LICENSE-BasicSR (LICENSE-BasicSR)), a Python port of the official
  MATLAB niqe_release.
 Model data: src/params.rs embeds the pristine-dataset MVG parameters
  (mu 36-dim, cov 36×36) plus the 7×7 Gaussian window. BasicSR's
  niqe_pris_params.npz is bit-identical to modelparameters.mat in
  LIVE's niqe_release.zip (http://live.ece.utexas.edu/research/quality/niqe_release.zip)
  (verified 2026-10-15, maxdiff 0.0 on both arrays). The LIVE notice — which
  permits use/copy/modification/distribution provided it appears in all
  copies — is kept verbatim in LICENSE-LIVE-niqe (LICENSE-LIVE-niqe).
  params.rs is regenerated by tests/gen_params.py; provenance sha256s
  are in that file's header.
 Paper: A. Mittal, R. Soundararajan and A. C. Bovik, "Making a 'Completely
  Blind' Image Quality Analyzer", IEEE Signal Processing Letters,
  vol. 20, no. 3, pp. 209–212, March 2013.
  <https://doi.org/10.1109/LSP.2012.2227726>

## Conventions (matching the oracle)

 RGB → gray is the bgr2ycbcr Y channel — the studio-swing BT.601
  luma MATLAB's rgb2ycbcr applies (NOT rgb2gray's full-range
  0.299/0.587/0.114):
  Y ≈ round(255·(16 + 65.481·r + 128.553·g + 24.966·b)/255) on
  [0,1]-normalised channels, preserving the oracle's f64-dot → f32 cast →
  f32 ×255 → f32-round chain. GRAY8 input is taken verbatim.
 Niqe::with_crop_border(k) mirrors the oracle's crop_border argument
  (default 0). The image is then cropped to the largest multiple of 96 px.
 Non-overlapping 96×96 blocks at scale 1; the image is halved by the
  MATLAB-compatible antialiased bicubic imresize and 48×48 blocks at
  scale 2. Features: 18 AGGD parameters per scale → 36-dim vector; blocks
  containing NaN are dropped from the covariance exactly like the oracle.
 Score: Mahalanobis distance of the block-feature MVG to the pristine MVG.
  Higher = worse (ScoreKind::Distortion); scalar output only (map
  requests are refused); typical range ~0–10 on natural images.
 Images with fewer than two complete 96×96 blocks after crop_border
  report ErrorKind::TooSmall — the oracle crashes there (np.cov of <2
  rows is NaN → np.linalg.pinv: "SVD did not converge"); this port
  surfaces it as an error instead (documented divergence). Content whose
  features go non-finite (e.g. a flat image, where the oracle propagates
  NaN through nanmean) reports ErrorKind::Degenerate.

## Usage

rust
use niqe::Niqe;
use zenmetrics_abstractions::NoReferenceMetric;

let mut metric = Niqe::new();          // crop_border = 0
// or Niqe::with_crop_border(k)


Output key: niqe. Through PairedNoReference, pairs additionally emit
niqe_ref.

## Parity

tests/parity.rs checks tests/goldens_niqe.tsv (42 ladder images + 128
KonJND reference/distorted images, generated by tests/golden_niqe.py
against basicsr==1.4.2) within JND/1000 of the oracle, and prints the
worst absolute delta and the worst delta in JND units. TSV paths are
relative to $ZENMETRICS_NR_GOLDEN_ROOT (default [local path]). JND (in
src/lib.rs) is the median |niqe − niqe_ref| over all 1008 KonJND pairs;
see benchmarks/nr_classic_2026-10-15/jnd.md.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/niqe/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/niqe/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/niqe/LICENSE-BasicSR](https://github.com/imazen/zenmetrics/blob/master/crates/niqe/LICENSE-BasicSR) · [crates/niqe/LICENSE-LIVE-niqe](https://github.com/imazen/zenmetrics/blob/master/crates/niqe/LICENSE-LIVE-niqe)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=niqe) · [Speed](https://metric-brief.pages.dev/#speed&q=niqe)

## nlpd

FR · Classical · CPU · Reported results

**Paper:** [Perceptual image quality assessment using a normalized Laplacian pyramid](https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103). Valero Laparra; Johannes Ballé; Alexander Berardino; Eero P Simoncelli. Electronic Imaging. 2016.

```bibtex
@misc{NLPD2016,
  title = {Perceptual image quality assessment using a normalized Laplacian pyramid},
  author = {Valero Laparra and Johannes Ballé and Alexander Berardino and Eero P Simoncelli},
  year = {2016},
  note = {Electronic Imaging},
  url = {https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103},
  doi = {10.2352/ISSN.2470-1173.2016.16.HVEI-103}
}
```

**Recorded paper:** Laparra et al., normalized Laplacian pyramid distance (laparra16a-preprint)

**Reference implementation:** Valerolaparra/NLPD_Pytorch (the authors' PyTorch reference; PyTorch 2.5.1 oracle)

**Reference identity:** Valerolaparra/NLPD_Pytorch (the authors' PyTorch reference; PyTorch 2.5.1 oracle)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** distance; 0 = identical

**Direction:** not recorded

**Cli contract:** Normalized Laplacian Pyramid Distance; lower is better.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric nlpd --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → RGB planes [0,1]; six levels, (Σ rms^0.6)^(1/0.6) pooling"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 1e-5 golden (97×99 pair); observed: within gate",
  "oracle": "Valerolaparra/NLPD_Pytorch (the authors' PyTorch reference; PyTorch 2.5.1 oracle)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/nlpd/LICENSE-IQA-optimization](https://github.com/imazen/zenmetrics/blob/master/crates/nlpd/LICENSE-IQA-optimization)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=nlpd) · [Speed](https://metric-brief.pages.dev/#speed&q=nlpd)

## nlpd-iqa

FR · Classical · CPU · Reported results

**Paper:** [Perceptual image quality assessment using a normalized Laplacian pyramid](https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103). Valero Laparra; Johannes Ballé; Alexander Berardino; Eero P Simoncelli. Electronic Imaging. 2016.

```bibtex
@misc{NLPD2016,
  title = {Perceptual image quality assessment using a normalized Laplacian pyramid},
  author = {Valero Laparra and Johannes Ballé and Alexander Berardino and Eero P Simoncelli},
  year = {2016},
  note = {Electronic Imaging},
  url = {https://doi.org/10.2352/ISSN.2470-1173.2016.16.HVEI-103},
  doi = {10.2352/ISSN.2470-1173.2016.16.HVEI-103}
}
```

**Recorded paper:** " — the dingkeyan93 IQA_pytorch configuration (NLPD(channels=1), alexhepburn nlpd-tensorflow lineage), NOT pyiqa and NOT the Laparra RGB reference

**Reference implementation:** IQA_pytorch source mirrored at [local path] semantics replicated to med |Δ| 0.0026 / max 0.0067 vs the AIC-4 NLPD column (n=53)

**Reference identity:** IQA_pytorch source mirrored at [local path] semantics replicated to med |Δ| 0.0026 / max 0.0067 vs the AIC-4 NLPD column (n=53)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** distance; 0 = identical

**Direction:** not recorded

**Cli contract:** NLPD with the IQA_pytorch / JPEG AIC-4 ingress — the
dingkeyan93/IQA-optimization NLPD(channels=1) variant the
published AIC-4 NLPD column was computed with, NOT the
Laparra-author RGB configuration that nlpd mirrors. Scored on
a single BT.709 luma plane (0.2126/0.7152/0.0722, [0,1],
re-quantized to the 8-bit grid), reflect-2 downsample, bilinear
align_corners upsample, six loop bands, mean-of-RMS pooling —
vs the nlpd 0.6-norm over RGB (which reads ≈20.5× higher).
med|Δ| = 0.0026 vs metrics_fullres.csv (53 pairs, 2026-09-26).
Emits nlpd_iqa so it never mixes with nlpd's column.
CPU-only.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric nlpd-iqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → BT.709 Y [0,1] re-quantized to the 8-bit grid (iqa::score_rgb_u8); iqa::score_y_f32 for caller-supplied planes"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 1e-4 golden (f64 replica → f32 ops); observed: ~4e-6",
  "oracle": "IQA_pytorch source mirrored at [local path] semantics replicated to med |Δ| 0.0026 / max 0.0067 vs the AIC-4 NLPD column (n=53)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=nlpd-iqa) · [Speed](https://metric-brief.pages.dev/#speed&q=nlpd-iqa)

## paq2piq

NR · Learned · CPU · Reported results

**Paper:** [From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality](https://arxiv.org/abs/1912.10088). Ying, Zhenqiang; Niu, Haoran; Gupta, Praful; Mahajan, Dhruv; Ghadiyaram, Deepti; Bovik, Alan. CVPR 2020. 2019 (arXiv submission year; publication venue may be later).

```bibtex
@misc{paq2piq2019,
  title = {From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality},
  author = {Ying, Zhenqiang and Niu, Haoran and Gupta, Praful and Mahajan, Dhruv and Ghadiyaram, Deepti and Bovik, Alan},
  year = {2019},
  note = {CVPR 2020},
  url = {https://arxiv.org/abs/1912.10088},
  eprint = {1912.10088},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Ying, Niu, Gupta, Mahajan, Ghadiyaram, Bovik. From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality. CVPR 2020

**Reference implementation:** https://github.com/baidut/paq2piq

**Reference identity:** git clone of baidut/paq2piq revision 48c91e844e9f7a768f6ddcfc744dddfdd1160fea (the first author's official inference repo; LICENSE.md added 2024-10-25)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** noncommercial

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** PaQ-2-PiQ — no-reference, authors' baidut/paq2piq (MIT code;
CC-BY-NC-4.0 weights, consent-gated; ResNet-18 + 20x20-block
RoIPool head, RoIPoolModel-fit.10.bs.120.pth); higher is
better, MOS-scale [0, 100]. Scored through
PairedNoReference: emits paq2piq (distorted) and
paq2piq_ref (reference). As arniqa for
ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE, plus
ZENMETRICS_NN_ACCEPT_WEIGHTS=paq2piq.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/paq2piq/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric paq2piq --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=paq2piq. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "noncommercial",
  "checkpoint": "https://github.com/baidut/PaQ-2-PiQ/releases/download/v1.0/RoIPoolModel-fit.10.bs.120.pth (~134 MiB; sha256 5ece3dcd18e7bfc619ef6eee0c6e2c3b94350a623d62409f5cac95a2979b5b32)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "torchvision ToTensor only (u8 -> f32 /255) — upstream applies NO resize and NO ImageNet normalisation",
  "resize": "none — fully convolutional graph at native resolution",
  "crop": "none — deterministic; the ROI grid is computed from (h,w) identically in oracle.py, export.py and paq2piq.rs (linspace(0,dim,21) block boundaries, input-pixel coordinates, upstream get_blockwise_rois)",
  "min_size": 1,
  "max_side": 4096
}
```

**Parity:**

```json
{
  "oracle": "baidut/paq2piq revision 48c91e8 RoIPoolModel + InferenceModel.predict semantics (t[0]), CPU fp32, RoIPoolModel-fit.10.bs.120.pth",
  "tolerance_jnd": 0.001,
  "jnd": 1.229401,
  "results": "export self-check (body->host RoIPool->head vs upstream end-to-end) |d| 3.8e-6; ONNX-CPU vs upstream goldens worst 3.087e-5 over 29 pairs (including 32x32 and 33x47). rten: 29 pairs worst |d| 3.087e-5 (2.51e-5 JND); ort-cuda: original 27 pairs worst |d| 1.572e-5 (1.28e-5 JND)."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/paq2piq/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/paq2piq/card.toml) · [crates/zenmetrics-nn/LICENSE-paq2piq](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-paq2piq)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=paq2piq) · [Speed](https://metric-brief.pages.dev/#speed&q=paq2piq)

## pieapp

FR · Learned · CPU · Reported results

**Paper:** [PieAPP: Perceptual Image-Error Assessment through Pairwise Preference](https://arxiv.org/abs/1806.02067). Prashnani, Ekta; Cai, Hong; Mostofi, Yasamin; Sen, Pradeep. CVPR 2018. 2018 (arXiv submission year; publication venue may be later).

```bibtex
@misc{pieapp2018,
  title = {PieAPP: Perceptual Image-Error Assessment through Pairwise Preference},
  author = {Prashnani, Ekta and Cai, Hong and Mostofi, Yasamin and Sen, Pradeep},
  year = {2018},
  note = {CVPR 2018},
  url = {https://arxiv.org/abs/1806.02067},
  eprint = {1806.02067},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Prashnani, Cai, Mostofi, Sen. PieAPP: Perceptual Image-Error Assessment through Pairwise Preference. CVPR 2018. https://arxiv.org/abs/1806.02067

**Reference implementation:** Official v0.1 public compute_score, release 0513613d1f575136ea52418e35f2cebc5a49f783

**Reference identity:** Official v0.1 public compute_score, release 0513613d1f575136ea52418e35f2cebc5a49f783

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** Modified BSD-2 for Non-Commercial Use Only, UC Regents

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** Learned error; card and provenance (../crates/zenmetrics-nn/models/pieapp/PROVENANCE.md); demo CLI sampling differs

**Direction:** lower-is-better

**Cli contract:** PieAPP v0.1, dense patches; noncommercial weights require consent.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/pieapp/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric pieapp --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "noncommercial",
  "terms": "Modified BSD-2 for Non-Commercial Use Only, UC Regents",
  "terms_url": "https://github.com/prashnani/PerceptualImageError/blob/master/LICENSE.txt",
  "checkpoint": "https://web.ece.ucsb.edu/~ekta/projects/PieAPPv0.1/weights/PieAPPv0.1.pth",
  "sha256": "0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0",
  "consent": "PieApp::WEIGHTS; ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp",
  "bundled": false
}
```

**Inputs:**

```json
{
  "format": "strided RGB8 sRGB",
  "graph_input": "NCHW RGB [0,1]",
  "network_input": "NCHW BGR code values [0,255]",
  "patch_size": 64,
  "stride": 6,
  "sampling_source": "Official release README documents deterministic dense stride 6 as the default; paper section 3 instead describes random sampling of 1024 patches.",
  "min_size": 64,
  "max_side": 4096,
  "max_pixels": 9000000,
  "max_pixels_32bit": 1048576
}
```

**Parity:**

```json
{
  "onnx_sha256": "e3905d3625402369174e1d52fa889f7fed0fe982eab1d47ed49498cbdd868653",
  "tolerance_jnd": 0.001,
  "identical_patch_oracle": -3.725290298461914e-09,
  "status": "Revision 3 qualified: 24 unchanged goldens per engine, exact repeats and warm/cold identity, all merged-tree gates pass. All 1008 KonJND scores and input hashes are unchanged.",
  "qualified_pairs_per_engine": 24,
  "rten_reported_worst_jnd": 1.84e-06,
  "ort_cuda_reported_worst_jnd": 2.69e-07,
  "qualification_receipt": "../../../../benchmarks/pieapp_2026-10-16_qualification.json",
  "previous_onnx_sha256": "7033ea6ec2b0de54c9eb8a2b34002a8f8e026cecf982a4cd51b74a13c2be0322",
  "jnd_scale": 0.5189487912492419,
  "konjnd_pairs": 1008,
  "jpeg_pairs": 504,
  "bpg_pairs": 504
}
```

**Conventions:**

```json
{
  "oracle_convention": {
    "entry_point": "Official PieAPP.compute_score(distorted_patches, reference_patches)",
    "grid": "Row-major valid 64x64 patches, origins 0,6,... <= side-64; no unaligned far-edge patch.",
    "selection": "Canonical parity targets this explicit stride-6 public-entry-point convention.",
    "cli_divergence_case": "1415_128",
    "cli_score": 1.6797861082006782,
    "entry_point_score": 1.6567797660827637,
    "cli_absolute_difference": 0.0230063421179145,
    "cli_equivalence": false,
    "cli_documented_sparse_flag": "--sampling_mode sparse",
    "cli_documented_sparse_score": 1.7122649515845207,
    "cli_explanation": "No documented option has established equivalence. Official source is not read; wrapper probing is restricted to documented options.",
    "cli_difference_jnd": 0.044332586385898314,
    "cli_panel_pairs": 24,
    "cli_panel_repetitions": 2,
    "cli_panel_engine": "Modern compatible official demo on CUDA; values retained at printed precision",
    "cli_panel_worst_case": "2207_2048",
    "cli_panel_worst_cli_score": 3.4582222,
    "cli_panel_worst_entry_point_score": 2.4828926609204816,
    "cli_panel_worst_absolute_difference": 0.9753295390795182,
    "cli_panel_worst_difference_jnd": 1.879433106938455,
    "cli_panel_receipt": "../../../../benchmarks/pieapp_review_2026-10-16/cli.json"
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/pieapp/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/pieapp/card.toml) · [benchmarks/pieapp_2026-10-16_qualification.json](https://github.com/imazen/zenmetrics/blob/master/benchmarks/pieapp_2026-10-16_qualification.json)

**Known divergences:** [PieAPP sampling and official entry point (2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#pieapp-sampling-and-official-entry-point-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=pieapp) · [Speed](https://metric-brief.pages.dev/#speed&q=pieapp)

## pieapp-pyiqa

FR · Learned · CPU · Reported results

**Paper:** [PieAPP: Perceptual Image-Error Assessment through Pairwise Preference](https://arxiv.org/abs/1806.02067). Prashnani, Ekta; Cai, Hong; Mostofi, Yasamin; Sen, Pradeep. CVPR 2018. 2018 (arXiv submission year; publication venue may be later).

```bibtex
@misc{pieapp2018,
  title = {PieAPP: Perceptual Image-Error Assessment through Pairwise Preference},
  author = {Prashnani, Ekta and Cai, Hong and Mostofi, Yasamin and Sen, Pradeep},
  year = {2018},
  note = {CVPR 2018},
  url = {https://arxiv.org/abs/1806.02067},
  eprint = {1806.02067},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Prashnani, Cai, Mostofi, Sen. PieAPP: Perceptual Image-Error Assessment through Pairwise Preference. CVPR 2018

**Reference implementation:** pyiqa 0.1.16 create_metric('pieapp'), CPU fp32, metric(dist, ref)

**Reference identity:** pyiqa 0.1.16 create_metric('pieapp'), CPU fp32, metric(dist, ref)

**Reference code license:** PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab)

**Rust license:** code: PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); weights: nc

**Weights license:** nc

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** Learned error; card and provenance (../crates/zenmetrics-nn/models/pieapp/PROVENANCE.md); demo CLI sampling differs

**Direction:** lower-is-better

**Cli contract:** PieAPP as pyiqa scores it (64×64 patches at stride 27; the AIC2026
PieAPP column, lower is better, ≥ 64²). Reads zenmetrics-nn's
pieapp.onnx. Needs feature pyiqa-ports and
ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp_pyiqa.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports,nn
# export the model once (see crates/zenmetrics-pyiqa-ports/models/pieapp-pyiqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric pieapp-pyiqa --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp_pyiqa. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "nc",
  "checkpoint": "PieAPPv0.1 (pyiqa: https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/PieAPPv0.1-0937b014.pth; official: https://web.ece.ucsb.edu/~ekta/projects/PieAPPv0.1/weights/PieAPPv0.1.pth)",
  "sha256": "0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0",
  "terms_url": "https://github.com/prashnani/PerceptualImageError/blob/master/LICENSE.txt",
  "consent": "PieAppPyiqa::WEIGHTS; ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp_pyiqa"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "graph": "zenmetrics-nn's pieapp.onnx unchanged (crates/zenmetrics-nn/models/pieapp/export.py): reference, distorted [n,3,64,64] RGB in [0,1] -> [n,2] = error, weight + 1e-6",
  "patches": "64x64 at stride 27, no padding: origins 0, 27, ... <= d - 64",
  "min_size": 64
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 create_metric('pieapp'), CPU fp32, metric(dist, ref)",
  "goldens": "goldens_ladder.tsv (common/ladder.py, 15 pairs 64x64..2048x1536)",
  "jnd": 0.514383,
  "tolerance_jnd": 0.001,
  "measured_ladder": "rten 1.19e-6 (2.3e-6 JND), ort-cpu 7.2e-7 (1.4e-6 JND), ort-cuda 4.8e-7 (9.3e-7 JND)",
  "measured_aic2026_61": "ort-cuda 3.5e-7 (6.8e-7 JND); pyiqa vs the published column: median 2.3e-5, max 1.9e-4",
  "goldens_note": "pyiqa runs all patches of an image in one batch (> 14 GiB at 2048x1536); goldens.py with PIQA_FEATURE_CHUNK=256 runs compute_features 256 patches at a time. Chunked and unchunked goldens are bit-identical on the four pairs both computed (64x64 .. 1024x1024).",
  "warm_reference": "keeps the reference's planar copy only (the network runs both patches of every pair); bit-identical to cold",
  "recorded_source_comments": "1 JND: pooled median score over the 1,008 KonJND-1k pairs, pyiqa CPU (JPEG 0.517011, BPG 0.509688).\nperf_deltas.py, i265 (Core Ultra 7 265K), rten, 1024x1024 ladder pair, cold evaluate per call;\nbenchmarks/iqa_complexity_2026-10-17/measured/pyiqa_ports_i265_1024.{tsv,meta}."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/pieapp-pyiqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/pieapp-pyiqa/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=pieapp-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=pieapp-pyiqa)

## piqe

NR · Classical · CPU · Not documented

**Paper:** [Blind image quality evaluation using perception based features](https://doi.org/10.1109/NCC.2015.7084843). Venkatanath N; Praneeth D; Maruthi Chandrasekhar Bh; Sumohana S. Channappayya; Swarup S. Medasani. 2015 Twenty First National Conference on Communications (NCC). 2015.

```bibtex
@misc{PIQE2015,
  title = {Blind image quality evaluation using perception based features},
  author = {Venkatanath N and Praneeth D and Maruthi Chandrasekhar Bh and Sumohana S. Channappayya and Swarup S. Medasani},
  year = {2015},
  note = {2015 Twenty First National Conference on Communications (NCC)},
  url = {https://doi.org/10.1109/NCC.2015.7084843},
  doi = {10.1109/NCC.2015.7084843}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PIQE — no-reference perception-based block distortion (in-tree
piqe crate, port of pypiqe 1.2 / MATLAB piqe). Score in
[0, 100]; higher is worse. Scored through PairedNoReference:
emits piqe (distorted) and piqe_ref (reference). Needs nr.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nr
target/release/zenmetrics score --metric piqe --reference ref.png --distorted dist.png
```

No-reference: the score describes the distorted image; the CLI still takes both paths.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PIQE — no-reference perception-based block distortion (in-tree\npiqe crate, port of pypiqe 1.2 / MATLAB piqe). Score in\n[0, 100]; higher is worse. Scored through PairedNoReference:\nemits piqe (distorted) and piqe_ref (reference). Needs nr."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/piqe/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/piqe/README.md)

# piqe

PIQE — Perception-based Image Quality Evaluator — as a pure-Rust, CPU,
no-reference metric for zenmetrics. Score is in [0, 100]; higher is
worse (ScoreKind::Distortion).

This is a port of pypiqe (https://github.com/michael-rutherford/pypiqe) 1.2
(MIT, Copyright (c) 2023 Michael Rutherford — notice kept in
LICENSE-pypiqe), a Python conversion of MATLAB's piqe that the author
tested against the MATLAB implementation. There are no learned parameters.

## Algorithm

1. Convert to gray: OpenCV COLOR_BGR2GRAY, i.e. the exact 15-bit
   fixed-point Y = (3735·B + 19235·G + 9798·R + 16384) >> 15.
2. Pad the image bottom/right with numpy symmetric (edge-inclusive
   mirror) to a multiple of 16; rescale to [0, 255] via
   round(255·v/max) (ties to even).
3. Normalize locally: imnorm = (im − mu)/(sigma + 1) where mu, sigma
   are the 7-tap Gaussian (σ = 7/6, replicate borders) mean/std in f64.
4. For each 16×16 block of imnorm: skip unless var > 0.1 (spatially
   active). Active blocks add impaired·(1 − var) + noisy·var to the
   distortion score, where impaired = any of the 44 overlapping 6-pixel
   edge segments has std < 0.1, and noisy = blockSigma > 2·blockBeta
   from the center/surround deviation ratio.
5. score = (distortion + 1)/(NHSA + 1)·100; a uniform image scores 100.

Deliberate divergence from the oracle: pypiqe scores inputs smaller than
one block by symmetric-padding them up; this port returns
ErrorKind::TooSmall below 16×16 instead.

## Usage

rust
use piqe::Piqe;
use zenmetrics_abstractions::NoReferenceMetric;

let mut m = Piqe::new();
// m.evaluate_image(image, opts) -> ScalarEval { "piqe": score }


The CLI exposes it as --metric piqe (feature nr) through
PairedNoReference, so a batch/score run reports piqe for the
distorted image and piqe_ref for the reference.

## Paper

> Venkatanath N., Praneeth D., Maruthi Chandrasekhar Bh., Channappayya S. S.,
> Medasani S. S., "Blind Image Quality Evaluation Using Perception Based
> Features", 2015 Twenty First National Conference on Communications
> (NCC), pp. 1–6. doi:10.1109/NCC.2015.7084843

## Parity

tests/golden_piqe.py regenerates tests/goldens_piqe.tsv (ladder PNGs +
64 KonJND images) with pypiqe==1.2 on opencv-python; tests/parity.rs
scores the same images and asserts each golden is within JND/1000, printing
the worst |Δ| in absolute and JND units. TSV paths are relative to
$ZENMETRICS_NR_GOLDEN_ROOT (default [local path]).

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/piqe/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/piqe/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/piqe/LICENSE-pypiqe](https://github.com/imazen/zenmetrics/blob/master/crates/piqe/LICENSE-pypiqe)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=piqe) · [Speed](https://metric-brief.pages.dev/#speed&q=piqe)

## proposal-butteraugli

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AIC2026 two-segment calibration of the libjxl Butteraugli 3-norm.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric proposal-butteraugli --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "native_output": "butteraugli pnorm3, existing CPU defaults",
  "output_map": "max(0.00459792N, 2.0938201N - 1.1033102)",
  "source": "crates/zenmetrics-cli/src/metrics/mod.rs"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=proposal-butteraugli) · [Speed](https://metric-brief.pages.dev/#speed&q=proposal-butteraugli)

## psnr

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Peak signal-to-noise ratio on encoded RGB8, in decibels.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Peak signal-to-noise ratio on encoded RGB8, in decibels."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr)

## psnr-cb

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PSNR of the studio-swing BT.601 Cb plane at full resolution (4:4:4),
the psnr-ycbcr611 conversion. JPEG AIC-4 PSNR-Cb: 61 pairs
median |Δ| 1.9e-3 dB, max 1.7e-2 dB (max 2.0e-3 JND).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-cb --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PSNR of the studio-swing BT.601 Cb plane at full resolution (4:4:4),\nthe psnr-ycbcr611 conversion. JPEG AIC-4 PSNR-Cb: 61 pairs\nmedian |Δ| 1.9e-3 dB, max 1.7e-2 dB (max 2.0e-3 JND)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-cb) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-cb)

## psnr-cr

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PSNR of the studio-swing BT.601 Cr plane at full resolution (4:4:4),
the psnr-ycbcr611 conversion. JPEG AIC-4 PSNR-Cr: 61 pairs
median |Δ| 2.1e-3 dB, max 5.7e-2 dB (max 8.8e-3 JND).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-cr --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PSNR of the studio-swing BT.601 Cr plane at full resolution (4:4:4),\nthe psnr-ycbcr611 conversion. JPEG AIC-4 PSNR-Cr: 61 pairs\nmedian |Δ| 2.1e-3 dB, max 5.7e-2 dB (max 8.8e-3 JND)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-cr) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-cr)

## psnr-y

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Peak signal-to-noise ratio on full-range BT.709 luma.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-y --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Peak signal-to-noise ratio on full-range BT.709 luma."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-y) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-y)

## psnr-y-libvmaf

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PSNR-Y on studio-swing BT.709 luma — round(16 + 219·L709) with
L709 = (0.2126R + 0.7152G + 0.0722B)/255 — the Y-plane
convention the CLI feeds the in-tree vmaf crate, i.e. what
libvmaf's psnr aux feature reported under the removed exec
adapter.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-y-libvmaf --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PSNR-Y on studio-swing BT.709 luma — round(16 + 219·L709) with\nL709 = (0.2126R + 0.7152G + 0.0722B)/255 — the Y-plane\nconvention the CLI feeds the in-tree vmaf crate, i.e. what\nlibvmaf's psnr aux feature reported under the removed exec\nadapter."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-y-libvmaf) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-y-libvmaf)

## psnr-y-studio601

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PSNR-Y on studio-swing BT.601 luma — round(16 + (65.481R +
128.553G + 24.966B)/255) per-pixel u8 — the JPEG/JFIF YUV420 Y
convention and the JPEG AIC-4 PSNR-Y column (the same plane
ssim-libvmaf/msssim-libvmaf/psnrhvs-daala build internally;
equal to --metric psnr-y --luma-ingress yuv601-studio).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-y-studio601 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PSNR-Y on studio-swing BT.601 luma — round(16 + (65.481R +\n128.553G + 24.966B)/255) per-pixel u8 — the JPEG/JFIF YUV420 Y\nconvention and the JPEG AIC-4 PSNR-Y column (the same plane\nssim-libvmaf/msssim-libvmaf/psnrhvs-daala build internally;\nequal to --metric psnr-y --luma-ingress yuv601-studio)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-y-studio601) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-y-studio601)

## psnr-y601

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PSNR-Y on full-range BT.601 luma (the MATLAB rgb2gray weights
0.299/0.587/0.114) — the most common "PSNR-Y" convention in the
literature; NOT the same numbers as psnr-y (BT.709).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-y601 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PSNR-Y on full-range BT.601 luma (the MATLAB rgb2gray weights\n0.299/0.587/0.114) — the most common \"PSNR-Y\" convention in the\nliterature; NOT the same numbers as psnr-y (BT.709)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-y601) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-y601)

## psnr-ycbcr611

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** PSNR-YCbCr 6:1:1 — weighted PSNR on studio-swing BT.601 YCbCr
(6·MSE_Y + MSE_Cb + MSE_Cr)/8 converted to dB. Matches JPEG AIC-4
PSNR-YCbCr611 column. Studio-swing conversion per pixel:
Y=round(16+(65.481R+128.553G+24.966B)/255),
Cb=round(128+(-37.797R-74.203G+112B)/255),
Cr=round(128+(112R-93.786G-18.214B)/255).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnr-ycbcr611 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "PSNR-YCbCr 6:1:1 — weighted PSNR on studio-swing BT.601 YCbCr\n(6·MSE_Y + MSE_Cb + MSE_Cr)/8 converted to dB. Matches JPEG AIC-4\nPSNR-YCbCr611 column. Studio-swing conversion per pixel:\nY=round(16+(65.481R+128.553G+24.966B)/255),\nCb=round(128+(-37.797R-74.203G+112B)/255),\nCr=round(128+(112R-93.786G-18.214B)/255)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnr-ycbcr611) · [Speed](https://metric-brief.pages.dev/#speed&q=psnr-ycbcr611)

## psnrhvs

FR · Classical · CPU · Reported results

**Paper: citation needed.**

**Recorded paper:** Ponomarenko/Egiazarian PSNR-HVS-M (VPQM 2007)

**Reference implementation:** authors' psnrhvsm.m

**Reference identity:** authors' psnrhvsm.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** dB, higher better

**Direction:** not recorded

**Cli contract:** PSNR-HVS / PSNR-HVS-M (Egiazarian et al. VPQM-06, Ponomarenko et al.
VPQM-07) — CPU implementation via the in-tree psnrhvs crate (a port
of the authors' psnrhvsm.m). PSNR-like: dB scale, higher = better,
100000 = identical/visually indistinguishable. Scored per sRGB
channel on the 0–255 scale and averaged (the multi-plane convention
codec harnesses report). Emits both psnrhvs_imazen_v and
psnrhvsm_imazen_v — the masked score is a byproduct of the same
pass. Not routed through the umbrella or the orchestrator (CPU-only,
no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnrhvs --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "RGB per-channel 8×8 DCT (-y = luma plane)"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 1e-3 dB; observed: ≤ 4e-4 dB",
  "oracle": "authors' psnrhvsm.m"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/psnrhvs/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/psnrhvs/README.md)

# psnrhvs

Pure-Rust CPU port of PSNR-HVS and PSNR-HVS-M —

- PSNR-HVS (K. Egiazarian, J. Astola, N. Ponomarenko, V. Lukin,
  F. Battisti, M. Carli, "New full-reference quality metrics based on
  HVS", VPQM-06): PSNR on 8×8 DCT coefficients weighted by a contrast
  sensitivity function.
- PSNR-HVS-M (N. Ponomarenko, F. Silvestri, K. Egiazarian, M. Carli,
  J. Astola, V. Lukin, "On between-coefficient contrast masking of DCT
  basis functions", VPQM-07): additionally subtracts a per-block
  masking threshold before weighting — the metric codec-evaluation
  reports (Daala/AV1-style psnr_hvs) usually mean.

The port follows the authors' reference implementation psnrhvsm.m
(Nikolay Ponomarenko, <http://ponomarenko.info>). That implementation is
licensed for educational/research use and is not redistributed here; the
port implements the published algorithm and constants, verified against
the reference under GNU Octave (see validation/ (validation/)).

## Algorithm

1. Tile into 8×8 blocks (default step 8, the reference's wstep;
   partial edge strips are dropped, as the reference does).
2. Orthonormal 2D DCT-II of both blocks (MATLAB dct2 convention).
3. Masking threshold (maskeff, HVS-M only): non-DC coefficient
   energy Σ z_dct²·MaskCof scaled by the ratio of the four 4×4-quadrant
   variances to the whole-block variance, sqrt(e·pop)/32; block mask =
   max(mask_ref, mask_dist).
4. Per coefficient u = |A_dct − B_dct|: PSNR-HVS accumulates
   (u·CSFCof)²; PSNR-HVS-M uses u' = max(u − mask/MaskCof, 0) on
   non-DC coefficients (DC is never masked).
5. 10·log10(255²/mean_energy), or 100000 when the energy is zero
   (identical / visually indistinguishable).

## API

rust
let s = psnrhvs::psnrhvs_luma8(&ref_rgb, &dist_rgb, w, h, w  3)?;
// s.psnr_hvs      — PSNR-HVS   (dB, higher = better)
// s.psnr_hvs_m    — PSNR-HVS-M (dB, >= psnr_hvs)


- psnrhvs_plane_f32(r, d, w, h, stride) — strided f32 planes on the
  0..255 scale (psnrhvs_plane_f32_step exposes wstep).
- psnrhvs_rgb8(r, d, w, h, stride) — interleaved sRGB; each channel is
  scored as a plane and the scores are averaged (the multi-plane
  convention codec harnesses report; RgbScore also carries the
  per-channel PlaneScores).
- psnrhvs_luma8(r, d, w, h, stride) — BT.601 luma
  (round(0.299R + 0.587G + 0.114B), same as gmsd's), the
  PSNR-HVS-Y codec-benchmark shape.

Images need at least one complete 8×8 block (Error::TooSmall below).

## Determinism

Per-block energies accumulate in f32 in fixed coefficient order; the
global sum is f64 in raster order, so every SIMD tier (scalar, v3
AVX2, v4 AVX-512 behind avx512, neon, wasm128) and every
parallel thread count produces bit-identical output. Final scoring
uses libm::log10 for cross-platform bit agreement.

Features: std (default), parallel (rayon band parallelism),
avx512, _dev (per-tier entry points for parity tests/disassembly).

## Verification

src/tests.rs pins 15 golden rows produced by psnrhvsm.m under
Octave (plane patterns covering step≠8, sub-8×8-multiple sizes,
constant regions, DC shifts, clamping; plus per-channel and luma rows
for the color entry points). All pass within 1e-3 dB — the residual is
the port's f32 block pipeline vs the reference's f64; the observed
delta is ≤ ~4e-4 dB.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/psnrhvs/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/psnrhvs/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/psnrhvs/LICENSE-xiph-daala](https://github.com/imazen/zenmetrics/blob/master/crates/psnrhvs/LICENSE-xiph-daala)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnrhvs) · [Speed](https://metric-brief.pages.dev/#speed&q=psnrhvs)

## psnrhvs-daala

FR · Classical · CPU · Reported results

**Paper: citation needed.**

**Recorded paper:** Daala/Xiph PSNR-HVS (integer bin-DCT variant adopted by libvmaf; distinct from the Ponomarenko MATLAB psnrhvs)

**Reference implementation:** libvmaf psnr_hvs feature @ f85a8536 — FFI oracle

**Reference identity:** libvmaf psnr_hvs feature @ f85a8536 — FFI oracle

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** dB; emits psnrhvs_daala_{y,cb,cr} + combined

**Direction:** not recorded

**Cli contract:** PSNR-HVS — the Daala/libvmaf dump_psnrhvs feature (integer
bin-DCT 8×8 blocks at step 7, CSF-masked error with variance-
ratio masking, 0.8·Y + 0.1·(Cb + Cr) combined) via
psnrhvs::daala, FFI-verified against vendored libvmaf f85a8536.
Input planes are studio-swing BT.601 Y + full-resolution Cb/Cr
(YUV444 — the convention the JPEG AIC-4 PSNR-HVS/
PSNR-HVS-{Y,Cb,Cr} columns reproduce under). NOT the Ponomarenko
psnrhvs metric (floating DCT, different CSF + masking —
~7.6 dB stricter scoring). Emits four columns:
psnrhvs_daala_{y,cb,cr} + combined psnrhvs_daala.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnrhvs-daala --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → studio-swing BT.601 YUV444 planes — the AIC-4 convention (psnr_hvs_daala_yuv420 also exposed for YUV420 pictures)"
}
```

**Parity:**

```json
{
  "reported": "reported gate: ≤ 2e-4 dB (420 + 444 cases); observed: ≪ gate",
  "oracle": "libvmaf psnr_hvs feature @ f85a8536 — FFI oracle"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnrhvs-daala) · [Speed](https://metric-brief.pages.dev/#speed&q=psnrhvs-daala)

## psnrhvs-y

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** Ponomarenko/Egiazarian PSNR-HVS-M (VPQM 2007)

**Reference implementation:** authors' psnrhvsm.m

**Reference identity:** authors' psnrhvsm.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** dB, higher better

**Direction:** not recorded

**Cli contract:** PSNR-HVS-Y — the same metric on BT.601 luma only (the codec-benchmark
staple; comparable to psnr_hvs_y in Daala/AV1-style reports).
Emits psnrhvsy_imazen_v and psnrhvsym_imazen_v.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric psnrhvs-y --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "RGB per-channel 8×8 DCT (-y = luma plane)"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "authors' psnrhvsm.m"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=psnrhvs-y) · [Speed](https://metric-brief.pages.dev/#speed&q=psnrhvs-y)

## ssim

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Single-scale structural similarity.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Single-scale structural similarity."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [PyIQA SSIM precision conventions observed 2026-10-19](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#pyiqa-ssim-precision-conventions-observed-2026-10-19)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim)

## ssim-libvmaf

FR · Classical · CPU · Reported results

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** Wang et al., IEEE TIP 2004 (as implemented by Z. Li / tdistler iqa)

**Reference implementation:** libvmaf float_ssim @ f85a8536 — FFI oracle (vmaf-head-sys dev-dep)

**Reference identity:** libvmaf float_ssim @ f85a8536 — FFI oracle (vmaf-head-sys dev-dep)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** SSIM — the libvmaf float_ssim feature (Wang et al. 2004 as
implemented by Z. Li / Netflix iqa: round(min_dim/256) box
decimation, 11×11 Gaussian, zli l·c·s pooling, L=255) via
msssim::float_ssim, FFI-verified against vendored libvmaf
f85a8536. Scored on the studio-swing BT.601 luma plane a YUV420
pipeline delivers to libvmaf
(Y = round(16 + (65.481R + 128.553G + 24.966B)/255)), NOT the
MATLAB rgb2gray luma msssim/ssim metrics use. This is the
metric behind the JPEG AIC-4 SSIM column. --luma-ingress
does not apply: the variant always builds its own studio-601 Y.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-libvmaf --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → studio-swing BT.601 luma — the YUV Y plane libvmaf consumes; NOT the MATLAB rgb2gray luma"
}
```

**Parity:**

```json
{
  "reported": "reported gate: ≤ 2e-4 (vs vendored f85a8536, 5 cases 8b+10b); observed: ≪ gate",
  "oracle": "libvmaf float_ssim @ f85a8536 — FFI oracle (vmaf-head-sys dev-dep)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-libvmaf) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-libvmaf)

## ssim-pyiqa

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AIC2026 ssim-pyiqa convention.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa convention."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa)

## ssim-pyiqa-f32

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AIC2026 ssim-pyiqa-f32 convention.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa-f32 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa-f32 convention."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa-f32) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa-f32)

## ssim-pyiqa-f32-k2

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=2. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa-f32-k2 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa-f32-k2 convention.",
  "fixed_factor_contract": {
    "kernel_size": 2,
    "definition": "Set downsample=True and kernel_size=2.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa-f32-k2) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa-f32-k2)

## ssim-pyiqa-f32-k3

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=3. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa-f32-k3 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa-f32-k3 convention.",
  "fixed_factor_contract": {
    "kernel_size": 3,
    "definition": "Set downsample=True and kernel_size=3.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa-f32-k3) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa-f32-k3)

## ssim-pyiqa-f32-k4

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=4. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa-f32-k4 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa-f32-k4 convention.",
  "fixed_factor_contract": {
    "kernel_size": 4,
    "definition": "Set downsample=True and kernel_size=4.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa-f32-k4) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa-f32-k4)

## ssim-pyiqa-f32-k5

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=5. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa-f32-k5 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa-f32-k5 convention.",
  "fixed_factor_contract": {
    "kernel_size": 5,
    "definition": "Set downsample=True and kernel_size=5.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa-f32-k5) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa-f32-k5)

## ssim-pyiqa-f32-k6

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=6. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim-pyiqa-f32-k6 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssim-pyiqa-f32-k6 convention.",
  "fixed_factor_contract": {
    "kernel_size": 6,
    "definition": "Set downsample=True and kernel_size=6.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim-pyiqa-f32-k6) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim-pyiqa-f32-k6)

## ssim2

FR · Classical · CPU · Reported results

**Paper: citation needed.**

**Recorded paper:** SSIMULACRA 2, Jon Sneyers (Cloudinary) (2022+, no paper)

**Reference implementation:** Cloudinary ssimulacra2 C++ (v2 lineage)

**Reference identity:** Cloudinary ssimulacra2 C++ (v2 lineage)

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** ~0–100, higher better

**Direction:** not recorded

**Cli contract:** SSIMULACRA2 — CPU implementation via Imazen's fast-ssim2 crate.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssim2 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "reported gate: bit-parity SIMD tiers; observed: —",
  "oracle": "Cloudinary ssimulacra2 C++ (v2 lineage)"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim2) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim2)

## ssim2-gpu

FR · Classical · GPU · Not documented

**Paper: citation needed.**

**Recorded paper:** SSIMULACRA 2, Jon Sneyers (Cloudinary) (2022+, no paper)

**Reference implementation:** fast-ssim2 (CPU) + third-party crates.io ssimulacra2 for parity

**Reference identity:** fast-ssim2 (CPU) + third-party crates.io ssimulacra2 for parity

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** ~0–100, higher better

**Direction:** not recorded

**Cli contract:** SSIMULACRA2 — GPU implementation via the ssim2-gpu crate
(dispatched through the zenmetrics-api umbrella).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features gpu-ssim2
target/release/zenmetrics score --metric ssim2-gpu --reference ref.png --distorted dist.png
```

Requires a GPU backend at run time.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "fast-ssim2 (CPU) + third-party crates.io ssimulacra2 for parity"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/ssim2-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/ssim2-gpu/README.md)

# ssim2-gpu

Multi-vendor GPU implementation of the SSIMULACRA2 perceptual image
quality metric, built on CubeCL (https://github.com/tracel-ai/cubecl).

One Rust kernel source, dispatchable across:
- CUDA (NVIDIA) via the cubecl CUDA runtime
- WGPU (cross-platform) — Vulkan / Metal / DX12 / WebGPU
- HIP (AMD ROCm) when the hip feature is enabled
- CPU (SIMD) reference path when cpu is enabled

Algorithmic parity target is the published ssimulacra2 (https://crates.io/crates/ssimulacra2)
v0.5.1 CPU crate (the canonical Rust port of cloudinary/ssimulacra2 (https://github.com/cloudinary/ssimulacra2)).

## Usage

rust
use cubecl::cuda::CudaRuntime;
use cubecl::Runtime;
use ssim2_gpu::Ssim2;

let client = CudaRuntime::client(&Default::default());
let mut s = Ssim2::<CudaRuntime>::new(client, width, height)?;

// One-off comparison.
let result = s.compute(&ref_srgb, &dis_srgb)?;
println!("score = {:.3}", result.score);

// Encoder rate-distortion search: cache the reference once.
s.set_reference(&ref_srgb)?;
for dis in distorted_candidates {
    let r = s.compute_with_reference(&dis)?;
    // ...
}


The Ssim2Batch<R> wrapper exposes the same API for batched scoring.

## Cargo features

- cuda / wgpu / hip / cpu — runtime backends; pick whichever
  your target platform supports. Default is cuda + wgpu + cpu.
- fast-reduction — opt-in since 2026-05-26 (task #52).
  Enables Atomic<f32>::fetch_add for the per-octave reduction.
  ~2-3× faster on CUDA at small image sizes but the atomic-add
  commit order varies across launches, so two runs of the same input
  diverge by ~5e-5 in the final score. The default portable path is
  bit-identical across runs and works on every cubecl backend
  including Metal. See Cargo.toml for the per-backend correctness
  matrix.
- fir — opt-in separable FIR D=5 Gaussian blur path per Kanetaka
  et al. IWAIT 2026. Off by default. When enabled, exposes the
  Ssim2Blur enum, with_blur / set_blur / blur() accessors,
  SSIM2_FIR_COLUMN_NAME, and column_name_for_blur. The FIR is a
  distinct metric — per-image scores diverge from the IIR's by
  design (different impulse-response support); sweep tooling lands FIR
  scores in a separate parquet column to avoid mixing.

## Score interpretation

Output is in roughly the 0–100 range:
- 100 = identical (or near-identical)
- 90+ = visually indistinguishable for most observers
- 70+ = high quality
- 30–60 = noticeable distortion
- <0 = the SSIMULACRA2 polynomial overshoot region for severely
  distorted images; not a bug, just how the curve behaves.

## Status

See PORT_STATUS.md (PORT_STATUS.md) and HANDOFF.md (HANDOFF.md).
Validated against CPU ssimulacra2 to ≤ 0.06 % relative error on the
JPEG q5..q90 corpus; cached vs direct path drift ≤ 8e-6.

## Memory modes

ssim2-gpu exposes the workspace's unified MemoryMode enum but has
no Strip implementation yet — strip support is a planned follow-up
(Phase 2 — see docs/STRIP_PROCESSING.md). MemoryMode::Auto (the
default) always resolves to Full; MemoryMode::Strip and
MemoryMode::Tile return Error::ModeUnsupported. When the image's
working set exceeds the VRAM cap (ZENMETRICS_VRAM_CAP_BYTES env
var, default 8 GB), Auto surfaces Error::TooBigForFull.

See the workspace README for the cross-crate matrix.

## Build

CUDA 13.2 required for cubecl 0.10's CUDA backend. RTX 50-series
(Blackwell, sm_120) needs CUDA 13 anyway.

bash
CUDA_PATH=/usr/local/cuda cargo build -p ssim2-gpu
CUDA_PATH=/usr/local/cuda cargo test --release -p ssim2-gpu


## License

MIT — same terms as the rest of turbo-metrics.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/ssim2-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/ssim2-gpu/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/ssim2-gpu/LICENSE-pu21](https://github.com/imazen/zenmetrics/blob/master/crates/ssim2-gpu/LICENSE-pu21) · [crates/ssim2-gpu/LICENSE-ssimulacra2](https://github.com/imazen/zenmetrics/blob/master/crates/ssim2-gpu/LICENSE-ssimulacra2)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssim2-gpu) · [Speed](https://metric-brief.pages.dev/#speed&q=ssim2-gpu)

## ssimc

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** JPEG AIC-4 SSIMc: SSIM on each RGB channel (11x11 Gaussian,
sigma 1.5, valid region, no downsampling, contrast-structure term
clipped at 0), averaged. Not the same as ssim (reflected borders,
no clipping).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "JPEG AIC-4 SSIMc: SSIM on each RGB channel (11x11 Gaussian,\nsigma 1.5, valid region, no downsampling, contrast-structure term\nclipped at 0), averaged. Not the same as ssim (reflected borders,\nno clipping)."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc)

## ssimc-pyiqa-f32

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** AIC2026 ssimc-pyiqa-f32 convention.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc-pyiqa-f32 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssimc-pyiqa-f32 convention."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc-pyiqa-f32) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc-pyiqa-f32)

## ssimc-pyiqa-f32-k2

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=2. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc-pyiqa-f32-k2 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssimc-pyiqa-f32-k2 convention.",
  "fixed_factor_contract": {
    "kernel_size": 2,
    "definition": "Set downsample=True and kernel_size=2.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc-pyiqa-f32-k2) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc-pyiqa-f32-k2)

## ssimc-pyiqa-f32-k3

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=3. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc-pyiqa-f32-k3 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssimc-pyiqa-f32-k3 convention.",
  "fixed_factor_contract": {
    "kernel_size": 3,
    "definition": "Set downsample=True and kernel_size=3.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc-pyiqa-f32-k3) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc-pyiqa-f32-k3)

## ssimc-pyiqa-f32-k4

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=4. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc-pyiqa-f32-k4 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssimc-pyiqa-f32-k4 convention.",
  "fixed_factor_contract": {
    "kernel_size": 4,
    "definition": "Set downsample=True and kernel_size=4.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc-pyiqa-f32-k4) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc-pyiqa-f32-k4)

## ssimc-pyiqa-f32-k5

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=5. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc-pyiqa-f32-k5 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssimc-pyiqa-f32-k5 convention.",
  "fixed_factor_contract": {
    "kernel_size": 5,
    "definition": "Set downsample=True and kernel_size=5.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc-pyiqa-f32-k5) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc-pyiqa-f32-k5)

## ssimc-pyiqa-f32-k6

FR · Classical · CPU · Not documented

**Paper:** [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861). Zhou Wang; A.C. Bovik; H.R. Sheikh; E.P. Simoncelli. IEEE Transactions on Image Processing. 2004.

```bibtex
@misc{SSIM2004,
  title = {Image quality assessment: from error visibility to structural similarity},
  author = {Zhou Wang and A.C. Bovik and H.R. Sheikh and E.P. Simoncelli},
  year = {2004},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2003.819861},
  doi = {10.1109/TIP.2003.819861}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Set downsample=True and kernel_size=6. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimc-pyiqa-f32-k6 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "AIC2026 ssimc-pyiqa-f32-k6 convention.",
  "fixed_factor_contract": {
    "kernel_size": 6,
    "definition": "Set downsample=True and kernel_size=6.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately.",
    "downsample": true
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimc-pyiqa-f32-k6) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimc-pyiqa-f32-k6)

## ssimulacra1

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Original Cloudinary SSIMULACRA; lower is better.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ssimulacra1 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Original Cloudinary SSIMULACRA; lower is better."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ssimulacra1) · [Speed](https://metric-brief.pages.dev/#speed&q=ssimulacra1)

## stlpips-alex

FR · Learned · CPU · Reported results

**Paper:** [Shift-tolerant Perceptual Similarity Metric](https://arxiv.org/abs/2207.13686). Ghildyal, Abhijay; Liu, Feng. ECCV 2022. 2022 (arXiv submission year; publication venue may be later).

```bibtex
@misc{stlpips2022,
  title = {Shift-tolerant Perceptual Similarity Metric},
  author = {Ghildyal, Abhijay and Liu, Feng},
  year = {2022},
  note = {ECCV 2022},
  url = {https://arxiv.org/abs/2207.13686},
  eprint = {2207.13686},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Ghildyal, Liu. Shift-tolerant Perceptual Similarity Metric. ECCV 2022. arXiv:2207.13686

**Reference implementation:** https://github.com/abhijay9/ShiftTolerant-LPIPS

**Reference identity:** stlpips_pytorch==0.0.2

**Reference code license:** BSD-2-Clause

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** ST-LPIPS, AlexNet (upstream stlpips_pytorch 0.0.2,
shift-tolerant); lower is better. As lpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/stlpips/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric stlpips-alex --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "weights": "stlpips_pytorch 0.0.2: weights/vST0.0/{alex,vgg}_shift_tolerant.pth (BSD-2-Clause)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "(v / 255)  2 - 1, then LPIPS ScalingLayer (inside the graph)",
  "resize": "none",
  "min_size": "alex 21, vgg 18 (stlpips_<net>.meta)"
}
```

**Parity:**

```json
{
  "oracle": "stlpips_pytorch 0.0.2, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs (3 images, 64-2048 px), i265:\nrten:     alex 4.2e-8, vgg 2.7e-8\nort-cuda: alex 6.0e-8, vgg 6.0e-8"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/stlpips/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/stlpips/card.toml) · [crates/zenmetrics-nn/LICENSE-stlpips](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-stlpips)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=stlpips-alex) · [Speed](https://metric-brief.pages.dev/#speed&q=stlpips-alex)

## stlpips-vgg

FR · Learned · CPU · Reported results

**Paper:** [Shift-tolerant Perceptual Similarity Metric](https://arxiv.org/abs/2207.13686). Ghildyal, Abhijay; Liu, Feng. ECCV 2022. 2022 (arXiv submission year; publication venue may be later).

```bibtex
@misc{stlpips2022,
  title = {Shift-tolerant Perceptual Similarity Metric},
  author = {Ghildyal, Abhijay and Liu, Feng},
  year = {2022},
  note = {ECCV 2022},
  url = {https://arxiv.org/abs/2207.13686},
  eprint = {2207.13686},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Ghildyal, Liu. Shift-tolerant Perceptual Similarity Metric. ECCV 2022. arXiv:2207.13686

**Reference implementation:** https://github.com/abhijay9/ShiftTolerant-LPIPS

**Reference identity:** stlpips_pytorch==0.0.2

**Reference code license:** BSD-2-Clause

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** lower-is-closer

**Cli contract:** ST-LPIPS, VGG16; as stlpips-alex.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/stlpips/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric stlpips-vgg --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "weights": "stlpips_pytorch 0.0.2: weights/vST0.0/{alex,vgg}_shift_tolerant.pth (BSD-2-Clause)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "(v / 255)  2 - 1, then LPIPS ScalingLayer (inside the graph)",
  "resize": "none",
  "min_size": "alex 21, vgg 18 (stlpips_<net>.meta)"
}
```

**Parity:**

```json
{
  "oracle": "stlpips_pytorch 0.0.2, CPU, fp32",
  "tolerance_jnd": 0.001,
  "recorded_source_comments": "Measured 2026-10-15, 18 pairs (3 images, 64-2048 px), i265:\nrten:     alex 4.2e-8, vgg 2.7e-8\nort-cuda: alex 6.0e-8, vgg 6.0e-8"
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/stlpips/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/stlpips/card.toml) · [crates/zenmetrics-nn/LICENSE-stlpips](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-stlpips)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=stlpips-vgg) · [Speed](https://metric-brief.pages.dev/#speed&q=stlpips-vgg)

## topiq-fr

FR · Learned · CPU · Declared gate

**Paper: citation needed.**

**Recorded paper:** Chen, Mo, Hou, Wu, Liao, Sun, Yan, Lin. TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment. IEEE TIP 2024

**Reference implementation:** https://github.com/chaofengc/IQA-PyTorch

**Reference identity:** pyiqa 0.1.16 (torch 2.14.1+cpu)

**Reference code license:** PolyForm-Noncommercial-1.0.0 + S-Lab-1.0 (LICENSE-pyiqa, LICENSE-S-Lab)

**Rust license:** code: PolyForm-Noncommercial-1.0.0 + S-Lab-1.0 (LICENSE-pyiqa, LICENSE-S-Lab); weights: https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE

**Weights license:** https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-closer

**Cli contract:** TOPIQ-FR, KADID-10k checkpoint (pyiqa topiq_fr; the AIC2026 TOPIQ
column): CFANet on ResNet50, no resize, higher is better. Needs
feature pyiqa-ports; code and weights are non-commercial, so
ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports,nn
# export the model once (see crates/zenmetrics-pyiqa-ports/models/topiq/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric topiq-fr --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "nc",
  "terms": "https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE",
  "topiq_fr": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_fr_kadid_res50-2c4cc61d.pth",
  "topiq_fr_sha256": "2c4cc61d1377b16d3ae85e19b7c0bb46ab70f285aada61a42692ef7908765c5e",
  "topiq_fr_pipal": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_fr_pipal_res50-69bbe5ba.pth",
  "topiq_fr_pipal_sha256": "69bbe5ba7a5a890343a717fa88ae422ed2c440f1821f99f038a73eaef7d532d3"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8 (8-bit PNG, as PIL convert('RGB'))",
  "normalisation": "v / 255 in Rust (engine::rgb8_to_planar_01); ImageNet mean/std inside the trunk graph",
  "resize": "none (pyiqa topiq_fr: num_crop=1, no test-time resize); cost grows with the head's attention token count",
  "min_size": 1,
  "max_pixels": 8388608
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 on CPU, fp32: metric(dist_path, ref_path)",
  "pairs": "82 gate pairs: 61 AIC2026 crop pairs (images not distributed) + 21 ladder pairs (3 images at 64, 256, 1024 and 2048x1536, 6 aspect-ratio crops, 3 identical pairs)",
  "tolerance_jnd": 0.001,
  "jnd_topiq_fr": 0.030642,
  "jnd_topiq_fr_pipal": 0.072797,
  "results_topiq_fr": "worst |d| rten 5.96e-7, ort-cpu 4.17e-7, ort-cuda 9.54e-7 (at most 3.1e-5 JND); warm == cold bitwise on every engine",
  "results_topiq_fr_pipal": "worst |d| rten 7.15e-7, ort-cpu 5.96e-7, ort-cuda 3.58e-7 (at most 9.8e-6 JND); warm == cold bitwise on every engine",
  "konjnd_notes": "topiq_fr scores the distorted image above its own identical-pair score in 77 of 1,008 KonJND pairs; topiq_fr_pipal in 0. Per-codec median JND: topiq_fr JPEG 0.0407, BPG 0.0206; topiq_fr_pipal JPEG 0.0802, BPG 0.0673."
}
```

**Conventions:**

```json
{
  "aic4": {
    "columns": [
      "TOPIQ",
      "TOPIQ-PIPAL"
    ],
    "results": "this port vs the published columns, 61 crop pairs, all three engines: TOPIQ median |d| 3.5e-5, max 2.73e-4, Spearman 1.0000; TOPIQ-PIPAL median 2.4e-5, max 1.54e-4, Spearman 1.0000. pyiqa 0.1.16 itself on the same pairs: 3.6e-5 / 2.7e-4 and 2.4e-5 / 1.5e-4, so the residual is the published run's, not the port's."
  }
}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/topiq/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/topiq/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=topiq-fr) · [Speed](https://metric-brief.pages.dev/#speed&q=topiq-fr)

## topiq-fr-pipal

FR · Learned · CPU · Declared gate

**Paper: citation needed.**

**Recorded paper:** Chen, Mo, Hou, Wu, Liao, Sun, Yan, Lin. TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment. IEEE TIP 2024

**Reference implementation:** https://github.com/chaofengc/IQA-PyTorch

**Reference identity:** pyiqa 0.1.16 (torch 2.14.1+cpu)

**Reference code license:** PolyForm-Noncommercial-1.0.0 + S-Lab-1.0 (LICENSE-pyiqa, LICENSE-S-Lab)

**Rust license:** code: PolyForm-Noncommercial-1.0.0 + S-Lab-1.0 (LICENSE-pyiqa, LICENSE-S-Lab); weights: https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE

**Weights license:** https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-closer

**Cli contract:** TOPIQ-FR, PIPAL checkpoint (pyiqa topiq_fr-pipal; AIC2026
TOPIQ-PIPAL). As topiq-fr, with
ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr_pipal.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports,nn
# export the model once (see crates/zenmetrics-pyiqa-ports/models/topiq/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric topiq-fr-pipal --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr_pipal. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "nc",
  "terms": "https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE",
  "topiq_fr": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_fr_kadid_res50-2c4cc61d.pth",
  "topiq_fr_sha256": "2c4cc61d1377b16d3ae85e19b7c0bb46ab70f285aada61a42692ef7908765c5e",
  "topiq_fr_pipal": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_fr_pipal_res50-69bbe5ba.pth",
  "topiq_fr_pipal_sha256": "69bbe5ba7a5a890343a717fa88ae422ed2c440f1821f99f038a73eaef7d532d3"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8 (8-bit PNG, as PIL convert('RGB'))",
  "normalisation": "v / 255 in Rust (engine::rgb8_to_planar_01); ImageNet mean/std inside the trunk graph",
  "resize": "none (pyiqa topiq_fr: num_crop=1, no test-time resize); cost grows with the head's attention token count",
  "min_size": 1,
  "max_pixels": 8388608
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 on CPU, fp32: metric(dist_path, ref_path)",
  "pairs": "82 gate pairs: 61 AIC2026 crop pairs (images not distributed) + 21 ladder pairs (3 images at 64, 256, 1024 and 2048x1536, 6 aspect-ratio crops, 3 identical pairs)",
  "tolerance_jnd": 0.001,
  "jnd_topiq_fr": 0.030642,
  "jnd_topiq_fr_pipal": 0.072797,
  "results_topiq_fr": "worst |d| rten 5.96e-7, ort-cpu 4.17e-7, ort-cuda 9.54e-7 (at most 3.1e-5 JND); warm == cold bitwise on every engine",
  "results_topiq_fr_pipal": "worst |d| rten 7.15e-7, ort-cpu 5.96e-7, ort-cuda 3.58e-7 (at most 9.8e-6 JND); warm == cold bitwise on every engine",
  "konjnd_notes": "topiq_fr scores the distorted image above its own identical-pair score in 77 of 1,008 KonJND pairs; topiq_fr_pipal in 0. Per-codec median JND: topiq_fr JPEG 0.0407, BPG 0.0206; topiq_fr_pipal JPEG 0.0802, BPG 0.0673."
}
```

**Conventions:**

```json
{
  "aic4": {
    "columns": [
      "TOPIQ",
      "TOPIQ-PIPAL"
    ],
    "results": "this port vs the published columns, 61 crop pairs, all three engines: TOPIQ median |d| 3.5e-5, max 2.73e-4, Spearman 1.0000; TOPIQ-PIPAL median 2.4e-5, max 1.54e-4, Spearman 1.0000. pyiqa 0.1.16 itself on the same pairs: 3.6e-5 / 2.7e-4 and 2.4e-5 / 1.5e-4, so the residual is the published run's, not the port's."
  }
}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/topiq/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/topiq/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=topiq-fr-pipal) · [Speed](https://metric-brief.pages.dev/#speed&q=topiq-fr-pipal)

## vif

FR · Classical · CPU · Reported results

**Paper:** [Image information and visual quality](https://doi.org/10.1109/TIP.2005.859378). H.R. Sheikh; A.C. Bovik. IEEE Transactions on Image Processing. 2006.

```bibtex
@misc{VIF2006,
  title = {Image information and visual quality},
  author = {H.R. Sheikh and A.C. Bovik},
  year = {2006},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2005.859378},
  doi = {10.1109/TIP.2005.859378}
}
```

**Recorded paper:** Sheikh & Bovik, IEEE TIP 15(2) 2006

**Reference implementation:** authors' vifp_mscale.m (MATLAB)

**Reference identity:** authors' vifp_mscale.m (MATLAB)

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,∞), 1 ≈ identical

**Direction:** not recorded

**Cli contract:** VIFp (pixel-domain Visual Information Fidelity, in-tree vif
crate, port of the authors' vifp_mscale.m): ratio score
≥ 0, ~1 = identical; can exceed 1 on enhanced inputs; flat
references yield NaN (denominator = 0 — matching the
reference). Emits vif_imazen_v. CPU-only — not routed
through the umbrella or the orchestrator.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vif --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → MATLAB-weights gray"
}
```

**Parity:**

```json
{
  "reported": "reported gate: goldens f64; observed: 5e-13",
  "oracle": "authors' vifp_mscale.m (MATLAB)"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/vif/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/vif/README.md)

# vif

Pure-Rust CPU implementation of VIFp — pixel-domain Visual
Information Fidelity (Sheikh & Bovik, Image Information and Visual
Quality, IEEE TIP 15(2), 2006), ported for behavioral fidelity to
the authors' multiscale scalar-GSM release vifp_mscale.m and
validated against it under GNU Octave.

- vif_plane_f32 — VIFp of two 0–255-scale planes.
- vif_rgb8 — on the unrounded MATLAB rgb2gray luma plane
  (0.2989/0.5870/0.1140); the reference is single-channel, so this
  is the house convention for RGB input.

## vifvec — a different metric sharing the name

vifvec is the original vector-GSM VIF formulation from the same
2006 paper — a 4-level SP5 steerable pyramid with 3×3+parent
vector-GSM neighbourhoods (vifvec.m + matlabPyrTools), not the
pixel-domain scalar-GSM vifp. pyiqa/IQA_pytorch's VIFs and the
JPEG AIC-4 VIF column are this variant.

- vifvec_plane_f64 — vifvec of two raw f64 planes (min dim ≥ 72 —
  the reference's maxPyrHt constraint; smaller inputs return Err
  as the .m does).
- vifvec_rgb8 — on u8-rounded round(0.299R+0.587G+0.114B) luma —
  the ingress the AIC-4 VIF column was computed under — med |Δ|
  1.9e-6 / max 6.3e-6 vs metrics_fullres.tab (n=54); unrounded
  luma misses by ~7e-4.

Validated against GNU Octave runs of the official .m release:
10 goldens < 1e-9 (odd dims, the 72px floor, identity, degenerate
inputs).

Pipeline, matching the reference: four scales; at each scale an
N-tap Gaussian (N = 2^(5−scale)+1 → 17, 9, 5, 3, σ = N/5,
rank-1 separable) produces filter2 'valid' mean/variance/
covariance statistics; a scalar GSM regression yields g and
sv_sq through the reference's exact masking order; the information
sums accumulate Σ log10(1 + g²·σ1²/(sv²+2)) and
Σ log10(1 + σ1²/2). Between scales the images are
filter2 'valid' + 1:2:end decimated by that scale's kernel.
vifp = num/den.

Degenerate behavior is reproduced, not guarded: a scale whose
'valid' map is empty contributes zero, a flat reference makes
sigma1_sq ≡ 0 so den = 0 and the score is NaN, and
min(w,h) < 17 leaves every scale empty → NaN. Identical
textured inputs score ~1.0 (0.999999999992 — the GSM gain stays
just under one), identical constant inputs score NaN. VIFp can
exceed 1 on sharpened inputs — the reference's own behavior.

## Validation

src/tests.rs checks 17 Octave goldens (even/odd dims, empty-scale
boundaries at 37×41/24×24/16×16, identical/constant/zero degenerates,
constant-distorted → 0.0, RGB luma) — see validation/ for the
generator and provenance. Worst observed delta ~5e-13.

f64 pipeline: the reference's 1e-10 masks sit ~4 orders above
f64's noise floor on a 0–255 signal; an f32 statistics plane
leaves ~1e-3 residual variance on flat input (E[x²]−μ²
cancellation) and would defeat every mask. The port therefore casts
f32/u8 inputs to f64 on gather and runs f64 end-to-end —
f64x4 lanes (f64x8 under avx512) via
archmage/magetypes, bit-identical across tiers and thread
counts.

## Features

- std (default) — std + alloc; without it the crate is no_std
  + alloc.
- avx512 — enable the x86-64 AVX-512 tier at runtime dispatch.
- parallel — rayon across the ref/dis filter pairs (bit-identical
  at any thread count).
- _dev — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2 / AVX-512 / NEON / wasm128), dispatched on
first call.

Optimization measurements and profiles: 2026-10-12 report (benchmarks/vif_2026-10-12.md).

## Third-party notices

This crate ports the LIVE/UT Austin VIF releases, whose copyright notice must
appear in all copies: LICENSE-live-vif (LICENSE-live-vif). vifvec reproduces
matlabPyrTools' sp5Filters taps (MIT): LICENSE-matlabPyrTools (LICENSE-matlabPyrTools).

### Thread-order correction under review

The unmerged fix/vifvec-thread-order branch keeps covariance accumulation
in the original one-worker raster order at every worker budget. Parallel
striped partial sums previously changed covariance and score bits. Other
independent parallel work remains enabled. The exact covariance regression
fails on the original path and passes on the correction; existing numerical
expectations are unchanged. The owner must review the measured speed/memory
tradeoff before merging and requalifying published benchmarks.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/vif/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/vif/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/vif/LICENSE-live-vif](https://github.com/imazen/zenmetrics/blob/master/crates/vif/LICENSE-live-vif) · [crates/vif/LICENSE-matlabPyrTools](https://github.com/imazen/zenmetrics/blob/master/crates/vif/LICENSE-matlabPyrTools)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vif) · [Speed](https://metric-brief.pages.dev/#speed&q=vif)

## vifvec

FR · Classical · CPU · Reported results

**Paper:** [Image information and visual quality](https://doi.org/10.1109/TIP.2005.859378). H.R. Sheikh; A.C. Bovik. IEEE Transactions on Image Processing. 2006.

```bibtex
@misc{VIF2006,
  title = {Image information and visual quality},
  author = {H.R. Sheikh and A.C. Bovik},
  year = {2006},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2005.859378},
  doi = {10.1109/TIP.2005.859378}
}
```

**Recorded paper:** Sheikh & Bovik, IEEE TIP 15(2) 2006 — the original vector-GSM formulation, a different algorithm from vifp

**Reference implementation:** authors' vifvec.m + matlabPyrTools sp5Filters/buildSpyr/corrDn/vifsub_est_M/ind2wtree — the implementation behind pyiqa/IQA_pytorch VIFs and the AIC-4 VIF column

**Reference identity:** authors' vifvec.m + matlabPyrTools sp5Filters/buildSpyr/corrDn/vifsub_est_M/ind2wtree — the implementation behind pyiqa/IQA_pytorch VIFs and the AIC-4 VIF column

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,∞), 1 = identical

**Direction:** not recorded

**Cli contract:** vifvec — the original steerable-pyramid vector-GSM VIF
(in-tree vif crate, clean-room of the authors' vifvec.m
release + matlabPyrTools sp5Filters/buildSpyr): 4-level
steerable pyramid, 3×3 vector-GSM neighbourhoods, subbands
[4 7 10 13 16 19 22 25], σn² = 0.4. A different metric from
vif (VIFp) that shares the name in the literature — this
is the implementation the JPEG AIC-4 VIF column was computed
with (u8-rounded round(0.299R+0.587G+0.114B) luma; Δ ≤ 5e-5
vs GNU Octave vifvec.m). Emits vifvec_imazen_v. CPU-only.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vifvec --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 → u8-rounded round(0.299R+0.587G+0.114B) gray (cv2/MATLAB convention); vifvec_plane_f64 for raw planes"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 10 goldens < 1e-9; observed: within gate",
  "oracle": "authors' vifvec.m + matlabPyrTools sp5Filters/buildSpyr/corrDn/vifsub_est_M/ind2wtree — the implementation behind pyiqa/IQA_pytorch VIFs and the AIC-4 VIF column"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vifvec) · [Speed](https://metric-brief.pages.dev/#speed&q=vifvec)

## vmaf

FR · Classical · CPU · Reported results

**Paper: citation needed.**

**Recorded paper:** Li et al., Netflix VMAF, 2016

**Reference implementation:** libvmaf 3.2.1 via FFI oracle

**Reference identity:** libvmaf 3.2.1 via FFI oracle

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** VMAF 0–100 + features (adm2, motion2, vif×4)

**Direction:** not recorded

**Cli contract:** VMAF v0.6.1 (in-tree pure-Rust vmaf crate, libvmaf v3.2.1 match).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vmaf --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "YUV420 planes (studio-swing ingress); v0 models luma-only"
}
```

**Parity:**

```json
{
  "reported": "reported gate: feats 1e-4, score 0.02; observed: ≪ gate",
  "oracle": "libvmaf 3.2.1 via FFI oracle"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/vmaf/LICENSE-vmaf](https://github.com/imazen/zenmetrics/blob/master/crates/vmaf/LICENSE-vmaf)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vmaf) · [Speed](https://metric-brief.pages.dev/#speed&q=vmaf)

## vmaf-4k

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** VMAF 0–100 + features (adm2, motion2, vif×4)

**Direction:** not recorded

**Cli contract:** VMAF v0.6.1 4K model (in-tree vmaf crate).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vmaf-4k --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "YUV420 planes (studio-swing ingress); v0 models luma-only"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vmaf-4k) · [Speed](https://metric-brief.pages.dev/#speed&q=vmaf-4k)

## vmaf-neg

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** VMAF 0–100 + features (adm2, motion2, vif×4)

**Direction:** not recorded

**Cli contract:** VMAF v0.6.1neg, with enhancement gain disabled (in-tree vmaf crate).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vmaf-neg --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "YUV420 planes (studio-swing ingress); v0 models luma-only"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vmaf-neg) · [Speed](https://metric-brief.pages.dev/#speed&q=vmaf-neg)

## vmaf-neg-studio601

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** VMAF 0–100 + features (adm2, motion2, vif×4)

**Direction:** not recorded

**Cli contract:** vmaf-neg on studio-swing BT.601 YUV (JPEG AIC-4 VMAF-neg).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vmaf-neg-studio601 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "YUV420 planes (studio-swing ingress); v0 models luma-only"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vmaf-neg-studio601) · [Speed](https://metric-brief.pages.dev/#speed&q=vmaf-neg-studio601)

## vmaf-studio601

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** VMAF 0–100 + features (adm2, motion2, vif×4)

**Direction:** not recorded

**Cli contract:** vmaf on studio-swing BT.601 YUV (the psnr-y-studio601 luma), the
convention behind the JPEG AIC-4 VMAF column.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vmaf-studio601 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "YUV420 planes (studio-swing ingress); v0 models luma-only"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vmaf-studio601) · [Speed](https://metric-brief.pages.dev/#speed&q=vmaf-studio601)

## vmaf-v1

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** VMAF 0–100 + features (adm2, motion2, vif×4)

**Direction:** not recorded

**Cli contract:** VMAF v1.0.16 3d0h model (in-tree vmaf crate).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vmaf-v1 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "YUV420 planes (studio-swing ingress); v0 models luma-only"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vmaf-v1) · [Speed](https://metric-brief.pages.dev/#speed&q=vmaf-v1)

## vsi

FR · Classical · CPU · Reported results

**Paper:** [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028). Lin Zhang; Ying Shen; Hongyu Li. IEEE Transactions on Image Processing. 2014.

```bibtex
@misc{VSI2014,
  title = {VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment},
  author = {Lin Zhang and Ying Shen and Hongyu Li},
  year = {2014},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2014.2346028},
  doi = {10.1109/TIP.2014.2346028}
}
```

**Recorded paper:** Zhang et al., IEEE TIP 23(10) 2014

**Reference implementation:** authors' VSI.m

**Reference identity:** authors' VSI.m

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** VSI — visual saliency-induced index, CPU-only in-tree vsi
crate (port of the authors' VSI.m: SDSP saliency via a
256×256 log-Gabor + Lab pipeline, opponent-channel decimation,
Scharr gradients, saliency-weighted pooling). Similarity on
~[0, 1]: 1 = identical (flat inputs return NaN, like the
reference). Emits vsi_imazen_v. Not routed through the
umbrella or the orchestrator (CPU-only, no GPU twin).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vsi --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (SDSP colour saliency)"
}
```

**Parity:**

```json
{
  "reported": "reported gate: 1e-4; observed: within gate",
  "oracle": "authors' VSI.m"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/vsi/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/vsi/README.md)

# vsi

Pure-Rust CPU implementation of VSI (visual saliency-induced
index), ported for behavioral fidelity to the authors' reference
VSI.m (Zhang, Shen & Li, IEEE TIP 23(10), 2014) and validated
against it under GNU Octave.

- vsi_rgb8 — VSI of two packed sRGB RGB8 pairs (the reference is
  RGB-only; there is no grayscale variant).

Pipeline, matching the reference: SDSP saliency maps — each RGB
channel resampled to 256×256 by the antialiased bilinear imresize
(triangle kernel broadened on shrink, symmetric whole-point padding,
full-kernel-sum normalisation), D50 CIE-Lab conversion, single-scale
log-Gabor bandpass via 2D FFT, centre and warm-colour priors, resize
back, mat2gray. Opponent channels L/M/N, box-mean conv2 'same'
decimation by F = max(1, round(min(w,h)/256)) then 1:F:end
subsample, Scharr gradients, and the quality map
gradSim^0.4 · VSSim · real((ISim·QSim)^0.02) pooled by
max(SM1, SM2) (NaN-propagating, MATLAB max semantics). Flat inputs
return NaN, as the reference's saliency normalisation degenerates.

The 2D FFT is a self-contained radix-2 + Bluestein implementation
vendored from this repository's crates/hdrvdp/src/fft.rs (same
provenance; HDR-specific helpers omitted). The imresize port is
verified bit-identical to Octave's imresize(A,[m n],'bilinear') at
17 scale combinations — see validation/imresize_check.m.

## Validation

src/tests.rs checks 14 Octave goldens (RGB pairs at even/odd dims,
F=1 and F=2 decimation regimes, identical and degenerate inputs,
replicated gray) — see validation/ for the generator and
provenance. Scalar and SIMD tiers are bit-identical by construction
(fixed-order f32 elementwise math, fixed-order f64 pooling).

## Features

- std (default) — std + alloc; without it the crate is no_std
  + alloc.
- avx512 — enable the x86-64 AVX-512 tier at runtime dispatch.
- parallel — rayon across the reference/distorted work pairs
  (bit-identical at any thread count).
- _dev — expose internals for in-workspace testing.

SIMD tiers (scalar / AVX2-FMA / AVX-512 / NEON / wasm128) via
archmage/magetypes, dispatched on first call.

## Fixed downscale variants

Vsi::with_downscale(Downscale::Factor(NonZeroU32::new(2).unwrap()))
selects a fixed opponent/saliency decimation factor for score_rgb8 and generic
Metric::evaluate, including maps. Saliency estimation retains its reference
256×256 processing. Default and free functions retain the
automatic factor. Factor 1 disables decimation; map geometry records the factor.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/vsi/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/vsi/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vsi) · [Speed](https://metric-brief.pages.dev/#speed&q=vsi)

## vsi-k2

FR · Classical · CPU · Not documented

**Paper:** [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028). Lin Zhang; Ying Shen; Hongyu Li. IEEE Transactions on Image Processing. 2014.

```bibtex
@misc{VSI2014,
  title = {VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment},
  author = {Lin Zhang and Ying Shen and Hongyu Li},
  year = {2014},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2014.2346028},
  doi = {10.1109/TIP.2014.2346028}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 2. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vsi-k2 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (SDSP colour saliency)",
  "fixed_factor_contract": {
    "kernel_size": 2,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 2.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vsi-k2) · [Speed](https://metric-brief.pages.dev/#speed&q=vsi-k2)

## vsi-k3

FR · Classical · CPU · Not documented

**Paper:** [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028). Lin Zhang; Ying Shen; Hongyu Li. IEEE Transactions on Image Processing. 2014.

```bibtex
@misc{VSI2014,
  title = {VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment},
  author = {Lin Zhang and Ying Shen and Hongyu Li},
  year = {2014},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2014.2346028},
  doi = {10.1109/TIP.2014.2346028}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 3. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vsi-k3 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (SDSP colour saliency)",
  "fixed_factor_contract": {
    "kernel_size": 3,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 3.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vsi-k3) · [Speed](https://metric-brief.pages.dev/#speed&q=vsi-k3)

## vsi-k4

FR · Classical · CPU · Not documented

**Paper:** [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028). Lin Zhang; Ying Shen; Hongyu Li. IEEE Transactions on Image Processing. 2014.

```bibtex
@misc{VSI2014,
  title = {VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment},
  author = {Lin Zhang and Ying Shen and Hongyu Li},
  year = {2014},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2014.2346028},
  doi = {10.1109/TIP.2014.2346028}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 4. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vsi-k4 --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (SDSP colour saliency)",
  "fixed_factor_contract": {
    "kernel_size": 4,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 4.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vsi-k4) · [Speed](https://metric-brief.pages.dev/#speed&q=vsi-k4)

## vsi-nodownscale

FR · Classical · CPU · Not documented

**Paper:** [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028). Lin Zhang; Ying Shen; Hongyu Li. IEEE Transactions on Image Processing. 2014.

```bibtex
@misc{VSI2014,
  title = {VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment},
  author = {Lin Zhang and Ying Shen and Hongyu Li},
  year = {2014},
  note = {IEEE Transactions on Image Processing},
  url = {https://doi.org/10.1109/TIP.2014.2346028},
  doi = {10.1109/TIP.2014.2346028}
}
```

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** [0,1]

**Direction:** not recorded

**Cli contract:** Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 1. All other pipeline steps remain unchanged. Settled definition, 2026-10-19; branch and implementation verification tracked separately.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric vsi-nodownscale --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8 (SDSP colour saliency)",
  "fixed_factor_contract": {
    "kernel_size": 1,
    "definition": "Replace only kernel_size = max(1, round(min(H,W)/256)) with kernel_size = 1.",
    "other_pipeline_steps": "unchanged",
    "qualification": "Settled definition, 2026-10-19; branch and implementation verification tracked separately."
  }
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=vsi-nodownscale) · [Speed](https://metric-brief.pages.dev/#speed&q=vsi-nodownscale)

## wadiqam-fr-live

FR · Learned · CPU · Reported results

**Paper:** [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697). Bosse, Sebastian; Maniry, Dominique; Müller, Klaus-Robert; Wiegand, Thomas; Samek, Wojciech. IEEE TIP 2018. 2016 (arXiv submission year; publication venue may be later).

```bibtex
@misc{wadiqam2016,
  title = {Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment},
  author = {Bosse, Sebastian and Maniry, Dominique and Müller, Klaus-Robert and Wiegand, Thomas and Samek, Wojciech},
  year = {2016},
  note = {IEEE TIP 2018},
  url = {https://arxiv.org/abs/1612.01697},
  eprint = {1612.01697},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Bosse, Maniry, Mueller, Wiegand, Samek. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment. IEEE TIP 27(1), 2018

**Reference implementation:** https://github.com/dmaniry/deepIQA

**Reference identity:** git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** per checkpoint, measured on a 12-source KonJND JPEG q95..q3 ladder (benchmarks/wadiqam_port_2026-10-16.md): fr_ are distance-like (higher = more degraded vs reference; an identical pair scores a nonzero image-dependent floor — KonJND medians 0.019 live / 12.5 tid, ranges -0.29..0.70 / 9.7..20.6 — since the (0,h,h) concat is not the head's zero); nr_live and nr_tid are both DMOS-like (higher = worse; median rho +0.98 / +0.79). nr_tid is TID2013 MOS-trained but rises with distortion anyway — the MOS labels do not transfer to this head's direction

**Cli contract:** WaDIQaM-FR LIVE — full-reference learned quality (upstream
dmaniry/deepIQA, MIT code and weights; shared conv trunk over
non-overlapping 32x32 patches + relu-weighted pooling,
fr_live_weighted.model); distance-like — higher is more
degraded versus the reference (an identical pair scores a
nonzero image-dependent floor; KonJND median 0.019, range
-0.29..0.70). Emits wadiqam_fr_live. As arniqa for
ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/wadiqam/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric wadiqam-fr-live --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "dmaniry/deepIQA models/{fr,nr}_{live,tid}_weighted.model (~9.5 MB each, chainer HDF5; hashes in export.py)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "none — raw u8 -> f32 (evaluate.py feeds imageio.imread output straight in)",
  "resize": "none — fully convolutional at native resolution",
  "patches": "non-overlapping 32x32 stride-32 grid, row-major (evaluate.py's extract_patches; partial edge patches dropped). README documents this as the eval protocol; the paper's randomly-sampled overlapping patch scheme is the training protocol, not eval",
  "min_size": 32,
  "max_patches": 32768
}
```

**Parity:**

```json
{
  "oracle": "dmaniry/deepIQA revision 51dece99 fr_model.FRModel / nr_model.Model, top='weighted', extract_patches + evaluate.py batching verbatim, chainer CPU fp32 (cuda.cupy shimmed to numpy after weight load)",
  "tolerance_jnd": 0.001,
  "jnd": "fr_live 29.932910 (JPEG 30.460, BPG 29.453), fr_tid 5.209576 (6.384/3.721), nr_live 23.551063 (24.828/21.711), nr_tid 2.757295 (3.007/2.525); fr_ = |score(pair)-score(ref,ref)|, nr_ = |score(dist)-score(ref)| — 1008 KonJND pairs, this port on ort-cuda, 2026-10-16",
  "results": "29 ladder pairs incl. non-square + 4096, both engines, gate 1e-3 JND: fr_live rten 7.65e-7 JND (worst |d| 2.29e-5) / ort-cuda 1.05e-6 JND (3.15e-5); fr_tid rten 7.83e-6 (4.08e-5) / ort-cuda 3.66e-6 (1.91e-5); nr_live rten 2.63e-7 (6.20e-6) / ort-cuda 1.03e-6 (2.43e-5); nr_tid rten 5.54e-6 (1.53e-5) / ort-cuda 5.54e-6 (1.53e-5). Export --goldens (ONNX vs chainer): worst 4.2e-5."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/wadiqam/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/wadiqam/card.toml) · [crates/zenmetrics-nn/LICENSE-wadiqam](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-wadiqam)

**Known divergences:** [wadiqam — deepIQA WaDIQaM FR+NR (2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#wadiqam--deepiqa-wadiqam-frnr-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=wadiqam-fr-live) · [Speed](https://metric-brief.pages.dev/#speed&q=wadiqam-fr-live)

## wadiqam-fr-tid

FR · Learned · CPU · Reported results

**Paper:** [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697). Bosse, Sebastian; Maniry, Dominique; Müller, Klaus-Robert; Wiegand, Thomas; Samek, Wojciech. IEEE TIP 2018. 2016 (arXiv submission year; publication venue may be later).

```bibtex
@misc{wadiqam2016,
  title = {Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment},
  author = {Bosse, Sebastian and Maniry, Dominique and Müller, Klaus-Robert and Wiegand, Thomas and Samek, Wojciech},
  year = {2016},
  note = {IEEE TIP 2018},
  url = {https://arxiv.org/abs/1612.01697},
  eprint = {1612.01697},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Bosse, Maniry, Mueller, Wiegand, Samek. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment. IEEE TIP 27(1), 2018

**Reference implementation:** https://github.com/dmaniry/deepIQA

**Reference identity:** git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** per checkpoint, measured on a 12-source KonJND JPEG q95..q3 ladder (benchmarks/wadiqam_port_2026-10-16.md): fr_ are distance-like (higher = more degraded vs reference; an identical pair scores a nonzero image-dependent floor — KonJND medians 0.019 live / 12.5 tid, ranges -0.29..0.70 / 9.7..20.6 — since the (0,h,h) concat is not the head's zero); nr_live and nr_tid are both DMOS-like (higher = worse; median rho +0.98 / +0.79). nr_tid is TID2013 MOS-trained but rises with distortion anyway — the MOS labels do not transfer to this head's direction

**Cli contract:** WaDIQaM-FR TID — as wadiqam-fr-live with the TID2013-trained
checkpoint (fr_tid_weighted.model; identical-pair floor
median 12.5, range 9.7..20.6). Emits wadiqam_fr_tid.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/wadiqam/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric wadiqam-fr-tid --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "dmaniry/deepIQA models/{fr,nr}_{live,tid}_weighted.model (~9.5 MB each, chainer HDF5; hashes in export.py)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "none — raw u8 -> f32 (evaluate.py feeds imageio.imread output straight in)",
  "resize": "none — fully convolutional at native resolution",
  "patches": "non-overlapping 32x32 stride-32 grid, row-major (evaluate.py's extract_patches; partial edge patches dropped). README documents this as the eval protocol; the paper's randomly-sampled overlapping patch scheme is the training protocol, not eval",
  "min_size": 32,
  "max_patches": 32768
}
```

**Parity:**

```json
{
  "oracle": "dmaniry/deepIQA revision 51dece99 fr_model.FRModel / nr_model.Model, top='weighted', extract_patches + evaluate.py batching verbatim, chainer CPU fp32 (cuda.cupy shimmed to numpy after weight load)",
  "tolerance_jnd": 0.001,
  "jnd": "fr_live 29.932910 (JPEG 30.460, BPG 29.453), fr_tid 5.209576 (6.384/3.721), nr_live 23.551063 (24.828/21.711), nr_tid 2.757295 (3.007/2.525); fr_ = |score(pair)-score(ref,ref)|, nr_ = |score(dist)-score(ref)| — 1008 KonJND pairs, this port on ort-cuda, 2026-10-16",
  "results": "29 ladder pairs incl. non-square + 4096, both engines, gate 1e-3 JND: fr_live rten 7.65e-7 JND (worst |d| 2.29e-5) / ort-cuda 1.05e-6 JND (3.15e-5); fr_tid rten 7.83e-6 (4.08e-5) / ort-cuda 3.66e-6 (1.91e-5); nr_live rten 2.63e-7 (6.20e-6) / ort-cuda 1.03e-6 (2.43e-5); nr_tid rten 5.54e-6 (1.53e-5) / ort-cuda 5.54e-6 (1.53e-5). Export --goldens (ONNX vs chainer): worst 4.2e-5."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/wadiqam/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/wadiqam/card.toml) · [crates/zenmetrics-nn/LICENSE-wadiqam](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-wadiqam)

**Known divergences:** [wadiqam — deepIQA WaDIQaM FR+NR (2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#wadiqam--deepiqa-wadiqam-frnr-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=wadiqam-fr-tid) · [Speed](https://metric-brief.pages.dev/#speed&q=wadiqam-fr-tid)

## wadiqam-nr-live

NR · Learned · CPU · Reported results

**Paper:** [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697). Bosse, Sebastian; Maniry, Dominique; Müller, Klaus-Robert; Wiegand, Thomas; Samek, Wojciech. IEEE TIP 2018. 2016 (arXiv submission year; publication venue may be later).

```bibtex
@misc{wadiqam2016,
  title = {Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment},
  author = {Bosse, Sebastian and Maniry, Dominique and Müller, Klaus-Robert and Wiegand, Thomas and Samek, Wojciech},
  year = {2016},
  note = {IEEE TIP 2018},
  url = {https://arxiv.org/abs/1612.01697},
  eprint = {1612.01697},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Bosse, Maniry, Mueller, Wiegand, Samek. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment. IEEE TIP 27(1), 2018

**Reference implementation:** https://github.com/dmaniry/deepIQA

**Reference identity:** git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** per checkpoint, measured on a 12-source KonJND JPEG q95..q3 ladder (benchmarks/wadiqam_port_2026-10-16.md): fr_ are distance-like (higher = more degraded vs reference; an identical pair scores a nonzero image-dependent floor — KonJND medians 0.019 live / 12.5 tid, ranges -0.29..0.70 / 9.7..20.6 — since the (0,h,h) concat is not the head's zero); nr_live and nr_tid are both DMOS-like (higher = worse; median rho +0.98 / +0.79). nr_tid is TID2013 MOS-trained but rises with distortion anyway — the MOS labels do not transfer to this head's direction

**Cli contract:** WaDIQaM-NR LIVE — no-reference learned quality (upstream
dmaniry/deepIQA, MIT code and weights;
nr_live_weighted.model); DMOS-like, higher is worse.
Scored through PairedNoReference: emits wadiqam_nr_live
(distorted) and wadiqam_nr_live_ref (reference). As arniqa
for ZENMETRICS_NN_MODEL_DIR/ZENMETRICS_NN_ENGINE.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/wadiqam/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric wadiqam-nr-live --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "dmaniry/deepIQA models/{fr,nr}_{live,tid}_weighted.model (~9.5 MB each, chainer HDF5; hashes in export.py)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "none — raw u8 -> f32 (evaluate.py feeds imageio.imread output straight in)",
  "resize": "none — fully convolutional at native resolution",
  "patches": "non-overlapping 32x32 stride-32 grid, row-major (evaluate.py's extract_patches; partial edge patches dropped). README documents this as the eval protocol; the paper's randomly-sampled overlapping patch scheme is the training protocol, not eval",
  "min_size": 32,
  "max_patches": 32768
}
```

**Parity:**

```json
{
  "oracle": "dmaniry/deepIQA revision 51dece99 fr_model.FRModel / nr_model.Model, top='weighted', extract_patches + evaluate.py batching verbatim, chainer CPU fp32 (cuda.cupy shimmed to numpy after weight load)",
  "tolerance_jnd": 0.001,
  "jnd": "fr_live 29.932910 (JPEG 30.460, BPG 29.453), fr_tid 5.209576 (6.384/3.721), nr_live 23.551063 (24.828/21.711), nr_tid 2.757295 (3.007/2.525); fr_ = |score(pair)-score(ref,ref)|, nr_ = |score(dist)-score(ref)| — 1008 KonJND pairs, this port on ort-cuda, 2026-10-16",
  "results": "29 ladder pairs incl. non-square + 4096, both engines, gate 1e-3 JND: fr_live rten 7.65e-7 JND (worst |d| 2.29e-5) / ort-cuda 1.05e-6 JND (3.15e-5); fr_tid rten 7.83e-6 (4.08e-5) / ort-cuda 3.66e-6 (1.91e-5); nr_live rten 2.63e-7 (6.20e-6) / ort-cuda 1.03e-6 (2.43e-5); nr_tid rten 5.54e-6 (1.53e-5) / ort-cuda 5.54e-6 (1.53e-5). Export --goldens (ONNX vs chainer): worst 4.2e-5."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/wadiqam/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/wadiqam/card.toml) · [crates/zenmetrics-nn/LICENSE-wadiqam](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-wadiqam)

**Known divergences:** [wadiqam — deepIQA WaDIQaM FR+NR (2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#wadiqam--deepiqa-wadiqam-frnr-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=wadiqam-nr-live) · [Speed](https://metric-brief.pages.dev/#speed&q=wadiqam-nr-live)

## wadiqam-nr-tid

NR · Learned · CPU · Reported results

**Paper:** [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697). Bosse, Sebastian; Maniry, Dominique; Müller, Klaus-Robert; Wiegand, Thomas; Samek, Wojciech. IEEE TIP 2018. 2016 (arXiv submission year; publication venue may be later).

```bibtex
@misc{wadiqam2016,
  title = {Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment},
  author = {Bosse, Sebastian and Maniry, Dominique and Müller, Klaus-Robert and Wiegand, Thomas and Samek, Wojciech},
  year = {2016},
  note = {IEEE TIP 2018},
  url = {https://arxiv.org/abs/1612.01697},
  eprint = {1612.01697},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Bosse, Maniry, Mueller, Wiegand, Samek. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment. IEEE TIP 27(1), 2018

**Reference implementation:** https://github.com/dmaniry/deepIQA

**Reference identity:** git clone of dmaniry/deepIQA revision 51dece994d73f43dbd2ea44d3f2ef581014194ad (the authors' official repo; the paper's 'code available' link)

**Reference code license:** MIT

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** per checkpoint, measured on a 12-source KonJND JPEG q95..q3 ladder (benchmarks/wadiqam_port_2026-10-16.md): fr_ are distance-like (higher = more degraded vs reference; an identical pair scores a nonzero image-dependent floor — KonJND medians 0.019 live / 12.5 tid, ranges -0.29..0.70 / 9.7..20.6 — since the (0,h,h) concat is not the head's zero); nr_live and nr_tid are both DMOS-like (higher = worse; median rho +0.98 / +0.79). nr_tid is TID2013 MOS-trained but rises with distortion anyway — the MOS labels do not transfer to this head's direction

**Cli contract:** WaDIQaM-NR TID — as wadiqam-nr-live with the TID2013-trained
checkpoint (nr_tid_weighted.model); measured higher is
worse despite TID's MOS labels (median rho +0.79 on a JPEG
q95..q3 ladder). Emits wadiqam_nr_tid and wadiqam_nr_tid_ref.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/wadiqam/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric wadiqam-nr-tid --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "dmaniry/deepIQA models/{fr,nr}_{live,tid}_weighted.model (~9.5 MB each, chainer HDF5; hashes in export.py)"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "none — raw u8 -> f32 (evaluate.py feeds imageio.imread output straight in)",
  "resize": "none — fully convolutional at native resolution",
  "patches": "non-overlapping 32x32 stride-32 grid, row-major (evaluate.py's extract_patches; partial edge patches dropped). README documents this as the eval protocol; the paper's randomly-sampled overlapping patch scheme is the training protocol, not eval",
  "min_size": 32,
  "max_patches": 32768
}
```

**Parity:**

```json
{
  "oracle": "dmaniry/deepIQA revision 51dece99 fr_model.FRModel / nr_model.Model, top='weighted', extract_patches + evaluate.py batching verbatim, chainer CPU fp32 (cuda.cupy shimmed to numpy after weight load)",
  "tolerance_jnd": 0.001,
  "jnd": "fr_live 29.932910 (JPEG 30.460, BPG 29.453), fr_tid 5.209576 (6.384/3.721), nr_live 23.551063 (24.828/21.711), nr_tid 2.757295 (3.007/2.525); fr_ = |score(pair)-score(ref,ref)|, nr_ = |score(dist)-score(ref)| — 1008 KonJND pairs, this port on ort-cuda, 2026-10-16",
  "results": "29 ladder pairs incl. non-square + 4096, both engines, gate 1e-3 JND: fr_live rten 7.65e-7 JND (worst |d| 2.29e-5) / ort-cuda 1.05e-6 JND (3.15e-5); fr_tid rten 7.83e-6 (4.08e-5) / ort-cuda 3.66e-6 (1.91e-5); nr_live rten 2.63e-7 (6.20e-6) / ort-cuda 1.03e-6 (2.43e-5); nr_tid rten 5.54e-6 (1.53e-5) / ort-cuda 5.54e-6 (1.53e-5). Export --goldens (ONNX vs chainer): worst 4.2e-5."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-nn/models/wadiqam/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/wadiqam/card.toml) · [crates/zenmetrics-nn/LICENSE-wadiqam](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-wadiqam)

**Known divergences:** [wadiqam — deepIQA WaDIQaM FR+NR (2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#wadiqam--deepiqa-wadiqam-frnr-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=wadiqam-nr-tid) · [Speed](https://metric-brief.pages.dev/#speed&q=wadiqam-nr-tid)

## wadiqam-pyiqa

FR · Learned · CPU · Reported results

**Paper:** [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment](https://arxiv.org/abs/1612.01697). Bosse, Sebastian; Maniry, Dominique; Müller, Klaus-Robert; Wiegand, Thomas; Samek, Wojciech. IEEE TIP 2018. 2016 (arXiv submission year; publication venue may be later).

```bibtex
@misc{wadiqam2016,
  title = {Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment},
  author = {Bosse, Sebastian and Maniry, Dominique and Müller, Klaus-Robert and Wiegand, Thomas and Samek, Wojciech},
  year = {2016},
  note = {IEEE TIP 2018},
  url = {https://arxiv.org/abs/1612.01697},
  eprint = {1612.01697},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Bosse, Maniry, Mueller, Wiegand, Samek. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment. IEEE TIP 2018

**Reference implementation:** pyiqa 0.1.16 create_metric('wadiqam_fr'), CPU fp32, metric(dist, ref)

**Reference identity:** pyiqa 0.1.16 create_metric('wadiqam_fr'), CPU fp32, metric(dist, ref)

**Reference code license:** PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab)

**Rust license:** code: PolyForm Noncommercial 1.0.0 + S-Lab 1.0 (../../LICENSE-pyiqa, ../../LICENSE-S-Lab); weights: nc

**Weights license:** nc

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** higher-is-better

**Cli contract:** WaDIQaM-FR as pyiqa scores it (KADID checkpoint, pyiqa's 32×32 patch
grid; the AIC2026 WaDIQaM column, higher is better). Needs
feature pyiqa-ports and ZENMETRICS_NN_ACCEPT_WEIGHTS=wadiqam_pyiqa.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features pyiqa-ports,nn
# export the model once (see crates/zenmetrics-pyiqa-ports/models/wadiqam-pyiqa/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric wadiqam-pyiqa --reference ref.png --distorted dist.png
```

After reviewing and accepting the linked weight terms, set ZENMETRICS_NN_ACCEPT_WEIGHTS=wadiqam_pyiqa. Consent is not implied by building the CLI.

**Weights:**

```json
{
  "licence_class": "nc",
  "checkpoint": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/WaDIQaM-kadid-f7541ea5.pth",
  "sha256": "f7541ea5dd4d742559e689889a0f3035bdff22fc444e5602a3082ed426ef20a8",
  "trained_on": "KADID-10k",
  "consent": "WadiqamPyiqa::WEIGHTS; ZENMETRICS_NN_ACCEPT_WEIGHTS=wadiqam_pyiqa"
}
```

**Inputs:**

```json
{
  "format": "packed sRGB8",
  "normalisation": "v / 255 (ToTensor), no mean/std",
  "patches": "32x32 at stride 32, origins range(0, d-32, 32) per axis: a side that is a multiple of 32 drops its last full patch (deepIQA keeps it)",
  "min_size": 33
}
```

**Parity:**

```json
{
  "oracle": "pyiqa 0.1.16 create_metric('wadiqam_fr'), CPU fp32, metric(dist, ref)",
  "goldens": "goldens_ladder.tsv (common/ladder.py, 15 pairs 64x64..2048x1536)",
  "jnd": 0.009666,
  "tolerance_jnd": 0.001,
  "measured_ladder": "rten 5.7e-8 (5.9e-6 JND), ort-cpu 1.10e-7 (1.1e-5 JND), ort-cuda 3.4e-8 (3.5e-6 JND)",
  "measured_aic2026_61": "rten 1.19e-8, ort-cpu 1.27e-8, ort-cuda 2.59e-8 (<= 2.7e-6 JND); pyiqa vs the published column: median 1.7e-6, max 1.2e-5",
  "warm_reference": "caches the reference's patch features; bit-identical to cold on every pair",
  "recorded_source_comments": "1 JND: pooled median |score(pair) - score(ref, ref)| over the 1,008 KonJND-1k pairs, pyiqa CPU\n(JPEG 0.008107, BPG 0.011477; distorted below its identical-pair score in 1006 of 1008).\nperf_deltas.py, i265 (Core Ultra 7 265K), rten, 1024x1024 ladder pair, cold evaluate per call;\nbenchmarks/iqa_complexity_2026-10-17/measured/pyiqa_ports_i265_1024.{tsv,meta}."
}
```

**Conventions:**

```json
{}
```

**Sources:** [crates/zenmetrics-pyiqa-ports/models/wadiqam-pyiqa/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-pyiqa-ports/models/wadiqam-pyiqa/card.toml)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=wadiqam-pyiqa) · [Speed](https://metric-brief.pages.dev/#speed&q=wadiqam-pyiqa)

## wd

FR · Learned · CPU · Reported results

**Paper:** [Wasserstein Distortion: Unifying Fidelity and Realism](https://arxiv.org/abs/2310.03629). Qiu, Yang; Wagner, Aaron B.; Ballé, Johannes; Theis, Lucas. CISS 2024. 2023 (arXiv submission year; publication venue may be later).

```bibtex
@misc{wd2023,
  title = {Wasserstein Distortion: Unifying Fidelity and Realism},
  author = {Qiu, Yang and Wagner, Aaron B. and Ballé, Johannes and Theis, Lucas},
  year = {2023},
  note = {CISS 2024},
  url = {https://arxiv.org/abs/2310.03629},
  eprint = {2310.03629},
  archivePrefix = {arXiv}
}
```

**Recorded paper:** Qiu, Wagner, Ballé, Theis. Wasserstein Distortion: Unifying Fidelity and Realism. CISS 2024. https://arxiv.org/abs/2310.03629

**Reference implementation:** https://github.com/Orange-OpenSource/Cool-Chic/blob/main/coolchic/training/metrics/wasserstein.py

**Reference identity:** ed9de16b8819146afeecc20b4d2d0f385f711a7fd9740288bfdbacd3e93e7b43

**Reference code license:** BSD-3-Clause; source and license notices in LICENSE-wd

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** permissive

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** five nonnegative distances, wd_s1…wd_s5; 25-pair both-engine gates and 1008-pair JND calibration (../benchmarks/wd_port_2026-10-16/README.md)

**Direction:** lower-is-better

**Cli contract:** Cool-Chic Wasserstein Distortion at sigma 2^1 through 2^5.
Raw sRGB VGG16 pre-pool features; lower is closer. Emits five columns.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli --features nn
# export the model once (see crates/zenmetrics-nn/models/wd/card.toml for weight terms and export command)
export ZENMETRICS_NN_MODEL_DIR=$PWD/nn-models
target/release/zenmetrics score --metric wd --reference ref.png --distorted dist.png
```

**Weights:**

```json
{
  "licence_class": "permissive",
  "checkpoint": "torchvision VGG16_Weights.IMAGENET1K_V1, vgg16-397923af.pth",
  "sha256": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
}
```

**Inputs:**

```json
{
  "format": "strided RGB8 sRGB",
  "normalisation": "u8 / 255 in f32; no ImageNet normalization, resize or colour conversion",
  "min_size": 8,
  "max_side": 4096,
  "max_pixels": 9000000,
  "max_pixels_32bit": 1048576,
  "peak_rss_at_cap": "3000x3000: rten 12,346,372 KiB, ort-cuda 11,880,128 KiB max RSS (/usr/bin/time -v); benchmarks/wd_port_2026-10-16/cap_memory.tsv"
}
```

**Parity:**

```json
{
  "tolerance_jnd": 0.001,
  "oracle": "Caller-supplied Cool-Chic wasserstein.py on CPU fp32: reference.py calls its unmodified wasserstein_distortion per tap and rebuilds multi_wasserstein_distortion's constant sigma map and f32 per-tap sum around it. Large feature planes are channel-chunked for bounded oracle memory; each upstream channel mean is combined by its channel count.",
  "results": "25 pairs x 5 columns per engine: rten worst 1.924787254e-6 JND, ort-cuda worst 3.443705654e-6 JND; 1e-3 gate. Includes 32-square, 64..2048 and non-square. Stride/warm/identity controls pass.",
  "jnd_source": "models/jnd_scale.tsv; native WD pooled medians on all 1008 KonJND pairs (504 JPEG, 504 BPG); benchmarks/wd_port_2026-10-16/konjnd_scales.tsv"
}
```

**Conventions:**

```json
{
  "convention": {
    "statistics": "Five levels of local mean and variance, zero-padded 3x3 outer([1,2,1]/4), decimation at indices ::2. Variances clamp to 5e-7 before square roots.",
    "columns": "wd_s3 is Cool-Chic's default WassersteinLoss (log2_sigma=3); wd_s1, wd_s2, wd_s4, wd_s5 run the same code at log2_sigma 1, 2, 4, 5 (an extension).",
    "variance": "E[x^2] - mu^2 in f32 before the 5e-7 clamp, as upstream; it cancels in smooth regions and is kept for parity.",
    "weighting": "Triangular weight relu(1-abs(log2_sigma-level)); sigma maps themselves use zero-padded stride-two convolution after each statistic level, including border attenuation. Spatial/channel mean per level, then f32 sums across levels and four VGG taps.",
    "aic": "The AIC-4 WD_s1..WD_s5 preprocessing is unidentified. This metric implements the pinned Cool-Chic convention; 113-combination selection on three pairs chose raw post-pool/rescaled-sigma/sum. Its 61-pair verification has median site errors 0.2310363687..0.3724274401 JND and worst 2.4017891693 JND, so it does not reproduce AIC. Full receipt: benchmarks/wd_port_2026-10-16/README.md."
  }
}
```

**Sources:** [crates/zenmetrics-nn/models/wd/card.toml](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/models/wd/card.toml) · [crates/zenmetrics-nn/LICENSE-wd](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-nn/LICENSE-wd)

**Known divergences:** [wd — Cool-Chic Wasserstein Distortion (2026-10-16)](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md#wd--cool-chic-wasserstein-distortion-2026-10-16)

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=wd) · [Speed](https://metric-brief.pages.dev/#speed&q=wd)

## ws-hv

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** no evidence

**Reference implementation:** no evidence

**Reference identity:** no evidence

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** not recorded

**Direction:** not recorded

**Cli contract:** Wavelet horizontal/vertical similarity; higher is better.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric ws-hv --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "cli_contract": "Wavelet horizontal/vertical similarity; higher is better."
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "no evidence"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=ws-hv) · [Speed](https://metric-brief.pages.dev/#speed&q=ws-hv)

## zensim

FR · Classical · CPU · Not documented

**Paper: citation needed.**

**Recorded paper:** —

**Reference implementation:** —

**Reference identity:** —

**Reference code license:** not recorded

**Rust license:** no evidence

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** 0–100

**Direction:** not recorded

**Cli contract:** Zensim — CPU implementation via the zensim crate.

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric zensim --reference ref.png --distorted dist.png
```

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "—"
}
```

**Conventions:**

```json
{}
```

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=zensim) · [Speed](https://metric-brief.pages.dev/#speed&q=zensim)

## zensim-gpu

FR · Classical · GPU · Not documented

**Paper: citation needed.**

**Recorded paper:** —

**Reference implementation:** —

**Reference identity:** —

**Reference code license:** not recorded

**Rust license:** AGPL-3.0-only OR LicenseRef-Imazen-Commercial

**Weights license:** not applicable

**Display:** No display requirement in CLI mapping; see input convention

**Output range:** 0–100

**Direction:** not recorded

**Cli contract:** Zensim — GPU implementation via the zensim-gpu crate
(dispatched through the zenmetrics-api umbrella).

**Reproduce from the CLI:**

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
bash scripts/ci/clone-siblings.sh
cargo build --release -p zenmetrics-cli
target/release/zenmetrics score --metric zensim-gpu --reference ref.png --distorted dist.png
```

Requires a GPU backend at run time.

**Weights:**

```json
{}
```

**Inputs:**

```json
{
  "convention": "sRGB8"
}
```

**Parity:**

```json
{
  "reported": "no evidence",
  "oracle": "—"
}
```

**Conventions:**

```json
{}
```

**Crate documentation:** [crates/zensim-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/zensim-gpu/README.md)

# zensim-gpu

Multi-vendor GPU implementation of the
zensim (https://github.com/imazen/zensim) perceptual similarity feature
extractor, built on CubeCL (https://github.com/tracel-ai/cubecl). One
Rust kernel source, runs on:

- CUDA (NVIDIA) via cubecl-cuda
- WGPU (cross-platform) via Vulkan / Metal / DX12 / WebGPU
- HIP (AMD ROCm) via cubecl-hip (hip feature)
- CPU fallback via cubecl-cpu for build-checking only — use the
  published zensim crate as the runtime CPU reference

Algorithmic parity target is zensim v0.2.8 with
ZensimProfile::latest() (= WEIGHTS_PREVIEW_V0_2, 228 features =
4 scales × 3 channels × 19 features). At the pyramid level this also
matches the existing zensim-cuda crate — same SIMD-padded layout,
same cbrt accuracy regime, same fused H-blur (mu1 / mu2 / sigma_sq /
sigma12) and fused V-blur + per-pixel feature kernels. The CUDA
crate stays in the workspace; this one extends reach to AMD / Intel /
Apple / WebGPU without giving anything up on NVIDIA.

## Single-image usage

rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use zensim_gpu::{Zensim, score_from_features};

let client = WgpuRuntime::client(&Default::default());
let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;

let ref_srgb: Vec<u8> = vec![0; 512  512  3];
let dis_srgb: Vec<u8> = vec![0; 512  512  3];

let features = z.compute_features(&ref_srgb, &dis_srgb)?;
let score = score_from_features(&features, &zensim::profile::WEIGHTS_PREVIEW_V0_2);
println!("zensim score = {:.4} (0-100, higher = better)", score);
# Ok::<(), zensim_gpu::Error>(())


## Cached-reference usage

When scoring many distorted candidates against one reference (encoder
rate-distortion search), call set_reference once and then
compute_with_reference per candidate. Skips the reference-side sRGB
→ XYB conversion, mirror-pad, and pyramid downscale on every call.

rust,no_run
use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use zensim_gpu::{Zensim, score_from_features};

let client = WgpuRuntime::client(&Default::default());
let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;

let ref_srgb: Vec<u8> = vec![0; 512  512  3];
z.set_reference(&ref_srgb)?;
# let candidates: Vec<Vec<u8>> = vec![];
for candidate in candidates {
    let f = z.compute_with_reference(&candidate)?;
    let s = score_from_features(&f, &zensim::profile::WEIGHTS_PREVIEW_V0_2);
    // ... use s ...
}
# Ok::<(), zensim_gpu::Error>(())


## Score interpretation

Output is a 228-entry [f64; TOTAL_FEATURES] feature vector. Apply the
trained weights from zensim::profile::WEIGHTS_PREVIEW_V0_2 (228
entries) and the standard 100 - 18·d^0.7 mapping to convert to a
0-100 score. The score_from_features(&features, &weights) helper
does exactly this.

## Features

| Feature | Default | Effect |
|---|---|---|
| cuda | yes | Compile the cubecl-cuda backend. |
| wgpu | yes | Compile the cubecl-wgpu backend. |
| cpu  | yes | Compile the cubecl-cpu backend (build-only — see above). |
| hip  | no  | Compile the cubecl-hip backend. |
| fast-reduction | yes | Reserved — current pipeline uses per-column slot writes (no atomics needed); the flag is kept for API symmetry with dssim-gpu / ssim2-gpu. |

For Metal targets:

bash
cargo build -p zensim-gpu --no-default-features --features wgpu


## Memory modes

zensim-gpu exposes the workspace's unified MemoryMode enum but has
no Strip implementation — the multi-channel, multi-scale,
Extended-regime allocator is interlocked enough that strip processing
would need a dedicated design pass. MemoryMode::Auto (the default)
resolves to Full; Strip / Tile return Error::ModeUnsupported.
When the working set exceeds the VRAM cap
(ZENMETRICS_VRAM_CAP_BYTES env var, default 8 GB), Auto surfaces
Error::TooBigForFull. The pre-existing
Zensim::new_with_regime_budget helper continues to handle the
Extended-regime persist-plane budget independently.

## Per-pixel diffmap (Phase 1)

zensim-gpu can also produce a per-pixel error map alongside the
scalar score, for encoder-loop integrations that need to know
where the distortion lives (e.g. JXL VarDCT quant-field
refinement). Mirror of the cvvdp-gpu Phase 1 API shape:

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use zensim_gpu::Zensim;

let client = CudaRuntime::client(&Default::default());
let mut z = Zensim::<CudaRuntime>::new(client, 512, 512)?;

let ref_srgb: Vec<u8> = vec![0; 512  512  3];
let dis_srgb: Vec<u8> = vec![0; 512  512  3];
let mut diffmap = Vec::new();

// One-shot score + diffmap. Returns butteraugli-direction normalised
// score (smaller = better; identity → 0).
let score = z.score_with_diffmap(&ref_srgb, &dis_srgb, &mut diffmap)?;
println!("score = {score:.4}, diffmap len = {}", diffmap.len());

// Or: warm-ref pattern for encoder loops with one reference, N
// distortion candidates per cell:
z.set_reference(&ref_srgb)?;
for candidate in [/ ... /] {
    let s = z.score_with_warm_ref_diffmap(&candidate, &mut diffmap)?;
}
# Ok::<(), zensim_gpu::Error>(())


### Linear-RGB planes input

For encoder-side integrations that already have the source in linear-
RGB f32 planes (skipping the sRGB-byte LUT roundtrip), the
score_from_linear_planes_ family takes three separate planes per
side:

rust,no_run
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use zensim_gpu::Zensim;

let client = CudaRuntime::client(&Default::default());
let mut z = Zensim::<CudaRuntime>::new(client, 512, 512)?;

let ref_r: Vec<f32> = vec![0.5; 512  512];
let ref_g: Vec<f32> = vec![0.5; 512  512];
let ref_b: Vec<f32> = vec![0.5; 512  512];

z.warm_reference_from_linear_planes(&ref_r, &ref_g, &ref_b)?;
let mut diffmap = Vec::new();

let dist_r: Vec<f32> = vec![0.49; 512  512];
let dist_g: Vec<f32> = vec![0.49; 512  512];
let dist_b: Vec<f32> = vec![0.49; 512  512];

let score = z.score_from_linear_planes_with_warm_ref_diffmap(
    &dist_r, &dist_g, &dist_b, &mut diffmap,
)?;
# Ok::<(), zensim_gpu::Error>(())


### Diffmap implementation note (Phase 1 + Phase 1b)

The per-pixel diffmap production defaults to zensim's canonical CPU
pipeline (zensim::compute_with_ref_and_diffmap_linear_planar) — the
GPU side stays the fast path for scalar feature extraction, and the
scalar score returned alongside the diffmap comes from the CPU pipeline
(NOT the GPU 372-feature MLP).

Phase 1b added a pure-GPU diffmap kernel chain (per-scale weighted
SSIM → power-of-2 upsample-add → trim) wired behind the env gate
ZENSIM_GPU_DIFFMAP=1. The GPU kernels are validated bit-close to the
CPU canonical (≤ 2.08e-4 pointwise — crates/zensim-gpu/tests/it/cpu_gpu_diffmap_parity.rs).
The gate is default-OFF: the scalar score must still come from the
CPU canonical path (the GPU-feature → V0_3 MLP score is unreliable on
the pinned zensim 0.3.0 — see docs/DIFFMAP_DIVERGENCES.md §2b + §9),
so GPU-diffmap + CPU-score is currently slower than CPU-only. The gate
ships the validated GPU diffmap infrastructure; flipping it to
default-ON (and crushing the diffmap wall overhead) is the chunk-N+1
score-path fix. See
docs/DIFFMAP_DIVERGENCES.md (docs/DIFFMAP_DIVERGENCES.md) §2 + §2b.

## Status

Initial port from zensim-cuda. See PORT_STATUS.md for the
per-kernel breakdown, runtime coverage, and validated parity numbers.

Phase 1 of the zensim-fork RFC arc: diffmap + linear-planes + warm-
ref-diffmap API surface shipped (CPU-fallback diffmap path). See
docs/DIFFMAP_DIVERGENCES.md.

**Sources:** [docs/METRIC_PROVENANCE.md](https://github.com/imazen/zenmetrics/blob/master/docs/METRIC_PROVENANCE.md) · [crates/zensim-gpu/README.md](https://github.com/imazen/zenmetrics/blob/master/crates/zensim-gpu/README.md) · [crates/zenmetrics-cli/src/metrics/mod.rs](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-cli/src/metrics/mod.rs) · [crates/zensim-gpu/LICENSE-libjxl](https://github.com/imazen/zenmetrics/blob/master/crates/zensim-gpu/LICENSE-libjxl) · [crates/zensim-gpu/LICENSE-pu21](https://github.com/imazen/zenmetrics/blob/master/crates/zensim-gpu/LICENSE-pu21)

**Known divergences:** [Consult the divergence ledger](https://github.com/imazen/zenmetrics/blob/master/DIVERGENCES.md); no exact metric heading joined.

[Model card and reproduction status](https://metric-brief.pages.dev/#models&m=zensim-gpu) · [Speed](https://metric-brief.pages.dev/#speed&q=zensim-gpu)
