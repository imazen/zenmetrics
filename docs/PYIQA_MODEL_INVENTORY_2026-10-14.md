# pyiqa (IQA-PyTorch) learned-metric inventory — 2026-10-14

Source read: `chaofengc/IQA-PyTorch` @ `09796cfe80b0` (2026-10-09), shallow clone. Nothing
from pyiqa was installed or run. File paths below are relative to that clone. Plan
built on this: [`PYIQA_PORT_PLAN.md`](PYIQA_PORT_PLAN.md). Licence statements are a
reading of the repos' licence files, not legal advice.

## 1. Licensing

**Superseded 2026-10-15 for per-model terms** by
[`PYIQA_UPSTREAM_LICENSES_2026-10-15.md`](PYIQA_UPSTREAM_LICENSES_2026-10-15.md), which
reads every upstream licence and weight host as text. Changes from this section: WaDIQaM
has official MIT weights (dmaniry/deepIQA); MUSIQ's weights are Apache-2.0; ST-LPIPS's
BlurPool files are CC BY-NC-SA 4.0 (Adobe); TReS's agreement forbids distribution.

**pyiqa itself**

| What | License | Evidence |
|---|---|---|
| Code, current | PolyForm Noncommercial 1.0.0 | `LICENSE`; `pyproject.toml:10`; PyPI 0.1.16 (since commit `ccac804386`, 2026-04-09) |
| Code, 2021-12 → 2026-04-09 | CC BY-NC-SA 4.0 | `LICENSE` at `24d8b00e76`, `c8100285f8` |
| "Applicable components" | NTU S-Lab License 1.0 (noncommercial) | `LICENSE-S-Lab`; which components is not stated |
| CLIP code, tokenizer, vocab | MIT (OpenAI) | `LICENSE-OpenAI-CLIP`; `pyiqa/archs/clip_api.py:1-4` |
| ST-LPIPS BlurPool code | CC BY-NC-SA 4.0 (Adobe antialiased-cnns) | `pyiqa/archs/stlpips_arch.py:518-520, 647-649` |
| Hugging Face weight mirror `chaofengc/IQA-PyTorch-Weights` | CC BY-NC-SA 4.0 | its README: some weights trained by pyiqa, others collected from official repos |

**Per model** (NC = noncommercial only). Port from the upstream repo; where the weights
are pyiqa's own training, the upstream licence does not cover them.

| Model family | Upstream code | Weights | Status |
|---|---|---|---|
| LPIPS (alex/vgg/squeeze) | richzhang/PerceptualSimilarity BSD-2 | official linear heads + torchvision trunks | OK |
| LPIPS+ | pyiqa code | LPIPS weights | OK if reimplemented from the TOPIQ paper |
| ST-LPIPS | abhijay9 BSD-2 | upstream | OK; write BlurPool clean-room |
| DISTS | dingkeyan93 MIT | upstream | OK |
| DeepDC | h4nwei MIT | none learned (VGG19 ImageNet) | OK |
| DreamSim | ssundaram21 MIT (not in pyiqa) | upstream; backbone weight licences unverified | port from upstream |
| AHIQ, CKDN, LIQE, PaQ2PiQ | MIT | upstream | OK |
| MANIQA, ARNIQA, UNIQUE, A-FINE, FGResQ, LAION-aes | Apache-2.0 | upstream | OK |
| MUSIQ | google-research Apache-2.0 | Google checkpoints, licence unverified | ? |
| PieAPP | "Modified BSD-2 — Non-Commercial Use Only" | official | **NC** |
| WaDIQaM | upstream has no licence | **trained by pyiqa** | **NC; retrain for permissive** |
| TOPIQ (all) | pyiqa's own | **trained by pyiqa** | **NC; retrain for permissive** |
| HyperIQA, DBCNN, NIMA | MIT | **retrained by pyiqa** | **NC weights** |
| CLIP-IQA(+) | S-Lab (NC) | — | **NC** |
| QualiCLIP, URanker | CC BY-NC | — | **NC** |
| TReS | academic noncommercial | — | **NC** |
| MS-SWD, MACLIP, CNNIQA, MetaIQA | no licence | — | **unlicensed** |
| Q-Align | S-Lab (NC); HF tag MIT | LLaMA-7B-class base | **NC**; base licence likely applies (unverified) |
| Q-ReAlign | — | HF tag Apache-2.0 | probably OK |

torchvision's ImageNet backbone weights may carry dataset terms; this touches almost every
model here — confirm once with counsel.

## 2. Learned metrics in `pyiqa/default_model_configs.py`

Classical (SSIM, PSNR, FSIM, VIF, GMSD, NLPD, VSI, CW-SSIM, MAD, MS-SSIM), statistical
(NIQE, ILNIQE, BRISQUE, NRQM, PI, PIQE, entropy) and set-level generative metrics
(FID/sFID, IS) are excluded. Parameter counts marked "computed" were derived from the
architecture code; file sizes are from the Hugging Face API.

### Full-reference

| Names | Backbone | Params / file | Input handling | Unusual ops |
|---|---|---|---|---|
| `lpips`, `lpips-vgg`, `lpips+`, `lpips-vgg+` | AlexNet / VGG16 (torchvision) | 2,469,696 / 14,714,688 + 1,152/1,472 linear (computed) | any size | `+` weights each layer's distance map by the **reference's** layer-2 activations (`lpips_arch.py:206-221`) |
| `stlpips`, `stlpips-vgg` | BlurPool AlexNet / VGG16 | 9.9 MB / 58.9 MB | any size | reflect pad + binomial depthwise conv, stride 2 |
| `dists` | VGG16, MaxPool → L2pooling (Hann) | 14.7M + 2,950 | any size | global mean/var/cov of features |
| `deepdc` | VGG19 to conv5_4, pre-ReLU | 20,024,384 (computed) | any size | C×C double-centred distance matrices via `bmm`, C ≤ 512 |
| `pieapp` | 11 conv + FC | 68,376,580 (one FC 61.9M), 273.5 MB | 64² patches, stride 27, border remainder dropped | BGR ×255 |
| `wadiqam_fr` | 10-conv VGG-like | 6,287,138, 25.2 MB | 32² non-overlapping patches; last row/col skipped even when sizes divide (`wadiqam_arch.py:193`) | weighted patch average |
| `topiq_fr`, `topiq_fr-pipal` | timm ResNet50 + small transformer | ≈36M, 144.6 MB | any size | `nn.MultiheadAttention`; adaptive avg-pool to stride-32 grid; bicubic-resized learned pos-embedding |
| `ahiq` | timm ResNet50 + ViT-B/8 | checkpoint 326 MB, only heads load from it | 20 uniform 224 crops; 28×28 grid hard-coded | **DeformConv2d** with offsets from the reference |
| `ckdn` | modified ResNet50 | 53.7 MB | shorter side → 329 (bicubic distorted, **nearest** reference), centre-crop 288 | — |
| `msswd` | 2 conv on CIELAB | 0.22 MB | shorter side → 256 | `torch.sort` |
| `dmm` | VGG16 | none learned | square non-aspect resize | `torch.svd` on 16² patches |
| `afine_*` | fine-tuned CLIP ViT-B/32 | 611 MB | asserts H, W multiples of 32 | — |
| `fgresq_pair` | 2× CLIP ViT-B/16 | 694 + 599 MB | 256² resize, crop 224 | — |

### No-reference

| Names | Backbone | File | Input handling | Unusual ops |
|---|---|---|---|---|
| `topiq_nr` ×3, face ×3, iaa ×2 | ResNet50 / Swin-B 384 | 181–508 MB | any size, or 384 / 384² resize; face variants use facexlib RetinaFace alignment | — |
| `musiq` ×4 | ResNet root + 14-layer transformer | 108.6 MB (≈27M) | longer side 224 and 384 (bicubic, no antialias) **plus every patch at native resolution** | hash position embedding, scale embeddings, masks, TF "SAME" weight-standardised conv |
| `maniqa` ×3 | ViT-B/8 + channel attention + Swin | 543 MB | 20 uniform 224 crops | `torch.roll` |
| `clipiqa` ×4 | CLIP RN50 / ViT-L/14 | 255.8 / 932.8 MB | **full resolution** | pos-embedding bicubic-resized as a flattened tokens×dim matrix (`clip_model.py:354-363`) |
| `liqe` ×2 | fine-tuned CLIP ViT-B/32 | 353.5 MB | stride-32 unfold of 224 patches, 15 evenly spaced | — |
| `qualiclip` ×5 | fine-tuned CLIP RN50 | 408 MB | full resolution | — |
| `hyperiqa` | ResNet50 + hypernetwork | 109.8 MB | 25 uniform 224 crops | generated weights applied with `bmm` |
| `dbcnn` | VGG16 + SCNN | 61.3 MB | any size | bilinear pooling |
| `nima` ×4 | inception-resnet-v2 / VGG16 | 218 / 58.9 MB | resize 299 + crop | — |
| `arniqa` ×8 | ResNet50 | 112 MB + TorchScript regressors | full res + 0.5× bilinear copy | — |
| `paq2piq` | ResNet18 | 46.9 MB | any size | RoIPool |
| `tres` ×2 | ResNet50 + transformer | 610 MB | 50 crops; a flipped pass is computed and discarded | — |
| `qalign` ×3, `compare2score` | mPLUG-Owl2 LLM | 16.41 GB | | |
| `qrealign` ×4 | Qwen3.5-VL 0.8B / 4B / 9B | 2.2 / 10.4 / 18.8 GB | | |
| others | `cnniqa`, `metaiqa`, `unique`, `uranker`, `laion_aes`, `clipscore`, `maclip` (hard-codes `.to('cuda')`), `fgresq` | | | |

## 3. Shared backbones

| Backbone | Metrics |
|---|---|
| **VGG16** | LPIPS-vgg(+), DISTS, ST-LPIPS-vgg, DBCNN, NIMA-vgg, DMM |
| VGG19 | DeepDC |
| AlexNet | LPIPS(+), ST-LPIPS |
| **ResNet50** | TOPIQ ×10, HyperIQA, AHIQ, ARNIQA ×8, TReS, CKDN |
| ResNet18 / ResNet34 | PaQ2PiQ / MetaIQA, UNIQUE |
| **ViT-B/8** | AHIQ, MANIQA |
| **CLIP RN50** | CLIP-IQA ×3, QualiCLIP ×5, MACLIP |
| **CLIP ViT-B/32** | LIQE ×2, A-FINE ×4, CLIPScore |
| ViT-B/16 | FGResQ, DreamSim |
| ViT-L/14 | `clipiqa+_vitL14_512`, LAION-aes |
| Swin-B | TOPIQ-IAA, TOPIQ face-swin |
| inception-resnet-v2 | NIMA ×3 |

CLIP-family prompts are fixed, so text features are constants: precompute them and ship
only the image tower.

## 4. Preprocessing that decides parity

- **Decode:** PIL `Image.open().convert('RGB')` → `to_tensor` (÷255)
  (`utils/img_util.py:49,61`). No ICC, no EXIF orientation, alpha dropped without
  compositing. Feed both sides identical RGB8 from PNG when generating goldens.
- **Normalisation per model:** LPIPS family `x*2-1` then
  `(x − [-.030,-.088,-.188]) / [.458,.448,.450]`; ImageNet mean/std (DISTS, DeepDC,
  TOPIQ, AHIQ, HyperIQA, DBCNN, ARNIQA, TReS, CKDN); 0.5/0.5 (MANIQA, NIMA);
  `(x-0.5)*2` (MUSIQ); OpenAI CLIP mean/std (CLIP family); BGR ×255 no mean (PieAPP);
  raw [0,1] (WaDIQaM, CNNIQA).
- **Resize hazards:** MUSIQ uses bicubic without antialias where Google's code uses TF
  gaussian resize — pyiqa's own calibration table shows up to 1.09 MOS (I19: 36.938
  official vs 38.0248). NIMA and MS-SWD call `TF.resize` without `antialias`, so results
  move with torchvision versions. CKDN resizes distorted bicubic, reference nearest. The
  crop helpers upscale small images by scale factor, not target size
  (`arch_util.py:61-67`). DMM and FGResQ ignore aspect ratio.
- **Multi-crop:** AHIQ 20, MANIQA 20, HyperIQA 25, TReS 50 uniform 224 crops; LIQE 15
  patches; TOPIQ 1. All deterministic in eval mode.
- **Determinism:** random crops only in training mode; the wrapper's `seed` is stored and
  never used (`models/inference_model.py:89`); cuDNN deterministic on, benchmark off
  (`:152`). pyiqa does not set TF32 flags (default behaviour not re-checked here).
  **Generate goldens on CPU fp32.** CLIP checkpoints store fp16 values, run in fp32.
- **Library drift:** AHIQ backbones come from whatever timm resolves
  (`ahiq_arch.py:254-255`, `timm>=0.8` unpinned), so its scores move with timm.
  MANIQA, TOPIQ, HyperIQA and NIMA load full checkpoints strictly and are unaffected.
- **Argument order:** pyiqa calls `net(target, ref)`. LPIPS+ weighting and AHIQ's deform
  offsets come from the reference.

## 5. ONNX exportability (judged from source; nothing exported)

- **Easy:** LPIPS ×4, DISTS, ST-LPIPS, DeepDC, DBCNN, ARNIQA, UNIQUE, CNNIQA; PaQ2PiQ
  (RoIPool → `MaxRoiPool`).
- **Easy with patching/cropping outside the graph:** PieAPP, WaDIQaM, HyperIQA, LIQE,
  NIMA, CKDN, MetaIQA, LAION-aes, MANIQA (replace forward hooks), TReS (drop flip pass).
- **Moderate:** TOPIQ (adaptive pool is plain avg-pool only when H, W are multiples of
  32; bicubic pos-embedding at dynamic size); CLIP-IQA / QualiCLIP / MACLIP
  (dynamic-resolution attention pool; export image tower only); MUSIQ (multiscale
  patching, hash positions and masks in host code); MS-SWD (sort → dynamic TopK);
  A-FINE (multiples of 32).
- **Problematic:** AHIQ (DeformConv2d — opset 19+, thin runtime support); DMM
  (`torch.svd`, no ONNX op); TOPIQ face (facexlib detector + OpenCV warp); Q-Align,
  Q-ReAlign, Compare2Score (LLMs); FID (set-level).

## 6. Accuracy and the published speed bar

**FR, PLCC / SRCC (pyiqa's tables):**

| metric | CSIQ | TID2013 |
|---|---|---|
| topiq_fr | 0.959 / 0.967 | 0.916 / 0.917 |
| wadiqam_fr | 0.909 / 0.922 | 0.842 / 0.826 |
| dists | 0.932 / 0.930 | 0.754 / 0.708 |
| lpips+ | 0.904 / 0.929 | 0.766 / 0.762 |
| pieapp | 0.838 / 0.897 | 0.720 / 0.844 |
| ahiq | 0.837 / 0.829 | 0.680 / 0.648 |
| stlpips | 0.823 / 0.895 | 0.715 / 0.737 |

**NR, KonIQ SRCC:** topiq_nr 0.930 > hyperiqa 0.904 ≈ dbcnn 0.903 > maniqa 0.893 >
clipiqa+_rn50_512 0.885 > musiq 0.865. Q-Align reports 0.941 at 1024 px.

**Speed bar** (`tests/Efficiency_benchmark.csv`, from `tests/test_efficiency.py`): one
call per image on 4 JPEGs of 0.86–1.0 MP, mean of calls 2–4, batch 1, fp32, **timing
includes Pillow decode and host→GPU copy**. Noisy: topiq_fr and topiq_fr-pipal share an
architecture and read 0.211 vs 0.115 s on V100.

| Metric | Xeon 6150 CPU s | V100 s | V100 peak GB | M1 Pro CPU s |
|---|---|---|---|---|
| lpips | 0.205 | 0.072 | 0.29 | 0.174 |
| lpips-vgg | 3.17 | 0.143 | 2.50 | 2.51 |
| dists | 2.93 | 0.125 | 1.53 | 2.95 |
| deepdc | — | 0.130 | 1.10 | — |
| stlpips | 1.31 | 0.180 | 0.91 | 1.45 |
| stlpips-vgg | 4.12 | 0.223 | 2.51 | 3.58 |
| pieapp | 7.50 | 0.199 | 3.54 | 8.46 |
| wadiqam_fr | 0.87 | 0.083 | 0.47 | 1.42 |
| topiq_fr | 1.74 | 0.115–0.211 | 1.02 | 1.25 |
| ahiq | 147.5 | 0.796 | 4.47 | 23.9 |
| topiq_nr | 1.12 | 0.060 | 0.67 | 0.84 |
| musiq | 0.61 | 0.065 | 0.41 | 0.73 |
| maniqa | 13.1 | 0.542 | 2.66 | 8.72 |
| clipiqa | 0.99 | 0.058 | 0.63 | 0.63 |
| liqe | 0.37 | 0.058 | 0.96 | 0.19 |
| hyperiqa | 1.27 | 0.045 | 0.47 | 3.19 |
| dbcnn | 1.53 | 0.064 | 0.59 | 1.44 |
| nima | 0.17 | 0.067 | 0.25 | 0.15 |
| arniqa | 0.76 | 0.062 | 0.33 | 0.49 |
| tres | 29.2 | 0.227 | 2.08 | 11.7 |
| qalign | — | 0.220 | 15.4 | — |

These are pyiqa's numbers on pyiqa's hardware, not re-measured here.

## 7. Not verified

- Which components the S-Lab licence covers inside pyiqa.
- Whether upstream repos host the ST-LPIPS, AHIQ, MANIQA-koniq/kadid and LIQE weights.
- Licences of Google's MUSIQ checkpoints, DreamSim's backbone weights, Q-Align's base.
- Whether upstream HyperIQA released commercially usable KonIQ weights.
- torchvision's antialias default history and PyTorch's TF32 default.
- Which weights timm resolves for AHIQ per timm version.
- Why CKDN (53.7 MB) and LIQE (353.5 MB) checkpoints don't match a plain fp32 count.
- The Xeon thread count behind the efficiency CSV.
