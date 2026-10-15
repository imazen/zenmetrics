# pyiqa's upstream sources and their licences (2026-10-15)

Question: for every model and metric that
[IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch) ships, what do the
**original** authors' code and weights allow, and is that better than taking it from
pyiqa? This decides where each zenmetrics port is sourced from and what licence our
crates need. It feeds [`PYIQA_PORT_PLAN.md`](PYIQA_PORT_PLAN.md) §2.

Method: pyiqa checkout `09796cfe80b0` (2026-10-09). Every origin's LICENSE file, README
licence section, source-file header or release-zip readme was read as text at a pinned
commit; GitHub's SPDX field was not trusted (most entries were `NOASSERTION` or `NONE`).
MATLAB release zips were downloaded and their headers read. Data files were matched by
sha256. Per-origin quotes and pinned URLs are in
[`pyiqa_licenses_2026-10-15/`](pyiqa_licenses_2026-10-15/).

This is a fact-finding record, not legal advice. Items marked **counsel** need one.

## 1. pyiqa's own terms

- Code: PolyForm Noncommercial 1.0.0 since 2026-04-09 (commit `ccac8043`); CC BY-NC-SA
  4.0 from its first commit (2021-12-06) until then. Also ships `LICENSE-S-Lab` (NTU
  S-Lab 1.0, noncommercial) and `LICENSE-OpenAI-CLIP` (MIT).
- Weight mirror ([`chaofengc/IQA-PyTorch-Weights`](https://huggingface.co/chaofengc/IQA-PyTorch-Weights)):
  card licence `cc-by-nc-sa-4.0`; the card says the files are partly the team's own
  trainings and partly "official weights collected from original model repositories" and
  asks users to "respect the original licenses".
- **For us, pyiqa is the wrong source whenever the original is better**, which is almost
  always: of 40 neural entries, 24 have permissive upstream code (§2).

Licence links, checked 2026-10-15:

| link | result |
|---|---|
| README → `https://polyformproject.org/licenses/noncommercial/1.0.0/` (trailing slash) | **HTTP 404** |
| `LICENSE` → `https://polyformproject.org/licenses/noncommercial/1.0.0` | 200 |
| `LICENSE` text "[Distribution License](#distribution-license)", "[Changes and New Works License](#changes-and-new-works-license)" | not links: GitHub shows an extensionless `LICENSE` as plain text |
| polyformproject.org page, same two anchors | dead: the page's `<h2>` headings carry no `id`, so `#distribution-license` has no target |
| [PolyForm's own markdown on GitHub](https://github.com/polyformproject/polyform-licenses/blob/1.0.0/PolyForm-Noncommercial-1.0.0.md) | anchors work (GitHub generates heading ids) |

## 2. Neural models

Verdict classes: **PERMISSIVE** (code and weights), **CODE-OK / WEIGHTS-UNKNOWN**
(permissive code; weights hosted with no stated terms), **CODE-OK / WEIGHTS-NC**,
**NC**, **NO-LICENCE** (no licence file: all rights reserved, so paper-only),
**MIXED**.

| model | upstream code licence | official weights and terms | verdict for us |
|---|---|---|---|
| LPIPS (alex/vgg/squeeze, LPIPS+) | BSD-2 (richzhang/PerceptualSimilarity) | in repo, BSD-2 | PERMISSIVE |
| DISTS | MIT (dingkeyan93/DISTS) | in repo, MIT | PERMISSIVE |
| DeepDC | MIT (h4nwei/DeepDC) | none needed (torchvision VGG19) | PERMISSIVE |
| DMM | MIT (Baoliang93/DMM) | none needed (torchvision VGG16) | PERMISSIVE |
| ST-LPIPS | BSD-2 (abhijay9/ShiftTolerant-LPIPS) except `models_lpf/*.py`, Adobe antialiased-cnns **CC BY-NC-SA 4.0** | in repo, BSD-2 | MIXED: rewrite BlurPool from the paper (a fixed binomial low-pass) and the rest is PERMISSIVE |
| WaDIQaM | official dmaniry/deepIQA: MIT; lidq92/WaDIQaM: no licence | deepIQA `models/{fr,nr}_{live,tid}_*.model` (Chainer), MIT | PERMISSIVE via deepIQA. pyiqa's weights are its own KADID/KonIQ retrain, so scores differ from pyiqa columns |
| ARNIQA | Apache-2.0 | GitHub release, Apache-2.0 | PERMISSIVE |
| MANIQA | Apache-2.0 | GitHub release | PERMISSIVE |
| MUSIQ | Apache-2.0 (google-research) | GCS / Kaggle, "Apache 2.0" | PERMISSIVE |
| A-FINE | Apache-2.0 | Google Drive; trained on the authors' DiffIQA, whose terms are noncommercial and cover "derived data" | PERMISSIVE code; weights **counsel** (dataset-derived) |
| FGResQ | Apache-2.0 | HF, Apache-2.0 | PERMISSIVE |
| LAION aesthetic | Apache-2.0 | in repo, Apache-2.0 | PERMISSIVE (CLIP caveat, §4) |
| CLIPScore | MIT | none (OpenAI CLIP) | PERMISSIVE (CLIP caveat) |
| Compare2Score | MIT | HF, MIT; LLaMA-2 base | PERMISSIVE; LLaMA-2 terms not read |
| Q-ReAlign | no file; `pyproject.toml` says MIT | HF, Apache-2.0 | PERMISSIVE (weights); code paper-first |
| DreamSim (not in pyiqa) | MIT | GitHub releases | PERMISSIVE (backbone caveats) |
| FID / IS | pytorch-fid Apache-2.0, clean-fid MIT | Inception-v3 Apache-2.0; FFHQ stats are CC BY-NC-SA (dataset) | PERMISSIVE except FFHQ stats |
| AHIQ | MIT; README says it borrows from anse3832/IQT (no licence) | Google Drive, no terms | CODE-OK / WEIGHTS-UNKNOWN; write from paper |
| CKDN | MIT | Google Drive, no terms | CODE-OK / WEIGHTS-UNKNOWN |
| HyperIQA | MIT | Google Drive, no terms (pyiqa uses its own retrain) | CODE-OK / WEIGHTS-UNKNOWN |
| LIQE | MIT | Google Drive, no terms | CODE-OK / WEIGHTS-UNKNOWN |
| NIMA | MIT (yunxiaoshi) | Google Drive, no terms (pyiqa uses its own retrain) | CODE-OK / WEIGHTS-UNKNOWN |
| UNIQUE | Apache-2.0 | Google Drive, no terms | CODE-OK / WEIGHTS-UNKNOWN |
| DBCNN | MIT | only the SCNN branch is public; full weights exist only as pyiqa's retrain | CODE-OK / WEIGHTS-NC |
| PaQ-2-PiQ | MIT (baidut/paq2piq) | baidut/PaQ-2-PiQ: README badge CC BY-NC-SA 4.0 | CODE-OK / WEIGHTS-NC |
| Q-Align | S-Lab 1.0 (NC) | HF `q-future/one-align`, MIT; LLaMA-2 base | weights permissive, code NC: paper-first |
| TOPIQ (FR, NR, Swin) | pyiqa's own (PolyForm NC); no other source | pyiqa HF only, CC BY-NC-SA | NC: paper-first code, weights NC (or retrain) |
| CLIP-IQA(+) | S-Lab 1.0 (NC) | GitHub release, S-Lab | NC |
| PieAPP | modified BSD-2 "for Non-Commercial Use Only" (UC Regents) | UCSB site | NC |
| QualiCLIP | CC BY-NC 4.0 | GitHub, CC BY-NC 4.0 | NC |
| URanker | CC BY-NC 4.0 | same | NC |
| TReS | CMU academic agreement: no distribution, derivatives assigned to CMU | same | **avoid**: don't read the code |
| CNNIQA, IQT, MA-CLIP, MetaIQA, MS-SWD | no licence file | none, or pyiqa-only (MS-SWD learned weights) | NO-LICENCE: paper only |

Shared backbones: torchvision (BSD-3), timm (Apache-2.0), Swin (MIT), OpenAI CLIP (MIT
code). Both torchvision's and timm's READMEs say ImageNet terms may apply to pretrained
weights (**counsel**). OpenAI's CLIP model card says "Any deployed use case of the model
- whether commercial or not - is currently out of scope" (**counsel** for every CLIP-based
model).

## 3. Classical metrics

| metric | origin terms | zenmetrics crate | note |
|---|---|---|---|
| VIF (vifp, vifvec) | LIVE/UT Austin: "for any purpose, provided that the copyright notice in its entirety appear in all copies"; matlabPyrTools MIT | `vif` | notices added 2026-10-15 (`LICENSE-live-vif`, `LICENSE-matlabPyrTools`) |
| NLPD | `NLP_dist.m` and Valerolaparra/NLPD_Pytorch: no licence text; IQA_pytorch MIT | `nlpd` | IQA_pytorch notice added 2026-10-15 for `iqa.rs` |
| GMSD | IQA_pytorch MIT; `GMSD.m` no licence text | `gmsd` (libgmsd, MIT, notice kept) | ok |
| MAD | STMAD files (archived in Netflix/vmaf `matlab/`): no licence text | `mad-iqa` | no grant and no restriction found |
| MS-SSIM | Waterloo `msssim.zip`: no licence text | `msssim` | low risk |
| SSIM | NYU `ssim_index.m`: "educational and research purposes only … shall not be used, rewritten, or adapted as the basis of a commercial software or hardware product without first obtaining permission" | `zenmetrics-cli` classical.rs (from the paper's window and constants) | keep it paper-sourced |
| **FSIM** | `FeatureSIM.m` (PolyU): same research-only / no-commercial-rewrite clause | `fsim`, which describes itself as ported "for behavioral fidelity" to it | **counsel, or ask Lin Zhang / Lei Zhang for permission** |
| **VSI** | `VSI.m` (Tongji, Lin Zhang): same clause | `vsi`, same description | **same** |
| CW-SSIM | MathWorks FEX zip: `license.txt` BSD-2, but `cwssim_index.m` header has the research-only clause | none | paper-only if ported |
| BRISQUE | LIVE notice (any purpose, keep notice); the LIVE SVM model matches piq's `brisque_svm_weights.pth` | none | use the LIVE zip's model with the notice; never `brisque_matlab.mat` (MathWorks-derived) |
| NIQE | LIVE notice; `modelparameters.mat` byte-identical to pyiqa's file | none | same; never `niqe_matlab_params.mat` |
| IL-NIQE | PolyU `ILNIQE.zip`: no licence text anywhere | none | needs our own pristine model, or permission |
| NRQM / PI | sr-metric MIT (bundles GPL randomforest-matlab); PIRM2018 scripts "academic research purposes only" | none | NRQM model usable with MIT notice; PI is a formula |
| PIQE | pypiqe MIT; MathWorks doc | none | port from the paper |

pyiqa's comments say official VSI and MAD code is "not available"; today `VSI.m` downloads
from the author's site and STMAD is in Netflix/vmaf.

## 4. What this means for zenmetrics

1. **Source every port from the original upstream, never from pyiqa files.** Read
   permissive upstream code; for NC, no-licence and TOPIQ models, write from the paper.
   TReS: don't read the code at all.
2. **Our crates keep one licence** (AGPL-3.0 + commercial). Noncommercial status
   belongs to **weights**, which we never redistribute unless their own licence is
   permissive; it is recorded per model in the model card and gated at runtime. A
   separately licensed crate is made for one specific model only if its code cannot be
   written independently (none so far). Plan §2 is revised accordingly.
3. **Upstream beats pyiqa for weights too**: WaDIQaM (deepIQA, MIT) and MUSIQ (Apache)
   move from NC to permissive. Their scores won't match pyiqa's retrained columns; where
   a pyiqa column is the target (AIC fits), that is a separate, NC-weights build.
4. **Oracles**: the upstream reference package per model (`lpips`, `DISTS_pytorch`,
   deepIQA, ...). pyiqa is needed only to reproduce pyiqa-retrained weights.
5. **Notices travel with ports**: every crate that transcribes code or constants keeps
   the origin's notice in a `LICENSE-<origin>` file (the gmsd convention).
6. **Open (counsel / permission)**: `fsim` and `vsi` against the research-only clause;
   ImageNet-pretrained backbone weights; OpenAI CLIP's model-card scope; A-FINE's
   DiffIQA-derived weights; LLaMA-2 terms for Compare2Score / Q-Align weights.

## 5. zenmetrics' own crates (audit 2026-10-15)

Every crate that ports or transcribes third-party code, constants or data was traced
to its origin. Missing notices were added verbatim as `LICENSE-<origin>` files
(commits `babd7861`, `c615004e`): vif, nlpd, msssim, iwssim, iwssim-gpu,
butteraugli-gpu, ssim2-gpu, zensim-gpu, zenmetrics-api, dssim-gpu, fsim, mad-iqa,
hdrvdp, zenmetrics-math, cvvdp-conformance, zenfleet-dash, psnrhvs, mdctpsnr.

ColorVideoVDP is **MIT** for every version cvvdp and cvvdp-gpu port (it was CC BY-NC
4.0 until 2024-06-17, commit `644abf6d`; v0.5.4 onward is MIT), and its MIT text is
already vendored. No change needed.

Open, for the owner (not changed):

| crate | origin terms | what the crate takes |
|---|---|---|
| `fsim` | `FR_FSIMc.m` / `FeatureSIM.m`: research-only, "shall not be used, rewritten, or adapted as the basis of a commercial software or hardware product without first obtaining permission" | pipeline "pass-for-pass", constants |
| `vsi` | `VSI.m`: same clause; octave-image `imresize.m` `conv_interp_vec`: GPL-3.0-or-later | pipeline and constants; a "bit-identical" port of the GPL helper |
| `psnrhvs` | Ponomarenko `psnrhvsm.m`: same research-only clause | CSF and masking tables, block loop |
| `hdrvdp` (v3) | follows jpeg-ai-qaf `VDP3/` (no licence file); reproduces the behaviour of three HDR-VDP helpers marked "internal use. Do not redistribute" | v3 pipeline. `THIRD-PARTY-NOTICES.md` §2's "no line reproduced" statement covers the v2 modules only |
| `mad-iqa` | STMAD release files: no licence text | `validation/ical_{std,stat}.m` are verbatim transcriptions of the C files |
| `msssim`, `iwssim`, `iwssim-gpu`, `nlpd` | Waterloo `msssim.zip`, `iwssim_iwpsnr.zip`, `NLPD_Pytorch`: no licence text (no grant, no restriction) | behaviour; low priority |
| `cvvdp` | licence field is `AGPL-3.0-or-later` without the commercial option the other crates carry | owner decision |
| `zenmetrics-corpus` | `data/source.png` is attributed to `imazen/dssim-cuda`, which returns 404; hash matches no kornelski/dssim test image | origin unrecorded |

The research-only clauses (fsim, vsi, psnrhvs, and SSIM's NYU code, which zenmetrics
does not port) restrict the authors' code; whether they bind an independent
reimplementation of a published algorithm is the counsel question in
[`PYIQA_PORT_PLAN.md`](PYIQA_PORT_PLAN.md) §9. Asking the authors for permission is the
cheaper route.

