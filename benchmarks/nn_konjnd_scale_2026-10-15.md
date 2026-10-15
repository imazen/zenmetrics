# One JND in LPIPS (alex, vgg, squeeze), DISTS, ST-LPIPS (alex, vgg) and DeepDC units, from KonJND-1k

Runs 2026-10-15 on `dev` (Zen 5, CPU fp32, torch 2.14.1+cpu, 12 threads, both through run-heavy):

- Run 1 (`konjnd_nn.py`): LPIPS and DISTS. rc=0, 1264 s, peak RSS 1.42 GiB. Packages `lpips==0.1.4`,
  `DISTS_pytorch==0.1`. The Rust port matches these to within 2e-7, so these values stand in for it.
- Run 2 (`konjnd_nn2.py`): ST-LPIPS and DeepDC on the same 1,008 pairs, appended as new columns.
  rc=0, 1386 s, peak RSS 1.42 GiB. Packages `stlpips_pytorch==0.0.2` (+ `torchsummary==1.5.1`) and
  `DeepDC_PyTorch==0.2`. The run-1 columns were carried over verbatim (checked equal on all 1,008 rows).

## Dataset

- Root (dev): `/mnt/v/datasets/KonJND-1k/KonJND-1k/`
  - `source_image/`: 1,008 pristine references, SRC0001–SRC1008, all 640×480 RGB PNG.
  - `jpeg/`: SRC0001–SRC0504 × JPEG QF 001–100 (`SRCxxxx_JPEG_NNN.jpg`, 50,400 files).
  - `bpg/`: SRC0505–SRC1008 × BPG QP 001–051, shipped **pre-decoded** as PNG (25,704 files).
    Higher QP means worse quality (`konjnd_full_scored.csv`: BPG_049 has ssim2 −27).
  - `subjective_ratings.csv` (sha256 `fb423f1c…`): per image `image_id, Compression type,
    No. of ratings, mean, std, ratings`. `mean` is the per-image mean picture-wise JND (PJND)
    threshold across 24–45 crowd observers per image (median 42), in codec-level units (JPEG QF, BPG QP).
- The set is 504 JPEG refs ∪ 504 BPG refs, with no overlap (`~/work/zen/zensim/docs/DATA_SPLITS.md`
  line 246: "the set is 504 JPEG refs ∪ 504 BPG refs, intersection 0"). The repo's
  `konjnd_jpeg_val` (504 pairs, `DATA_PROVENANCE.md` line 1122;
  `/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv`, sha256 `70148a39…`)
  is exactly the JPEG half. This run keeps that split as the "JPEG" codec row; "BPG" is the
  other 504. Authors define no split: whole-set benchmark.

## Pairs

Pair = (reference, the distorted file at level `floor(mean + 0.5)`), i.e. round-half-up. This is the
repo's corrected rule (`/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06/pairs-corrected-2026-09-06/README.md`:
"konjnd.tsv — round-half-UP"). Cross-checked: 1,008/1,008 distorted filenames match that `konjnd.tsv`.
Counts: JPEG 504, BPG 504, pooled 1,008. JPEG levels at JND span QF 22–70 (median 56); BPG
QP 25–34 (median 30). JPEG decoded with Pillow 12.3.0 (libjpeg-turbo 3.1.4.1); BPG files are the shipped PNGs.

## Metrics

- LPIPS: `lpips.LPIPS(net=alex|vgg|squeeze, version='0.1')`, input `uint8/255*2-1` float32, NCHW, no resize, spatial mean.
- DISTS: `DISTS_pytorch.DISTS()`, input `uint8/255` float32, no resize; `weights.pt` loaded from `sys.prefix` (the venv
  root, byte-identical to the copy inside the package dir). Backbones: torchvision ImageNet weights
  (`alexnet-owt-7be5be79`, `vgg16-397923af`, squeezenet1_1).
- ST-LPIPS: `IPython` stubbed before import (the package imports `IPython.embed`), then
  `stlpips.LPIPS(net=alex|vgg, variant='shift_tolerant', verbose=False)`, input `uint8/255*2-1` float32, no resize,
  spatial mean. The package loads `weights/vST0.0/<net>_shift_tolerant.pth` with `strict=False`; checked: every
  tensor in the checkpoint lands in the model, the checkpoint covers every model key, and the 5 reported
  "missing" keys are `lins.*`, which alias the same modules as `lin0..4`. `model.version` stays `0.1`, so the
  scaling layer is applied.
- DeepDC: `imp` stubbed (removed in Python 3.12), `DeepDC()(ref, dist)`, input `uint8/255` float32, no resize
  (upstream's demo resizes the short side to 224; not done here). **ImageNet input normalisation ON**
  (coordinator definition, matches the JPEG AIC-4 DeepDC values): `features_extractor.use_input_norm = True`,
  `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` registered as buffers, then `eval()`. Package
  default is normalisation OFF; on SRC0001 OFF gives 0.036124 and ON gives 0.042176. A first run 2 started with
  normalisation OFF was stopped at about 130 pairs and discarded (`pairs_scores.aborted-nonorm.tsv.partial`,
  `run2.aborted-nonorm.log`); its ST-LPIPS values were bit-identical to the final run. Backbone: torchvision
  `vgg19-dcbb9e9d`.
- Identity check on SRC0001 vs itself: LPIPS 0.0 for all three nets, DISTS −1.19e-7, ST-LPIPS alex 0.0,
  ST-LPIPS vgg 0.0, DeepDC 0.0.

## Definition

**1 JND (metric units) := the pooled median of the metric over the 1,008 (reference, image-at-mean-PJND) pairs.**
Percentiles are numpy linear interpolation. The spread column is p75/p25.

## Results

| metric | codec | n | p25 | median | p75 | mean | p75/p25 |
|---|---|---|---|---|---|---|---|
| lpips_alex | JPEG | 504 | 0.047362 | 0.067057 | 0.091202 | 0.073344 | 1.926 |
| lpips_alex | BPG | 504 | 0.044566 | 0.063697 | 0.089724 | 0.072027 | 2.013 |
| lpips_alex | pooled | 1008 | 0.046107 | **0.065220** | 0.090613 | 0.072685 | 1.965 |
| lpips_vgg | JPEG | 504 | 0.175627 | 0.215482 | 0.253103 | 0.214598 | 1.441 |
| lpips_vgg | BPG | 504 | 0.142629 | 0.173066 | 0.211996 | 0.180071 | 1.486 |
| lpips_vgg | pooled | 1008 | 0.157719 | **0.193925** | 0.236552 | 0.197335 | 1.500 |
| lpips_squeeze | JPEG | 504 | 0.071734 | 0.093239 | 0.113755 | 0.094885 | 1.586 |
| lpips_squeeze | BPG | 504 | 0.049897 | 0.064681 | 0.086120 | 0.071372 | 1.726 |
| lpips_squeeze | pooled | 1008 | 0.057006 | **0.079601** | 0.103903 | 0.083129 | 1.823 |
| dists | JPEG | 504 | 0.087611 | 0.105177 | 0.123957 | 0.108368 | 1.415 |
| dists | BPG | 504 | 0.068635 | 0.083443 | 0.102237 | 0.086683 | 1.490 |
| dists | pooled | 1008 | 0.076373 | **0.094498** | 0.114473 | 0.097526 | 1.499 |
| stlpips_alex | JPEG | 504 | 0.014748 | 0.020606 | 0.029211 | 0.023975 | 1.981 |
| stlpips_alex | BPG | 504 | 0.018063 | 0.027950 | 0.040027 | 0.032450 | 2.216 |
| stlpips_alex | pooled | 1008 | 0.016004 | **0.023365** | 0.034415 | 0.028213 | 2.150 |
| stlpips_vgg | JPEG | 504 | 0.053745 | 0.069698 | 0.089607 | 0.074689 | 1.667 |
| stlpips_vgg | BPG | 504 | 0.043573 | 0.060776 | 0.084281 | 0.068395 | 1.934 |
| stlpips_vgg | pooled | 1008 | 0.048183 | **0.065762** | 0.087483 | 0.071542 | 1.816 |
| deepdc | JPEG | 504 | 0.064175 | 0.080536 | 0.099321 | 0.084171 | 1.548 |
| deepdc | BPG | 504 | 0.035697 | 0.047554 | 0.064147 | 0.052494 | 1.797 |
| deepdc | pooled | 1008 | 0.044892 | **0.064153** | 0.085502 | 0.068333 | 1.905 |
| ckdn | JPEG | 504 | 0.003011 | 0.006051 | 0.010098 | 0.006987 | 3.354 |
| ckdn | BPG | 504 | 0.002156 | 0.004728 | 0.008025 | 0.005915 | 3.723 |
| ckdn | pooled | 1008 | 0.002575 | **0.005276** | 0.009086 | 0.006451 | 3.528 |
| afine | JPEG | 504 | 0.256606 | 0.498062 | 0.756067 | 0.364956 | 2.946 |
| afine | BPG | 504 | 0.241244 | 0.512932 | 0.754651 | 0.480715 | 3.128 |
| afine | pooled | 1008 | 0.247775 | **0.507388** | 0.755332 | 0.422836 | 3.048 |
| dreamsim | JPEG | 504 | 0.017586 | 0.026361 | 0.034738 | 0.027194 | 1.975 |
| dreamsim | BPG | 504 | 0.007528 | 0.012349 | 0.017801 | 0.013551 | 2.371 |
| dreamsim | pooled | 1008 | 0.010763 | **0.017641** | 0.028215 | 0.020373 | 2.618 |
| ahiq | JPEG | 504 | 0.015013 | 0.022583 | 0.032845 | 0.024553 | 2.188 |
| ahiq | BPG | 504 | 0.019683 | 0.026778 | 0.033226 | 0.027219 | 1.688 |
| ahiq | pooled | 1008 | 0.017611 | **0.024882** | 0.032989 | 0.025886 | 1.873 |

Estimated 1 JND: LPIPS-alex 0.0652, LPIPS-vgg 0.194, LPIPS-squeeze 0.0796, DISTS 0.0945,
ST-LPIPS-alex 0.0234, ST-LPIPS-vgg 0.0658, DeepDC (input norm on) 0.0642, CKDN 0.0053,
A-FINE (raw `afine_all`) 0.5074, DreamSim ensemble distance 0.0176, AHIQ 0.0249.

The `ckdn` rows are not from an upstream package: they were scored by the Rust port
(`zenmetrics batch --metric ckdn --features nn,nn-ort`, engine ort-cuda, ONNX Runtime
1.31, i265, 2026-10-15, in the commit that adds the `ckdn` row to `jnd_scale.tsv`).
CKDN is higher-is-better and its identical-pair score is image-dependent (median
0.614, range 0.434–0.798 on these references — qse ≠ dte), so each value is
`|score(ref, dist) − score(ref, ref)|`, the distance from the perfect score for that
reference, and the table entries are its quartiles and median. In 416 of 1,008 pairs
the distorted image scored above its own reference. Measured with the AIC-4 eval
transform in place (each image resized short-side→329 — bicubic+antialias for the
distorted side, nearest for the reference — then centre-cropped to 288), which is
the convention this column scores with; the earlier native-feed run is superseded.

The `afine` rows are likewise scored by the Rust port (`--metric afine`, same
engine/host/date, in the commit that adds the `afine` row to `jnd_scale.tsv`).
A-FINE is lower-is-better and its identical-pair score is image-dependent
(median −2.996, range −3.888…−2.350 on these references — the naturalness term
rates even pristine references image-dependently), so each value is
`score(ref, dist) − score(ref, ref)`, the distance above the per-reference
perfect score, and the table entries are its quartiles and median. In 123 of
1,008 pairs the distorted image scored below (better than) its own reference.
Scored at native resolution with upstream preprocessing (CLIP normalisation,
bicubic upsample of each side to the next multiple of 32).

The `dreamsim` rows are likewise scored by the Rust port (`--metric dreamsim`,
same engine/host/date, in the commit that adds the `dreamsim` row to
`jnd_scale.tsv`). DreamSim is lower-is-better and its identical-pair score is
exactly 0 on every reference (cosine distance of equal embeddings), so each
value is the raw distance `score(ref, dist)` and the table entries are its
quartiles and median. Scored at native resolution with upstream preprocessing
(Pillow's fixed-point anamorphic bicubic resize to 224×224 — what
torchvision `Resize(BICUBIC)` on a PIL image runs — then the ensemble's
per-extractor normalisation inside the graph).

The `ahiq` rows are likewise scored by the Rust port (`--metric ahiq`, same
engine/host/date, in the commit that adds the `ahiq` row to `jnd_scale.tsv`).
AHIQ is higher-is-better and its identical-pair score is image-dependent
(median 0.644, range 0.591–0.718 on these references — the pixel regressor
rates even pristine references image-dependently), so each value is
`|score(ref, dist) − score(ref, ref)|`, the distance from the perfect score
for that reference, and the table entries are its quartiles and median.
Scored at native resolution with upstream's protocol (no resize; mean over
the nine deterministic 224×224 `five_point_crop` crops; the 640×480 KonJND
images are comfortably ≥224).

## Caveats

- **Codec dependence.** For vgg, squeeze and DISTS the BPG median is 20–31 % below the JPEG median
  (vgg 0.173 vs 0.215, squeeze 0.065 vs 0.093, DISTS 0.083 vs 0.105). So the same JND reads differently per
  artifact type. Only alex is close (0.064 vs 0.067), and it has the widest spread. The pooled median
  is a 50/50 JPEG/BPG blend, not a codec-free constant. DeepDC has the largest gap: the BPG median is 41 % below
  JPEG (0.048 vs 0.081). ST-LPIPS-vgg is 13 % lower for BPG (0.061 vs 0.070). **ST-LPIPS-alex goes the other way**:
  BPG 36 % above JPEG (0.028 vs 0.021), the only one of the seven metrics where BPG reads higher.
- **Wide per-image spread.** p75/p25 is 1.4–2.0. Pooled 5th–95th percentile: alex 0.024–0.147,
  vgg 0.103–0.301, squeeze 0.034–0.148, DISTS 0.051–0.153, ST-LPIPS alex 0.0085–0.064, ST-LPIPS vgg 0.027–0.134,
  DeepDC 0.025–0.126. ST-LPIPS-alex has the widest spread of all seven (pooled p75/p25 2.15). One JND is a population median, not a
  per-image threshold.
- **Level vs metric.** Spearman(PJND level, metric) within JPEG is +0.34 to +0.55: images whose JND sits at
  a higher QF also score higher at that threshold. This is a content effect, since the level and the metric share the image.
  Within BPG it is +0.07 to +0.15, with the QP range narrow (25–34). Run-2 metrics: ST-LPIPS-vgg +0.497 (JPEG) /
  +0.143 (BPG), DeepDC +0.318 / +0.084, ST-LPIPS-alex only +0.047 / +0.150.
- **Discrete levels.** The threshold image is the nearest available level to a mean of 24–45
  observers. One QF/QP step shifts the value, and the 4 exact-.5 means depend on the rounding rule.
- **Content and size.** All content is 640×480 natural photos, no resize. LPIPS/DISTS are not scale-invariant,
  so these numbers apply to this size and viewing setup (crowdsourced, the authors' protocol).
- **Cross-metric agreement at JND.** Spearman over the 1,008 values: alex–vgg 0.733, alex–squeeze 0.787,
  alex–DISTS 0.526, vgg–squeeze 0.922, vgg–DISTS 0.721, squeeze–DISTS 0.694. DeepDC tracks DISTS most closely
  (0.828), then LPIPS-vgg (0.741) and squeeze (0.737). ST-LPIPS-vgg tracks LPIPS-alex (0.805). ST-LPIPS-alex is
  nearly unrelated to the rest at the JND (LPIPS-alex 0.571, ST-LPIPS-vgg 0.638, LPIPS-vgg 0.197, DISTS 0.125,
  DeepDC 0.046).
- Holdout note: zensim registers KonJND-JPEG as an evaluation set (SELECT/TERMINAL). This run reads only the
  dataset's PJND levels to place pairs, and it trains or selects nothing.

## Files

- `nn_konjnd_scale_2026-10-15.tsv` (next to this file): the full table above, all codec rows and
  quartiles. The pooled medians are copied into `crates/zenmetrics-nn/models/jnd_scale.tsv`, which
  `tests/parity.rs` reads to express engine tolerance in JND units.
- Per-pair scores (244,700 bytes, too large to commit): `pairs_scores.tsv`, sha256
  `9d10349e2bcaa570a61c79cca11b8ec3ec77be3ce48828e8fb9a51083ce44b55`, columns `src, codec, pjnd_mean,
  level, n_ratings, ref_path, dist_path, width, height, lpips_alex, lpips_vgg, lpips_squeeze, dists,
  stlpips_alex, stlpips_vgg, deepdc`. Stored on dev at `/mnt/v/output/zenmetrics/nn-konjnd-2026-10-15/`
  and mirrored to tower `/mnt/tower/output/zenmetrics/nn-konjnd-2026-10-15/` (checked on dev,
  2026-10-15), with the run-1 version `pairs_scores.v1.tsv` (13 columns, sha256 `2510cb04…`).
- Scripts: `nn_konjnd_scale_2026-10-15/konjnd_nn.py` (run 1), `konjnd_nn2.py` (run 2),
  `konjnd_summary.py` (the table). They read the dataset at the dev path above.
