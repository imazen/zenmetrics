# AIC2026 columns: canonical-configuration search (2026-10-17)

Owner request: configure the canonical implementations to reproduce the published
AIC2026 columns; whatever a bounded search cannot reproduce becomes a question for the
dataset authors. Scope (worst rank agreement first): `A-FINE`, `WD_s1`–`WD_s5`, `DMM`,
`HDR_VDP_3`, `HDR_VDP_2`, then (added by the lead) `proposal-Butteraugli` and `CAMBI`.

Data: the 61 public 840×944 crop pairs and their published values (`metrics_subset.csv`,
not committed). Every number below is measured over those 61 pairs unless `n` says
otherwise. "site-JND" uses the AIC site's per-metric mapping
`jnd(x) = a·|sign·(x−b)|^c` (site `config.json`), so 1e-3 site-JND is the brief's match
gate. TSVs in this directory hold aggregates only (median/max |Δ|, site-JND, Pearson,
Spearman and the least-squares affine map ours→published); no per-pair published values.

| column | result | best configuration | median / max site-JND | Spearman |
|---|---|---|---|---|
| `proposal-Butteraugli` | **reproduced** | `max(0.00459792·N, 2.0938201·N − 1.1033102)`, N = our libjxl `butteraugli_pnorm3` | 3.2e-8 / 5.1e-5 | 1.0000 |
| `CAMBI` | **reproduced** | libvmaf CAMBI defaults on ffmpeg/swscale `yuv420p` luma | 8.2e-8 / 2.1e-3† | 1.0000 |
| `DMM` | **reproduced** | pyiqa 0.1.16 `dmm`, CUDA, fp32 | 2.6e-4 / 2.4e-3 | 0.9999 |
| `HDR_VDP_3` | **reproduced (identity map, median 1.4e-3)** | HDR-VDP-3.0.7 `res.Q`, sRGB→200 cd/m² XYZ, ppd 37.84 | 1.36e-3 / 1.29e-2 | 1.0000 |
| `HDR_VDP_2` | near (affine ≈ identity) | HDR-VDP-2.2.2 `res.Q`, sRGB→200 cd/m², ppd 37.84 | 3.65e-3 / 5.15e-2 | 0.9998 |
| `WD_s1`–`WD_s5` | not reproduced | — | best 0.13–0.20 | best 0.98–0.99 |
| `A-FINE` | not reproduced | — | best 0.10–0.14 | best 0.94 |

† CAMBI's max raw |Δ| is 4.8e-7 (the published 6-decimal rounding); the site map's
exponent c < 1 makes the JND slope steep near 0.

## proposal-Butteraugli: a fixed map of libjxl's 3-norm

The site CSVs also carry `Butteraugli-3norm` and `Butteraugli-max`. Over all 9,618 rows of
both the crop and full-resolution CSVs, `proposal-Butteraugli` is a two-segment linear
function of `Butteraugli-3norm` (`scripts/butter_map_fit.py`):

    P = max(0.00459792·N, 2.0938201·N − 1.1033102)        breakpoint N ≈ 0.52810

The low segment is exact to 4e-19 and the high one to the 3-norm column's printed
precision (median 4.2e-6, p99 1.06e-5, max 1.06e-4 where that column has fewer decimals).
The lead's affine `2.09·x − 1.10` was this map with the low segment mixed in.

Our CLI `butteraugli` (libjxl `butteraugli` crate, defaults) reproduces the site's
`Butteraugli-3norm` and `Butteraugli-max` on the 61 crops to median 2e-6, max 4.7e-5
(both CSV-rounded), and the map applied to our `butteraugli_pnorm3`
(`butteraugli_configs.tsv`):

| ours | median / max abs | median / max site-JND | Pearson | Spearman |
|---|---|---|---|---|
| **map(`butteraugli_pnorm3`)** | **6.6e-8 / 1.0e-4** | **3.2e-8 / 5.1e-5** | **1.0000** | **1.0000** |
| raw `butteraugli_pnorm3` | 1.05 / 3.42 | 0.468 / 1.22 | 0.9999 | 1.0000 |

So no intensity-target, linear-RGB or max/3-norm combination search was needed: the column
is the default libjxl 3-norm passed through that map. To expose it: a `proposal-butteraugli`
column = the map above on `butteraugli_pnorm3`. The map's provenance (a fitted JND
calibration of the proposal?) is the only open question.

## CAMBI: the luma comes from ffmpeg's swscale

`cambi_configs.tsv`, `scripts/cambi_probe.py`. The pinned libvmaf 3.2.1 `vmaf` binary with
`--feature cambi` (standalone defaults: window 65, topk 0.6, TVI 0.019, max log contrast 2,
BT.1886, VLT 0, no high-res speedup) on the distorted image, with the PNG → YUV step varied:

| PNG → YUV | median / max abs | median / max site-JND | Spearman |
|---|---|---|---|
| exact `round(16 + (65.481R + 128.553G + 24.966B)/255)` (earlier rounds) | 1.9e-4 / 0.178 | 1.7e-4 / 0.065 | 0.9972 |
| **`ffmpeg -pix_fmt yuv420p`** (also with `accurate_rnd`, also `yuv444p`) | **2.4e-7 / 4.8e-7** | **8.2e-8 / 2.1e-3** | **1.0000** |
| ffmpeg BT.709 matrix | 6.0e-3 / 0.56 | 7.1e-3 / 1.33 | 0.9903 |
| ffmpeg full range | 1.4e-2 / 0.84 | 1.8e-2 / 1.36 | 0.9611 |
| ffmpeg `yuv420p10le`, `-b 10` | 1.8e-2 / 5.33 | 2.8e-2 / 1.38 | 0.8824 |

The two conversions differ on 0.45 % of pixels by one code, which CAMBI's banding detector
amplifies (19 pairs above 1e-3 with the exact formula, none with ffmpeg; S27: 0.0388 vs
published 0.216892, ffmpeg 0.2168916). ffmpeg 8.0.1's luma equals, on all 48,370,560
pixels of the 61 distorted images (`scripts/sws_luma_check.py`, 0 mismatches):

    t = (8414·R + 16519·G + 3208·B + (32 << 14) + (1 << 8)) >> 9
    Y = (t + 32) >> 6

(coefficients `rint({0.299, 0.587, 0.114}·219/255·2^15)`). Our `vmaf::cambi_from_luma`
already equals libvmaf to 7e-18 (`../aic_more_2026-10-15/`), so the `cambi` CLI needs only
this luma ingress to reproduce the column.

## HDR_VDP_3 and HDR_VDP_2: the display model and the output term

The JPEG-AI quality framework's own HDR-VDP path (`jpeg-ai-qaf` @ `0628a6b`, the lineage
our `hdrvdp::v3` port verifies against) at its defaults (8-bit codes decoded as PQ,
0.5 % reflection of 100 lx, 64.5″ 4K at 1.325 m → ppd 64.06) misses both columns:
`HDR_VDP_3` median |Δ| 0.62 JOD, Spearman 0.83; `HDR_VDP_2` (its Python VDP2 port
returns the removed logistic `Q_MOS`) median |Δ| 31.5, Spearman 0.81.

What closes them, step by step (`hdrvdp3_configs.tsv`, `hdrvdp2_configs.tsv`):

1. **ppd** is the strongest single knob (VDP3 median 0.62 → 0.15–0.30 at ppd 28–42), but
   alone it leaves Spearman at 0.84–0.86.
2. **The CVVDP `standard_fhd` display as ingress.** The paper scores CVVDP with
   `cvvdp -d standard_fhd` (200 cd/m², 37.84 ppd). Decoding the sRGB codes to 200 cd/m²
   with the sRGB EOTF and scoring at ppd 37.84 raises Spearman to 0.999 (VDP3) and 0.996
   (VDP2). With the CVVDP black level (0.2) and reflection (0.3979 cd/m²) included the
   level is still off; **with no black level and no reflection** it is right:
   `L = 200 · sRGB_EOTF(V)` cd/m².
3. **`HDR_VDP_3` is `res.Q`, not `res.Q_JOD`.** Inverting 3.0.7's
   `Q_JOD = 10 − 0.52·(10−Q)^1.2812` (`hdrvdp3.m` 724–726) on our scores turns a
   slope-1.41 relation into the identity. Our `v3` result already carries `q`.
4. **Emission spectra `oled`**, i.e. MATLAB's default for `xyz` or `rgb-bt.2020` input:
   XYZ (D65 sRGB matrix), BT.2020 and BT.709-with-`oled` agree to 1e-4 Q. The `rgb-bt.709`
   default `led-lcd-srgb` gives median 2.1e-3 site-JND.

| column | configuration | median / max abs | median / max site-JND | Pearson | Spearman | affine (slope, intercept) |
|---|---|---|---|---|---|---|
| `HDR_VDP_3` | jpeg-ai-qaf default (PQ, ppd 64.06, Q_JOD) | 0.622 / 1.72 | 0.616 / 1.74 | 0.8900 | 0.8315 | 1.235, −2.77 |
| `HDR_VDP_3` | sRGB 200 cd/m², black 0.2, refl 0.398, ppd 37.84, Q_JOD | 0.728 / 0.952 | 0.717 / 1.03 | 0.9962 | 0.9990 | 1.410, −4.32 |
| `HDR_VDP_3` | same, `res.Q` | 0.0196 / 0.133 | 0.0202 / 0.141 | 0.99961 | 0.9990 | 1.014, −0.14 |
| `HDR_VDP_3` | sRGB 200, black 0, refl 0, ppd 37.84, rgb-bt.709/led-lcd-srgb, `res.Q` | 0.00226 / 0.0242 | 0.00213 / 0.0258 | 0.99998 | 0.9997 | 1.0006, −0.007 |
| `HDR_VDP_3` | **same, XYZ input (oled)** | **0.00140 / 0.0125** | **0.00136 / 0.0129** | **0.99999** | **1.0000** | **0.9997, 0.0023** |
| `HDR_VDP_2` | jpeg-ai-qaf VDP2 port default (Q_MOS) | 31.5 / 44.3 | 2.04 / 3.83 | 0.7742 | 0.8125 | 0.711, −4.78 |
| `HDR_VDP_2` | 2.2.2 `res.Q`, sRGB 200, black 0.2, refl 0.398, ppd 37.84 | 0.284 / 1.22 | 0.0333 / 0.326 | 0.99873 | 0.9962 | 0.999, −0.31 |
| `HDR_VDP_2` | same, black 0, refl 0 | 0.0291 / 0.222 | 0.00498 / 0.0498 | 0.99997 | 0.9998 | 0.997, 0.19 |
| `HDR_VDP_2` | **same, codes not rounded to 10 bits** | **0.0278 / 0.230** | **0.00365 / 0.0515** | **0.99997** | **0.9998** | **0.997, 0.18** |

Exact ppd: the paper's 37.84 is CVVDP's centre ppd for `standard_fhd`,
37.842505866…; a rerun at that exact value is queued on i265 and not yet measured.

Searched and worse (all in the TSVs): EOTF PQ / sRGB / gamma 2.2 / raw codes; peak
100–10000 cd/m²; black 0–0.5; reflection on/off and ρ = 0.01; ppd 15–128 (the qaf port
crashes in `fast_gauss` below ppd ≈ 20; the Rust port does not); 8/10-bit/no rounding;
tasks quality / side-by-side / flicker; ref/test swap; observer age 24–60; surround
none/mean; emission ccfl-lcd / crt / led-lcd / led-lcd-srgb / oled; age-related loss and
masking off; v2 single-channel luminance input. HDR-VDP 3.0.6 → 3.0.7 differs only
structurally (Octave/GPU support, file paths, a padding helper); 3.0.6 was diffed, not run.

What the ports would need to expose (no API change made here):
- `hdrvdp3`: an SDR display-model ingress `L = peak·sRGB_EOTF(V)` (peak 200, no black,
  no reflection) converted to XYZ; `--hdrvdp3-input xyz` (emission defaults to `oled`),
  `--hdrvdp3-ppd 37.84`, task `quality`; and a `res.Q` output column next to Q_JOD.
- `hdrvdp`: the same ingress in BT.709, ppd 37.84, the existing `q` (`res.Q`) output.

Remaining `HDR_VDP_2` residual (median 0.028 Q) is ~35× our port's validated distance
to MATLAB 2.2.2 (`res.Q` 7.8e-4), so some small setting still differs.

## DMM: pyiqa's square resize, on CUDA, in fp32

`dmm_configs.tsv`. pyiqa 0.1.16 `dmm` resizes each image to a **square**
`tar × tar` with `F.interpolate(size=tar, mode='bilinear', align_corners=False)`,
`tar = max(int(min(h,w)/48)·32, 128)` (840×944 → 544×544); upstream `DMM_PyTorch`
resizes the short side to `tar` keeping aspect (544×611).

| variant | median / max abs | median / max site-JND | Spearman |
|---|---|---|---|
| **pyiqa square, CUDA fp32** | **6.3e-5 / 1.36e-3** | **2.6e-4 / 2.4e-3** | **0.9999** |
| pyiqa square, CUDA fp64 | 1.5e-3 / 6.9e-3 | 4.1e-3 / 1.1e-2 | 1.0000 |
| pyiqa square, CPU fp32 (49 of 61 pairs*) | 7.1e-3 / 0.114 | 2.2e-2 / 0.215 | 0.9971 |
| same 49 pairs, CUDA fp32 | 6.6e-5 / 1.1e-3 | 2.8e-4 / 2.4e-3 | 0.9999 |
| pyiqa, aspect-preserving tensor resize, CUDA | 0.045 / 0.195 | 0.10 / 0.30 | 0.998 |
| pyiqa, PIL square / antialiased square, CUDA | 0.13 / 0.63 | 0.37 / 1.18 | 0.977 |
| upstream `DMM_PyTorch` (PIL aspect), CUDA | 0.084 / 0.43 | 0.24 / 1.07 | 0.978 |

\* stopped at 49 pairs when dev was reclaimed; the 49-pair CUDA row is the paired control.

The CPU→CUDA step is the SVD: DMM's `wt` term depends on the basis the solver picks in
degenerate singular-value clusters (see `../dmm_aic_probe/`), and torch's batched CUDA
SVD picks differently from LAPACK. GPU: RTX 5070, torch 2.11.0+cu128. The remaining
max 2.4e-3 site-JND may be GPU/driver-specific. Our `dmm` port (CPU SVD, upstream aspect
resize) cannot reproduce this column bit-for-bit; a pyiqa-convention port would need the
square resize and would still differ on degenerate patches unless it reproduces cuSOLVER's
basis choice.

## WD_s1–WD_s5: not reproduced

The paper names "Wasserstein Distortion (VGG-16, σ = 2⁴)", so `WD_sk` is log2σ = k. Our
`wd` follows Cool-Chic's `wasserstein.py`, which credits google/codex
`codex/loss/wasserstein.py` (Ballé et al.). Three differences between them, plus the
CVPR'25 paper's description, define the grid `scripts/wd_probe.py` searched (VGG16
features computed once per input variant; σ weighting applied to channel-mean level maps,
which equals Cool-Chic's mean over (C,H,W); the probe reproduces shipped `wd` exactly,
S01 s1 0.0506557 vs 0.050655):

- input raw [0,1] / ImageNet-normalised; distance per location squared (both codes) or √
  of it (paper); variance floor 5e-7 (Cool-Chic) / 0 (codex); σ constant (Cool-Chic) /
  codex `relu(log2σ − mean log2(image/feature size))`; taps 3,8,15,22 / +29; the pixels
  as a 0th feature layer (paper) on/off; image scales 1 / 1+½+¼ (paper's "two downsampled
  versions", 2×2 average pooling); log2σ 0–6 matched against every column. 4,480 rows.
- Follow-up (52 pairs, ImageNet/squared/floor 0 only): 7 pyramid levels instead of 5.
  The antialiased-bilinear downscale variant was queued but not run (dev reclaimed).

| column | shipped Cool-Chic (median site-JND / Spearman) | best median site-JND | best Spearman |
|---|---|---|---|
| WD_s1 | 1.08 / 0.944 | 0.138 (imnet, rescale, 4 taps, log2σ 4) | 0.983 |
| WD_s2 | 1.24 / 0.923 | 0.127 (imnet, rescale, log2σ 5) | 0.984 |
| WD_s3 | 1.21 / 0.894 | 0.149 (imnet, const, log2σ 4) | 0.982 |
| WD_s4 | 1.05 / 0.859 | 0.169 (raw, rescale, 3 scales, log2σ 4) | 0.986 |
| WD_s5 | 0.91 / 0.858 | 0.195 (raw, rescale, 3 scales, log2σ 5) | 0.981 |

The best-ranking family (ImageNet input, codex σ rescale, three image scales) reaches
Spearman 0.98–0.99 with published/ours ≈ 0.25 on every column, but its best log2σ does
not follow k (s1→1, s2→3, s3→4, s4→5, s5→6). Top rows: `wd_conventions_top.tsv`; full
summaries (`wd_summary.tsv` sha256 `42f964d1…22fe350`, `wd_summary_n7avg.tsv`
`df0c5661…6622e00`) at `/mnt/v/output/zenmetrics/aic_confsearch_2026-10-17/` on dev.

## A-FINE: not reproduced

`afine_configs.tsv`. Upstream QuickInference modules + `afine.pth` (sha256 `0fd31f02…`,
identical to pyiqa's mirror and to the only file in upstream's model folder, 582.5 MiB,
2025-02-28). Upstream history has one inference convention (pad32 by bicubic resize, CLIP
normalisation); pyiqa 0.1.15/0.1.16 `afine_arch` differ only in docstrings and asserts
H, W divisible by 32 — which 840×944 is not, so a pyiqa user had to crop/resize/pad first.

Variants (each scored for fidelity `F`, naturalness, `afine_all`, `F + N_d − N_r`,
scaled): native, ref/dis swap, BGR, no normalisation, ImageNet normalisation, ×255 input,
in-place normalisation applied twice/three times (pyiqa normalises the caller's tensor in
place), to only one image; centre/top-left crop to 832×928; reflect/replicate/zero
padding to 864×960 (zero before and after normalisation); bicubic/PIL resize to 832×928;
bicubic up to 864×960; half-resolution; 224/448 tiles (stride 224/112/224); CUDA fp32;
CUDA autocast fp16.

| best rows | median / max site-JND | Spearman |
|---|---|---|
| resize 832×928 bicubic, `F + N_d − N_r` | 0.109 / 0.78 | 0.879 |
| centre crop 832×928, `F` | 0.137 / 1.05 | 0.938 |
| native, `F` | 0.135 / 0.88 | 0.906 |
| native, `afine_all` | 1.05 / 1.95 | 0.819 |

CUDA and autocast change nothing material (native `F` 0.13548 → 0.13544). With the
fidelity map's trained constants, the published values on near-lossless pairs (≈ −1.94)
imply a raw fidelity ≈ 0.097 where every variant gives ≈ 0.015, so the column comes from a
different input or model state than any configuration tried.

## Questions for the dataset authors

1. **A-FINE**: Which entry point and output were used (upstream `afine.py`, pyiqa
   `afine_all` / `afine_fr`, other)? pyiqa asserts H, W divisible by 32: how were the
   840×944 crops (and full-resolution images) brought to that — crop, pad, resize? Which
   checkpoint (sha256)? GPU, dtype?
2. **WD**: Which implementation (codex JAX, Cool-Chic PyTorch, own)? Feature extractor
   and taps, input normalisation, whether pixels and downsampled copies are included,
   number of levels, whether σ is rescaled per feature map, and how `WD_s1`–`WD_s5` map to
   σ.
3. **HDR_VDP_2 / HDR_VDP_3** (to confirm, not to find): display model 200 cd/m² sRGB with
   no black level or reflection, ppd 37.84; HDR-VDP-3 input encoding (`xyz`?) and that the
   column is `res.Q` rather than `Q_JOD`; HDR-VDP-2 release (2.2.2?) and any option that
   would explain the remaining 0.028 Q.
4. **DMM** (to confirm): pyiqa `dmm` on CUDA; GPU model and torch version.
5. **proposal-Butteraugli** (provenance only): where do the constants of
   `max(0.00459792·N, 2.0938201·N − 1.1033102)` come from?
6. **CAMBI** (to confirm): PNG → YUV via ffmpeg `-pix_fmt yuv420p`, and which ffmpeg version.

## Reproduce

Data layout: `~/tmp/aic-confsearch/{pairs_dev.tsv,data/crop/metrics_subset.csv,data/config.json}`;
oracles under `~/tmp/aic-confsearch/ext/` (jpeg-ai-qaf `0628a6b` `git archive`, upstream
A-FINE `1b49be7` QuickInference + checkpoints, `Baoliang93/DMM` `DMM_PyTorch`). venvs:
`venv38` (Python 3.8, numpy 1.24.4, scipy 1.10.1 — the qaf port needs numpy < 1.25),
`venv` (torch 2.14.1 CPU), `venv12` (torch 2.11.0+cu128, pyiqa 0.1.16).

- HDR-VDP (qaf port): `scripts/run_cfgs.sh configs/cfg_*.txt` → `scripts/evalcol.py`;
  Q_JOD → Q: `scripts/jod2q.py`. Our ports: `vdpprobe/build.sh <zenmetrics> <scratch>`
  (verified to reproduce the probe output byte for byte), `scripts/run_rs.sh configs/cfg_rs_*.txt`.
- DMM: `scripts/dmm_probe.py ext/dmm_up pairs.tsv out.tsv pyiqa_tsquare_cuda,...`.
- A-FINE: `scripts/run_afine.sh <variant>...` (`DEV=cuda AMP=1` for GPU), `scripts/afine_eval.sh`.
- WD: `scripts/wd_probe.py pairs.tsv out.tsv` (env `DISTS FLOORS NLEV DOWN TAG`),
  `scripts/wd_eval.py`.
- proposal-Butteraugli: `zenmetrics batch --metric butteraugli` (`scripts/i265_butter.sh`), the map
  in the README, `scripts/butter_map_fit.py <site CSVs>` for the map itself.
- CAMBI: `scripts/cambi_probe.py <libvmaf 3.2.1 vmaf> pairs.tsv out.tsv <scratch> ff_420,...`
  (`scripts/i265_cambi.sh`); `scripts/sws_luma_check.py pairs.tsv` for the luma identity.
- `configs/cfg_g1.txt` was stopped part-way and not evaluated (superseded by the Rust grids).

Hosts: compute ran on dev (Ryzen 9 9950X3D, RTX 5070) under `run-heavy`; after the owner
reclaimed dev, the proposal-Butteraugli and CAMBI runs ran niced on i265 (41.6 s wall,
159 MiB max RSS, `/usr/bin/time -v`), and the exact-ppd check is queued there. dev was reclaimed by the owner before the WD bilinear variant and
the last 12 CPU DMM pairs ran. Wrapper receipts (peak RSS): HDR-VDP qaf grids ≤ 0.53 GiB;
Rust probe ≤ 1.47 GiB; A-FINE CPU batch 1.71 GiB; WD probe 5.49 GiB; DMM CUDA 2.26 GiB.
