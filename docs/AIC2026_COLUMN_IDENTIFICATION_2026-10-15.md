# AIC2026: identifying the less-documented metric columns (2026-10-15)

`metrics_fullres.csv` in the AIC2026 dataset ([DOI 10.18419/DARUS-6156](https://doi.org/10.18419/DARUS-6156))
has no column dictionary, and the paper ([arXiv:2607.22783](https://arxiv.org/abs/2607.22783), Table VI)
names only some columns. This records what each remaining column is, how each claim was
established, and what it means for porting.

Labels: **stated** (a cited source says so), **measured** (recomputed here and matched to the
CSV, or computed from the CSV), **inferred** (reasoning, not confirmed).

Sources beyond the paper: the project site's metric configuration,
[`/aic2026/api/metric-eval/config`](https://merkur242.inf.uni-konstanz.de/aic2026/api/metric-eval/config),
gives a JND mapping `{a, b, c, sign}` per column in the CTC form `a·max(0, b − x)^c`
([WG1N101246](https://ds.jpeg.org/documents/jpegaic/wg1n101246-108-ICQ-Common_Test_Conditions_on_Objective_Quality_Assessment_v2_0.pdf), §4).
Its `sign` is read here as the authors' score direction (−1 with `b = 1`: higher is better;
+1 with `b = 0`: lower is better). Its excluded-experiment names (`SSIM-pyiqa-f32-k2…k6`,
`SSIMc-pyiqa-f32-k*`) show that the `SSIM*` columns came from pyiqa. The paper's own `SSIM`
and `MS-SSIM` rows are the libvmaf versions.

| column | what it is | basis | direction | porting route |
|---|---|---|---|---|
| `WD_s1` … `WD_s5` | Wasserstein Distortion (Qiu, Wagner, Ballé, Theis, CISS 2024, [arXiv:2310.03629](https://arxiv.org/abs/2310.03629)) on VGG-16 features, σ = 2^k | `WD_s4` = the paper's "VGG-16, σ = 2^4" row: its SRCCs against 30 metrics reproduce Table VI to rounding (mean 0.0024, max 0.005; other scales 0.024–0.060). σ = 2^k is inferred from the reference code's `log2_sigma` and `num_levels = 5` | lower is better | permissive: [google/codex `wasserstein.py`](https://github.com/google/codex/blob/main/codex/loss/wasserstein.py) (Apache-2.0, JAX, formula only) and [Cool-Chic `wasserstein.py`](https://github.com/Orange-OpenSource/Cool-Chic/blob/main/coolchic/training/metrics/wasserstein.py) (BSD-3-Clause, PyTorch, torchvision VGG16 features before each max-pool). Which one AIC used is not stated |
| `SSIMc` | SSIM per RGB channel (0–255, rounded), 11×11 Gaussian σ = 1.5, valid region, no downsampling, contrast-structure term clipped at 0, mean over channels and pixels | measured: 61 pairs, median \|Δ\| 1.1e-7, max 4.4e-7 | higher | written from Wang et al. 2004 with those conventions (aic-classic lane) |
| `SSIM-pyiqa` | the same SSIM on luma Y = 0.299R + 0.587G + 0.114B, rounded, no downsampling | measured on 3 pairs to 1e-6 (BT.709 luma does not match) | higher | as SSIMc on that luma |
| `MS-SSIM-pyiqa` | MS-SSIM (Wang, Simoncelli, Bovik 2003) on the same luma: 5 scales, weights [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], cs clipped at 0, 2×2 average pooling between scales (zero-padded when odd), product combination | measured on 3 pairs to 2e-7 | higher | from the paper with those conventions |
| `CW-SSIM` | complex-wavelet SSIM (Wang & Simoncelli, ICASSP 2005; Sampat et al., TIP 2009) on the same luma, complex steerable pyramid, 4 levels × 8 orientations, 7×7 window, scored at one coarse level | resolved 2026-10-20: pyiqa 0.1.16 `cw_ssim` convention; `cw-ssim-pyiqa` reproduces the full 9,618 rows to max |Δ| 5e-7 | higher | `cw-ssim-pyiqa` (`zenmetrics-pyiqa-ports`) |
| `MSSWD` | learned MS-SWD (He et al., ECCV 2024, [arXiv:2407.10181](https://arxiv.org/abs/2407.10181)); images resized to a 256 px short side | measured privately: median 4.1e-6, max 2.4e-4, Spearman 0.9999 | lower | private port, aggregates only; the official repo ([real-hjq/MS-SWD](https://github.com/real-hjq/MS-SWD)) has no licence file and the learned weights are distributed by pyiqa (noncommercial) |
| `AIMOS` | excluded (proprietary, unreleased) | Owner clarification, 2026-10-19: proprietary Cloudinary metric, never publicly described or released | outside scope | No implementation, reproduction or author-question task |
| `proposal-IDFIQA` | one of the four AIC-4 proposals (inferred from the `proposal-` prefix and the [109th meeting report](https://records.sigmm.org/?p=14852)); the AIC-4 notes in [`AIC2026_METRICS_AND_FITTING.md`](AIC2026_METRICS_AND_FITTING.md) attribute it to Khakpour, Yademellat & Mansouri (ICCKE 2025) | private port: median 4.2e-6, max 3.33e-4 JND, Spearman 1.0 | lower (config) | private aggregates only; request upstream code licence |

Measured CSV ranges (9,618 rows): `WD_s1` 0.048–9.55 (median 1.43), `WD_s2` 0.034–8.10 (1.17),
`WD_s3` 0.018–7.13 (0.81), `WD_s4` 0.0076–5.81 (0.52), `WD_s5` 0.0035–4.28 (0.33); `SSIM-pyiqa`
0.576–0.9988 (median 0.932, against 0.997 for libvmaf `SSIM`, which downsamples first);
`SSIMc` 0.550–0.9984; `MS-SSIM-pyiqa` 0.909–0.9997; `CW-SSIM` 0.647–0.99999 (median 0.9937);
`MSSWD` 0.0045–1.44 (median 0.044); `proposal-IDFIQA` 0.0014–4.69 (median 0.44).

AIMOS is excluded from implementation and reproduction scope. The other identified columns here are full-reference.

Licences read from LICENSE files on 2026-10-15: google/codex Apache-2.0; Cool-Chic
BSD-3-Clause; real-hjq/MS-SWD none. The SSIM-family conventions above are facts about the
published numbers; implementations here are written from the original papers, not from pyiqa
(PolyForm Noncommercial).

Open: ask the dataset contacts (README) which WD code computed `WD_s*`.
Nobody has been contacted.
