# AIC2026 published columns: what we reproduce, and what to ask the dataset authors

Status of every metric column published with the JPEG AIC2026 dataset
(`metrics_fullres.csv` / `metrics_cropped.csv`, 9,618 rows) against the
zenmetrics implementation of the same method, as of 2026-10-17. "Reproduced"
means our implementation matches the published per-image values within the
stated error on the 61-pair subset (or more); a mismatch is a question about
the published pipeline, not a claim that either side is wrong. Evidence links
point at the committed receipts; the column-to-implementation map is
`scripts/complexity/aic2026_columns.json`, and the wider ledger is
[AIC2026_METRICS_AND_FITTING.md](AIC2026_METRICS_AND_FITTING.md).

Timing for each implementation: [IQA_COMPLEXITY.md](IQA_COMPLEXITY.md)
(controlled runs on one host) and the draft paper
[docs/paper/IQA_COMPLEXITY_VS_RUNTIME.md](paper/IQA_COMPLEXITY_VS_RUNTIME.md).

## Implemented but not reproduced: questions for the dataset authors

| Column | Ours | Measured gap | What to ask | Evidence |
|---|---|---|---|---|
| `A-FINE` | `afine` (upstream-verified `afine_all`) | not reproduced. No pyiqa release scores these 840×944 crops as given (0.1.15 and 0.1.16 assert H and W divisible by 32). Upstream and pyiqa inference across crop/pad/resize to a multiple of 32, tiling, every output term, input scaling and normalisation variants, CPU and CUDA: best Spearman 0.94 (fidelity term), best median 0.11 site-JND. On near-lossless pairs the column implies a raw fidelity ≈ 0.097 where every variant gives ≈ 0.015 (2026-10-17) | which A-FINE entry point and output, how inputs were brought to a multiple of 32, which checkpoint | [aic_confsearch_2026-10-17](../benchmarks/aic_confsearch_2026-10-17/README.md), [afine_aic_2026-10-16](../benchmarks/afine_aic_2026-10-16/README.md), [afine_aic_probe](../benchmarks/afine_aic_probe/README.md) |
| `WD_s1`–`WD_s5` | `wd` (`wd_s1`–`wd_s5`, Cool-Chic convention) | not reproduced. A 4,480-row grid over the Cool-Chic and google/codex conventions (input normalisation, squared or √ distance, variance floor, σ rescale, VGG taps, pixels as a layer, image pyramid, log2σ 0–6) reaches Spearman 0.98–0.99 with a near-constant scale (published/ours ≈ 0.25), but the best log2σ does not follow the column index (s1→1, s2→3, s3→4, s4→5, s5→6); best median 0.13–0.20 site-JND (2026-10-17) | which implementation, layers, normalisation and pyramid, and how the five columns map to σ | [aic_confsearch_2026-10-17](../benchmarks/aic_confsearch_2026-10-17/README.md), [wd_port_2026-10-16](../benchmarks/wd_port_2026-10-16/README.md) |

## Identified by configuration search (ports need the option exposed)

A bounded search over each canonical implementation's options (2026-10-17, [aic_confsearch_2026-10-17](../benchmarks/aic_confsearch_2026-10-17/README.md)) reproduces
these columns. site-JND is the AIC site's per-metric map; the match gate is 1e-3 site-JND.

| Column | Configuration that reproduces it | median / max site-JND | Spearman | What the zenmetrics port needs |
|---|---|---|---|---|
| `proposal-Butteraugli` | `max(0.00459792·N, 2.0938201·N − 1.1033102)` with N = libjxl `butteraugli` 3-norm at defaults; the same map holds on all 9,618 rows of both CSVs | 3.2e-8 / 5.1e-5 | 1.0000 | a column applying this map to `butteraugli_pnorm3`; the constants' provenance is unknown |
| `CAMBI` | libvmaf 3.2.1 CAMBI defaults on luma from ffmpeg/swscale `-pix_fmt yuv420p` (an exact BT.601 formula differs by one code on 0.45 % of pixels, which CAMBI amplifies) | 8.2e-8 / 2.1e-3 (max raw 4.8e-7, the 6-decimal rounding) | 1.0000 | the swscale luma ingress; `cambi_from_luma` already equals libvmaf |
| `DMM` | pyiqa 0.1.16 `dmm` on CUDA, fp32 (square `tar×tar` bilinear resize; torch's CUDA SVD picks a different basis than LAPACK in degenerate clusters) | 2.6e-4 / 2.4e-3 | 0.9999 | pyiqa's square resize, and cuSOLVER's basis choice for an exact match; `dmm` (upstream aspect resize, CPU SVD) differs by median 0.075 |
| `HDR_VDP_3` | HDR-VDP 3.0.7 `res.Q` (not `Q_JOD`), XYZ input (emission `oled`), display `L = 200·sRGB_EOTF(V)` cd/m² with no black level or reflection, ppd 37.84 (CVVDP `standard_fhd`), task `quality` | 1.36e-3 / 1.29e-2 | 1.0000 | that SDR display ingress, XYZ input, ppd option and a `res.Q` output in `hdrvdp3` |
| `HDR_VDP_2` (near) | HDR-VDP 2.2.2 `res.Q`, same display and ppd, BT.709 input, codes not rounded to 10 bits | 3.65e-3 / 5.15e-2 (median 0.028 Q) | 0.9998 | the same ingress in `hdrvdp`; the remaining 0.028 Q is ~35× our port's validated distance to MATLAB, so a small setting still differs |
| `proposal-DVIFM`, `proposal-DVIFM-0.2`, `proposal-DVIFM-0.2-use_chroma` | `dvifmish` port mode (unpublished; ports of the author's four released C++ variants) | no released variant reproduces any column: best Spearman 0.941 (`port-lap-v0.3` vs `proposal-DVIFM-0.2`), 0.751 vs `proposal-DVIFM`, 0.865 vs `-use_chroma` (61 crop pairs, 2026-10-17); the releases were retuned on AIC2026 after the columns were computed. We use the current releases (dvifmish port mode) and do not try to reproduce these older columns | which DVIFM build, constants and JND map produced each column | [aic2026_repro_2026-10-17](../benchmarks/aic2026_repro_2026-10-17/README.md) |

## Identified: the column matches pyiqa defaults

Scoring the 61 crop pairs with pyiqa 0.1.16 (`create_metric(name)` defaults, CPU fp32,
2026-10-17) reproduces these columns. This identifies the published pipeline; the
zenmetrics implementations below follow the upstream releases, so where they differ
from the column the difference is the convention, not a defect. Ports that follow the
column's convention are being added in `crates/zenmetrics-pyiqa-ports`, the one crate under
pyiqa's non-commercial terms (TOPIQ, TOPIQ-PIPAL, WaDIQaM, AHIQ, CW-SSIM, DMM). MSSWD has no
licence and is ported outside this repository.

| Column | pyiqa metric | median \|Δ\| | max \|Δ\| | Spearman | zenmetrics |
|---|---|---|---|---|---|
| `TOPIQ` | `topiq_fr` (KADID) | 3.6e-5 | 2.7e-4 | 1.0000 | port in progress in `zenmetrics-pyiqa-ports` (non-commercial crate) |
| `TOPIQ-PIPAL` | `topiq_fr-pipal` | 2.4e-5 | 1.5e-4 | 1.0000 | port in progress in `zenmetrics-pyiqa-ports` (non-commercial crate) |
| `MSSWD` | `msswd` (learned) | 4.2e-6 | 2.4e-4 | 0.9999 | not implemented (no licence); the official random-projection MS-SWD is a different method |
| `WaDIQaM` | `wadiqam_fr` (KADID-trained) | 1.7e-6 | 1.2e-5 | 1.0000 | `wadiqam-*` use the official deepIQA LIVE/TID weights, a different model (best Spearman −0.74) |
| `AHIQ` | `ahiq` | 2.5e-6 | 1.0e-5 | 1.0000 | `ahiq` follows upstream `test.py` nine-crop (median \|Δ\| 0.14) |
| `PieAPP` | `pieapp` | 2.3e-5 | 1.9e-4 | 0.9999 | `pieapp` (upstream convention): median 6.6e-3, max 0.042, Spearman 0.9975 |
| `CKDN` | `ckdn` | 4.0e-5 | 1.4e-4 | 0.9999 | `ckdn` matches to the same residual |
| `CW-SSIM` | `cw_ssim` | 0 | 2.2e-16 | 1.0000 | `cw-ssim` max 1.1e-4 |

Controls scored the same way also match their columns (median \|Δ\| ≤ 3.4e-6, Spearman 1.0000):
`DISTS`, `DeepDC`, `LPIPS-alex`, `LPIPS-VGG`, `STLPIPS`, `STLPIPS-VGG`, `SSIM-pyiqa`,
`MS-SSIM-pyiqa`, `NLPD`, `VSI`, `GMSD`, `VIF`, `PSNR`, `SSIMc`. pyiqa's `mad` is not
`AIMOS` (Spearman −0.83).

## Close but not exact (minor; ask only if convenient)

| Column | Ours | Residual | Open detail |
|---|---|---|---|
| `SSIMULACRA1` | `ssimulacra1` | native max 1.9e-5 | exact published convention |
| `WS-HV` | `ws-hv` | native max 0.021 | exact published convention |
| `FSIM`, `FSIMc` | `fsim-y`, `fsim` | one JPEG tail pair at 0.3 JND | downscale factor used |
| `CKDN` | `ckdn` | median 4.0e-5, max 1.4e-4 (0.026 JND) | resize implementation (pyiqa defaults give the same residual). The published column itself shows no agreement with the AIC2026 subjective JND (within-source Spearman −0.005); our port reproduces it, so that is a property of the method on this data, not of the port |
| `NLPD` | `nlpd-iqa` | median 2.6e-3, 2.6 % relative (f32 order and bilinear edges) | none needed |
| `PSNR-Cb`, `PSNR-Cr` | `psnr-cb`, `psnr-cr` | max 1.7e-2 / 5.7e-2 dB | chroma conversion details |
| `PSNR-YCbCr611` | `psnr-ycbcr611` | max 3.5e-3 dB | none needed |

## Not yet compared against the published column

None: every implemented column has been compared (2026-10-17).

## Not implemented: what we would need

| Column | Reason | What to ask |
|---|---|---|
| `AIMOS` | not identified in any public source | what AIMOS is, and its reference |
| `proposal-IDFIQA` | reference code has no licence. The authors' AIC-4 evaluation script reproduces the column (61 crop pairs: median \|Δ\| 4.7e-6, max 3.3e-4 JND, Spearman 1.0; 2026-10-17); a port is being written outside this repository | a licence for the reference code |
| `HaarXYB-*`, `HaarY-*`, `HaarYCoCg-* deltaL1248` (site-only names) | unidentified; no per-image values in any CSV and listed as excluded in the site config, yet the authors' rank-distribution figure (792 five-source subsets) shows six of them taking the top six ranks. Per the 2026-10-09 AHG discussion these are experimental metrics tuned on this data, to be ignored | what they are (colour space, masking, "deltaL1248"), and the code or per-image values |
| FSIM, FSIMc and VSI at `-k2`/`-k3`/`-k4` and `-nodownscale`; `SSIM-pyiqa-f32` and `SSIMc-pyiqa-f32` at `-k2` to `-k6` (site-only names) | downscale-factor variants with fitted JND maps in the site config but no per-image values in any CSV; FSIMc-k4 ranks ninth in the authors' rank-distribution figure, above FSIMc | the per-image values, and which factor the published FSIM, FSIMc and VSI columns used |
| `PSNR-HVS-vqmt`, `PSNR-HVS-M-vqmt` (site-only names) | MSU VQMT implementation; the site config has a fitted JND map for both, so values exist, but none are published | the per-image values |

## Reproduced

`PSNR`, `PSNR-Y` (studio-601), `SSIM` (libvmaf), `MS-SSIM` (libvmaf),
`SSIM-pyiqa`, `MS-SSIM-pyiqa`, `SSIMc`, `PSNR-HVS` family (Daala),
`VIF` (vifvec), `VIF_vmaf`, `VMAF`, `VMAF-neg`, `ADM2`, `CVVDP` (with the
`standard_fhd` display), `SSIMULACRA2`, `GMSD`, `HaarPSI`, `VSI`, `DSSIM`,
`IW-SSIM` (piq convention), `mDCT-PSNR`, `proposal-mDCTPSNR`, `FLIP`
(`flip-v17` exact), `CIEDE2000`, `Butteraugli-original`, `LPIPS-alex`,
`LPIPS-VGG`, `DISTS`, `STLPIPS`, `STLPIPS-VGG`, `DeepDC`, `DreamSim`.
