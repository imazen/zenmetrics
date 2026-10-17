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
| `A-FINE` | `afine` (upstream-verified `afine_all`) | column sits 0.5–1.4 above ours (median 0.75) on 61 pairs; a structured search over sub-scores, resizes, normalisation and λ found no match. No pyiqa release scores these 840×944 crops as given (0.1.15 and 0.1.16 assert H and W divisible by 32). Running pyiqa's model on 12 resize/crop × output variants, the fidelity term comes closest (median \|Δ\| 0.09, Spearman 0.94) and the full score stays 0.71–0.73 off, so the column is not pyiqa's A-FINE at these inputs (2026-10-17) | which A-FINE entry point and output (full `afine_all`, fidelity or naturalness term), checkpoint, input resize and normalisation | [afine_aic_2026-10-16](../benchmarks/afine_aic_2026-10-16/README.md), [afine_aic_probe](../benchmarks/afine_aic_probe/README.md) |
| `DMM` | `dmm` | median \|Δ\| 0.075, max 0.417 under the upstream convention; pyiqa 0.1.16 defaults come closer (median 9.1e-3, max 0.114, Spearman 0.9975) but not exact | which DMM implementation and version, SVD library and precision | [dmm_aic_probe](../benchmarks/dmm_aic_probe/README.md) |
| `WD_s1`–`WD_s5` | `wd` (`wd_s1`–`wd_s5`) | best of a bounded convention search still errs by median 0.23–0.37 site JND per column | per-column scale (σ) mapping, VGG feature layers, normalisation, resolution handling | [wd_port_2026-10-16](../benchmarks/wd_port_2026-10-16/README.md) |
| `HDR_VDP_3` | `hdrvdp3` (bit-faithful to MATLAB 3.0.7) | column not reproduced; a different run configuration is likely | task, display model (peak luminance, contrast, pixels per degree), colour encoding and EOTF used | [AIC2026_METRICS_AND_FITTING.md](AIC2026_METRICS_AND_FITTING.md), DIVERGENCES.md |
| `HDR_VDP_2` | `hdrvdp` | needs the absolute-luminance display configuration | HDR-VDP-2 version and display parameters | same |
| `proposal-Butteraugli` | `butteraugli` (3-norm column) | tracks our libjxl 3-norm with Pearson 0.99997 after an affine map (pub ≈ 2.09·ours − 1.10); exact values not recovered | libjxl revision, intensity target, aggregation and any JND mapping applied | AIC2026_METRICS_AND_FITTING.md |
| `CAMBI` | `cambi` | native median 1.3e-4, max 0.178; settings unresolved | CAMBI options (window, top-k, thresholds, EOTF, bit depth, any scaling or crop) | [aic_more_2026-10-15](../benchmarks/aic_more_2026-10-15/README.md) |
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
