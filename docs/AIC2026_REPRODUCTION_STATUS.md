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
| `A-FINE` | `afine` (upstream-verified `afine_all`) | column sits 0.5–1.4 above ours (median 0.75) on 61 pairs; a structured search over sub-scores, resizes, normalisation and λ found no match | which A-FINE entry point and output (full `afine_all`, fidelity or naturalness term), checkpoint, input resize and normalisation | [afine_aic_2026-10-16](../benchmarks/afine_aic_2026-10-16/README.md), [afine_aic_probe](../benchmarks/afine_aic_probe/README.md) |
| `AHIQ` | `ahiq` (upstream nine-crop) | median \|Δ\| 0.14 (≈5.6 JND), max 0.21, SRCC ≈0.76 under upstream's own `test.py` path | crop protocol (count, positions, seeds), checkpoint epoch, timm backbone versions | [ahiq_aic_probe](../benchmarks/ahiq_aic_probe/README.md) |
| `DMM` | `dmm` | median \|Δ\| 0.075, max 0.417 under the upstream convention | preprocessing (resize/crop), SVD library and precision, checkpoint | [dmm_aic_probe](../benchmarks/dmm_aic_probe/README.md) |
| `WD_s1`–`WD_s5` | `wd` (`wd_s1`–`wd_s5`) | best of a bounded convention search still errs by median 0.23–0.37 site JND per column | per-column scale (σ) mapping, VGG feature layers, normalisation, resolution handling | [wd_port_2026-10-16](../benchmarks/wd_port_2026-10-16/README.md) |
| `HDR_VDP_3` | `hdrvdp3` (bit-faithful to MATLAB 3.0.7) | column not reproduced; a different run configuration is likely | task, display model (peak luminance, contrast, pixels per degree), colour encoding and EOTF used | [AIC2026_METRICS_AND_FITTING.md](AIC2026_METRICS_AND_FITTING.md), DIVERGENCES.md |
| `HDR_VDP_2` | `hdrvdp` | needs the absolute-luminance display configuration | HDR-VDP-2 version and display parameters | same |
| `proposal-Butteraugli` | `butteraugli` (3-norm column) | tracks our libjxl 3-norm with Pearson 0.99997 after an affine map (pub ≈ 2.09·ours − 1.10); exact values not recovered | libjxl revision, intensity target, aggregation and any JND mapping applied | AIC2026_METRICS_AND_FITTING.md |
| `CAMBI` | `cambi` | native median 1.3e-4, max 0.178; settings unresolved | CAMBI options (window, top-k, thresholds, EOTF, bit depth, any scaling or crop) | [aic_more_2026-10-15](../benchmarks/aic_more_2026-10-15/README.md) |
| `CW-SSIM` | `cw-ssim` | median 2.9e-7, max 1.1e-4, but no pair matches exactly (20 of 61 off by more than 1e-6); variant chosen in-sample | implementation (pyrtools version), levels, orientations, K, window | AIC2026_METRICS_AND_FITTING.md |
| `WaDIQaM` | `wadiqam-fr-live`, `-fr-tid`, `-nr-live`, `-nr-tid` (official deepIQA weights) | none of the four checkpoints reproduces the column: published values lie in −0.039…0.018, ours on the checkpoints' own DMOS scales; best rank agreement is FR-LIVE at Spearman −0.74 (sign expected, ours is higher-is-worse); NR variants and NR dist−ref differences are weaker (61 crop pairs, 2026-10-17) | which WaDIQaM model and implementation produced the column (a retrained model such as pyiqa's?), its training set and output normalisation | [aic2026_repro_2026-10-17](../benchmarks/aic2026_repro_2026-10-17/README.md) |
| `proposal-DVIFM`, `proposal-DVIFM-0.2`, `proposal-DVIFM-0.2-use_chroma` | `dvifmish` port mode (unpublished; ports of the author's four released C++ variants) | no released variant reproduces any column: best Spearman 0.941 (`port-lap-v0.3` vs `proposal-DVIFM-0.2`), 0.751 vs `proposal-DVIFM`, 0.865 vs `-use_chroma` (61 crop pairs, 2026-10-17); the releases were retuned on AIC2026 after the columns were computed | which DVIFM build, constants and JND map produced each column | [aic2026_repro_2026-10-17](../benchmarks/aic2026_repro_2026-10-17/README.md) |

## Close but not exact (minor; ask only if convenient)

| Column | Ours | Residual | Open detail |
|---|---|---|---|
| `SSIMULACRA1` | `ssimulacra1` | native max 1.9e-5 | exact published convention |
| `WS-HV` | `ws-hv` | native max 0.021 | exact published convention |
| `FSIM`, `FSIMc` | `fsim-y`, `fsim` | one JPEG tail pair at 0.3 JND | downscale factor used |
| `CKDN` | `ckdn` | median 4.0e-5, max 1.4e-4 (0.026 JND) | resize implementation. The published column itself shows no agreement with the AIC2026 subjective JND (within-source Spearman −0.005); our port reproduces it, so that is a property of the method on this data, not of the port |
| `NLPD` | `nlpd-iqa` | median 2.6e-3, 2.6 % relative (f32 order and bilinear edges) | none needed |
| `PSNR-Cb`, `PSNR-Cr` | `psnr-cb`, `psnr-cr` | max 1.7e-2 / 5.7e-2 dB | chroma conversion details |
| `PSNR-YCbCr611` | `psnr-ycbcr611` | max 3.5e-3 dB | none needed |

## Not yet compared against the published column

| Column | Ours | Status |
|---|---|---|
| `PieAPP` | `pieapp` | implemented and qualified against upstream; comparison against the column is running (2026-10-17) |

## Not implemented: what we would need

| Column | Reason | What to ask |
|---|---|---|
| `TOPIQ`, `TOPIQ-PIPAL` | weights are non-commercial and distributed through pyiqa, which this project does not use | an alternative source of the weights and their terms |
| `MSSWD` | no upstream licence; weights only through pyiqa | licence and a weights source |
| `AIMOS` | not identified in any public source | what AIMOS is, and its reference |
| `proposal-IDFIQA` | reference code has no licence. The authors' AIC-4 evaluation script reproduces the column (61 crop pairs: median \|Δ\| 4.7e-6, max 3.3e-4 JND, Spearman 1.0; 2026-10-17); a port is being written outside this repository | a licence for the reference code |
| `HaarXYB-*`, `HaarY-*`, `HaarYCoCg-* deltaL1248` (site-only names) | unidentified, no published values | what these are |
| `PSNR-HVS-vqmt`, `PSNR-HVS-M-vqmt` (site-only names) | MSU VQMT implementation, no published values | none unless the values are published |

## Reproduced

`PSNR`, `PSNR-Y` (studio-601), `SSIM` (libvmaf), `MS-SSIM` (libvmaf),
`SSIM-pyiqa`, `MS-SSIM-pyiqa`, `SSIMc`, `PSNR-HVS` family (Daala),
`VIF` (vifvec), `VIF_vmaf`, `VMAF`, `VMAF-neg`, `ADM2`, `CVVDP` (with the
`standard_fhd` display), `SSIMULACRA2`, `GMSD`, `HaarPSI`, `VSI`, `DSSIM`,
`IW-SSIM` (piq convention), `mDCT-PSNR`, `proposal-mDCTPSNR`, `FLIP`
(`flip-v17` exact), `CIEDE2000`, `Butteraugli-original`, `LPIPS-alex`,
`LPIPS-VGG`, `DISTS`, `STLPIPS`, `STLPIPS-VGG`, `DeepDC`, `DreamSim`.
