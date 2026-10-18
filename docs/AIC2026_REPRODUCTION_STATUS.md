# AIC2026 published columns: what we reproduce, and what to ask the dataset authors

Status of every metric column published with the JPEG AIC2026 dataset
(`metrics_fullres.csv` / `metrics_cropped.csv`, 9,618 rows) against the
zenmetrics implementation of the same method, as of 2026-10-18. "Reproduced"
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
| `proposal-IDFIQA` | private `idfiqa` port | median 4.2e-6, max 3.33e-4 JND, Spearman 1.0 on rten; native oracle 1e-3 JND gate passes on rten, ORT CPU and CUDA | upstream code licence; no code or weights published | aggregate card verified 2026-10-18 |
| FSIM/FSIMc/VSI fixed-factor and SSIM-family f32 variants | `fsim-k2/k3/k4/nodownscale` (both FSIM outputs), `vsi-k*`, `ssim-pyiqa-f32-k2`–`k6`, `ssimc-pyiqa-f32-k2`–`k6`, `ms-ssim-pyiqa-f32-k2`–`k6` | implemented, counted and timed; published parity unverifiable because no per-image values are available | per-image values and published base downscale factors | [CLI](../crates/zenmetrics-cli/src/metrics/mod.rs), [counts and timings](../benchmarks/iqa_complexity_2026-10-17/README.md) |

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
column's convention are implemented in `crates/zenmetrics-pyiqa-ports`, the one crate under
pyiqa's non-commercial terms (TOPIQ, TOPIQ-PIPAL, WaDIQaM, AHIQ, CW-SSIM, DMM). MSSWD is ported privately; only aggregates can be published.
The native `dmm-pyiqa` port fails its own 1e-3 JND gate (worst 0.071 JND, SVD);
matching the resize convention does not qualify native numerical parity.
See [the port receipt](../crates/zenmetrics-pyiqa-ports/README.md).

| Column | pyiqa metric | median \|Δ\| | max \|Δ\| | Spearman | zenmetrics |
|---|---|---|---|---|---|
| `TOPIQ` | `topiq_fr` (KADID) | 3.6e-5 | 2.7e-4 | 1.0000 | `topiq-fr`: public non-commercial port; all three engines pass 1e-3 JND |
| `TOPIQ-PIPAL` | `topiq_fr-pipal` | 2.4e-5 | 1.5e-4 | 1.0000 | `topiq-fr-pipal`: public non-commercial port; all three engines pass 1e-3 JND |
| `MSSWD` | `msswd` (learned) | 4.1e-6 | 2.4e-4 | 0.9999 | ported privately; aggregates only; redistribution terms unresolved |
| `WaDIQaM` | `wadiqam_fr` (KADID-trained) | 1.7e-6 | 1.2e-5 | 1.0000 | `wadiqam-pyiqa` passes 1e-3 JND (worst 1.1e-5 ladder / 2.7e-6 AIC JND); official LIVE/TID variants are distinct |
| `AHIQ` | `ahiq` | 2.5e-6 | 1.0e-5 | 1.0000 | `ahiq-pyiqa` passes 1e-3 JND (worst 1.2e-5 AIC JND); `ahiq` keeps upstream nine-crop |
| `PieAPP` | `pieapp` | 2.3e-5 | 1.9e-4 | 0.9999 | `pieapp-pyiqa` stride 27 passes 1e-3 JND (worst 2.3e-6); upstream `pieapp` is distinct |
| `CKDN` | `ckdn` | 4.0e-5 | 1.4e-4 | 0.9999 | `ckdn` matches to the same residual |
| `CW-SSIM` | `cw_ssim` | 0 | 2.2e-16 | 1.0000 | `cw-ssim-pyiqa` passes 1e-3 JND (worst 5.4e-13 JND); `cw-ssim` remains distinct |

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
| `AIMOS` | still unidentified: private probe scored 16 of 34 no-reference metrics with no match (best hyperiqa ratio, Spearman 0.82) | what AIMOS is, and its reference |
| `HaarXYB-*`, `HaarY-*`, `HaarYCoCg-* deltaL1248` (site-only names) | unidentified; no per-image values in any CSV and listed as excluded in the site config, yet the authors' rank-distribution figure (792 five-source subsets) shows six of them taking the top six ranks. Per the 2026-10-09 AHG discussion these are experimental metrics tuned on this data, to be ignored | what they are (colour space, masking, "deltaL1248"), and the code or per-image values |
| `PSNR-HVS-vqmt`, `PSNR-HVS-M-vqmt` (site-only names) | MSU VQMT implementation; the site config has a fitted JND map for both, so values exist, but none are published | the per-image values |

## Reproduced

`PSNR`, `PSNR-Y` (studio-601), `SSIM` (libvmaf), `MS-SSIM` (libvmaf),
`SSIM-pyiqa`, `MS-SSIM-pyiqa`, `SSIMc`, `PSNR-HVS` family (Daala),
`VIF` (vifvec), `VIF_vmaf`, `VMAF`, `VMAF-neg`, `ADM2`, `CVVDP` (with the
`standard_fhd` display), `SSIMULACRA2`, `GMSD`, `HaarPSI`, `VSI`, `DSSIM`,
`IW-SSIM` (piq convention), `mDCT-PSNR`, `proposal-mDCTPSNR`, `FLIP`
(`flip-v17` exact), `CIEDE2000`, `Butteraugli-original`, `LPIPS-alex`,
`LPIPS-VGG`, `DISTS`, `STLPIPS`, `STLPIPS-VGG`, `DeepDC`, `DreamSim`.

## Explicit AIC routes (2026-10-18)

The [native 61-crop qualification](../benchmarks/aic_refusals_2026-10-18/README.md)
adds `hdrvdp3-aic2026` (Q and Q_JOD), `hdrvdp-aic2026`, `cambi-aic2026` and
`proposal-butteraugli`. Existing metric names retain their behavior.

HDR-VDP 3 uses 10-bit code quantization, 200-nit sRGB, XYZ/OLED and
37.842505866 ppd: median/max 0.00133569/0.0128073 site-JND, Spearman 1.
CAMBI uses integer swscale luma: median 8.17689e-8 site-JND, Spearman 1.
Both pass the specified crop gates.

HDR-VDP 2 retains median 0.0277768 Q residual. An independent Octave 2.2.2
RGB panel also has median 0.02651 Q residual against the published column;
Rust-versus-Octave median is 0.00869019 Q. The older luminance-only golden
bound does not establish this RGB panel's accuracy or identify the setting gap.
The strict-arithmetic control also retains the gap: native strict versus
Octave median 0.008689808892256679 Q, maximum 0.014212052720857571 Q.
Strict versus relaxed native differs by median 1.6615683477994025e-6 Q;
this control does not identify the missing setting. Full rows are in the
[retained strict panel](../benchmarks/aic_refusals_2026-10-18/hdrvdp2-strict-comparison.tsv).

The native proposal-Butteraugli panel has median 3.06732e-7 site-JND, above
the requested 3.2e-8 gate. The 9,618-row mapping of published columns must
not be presented as 9,618 independently computed native scores.
