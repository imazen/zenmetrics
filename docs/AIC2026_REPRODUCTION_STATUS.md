# AIC2026 published columns: what we reproduce, and what to ask the dataset authors

Status of every metric column published with the JPEG AIC2026 dataset
(`metrics_fullres.csv` / `metrics_cropped.csv`, 9,618 rows) against the
zenmetrics implementation of the same method, as of 2026-10-20 (all 9,618
cropped rows scored; the online-dashboard CSV of 2026-10-20 is now also
compared — see the online section). "Reproduced"
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
| `VMAF v1` (online) | `vmaf-v1` (`vmaf_v1.0.16_3d0h`, in-tree `vmaf` crate, `cpu-vmaf`) | not reproduced. Full set (9,618, 2026-10-20): raw median \|Δ\| 1.33, max 14.7, median site-JND 0.179, Spearman 0.969 — same family but a clearly different model than either `vmaf_v1.0.16_3d0h` or July `VMAF` (≈ our `vmaf-studio601`, median 1.75 apart). Variant probe 2026-10-21 (193-pair spread, `examples/v1_variant_probe.rs`): all eight vendored `vmaf_v1.0.16_*`/`_hfr_*` models miss (best is `3d0h` at median \|Δ\| 1.40, all systematically high — the column is lower and steeper at the midrange); studio-601 vs BT.709 vs full-range-601 ingress, neutral-128 chroma, 10-bit planes, adm3→adm2 substitution and 2-frame motion feeds each move the median ≤0.2. Scoring at full-res misses the same way (n=40, median 1.50). The cropped and full-res `VMAF v1` columns are exact-equal on 54.6 % of rows and differ up to 14.9 on the rest — the values are per-geometry, and ours miss both. Which model/ingress produced `VMAF v1` remains an authors question; libvmaf ≥3.2.0 ships only these eight v1 models and their JSONs are byte-identical to ours | which model and ingress produced the `VMAF v1` column | [full-set diff](../benchmarks/aic2026_full_repro_2026-10-20/compare_cropped-online.tsv) |
 1 of 1 ends

## Identified by configuration search (ports need the option exposed)

A bounded search over each canonical implementation's options (2026-10-17, [aic_confsearch_2026-10-17](../benchmarks/aic_confsearch_2026-10-17/README.md)) reproduces
these columns. site-JND is the AIC site's per-metric map; the match gate is 1e-3 site-JND.

| Column | Configuration that reproduces it | median / max site-JND | Spearman | What the zenmetrics port needs |
|---|---|---|---|---|
| `proposal-Butteraugli` | `max(0.00459792·N, 2.0938201·N − 1.1033102)` with N = libjxl `butteraugli` 3-norm at defaults; the same map holds on all 9,618 rows of both CSVs | 3.2e-8 / 5.1e-5 | 1.0000 | done: `proposal-butteraugli` applies the map; verified on all 9,618 crops 2026-10-20 — median site-JND 2.9e-7, max raw \|Δ\| 0.18 (JND tail), Pearson/Spearman 1.00000. The constants' provenance is still unknown |
| `CAMBI` | libvmaf 3.2.1 CAMBI defaults on luma from ffmpeg/swscale `-pix_fmt yuv420p` (an exact BT.601 formula differs by one code on 0.45 % of pixels, which CAMBI amplifies) | 8.2e-8 / 2.1e-3 (max raw 4.8e-7, the 6-decimal rounding) | 1.0000 | done: `cambi-aic2026` verified on all 9,618 crops 2026-10-20 — median site-JND 1.3e-7, max raw \|Δ\| 5e-7 (the published 6-decimal rounding); the generic `cambi` route also passes (median 1.1e-4) but is ~1000× further off |
| `DMM` | pyiqa 0.1.16 `dmm` on CUDA, fp32 (square `tar×tar` bilinear resize; torch's CUDA SVD picks a different basis than LAPACK in degenerate clusters) | 2.6e-4 / 2.4e-3 | 0.9999 | pyiqa's square resize, and cuSOLVER's basis choice for an exact match; `dmm` (upstream aspect resize, CPU SVD) differs by median 0.075 |
| `HDR_VDP_3` | HDR-VDP 3.0.7 `res.Q` (not `Q_JOD`), XYZ input (emission `oled`), display `L = 200·sRGB_EOTF(V)` cd/m² with no black level or reflection, ppd 37.84 (CVVDP `standard_fhd`), task `quality` | 1.36e-3 / 1.29e-2 | 1.0000 | `hdrvdp3-aic2026` carries it; full set (9,618, 2026-10-20): median raw \|Δ\| 1.56e-3 Q, median site-JND 1.53e-3, Spearman 0.99998 — just over the 1e-3 gate |
| `HDR_VDP_2` (near) | HDR-VDP 2.2.2 `res.Q`, same display and ppd, BT.709 input, codes not rounded to 10 bits | 3.65e-3 / 5.15e-2 (median 0.028 Q) | 0.9998 | `hdrvdp-aic2026` carries it; full set: median raw \|Δ\| 0.0281 Q, median site-JND 4.8e-3, Spearman 0.9999. Investigation closed 2026-10-20: residual is reference-dependent (per-source medians −0.062..+0.246, +0.42 with gradient energy); native-vs-Octave-2.2.2 gap is only 0.0087 Q, so the published column is a third implementation — most plausibly actual MATLAB or an unknown port (all documented settings verified or ruled out: alt weights, surround modes, borders, ppd sweep, strict arithmetic). Older versions ruled out by a SourceForge sweep on the 200-pair panel: 2.1/2.1.1/2.1.3/2.2.0 all sit ~59+ Q away (2.2.0 near-perfectly anti-correlated, pre-2.2.1 unreversed scale) vs 2.2.2's 0.07 |
| `proposal-DVIFM`, `proposal-DVIFM-0.2`, `proposal-DVIFM-0.2-use_chroma` | `dvifmish` port mode (unpublished; ports of the author's four released C++ variants) | no released variant reproduces any column: best Spearman 0.941 (`port-lap-v0.3` vs `proposal-DVIFM-0.2`), 0.751 vs `proposal-DVIFM`, 0.865 vs `-use_chroma` (61 crop pairs, 2026-10-17); the releases were retuned on AIC2026 after the columns were computed. We use the current releases (dvifmish port mode) and do not try to reproduce these older columns | which DVIFM build, constants and JND map produced each column | [aic2026_repro_2026-10-17](../benchmarks/aic2026_repro_2026-10-17/README.md) |
| `proposal-DVIFM-2d`, `proposal-DVIFM-min` (online CSV, aggregates only) | `dvifmish` `port-2d-v0.2` / `port-min-v0.2` | all 9,618 crops (2026-10-20): Spearman 0.923 vs `proposal-DVIFM-2d` (median site-JND 1.5), 0.989 vs `proposal-DVIFM-min` (median site-JND 1.46) — rank agreement only, not reproduced | which DVIFM build and constants produced these columns | aggregates only; per-image values never committed |

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
| `CW-SSIM` | `cw_ssim` | 0 | 2.2e-16 | 1.0000 | `cw-ssim-pyiqa` verified on all 9,618 crops 2026-10-20 — median raw \|Δ\| 2.5e-7, max 5e-7 (published 6-decimal rounding), median site-JND 1.9e-5; `cw-ssim` also passes (median 5.4e-5) but remains the distinct upstream convention |

Controls scored the same way also match their columns on that 61-pair panel (median \|Δ\| ≤ 3.4e-6, Spearman 1.0000; the full 9,618-crop run moves `NLPD` and `VSI` over the site-JND gate, see Close but not exact):
`DISTS`, `DeepDC`, `LPIPS-alex`, `LPIPS-VGG`, `STLPIPS`, `STLPIPS-VGG`, `SSIM-pyiqa`,
`MS-SSIM-pyiqa`, `NLPD`, `VSI`, `GMSD`, `VIF`, `PSNR`, `SSIMc`.

## Close but not exact (minor; ask only if convenient)

| Column | Ours | Residual | Open detail |
|---|---|---|---|
| `SSIMULACRA1` | `ssimulacra1` | native max 1.9e-5 | exact published convention |
| `WS-HV` | `ws-hv` | native max 0.021; full set vs the online `WS-HV` column: raw median 7.9e-4, max 5.5e-3, median site-JND 5.1e-3, Spearman 0.99998 — over the 1e-3 gate on the steep JND tail | exact published convention; which libvmaf version produced the column |
| `FSIM`, `FSIMc` | `fsim-y`, `fsim` | full set (9,618 crops, 2026-10-20): raw median 4.7e-5 / 4.9e-5, max 9.7e-3; median site-JND 1.05e-2 / 9.8e-3, over the 1e-3 gate; ours higher on every differing row, all at scores above 0.96. Ours matches the authors' FSIM.m to a 4.5e-8 maximum on 61 pairs, so the column most likely came from another implementation | which FSIM implementation produced the column; pyiqa `fsim` on all crops is being run to test it ([full-set diff](../benchmarks/aic2026_full_repro_2026-10-20/README.md)) |
| `VSI` | `vsi` | full set: raw median 1.5e-5, max 4.2e-3; median site-JND 7.4e-3 (the 61-pair panel matched at ≤ 3.4e-6). Ours matches the authors' VSI.m to a 1.1e-8 maximum on 61 pairs, with the resize noted in the VSI known-bug entry | which implementation and resize produced the column |
| `VMAF` | `vmaf-studio601` | full set: raw median 6.5e-3, median site-JND 1.07e-3, just over the 1e-3 gate; Pearson 1.00000 | libvmaf version used for the column |
| `CKDN` | `ckdn` | median 4.0e-5, max 1.4e-4 (0.026 JND) | resize implementation (pyiqa defaults give the same residual). The published column itself shows no agreement with the AIC2026 subjective JND (within-source Spearman −0.005); our port reproduces it, so that is a property of the method on this data, not of the port |
| `NLPD` | `nlpd-iqa` | median 2.6e-3, 2.6 % relative (f32 order and bilinear edges) | none needed |
| `PSNR-Cb`, `PSNR-Cr` | `psnr-cb`, `psnr-cr` | max 1.7e-2 / 5.7e-2 dB | chroma conversion details |
| `PSNR-YCbCr611` | `psnr-ycbcr611` | max 3.5e-3 dB | none needed |
| `VMAF v0` (online) | `vmaf-studio601` | full set: raw median 0.035, max 1.22, median site-JND 5.7e-3; the column is NOT the July `VMAF` column (median 0.035 apart) — a nearby but different model/ingress | which model and ingress produced `VMAF v0` |
| `VMAF-neg v0` (online) | `vmaf-neg-studio601` | full set: raw median 0.039, max 0.87, median site-JND 6.3e-3 | same question as `VMAF v0` |
| `VIF-vmaf-scale0..3` (online) | `aic_vmaf_features` (`vif_s0`–`vif_s3`) | full set: raw medians 2.6e-3 / 3.9e-4 / 1.9e-4 / 1.2e-4, median site-JND 9.9e-3 / 4.2e-3 / 2.8e-3 / 2.1e-3, Spearman ≥ 0.9994 each — same libvmaf-version-noise family as `VIF_vmaf` | which libvmaf version produced the scale columns |

## Settled fixed-factor semantics (owner clarification, 2026-10-19)

- FSIM/FSIMc and VSI `-kN` replace only `kernel_size = max(1, round(min(H,W)/256))` with `kernel_size = k`; every other pipeline step is unchanged.
- The pyiqa SSIM-family `-kN` variants set `downsample=True` and `kernel_size=k`; every other pipeline step is unchanged.
- These are the settled definitions. Branch and implementation qualification belong to the separate verification lane; this docs/site update changes no metric implementation or numerical gate.
- Verified 2026-10-20 against the primary oracles patched exactly as above: 1,952 scalar comparisons on all 61 staged crop pairs, 0 above bound (FSIM/FSIMc `-kN` maximum |Δ| 1.1e-6, VSI `-kN` 1.1e-8, pyiqa SSIM family 1.3e-5; MS-SSIM `-k5`/`-k6` refuse every pair because the 11×11 window does not fit their last pooled view) ([aic61](../benchmarks/fixed_factor_verify_2026-10-20/aic61/README.md), [retained evidence](../benchmarks/fixed_factor_verify_2026-10-20/README.md)). The July CSV has no fixed-factor columns, so there is nothing to ask the authors here; these variants are settled.

## Not yet compared against the published column

None: every implemented column has been compared (2026-10-17).

## Excluded from reproduction and implementation

| Column | Status | Reason |
|---|---|---|
| `AIMOS` | excluded (proprietary, unreleased) | Proprietary Cloudinary metric, never publicly described or released. No implementation, reproduction or author-question task. |

## Not implemented: what we would need

| Column | Reason | What to ask |
|---|---|---|
| `HaarXYB-*`, `HaarY-*`, `HaarYCoCg-* deltaL1248` (site-only names) | unidentified; no per-image values in any CSV and listed as excluded in the site config, yet the authors' rank-distribution figure (792 five-source subsets) shows six of them taking the top six ranks. Per the 2026-10-09 AHG discussion these are experimental metrics tuned on this data, to be ignored | what they are (colour space, masking, "deltaL1248"), and the code or per-image values |
| `PSNR-HVS-vqmt`, `PSNR-HVS-M-vqmt` (site-only names) | MSU VQMT implementation; the site config has a fitted JND map for both, so values exist, but none are published | the per-image values |

## Reproduced

`PSNR`, `PSNR-Y` (studio-601), `SSIM` (libvmaf), `MS-SSIM` (libvmaf),
`SSIM-pyiqa`, `MS-SSIM-pyiqa`, `SSIMc`, `PSNR-HVS` family (Daala),
`VIF` (vifvec), `VIF_vmaf`, `VMAF-neg`, `ADM2`, `CVVDP` (with the
`standard_fhd` display), `SSIMULACRA2`, `GMSD`, `HaarPSI`, `DSSIM`,
`IW-SSIM` (piq convention), `mDCT-PSNR`, `proposal-mDCTPSNR`, `FLIP`
(`flip-v17` exact), `CIEDE2000`, `Butteraugli-original`, `LPIPS-alex`,
`LPIPS-VGG`, `DISTS`, `STLPIPS`, `STLPIPS-VGG`, `DeepDC`, `DreamSim`.
Full-set (9,618) verified 2026-10-20 additionally: `proposal-Butteraugli`
(`proposal-butteraugli`, median site-JND 2.9e-7), `CAMBI` (`cambi-aic2026`,
median site-JND 1.3e-7), `CW-SSIM` (`cw-ssim-pyiqa`, median site-JND
1.9e-5), and on the online CSV `Butteraugli-max` and `Butteraugli-3norm`
(`butteraugli` max/pnorm3 outputs), `PSNR-HVS-vmaf` and `PSNR-HVS-Y-vmaf`
(`psnrhvs-daala`, the renamed Daala columns), `SSIMULACRA1`
(`ssimulacra1`, median |Δ| 3e-7 on the scored rows). Private ports, on the 61-crop panel:
`proposal-IDFIQA` (`idfiqa`, rten: median |Δ| 4.2e-6 JND, max 3.33e-4,
Spearman 1.0; the native 1e-3 JND oracle gate passes on rten, ORT CPU and
ORT CUDA; aggregate card verified 2026-10-18).

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

Data provenance for both HDR columns (2026-10-19,
[PROVENANCE.md](../benchmarks/hdrvdp_canonical_2026-10-19/PROVENANCE.md)): the
cropped `HDR_VDP_2`/`HDR_VDP_3` values are identical in DaRUS v2.0, the site's
2026-10-05 CSV and our 61-row subset. Fullres differs only on the 32 large
(category-3) sources, so the cropped columns were scored on the crops. The 122
crop PNGs are byte-identical to the DaRUS `PTC-crops-all/` members. 27 of the
61 pairs are category 3. The July label table has no HDR columns.

The native proposal-Butteraugli panel has median 3.06732e-7 site-JND, above
the requested 3.2e-8 gate. The 9,618-row mapping of published columns must
not be presented as 9,618 independently computed native scores.

### Additional HDR-VDP canonical-configuration evidence (2026-10-19)

The [successor research](../benchmarks/hdrvdp_canonical_2026-10-19/README.md)
has rerun the complete 61-pair upstream 3.0.7 closest route: median
0.0013356867286653529/max0.012807267024098623 site-JND, with 0/61 exact
six-decimal matches. Upstream and the retained Rust Q values differ by at
most 4.973799150320701e-14. The fresh Rust replay matches those retained
native values exactly. This separates the configuration residual from port
arithmetic. The retained HDR2 Octave baseline also replays exactly.
The [written CTC evidence](../benchmarks/hdrvdp_canonical_2026-10-19/UPSTREAM.md)
specifies 203 cd/m² for SDR HDR-VDP, distinct from the AIC2026 paper's
200-nit CVVDP statement. The exact HDR dataset calls remain unconfirmed;
all original gates remain in force.

The [symmetric-IFFT continuation](../benchmarks/hdrvdp_canonical_2026-10-19/SYMMETRIC_IFFT.md)
completes two additional 61-pair HDR2 oracle controls, retaining nonnegative
rows or columns before Hermitian completion. Both remain 0/61 exact; their
Q changes versus the original Octave adapter are at most 9.947598300641403e-14.
This adaptation does not explain the HDR2 column residual. Actual MATLAB
runtime identity and the executable dataset calls remain unqualified; no
metric implementation or original acceptance gate changed.

The [decoded-pixel/spectral continuation](../benchmarks/hdrvdp_canonical_2026-10-19/STAGE_CONTROLS.md)
finds exact 122-image RGB8 decoding parity across the tested paths. Replacing
upstream HDR2 spectral mixing with native coefficients changes its 61 scores
by at most 7.105427357601002e-14 Q. The modeled modern cubic/spline control
still has median 0.026448804274338045 Q error and 0/61 exact matches. These
controls explain neither the published-column gap nor the separate native
HDR2 residual. Original gates, implementation and qualification labels remain.

## The online-dashboard CSV (second oracle, 2026-10-20)

`metrics_*.online-2026-10-20.csv` (same 9,618 rows and order as July; every
shared column is numerically value-identical) adds 17 fully populated columns
and renames three: `PSNR-HVS`→`PSNR-HVS-vmaf`, `PSNR-HVS-Y`→`PSNR-HVS-Y-vmaf`,
`VMAF`→`VMAF v0`, `VMAF-neg`→`VMAF-neg v0`. Verified on the full cropped set:
the renames are value-identical for the PSNR-HVS pair but `VMAF v0`/`VMAF-neg
v0` differ from the July `VMAF`/`VMAF-neg` columns by raw medians 0.035/0.039 —
the online v0 columns are a nearby but different model or ingress.

Of the 17 new columns, at the 1e-3 median site-JND gate:

- **Match:** `Butteraugli-max`, `Butteraugli-3norm` (`butteraugli` outputs —
  the dashboard's `proposal-Butteraugli` affine applies to `Butteraugli-3norm`),
  `Butteraugli-original` (`butteraugli-original`), `SSIMULACRA1`
  (`ssimulacra1`), `PSNR-HVS-vmaf`, `PSNR-HVS-Y-vmaf` (`psnrhvs-daala`).
- **Close but over gate:** `WS-HV` (`ws-hv`, median JND 5.1e-3), `VMAF v0`
  (5.7e-3), `VMAF-neg v0` (6.3e-3), `VIF-vmaf-scale0..3` (2.1e-3–9.9e-3 —
  same libvmaf-version family as the `VMAF`/`VIF_vmaf` residuals).
- **Not reproduced:** `VMAF v1` (`vmaf-v1` 1.0.16-3d0h: raw median |Δ| 1.33,
  Spearman 0.969 — a different model; see the authors-question table),
  `proposal-DVIFM-2d`, `proposal-DVIFM-min` (dvifmish `port-2d-v0.2`/
  `port-min-v0.2`: Spearman 0.923/0.989, median site-JND ~1.5 — rank agreement
  only; per-image values stay unpublished).

Committed artifacts: `benchmarks/aic2026_full_repro_2026-10-20/`
(`metrics_cropped-online.zenmetrics.csv.gz`, `provenance_cropped-online.json`,
`compare_cropped-online.tsv`); the July CSV is `metrics_cropped.zenmetrics.csv.gz`
with `provenance_cropped.json`, `compare_cropped.tsv`. `scripts/aic2026/`
(`assemble.py` + `aic_columns.py` + `july_diff.py` + `repro_check.py`,
`just aic-repro-check`) rebuilds and verifies both.

Post-`4d833195` metric-code changes (`flip-v17`, `psnr`, `psnr-cb`, `psnr-cr`,
`psnr-y-studio601`, `psnr-ycbcr611`) were rescored on the `22bff298` master
build and every value is byte-identical to the committed CSV — the commits'
bit-exact claims hold on the full 9,618-row set.
