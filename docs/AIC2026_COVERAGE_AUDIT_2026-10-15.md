# AIC2026 site metrics vs zenmetrics: coverage audit (2026-10-15, i265)

Scope: every name in the site's `config.json` (`mappings`, 124; `excluded`, 52), the 2026 dataset CSV
columns, and the screenshot names. Measured on the 61 full-resolution pairs in `~/tmp/aic-sub/img/`
against `site_fullres_subset61.csv` (site) and `metrics_subset.csv` (dataset). Binary:
`~/tmp/zenmetrics-lock-snap-nn/target/release/zenmetrics` (zenmetrics-cli 0.6.0, built from the
`zenmetrics--nn` checkout), `batch --use-legacy-scheduler`. "max |Δ| in JND" applies the site's fitted
map for that column to both our value and the published one.

## Summary

131 rows (screenshot aliases share a row with the site name they point to):

| status | rows |
|---|---|
| reproduced | 27 |
| close | 13 |
| differs | 6 |
| missing | 48 |
| variant-needed | 25 |
| unverified (no published values) | 6 |
| n/a (JND display columns) | 6 |

Reproduced (max |Δ| ≤ ~0.1% of range, ≤ 0.05 JND): PSNR, PSNR-Y, SSIM, MS-SSIM, PSNR-HVS(-Y/-Cb/-Cr),
SSIMULACRA2, GMSD, HaarPSI, VIF (vifvec), DSSIM, mDCT-PSNR, proposal-mDCTPSNR (via the remap),
CVVDP, Butteraugli-max, Butteraugli-3norm, LPIPS-alex/-VGG, STLPIPS/-VGG, DISTS, DeepDC, dataset
VMAF/VMAF-neg and VIF_vmaf (vmaf crate only, not the CLI).

Close (tiny median, one or two outlier pairs): FSIM/FSIMc (two JPG_07 pairs at 0.94 JND), VSI (one
pair 0.65 JND), MS-SSIM-pyiqa via `msssim` (odd-size source 0.40 JND), IW-SSIM (one near-lossless pair
0.13 JND), NLPD (6% of range), SSIMc via `ssim` (0.07 JND), ADM2 and VIF-vmaf-scale0…3 (crate only),
VMAF v0 / VMAF-neg v0 via the crate with studio-601 Y.

Differs: CLI `vmaf`, `vmaf-neg`, `vmaf-v1` (BT.709 luma ingress, `--luma-ingress` ignored; max 2.6 / 3.0 /
8.7 points), proposal-Butteraugli (affine of our pnorm-3 only), HDR_VDP_2 / HDR_VDP_3 (docs; not
re-measured: CLI needs HDR sources).

### Missing, ranked by how many site/screenshot entries one piece of work would cover

1. **Fixed downscale factor + unrounded-input options** (25 variant-needed rows: 22 site names + 9 screenshot-only names; plus the missing SSIM-pyiqa base):
   FSIM/FSIMc/VSI `-k2/-k3/-k4/-nodownscale` (12), SSIM-pyiqa-f32-k2…k6 and SSIMc-pyiqa-f32-k2…k6 (10),
   pyiqa-VSI-f32, pyiqa-SSIM-f32, pyiqa-SSIMc-f32, pyiqa-MS-SSIM-f32(-k2…k6). Needs a factor override in
   `fsim`/`vsi` (both compute `max(1, round(min/256))` internally) and a luma SSIM without downscale
   (SSIM-pyiqa, also missing as a base metric).
2. **DVIFM** (13 entries: 9 site names + 4 screenshot aliases): proposal-DVIFM, -0.2, -0.2-use_chroma,
   -2d, -min, -min-v0.2, -2d-v0.2, -lap-v0.2, -lap-v0.3. dvifmish (dev `~/work/dvifmish`, publish=false)
   already ports the author's v0.2/v0.3 builds under exactly these preset names; not wired into
   zenmetrics and not compared to these columns here.
3. **HaarXYB / HaarY / HaarYCoCg -deltaL1248** (11, all excluded, no values): unidentified.
4. **Learned models** (10, one each): TOPIQ, TOPIQ-PIPAL, AHIQ, PieAPP, WaDIQaM, DMM, A-FINE, DreamSim,
   MSSWD, CKDN (CKDN is in progress in the zenmetrics--nnport lane).
5. **WD_s1…s5** (5): Wasserstein Distortion on VGG-16 features (another lane is probing it).
6. **Classic lane, unlanded** (5): PSNR-YCbCr611, CIEDE2000, SSIMc exact (commits 94574b8b, ac757d73,
   0a7fa500 exist in the object store, on no branch) plus PSNR-Cb/PSNR-Cr. My numpy probes reproduce
   all five (max |Δ|: 3.5e-3 dB, 3.0e-4 ΔE, 4.4e-7, 0.017 dB, 0.057 dB), so the conventions are known.
7. **VMAF family plumbing** (not missing, but the CLI path is wrong for 5 rows and absent for 7):
   feed `vmaf*` studio-601 Y (the crate then reproduces the dataset columns to 0.08), accept odd sizes
   (8 of 13 sources are odd; the CLI aborts the whole batch), and expose ADM2, VIF scales, VIF_vmaf and
   CAMBI (`vmaf::cambi_v1_from_luma` exists) as CLI columns.
8. **PSNR-HVS-vqmt / -M-vqmt** (2, excluded, no values).
9. **Singles**: Butteraugli-original (≠ libjxl: Pearson 0.87 vs our max), SSIMULACRA1, WS-HV
   (unidentified), CW-SSIM, FLIP, proposal-IDFIQA, AIMOS (unidentified).

## Findings worth acting on

- **kN = fixed downscale factor N; nodownscale = factor 1; f32 = no u8 rounding.** Evidence: (a) the
  site maps for FSIM, FSIMc and VSI equal their `-k3` maps to 7 digits, so the fit used the k3 values;
  (b) those base maps fit the cropped 840×944 set (automatic factor 3) far better than full-res values
  (RMS vs JND_CVVDP over 9,618 rows: FSIM 0.91 vs 1.38, VSI 0.98 vs 1.33; other metrics equal);
  (c) the published full-res FSIM/VSI columns are the automatic factor (our auto-factor ports match them);
  (d) SSIM*-f32-kN: our numpy values after an N×N downscale put map_kN within +0.01…+0.19 JND (median)
  of the base map; without the downscale the bias is +1.3…+10.5 JND (`variant_test.tsv`). The resampling
  kernel (box, bicubic, Lanczos) cannot be told apart this way. SSIM-pyiqa base uses rounded luma
  (matches to 3.9e-6; unrounded is off by up to 7e-4), so "f32" means unrounded.
- **Stale DeepDC numbers:** `~/tmp/nn-phase0/aic/deepdc.tsv` (ort-cuda, full res) was scored before the
  ImageNet-normalised re-export and is off by up to 0.10. Rescored here on rten: max 3.0e-5.
- **CLI `vmaf` ingress:** it feeds studio BT.709 Y; the crate fed studio-601 Y matches the dataset
  VMAF/VMAF-neg (max 0.076 / 0.12). The site's newer `VMAF v0`/`VMAF-neg v0`/VIF-scale columns differ from
  the dataset ones by up to 0.26 and no ingress tried here closes that.
- **Memory:** a 61-pair rten batch of `lpips-vgg`/`dists` peaks at 16.3 / 16.1 GB; `stlpips-vgg` was
  OOM-killed at the 16 GB cap as one batch and passed one process per pair (peak 15.9 GB).

## Files

- Per-pair comparisons: `cmp/<label>.tsv` (ours, published, |Δ|, both JND values, |ΔJND|);
  summary: `cmp_summary.tsv`; variant test: `variant_test.tsv`.
- Our raw scores: `raw/*.tsv` (CLI), `raw/py_probe.tsv` (numpy probes, `py_probe.py`),
  `raw/vmaf_probe.txt` (vmaf crate `aic_probe`, studio-601 Y), `raw/nnpairs/` (per-pair rten).
- Scripts: `run_classic.sh`, `run_vmaf_even.sh`, `run_vmaf_601.sh`, `run_nn.sh`, `run_nn_pairs.sh`,
  `run_deepdc_pairs.sh`, `compare.py`, `variant_test.py`, `make_coverage.py` + `coverage_rows.py`.
- run-heavy peaks: classic CLI set 2.06 GiB; numpy probes 3.24 GiB; NN per-pair 15.93 GiB.

## Table

Status rule: reproduced = max |Δ| ≤ ~0.1% of the column range and ≤ 0.05 JND; close = tiny median with a tail up to ~1 JND or a few % of range; differs = wrong family/config; unverified = no published values exist for the name. "(excluded)" marks names the site hides.

## Classical (full-reference, non-learned)

| name | in CSV? | zenmetrics metric | n | median \|Δ\| | max \|Δ\| | max \|Δ\| in JND | status | note |
|---|---|---|---|---|---|---|---|---|
| PSNR | site+dataset | `psnr` | 61 | 9.09e-05 | 0.00128 | 5.48e-05 | reproduced | RGB mean-MSE PSNR |
| PSNR-Y / pyiqa-PSNR-Y | site+dataset | `psnr-y-studio601` | 61 | 0.000313 | 0.00328 | 0.000576 | reproduced | studio-swing BT.601 Y, u8-rounded; `psnr-y` (BT.709) and `psnr-y601` (full-range) are 1.3 dB off |
| PSNR-YCbCr611 | site+dataset | missing on master (lane commit 94574b8b `psnr-ycbcr611`, unlanded) | 61 | 0.000311 | 0.0035 | 0.000532 | missing | numpy probe reproduces: (6·MSE_Y+MSE_Cb+MSE_Cr)/8 on rounded studio-601 YCbCr, then dB (PSNR-average form is 0.65 dB off) |
| PSNR-Cb (excluded) | dataset | missing (no CLI chroma PSNR) | 61 | 0.00192 | 0.0169 | 0.00197 | missing | numpy probe: rounded studio-601 Cb at full resolution reproduces |
| PSNR-Cr (excluded) | dataset | missing (no CLI chroma PSNR) | 61 | 0.00211 | 0.0573 | 0.00882 | missing | as PSNR-Cb; one pair 0.057 dB off |
| PSNR-HVS-vmaf (= dataset PSNR-HVS) | site | `psnrhvs-daala` (`psnrhvs_daala`) | 61 | 0.000871 | 0.0154 | 0.00449 | reproduced | dataset `PSNR-HVS` is the same column (identical values) |
| PSNR-HVS-Y-vmaf (= dataset PSNR-HVS-Y) | site | `psnrhvs-daala` (`psnrhvs_daala_y`) | 61 | 0.00121 | 0.0261 | 0.00659 | reproduced |  |
| PSNR-HVS-Cb-vmaf (dataset PSNR-HVS-Cb) | dataset | `psnrhvs-daala` (`psnrhvs_daala_cb`) | 61 | 0.00264 | 0.0382 |  | reproduced | max 0.038 dB of a 27 dB range |
| PSNR-HVS-Cr-vmaf (dataset PSNR-HVS-Cr) | dataset | `psnrhvs-daala` (`psnrhvs_daala_cr`) | 61 | 0.00306 | 0.0691 |  | reproduced | max 0.069 dB of a 24 dB range |
| SSIM | site+dataset | `ssim-libvmaf` | 61 | 1e-06 | 3.2e-05 | 0.00618 | reproduced | libvmaf float_ssim on studio-601 Y |
| MS-SSIM | site+dataset | `msssim-libvmaf` | 61 | 1e-06 | 0.000141 | 0.0119 | reproduced | libvmaf float_ms_ssim; worst pair 0.012 JND |
| SSIM-pyiqa / pyiqa-SSIM | site+dataset | missing (`ssim` is a different variant: max 0.074) | 61 | 3.31e-07 | 3.86e-06 | 5.63e-05 | missing | numpy probe reproduces (3.9e-6 max): SSIM on round(0.299R+0.587G+0.114B), 11×11 σ1.5 Gaussian, valid region, cs≥0, no downscale |
| SSIMc / pyiqa-SSIMc | site+dataset | `ssim` (close); exact form not in CLI (lane commit 0a7fa500 `ssimc`, unlanded) | 61 | 0.0002 | 0.00542 | 0.0681 | close | `ssim` med 2e-4, max 5.4e-3 (0.07 JND). numpy probe (per-RGB SSIM, valid region, cs≥0) reproduces to 4.4e-7 |
| MS-SSIM-pyiqa / pyiqa-MS-SSIM | site+dataset | `msssim` (house luma) | 61 | 0.000116 | 0.00497 | 0.398 | close | med 1.2e-4; worst S53_JAIB_07 (odd-size source) 5e-3 = 0.40 JND: odd-dimension pooling differs. `--luma-ingress yuv601-studio` is worse (med 1.7e-3) |
| IW-SSIM / IW-SSIM-piq | site+dataset | `iwssim-piq` | 61 | 6.92e-06 | 0.000666 | 0.132 | close | med 6.9e-6; one near-lossless pair (S70_VVC_03) 6.7e-4 = 0.13 JND; next worst 2.7e-4 |
| SSIMULACRA2 | site+dataset | `ssim2` | 61 | 2.6e-07 | 4.9e-07 | 3.63e-08 | reproduced | max 4.9e-7 (output rounding); the old 6e-3 drift is gone |
| GMSD / pyiqa-GMSD | site+dataset | `gmsd` | 61 | 3.87e-07 | 5.36e-06 | 0.000365 | reproduced |  |
| HaarPSI / pyiqa-HaarPSI | site+dataset | `haarpsi` | 61 | 7.9e-06 | 0.000182 | 0.00414 | reproduced | max 1.8e-4 = 0.004 JND |
| VIF / pyiqa-VIF | site+dataset | `vifvec` | 61 | 2.56e-06 | 5.61e-05 | 0.00026 | reproduced | `vif` (VIFp) is a different metric: max 0.37 |
| NLPD / pyiqa-NLPD | site+dataset | `nlpd-iqa` | 61 | 0.00234 | 0.0118 | 0.231 | close | med 2.3e-3, max 1.2e-2 (6% of range, 0.23 JND); `nlpd` is the RGB/Laparra variant |
| FSIM / pyiqa-FSIM | site+dataset | `fsim` (`fsim_*`) | 61 | 2.3e-05 | 0.00649 | 0.941 | close | med 2.3e-5; S55_JPG_07 6.5e-3 and S54_JPG_07 4.8e-3 = 0.94 JND each; all others ≤2.4e-4 |
| FSIMc / pyiqa-FSIMc | site+dataset | `fsim` (`fsimc_*`) | 61 | 2.96e-05 | 0.00645 | 0.925 | close | same two JPG_07 pairs (0.92 JND) |
| VSI / pyiqa-VSI | site+dataset | `vsi` | 61 | 1.81e-05 | 0.00185 | 0.654 | close | med 1.8e-5; S15_PNG_15 1.9e-3 = 0.65 JND; next worst 1.4e-4 |
| DSSIM | site+dataset | `dssim` | 61 | 2.3e-07 | 2.33e-06 | 0.000691 | reproduced |  |
| CIEDE2000 | site+dataset | missing on master (lane commit ac757d73 `ciede2000`, unlanded) | 61 | 3.02e-05 | 0.000296 | 0.000157 | missing | numpy probe reproduces (max 3e-4): sRGB→linear→XYZ D65→Lab, mean ΔE00, kL=kC=kH=1 |
| mDCT-PSNR / mDCTpsnr | site+dataset | `mdctpsnr` | 61 | 1.9e-06 | 1.17e-05 | 5.86e-07 | reproduced |  |
| CVVDP | site+dataset | `cvvdp --display-model standard_fhd` | 61 | 0.000108 | 0.000887 | 0.00229 | reproduced | max 8.9e-4 JOD = 0.002 JND |
| Butteraugli-max | site | `butteraugli` (`butteraugli_max`) | 61 | 4e-06 | 5.1e-05 | 9.96e-06 | reproduced | libjxl butteraugli, max-norm |
| Butteraugli-3norm | site | `butteraugli` (`butteraugli_pnorm3`) | 61 | 2e-06 | 9.6e-05 | 7.41e-05 | reproduced |  |
| Butteraugli-original | site | missing | 61 | 1.09 | 11.7 | 2.03 | missing | vs our libjxl max: med |Δ| 1.09, Pearson 0.87: a different implementation (presumably the pre-libjxl google/butteraugli tool); not identified further |
| SSIMULACRA1 | site | missing |  |  |  |  | missing | site values only (0.004–0.12, lower better); no implementation anywhere in the workspace |
| WS-HV | site | missing |  |  |  |  | missing | unidentified (0.45–0.98, higher better) |
| CW-SSIM | site+dataset | missing |  |  |  |  | missing | complex steerable pyramid; conventions not pinned (see AIC2026_COLUMN_IDENTIFICATION) |
| FLIP | site+dataset | missing |  |  |  |  | missing | NVIDIA FLIP (BSD-3 reference exists) |
| MAD (excluded) | neither | `mad` |  |  |  |  | unverified | excluded on the site; no published values to compare. Site map has only a, b |
| HDR_VDP_2 / HDR-VDP-2 Q | site+dataset | `hdrvdp` (HDR-nits input only) |  |  |  |  | differs | not re-measured here (CLI needs `--hdr` sources). Docs: published column misses our v2.2.2 at ppd 30 and 64 by up to 5.7 → unknown display config |
| HDR_VDP_3 / HDR-VDP-3 Q | site+dataset | `hdrvdp3` (HDR-nits input only) |  |  |  |  | differs | not re-measured here. Docs: ours = MATLAB 3.0.7 bit-for-bit, but published reads 0.36–0.92 JOD lower → different run config |
| PSNR-HVS-ref, PSNR-HVS-M-ref | neither | `psnrhvs` / `psnrhvs-y` (Ponomarenko `psnrhvsm.m` port) — probably |  |  |  |  | unverified | excluded; no values; site map is a placeholder (a=1, c≈50.8/64.1). "ref" reads as the authors' MATLAB; ours ports it |
| PSNR-HVS-vqmt, PSNR-HVS-M-vqmt | neither | missing (MSU VQMT implementation) |  |  |  |  | unverified | excluded; no values; placeholder maps |
| HaarXYB-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarXYB-hmask-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarXYB-mask-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarXYB-SSIM-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarXYB-SSIM-hmask-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarXYB-SSIM-mask-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarXYB-SSIM-mask-edge-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarY-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarY-hmask-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarYCoCg-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |
| HaarYCoCg-hmask-deltaL1248 (excluded) | neither | missing |  |  |  |  | missing | excluded; no values, placeholder map; unidentified Haar-wavelet experiment (colour space XYB/Y/YCoCg, ΔL at scales 1,2,4,8?) |

## libvmaf family

| name | in CSV? | zenmetrics metric | n | median \|Δ\| | max \|Δ\| | max \|Δ\| in JND | status | note |
|---|---|---|---|---|---|---|---|---|
| VMAF v0 / VMAF-v0.6.1 | site | `vmaf` (differs); vmaf crate on studio-601 Y is close | 61 | 0.279 | 2.61 | 0.427 | differs | CLI feeds studio BT.709 Y (`--luma-ingress` is ignored for vmaf): med 0.28, max 2.6. The vmaf crate fed studio-601 Y (aic_probe) gives med 0.033, max 0.26 (0.04 JND). CLI also rejects odd sizes: scored on even-cropped copies |
| VMAF v0 via vmaf crate + studio-601 Y | site | (not a CLI path: crates/vmaf/examples/aic_probe.rs) | 61 | 0.0327 | 0.256 | 0.0408 | close | the residual vs the site column is not ingress: the same probe matches the older dataset `VMAF` to med 0.0066 / max 0.076 |
| VMAF-neg v0 / VMAF-v0.6.1neg | site | `vmaf-neg` (differs); crate + studio-601 Y close | 61 | 0.3 | 3.05 | 0.484 | differs | CLI med 0.30, max 3.0; crate+601: med 0.046, max 0.25 (0.04 JND) |
| VMAF v1 / VMAF-v1 | site | `vmaf-v1` | 61 | 1.1 | 8.67 | 1.2 | differs | med 1.1, max 8.7 (1.2 JND). Same BT.709 ingress as `vmaf`; the 601 path was not tried for v1 (aic_probe is v0 only) |
| VMAF (dataset) | dataset | vmaf crate + studio-601 Y (CLI `vmaf` differs: max 2.7) | 61 | 0.00661 | 0.0763 | 0.0125 | reproduced | med 0.0066, max 0.076 (0.013 JND); older dataset column, ≠ site `VMAF v0` by up to 0.26 |
| VMAF-neg (dataset) / JND_VMAF-neg | dataset | vmaf crate + studio-601 Y (CLI `vmaf-neg` differs: max 2.9) | 61 | 0.00265 | 0.123 | 0.0194 | reproduced | med 0.0027, max 0.12 (0.02 JND). `JND_VMAF-neg` (excluded) is a fixed remap of this column |
| VMAF_420 | neither | `vmaf` |  |  |  |  | unverified | site map only, no values; v0.6.1 features are luma-only, so 4:2:0 chroma should not change the score |
| ADM2 | site+dataset | vmaf crate feature, not a CLI column | 61 | 1.2e-05 | 0.000956 | 0.0213 | close | aic_probe (studio-601 Y): med 1.2e-5, max 9.6e-4 (1% of range, 0.02 JND) |
| VIF-vmaf-scale0 | site | vmaf crate feature, not a CLI column | 61 | 0.00323 | 0.013 | 0.0542 | close | aic_probe; site scale columns come from the same recomputation as `VMAF v0` (their sum ≠ dataset VIF_vmaf by up to 0.013) |
| VIF-vmaf-scale1 | site | vmaf crate feature, not a CLI column | 61 | 0.000418 | 0.00319 | 0.037 | close | aic_probe; site scale columns come from the same recomputation as `VMAF v0` (their sum ≠ dataset VIF_vmaf by up to 0.013) |
| VIF-vmaf-scale2 | site | vmaf crate feature, not a CLI column | 61 | 0.00019 | 0.00325 | 0.0582 | close | aic_probe; site scale columns come from the same recomputation as `VMAF v0` (their sum ≠ dataset VIF_vmaf by up to 0.013) |
| VIF-vmaf-scale3 | site | vmaf crate feature, not a CLI column | 61 | 0.000154 | 0.00309 | 0.0677 | close | aic_probe; site scale columns come from the same recomputation as `VMAF v0` (their sum ≠ dataset VIF_vmaf by up to 0.013) |
| VIF_vmaf (dataset) | dataset | vmaf crate feature (Σ of 4 scales) | 61 | 7.2e-05 | 0.00386 | 0.0182 | reproduced | dataset VIF_vmaf = sum of the 4 site scale columns (max diff 0.013) |
| VIF-vmaf-fitted (excluded) | neither | missing |  |  |  |  | unverified | excluded; placeholder map, no values; presumably a fitted combination of the 4 scales |
| CAMBI | site+dataset | `vmaf::cambi_v1_from_luma` in the crate; not a CLI metric |  |  |  |  | missing | CSV has values; not measured here (no CLI or example emits it) |

## Learned (deep)

| name | in CSV? | zenmetrics metric | n | median \|Δ\| | max \|Δ\| | max \|Δ\| in JND | status | note |
|---|---|---|---|---|---|---|---|---|
| LPIPS-alex | site+dataset | `lpips-alex` (nn) | 61 | 2.3e-07 | 4.87e-07 | 4.34e-05 | reproduced | full-res, ort-cuda run from ~/tmp/nn-phase0/aic (same binary, same pairs) |
| STLPIPS | site+dataset | `stlpips-alex` (nn) | 61 | 3.34e-06 | 9.27e-05 | 0.00356 | reproduced | as LPIPS-alex |
| LPIPS-VGG | site+dataset | `lpips-vgg` (nn) | 61 | 3.79e-06 | 0.000102 | 0.000756 | reproduced | rten CPU, full res, one batch, peak RSS 16.3 GB |
| DISTS | site+dataset | `dists` (nn) | 61 | 2.55e-06 | 1.65e-05 | 0.00033 | reproduced | rten CPU, full res |
| STLPIPS-VGG | site+dataset | `stlpips-vgg` (nn) | 61 | 3.4e-06 | 8.18e-05 | 0.000457 | reproduced | 10 pairs ort-cuda + 51 pairs rten, one process per pair (a single 61-pair rten batch hit the 16 GB cap) |
| DeepDC | site+dataset | `deepdc` (nn) | 61 | 1.86e-06 | 3.02e-05 | 0.000366 | reproduced | rten, one process per pair. The earlier ort-cuda full-res rows in ~/tmp/nn-phase0/aic were scored with the pre-normalisation export and are off by up to 0.10: discard them |
| CKDN (excluded) | dataset | missing on master (`ckdn` in the zenmetrics--nnport lane, unlanded) |  |  |  |  | missing | dataset-only column; not measured here |
| WD_s1 | site+dataset | missing |  |  |  |  | missing | Wasserstein Distortion on VGG-16 features (another lane is probing it). |
| WD_s2 | site+dataset | missing |  |  |  |  | missing | Wasserstein Distortion on VGG-16 features (another lane is probing it). |
| WD_s3 | site+dataset | missing |  |  |  |  | missing | Wasserstein Distortion on VGG-16 features (another lane is probing it). |
| WD_s4 | site+dataset | missing |  |  |  |  | missing | Wasserstein Distortion on VGG-16 features (another lane is probing it). the paper's VGG-16 σ=2^4 row |
| WD_s5 | site+dataset | missing |  |  |  |  | missing | Wasserstein Distortion on VGG-16 features (another lane is probing it). |
| TOPIQ | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| TOPIQ-PIPAL | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| AHIQ | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| PieAPP | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| WaDIQaM | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| DMM | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| A-FINE | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| DreamSim | site+dataset | missing |  |  |  |  | missing | queued in PYIQA_PORT_PLAN.md |
| MSSWD | site+dataset | missing |  |  |  |  | missing | learned MS-SWD; on hold: weights only via pyiqa (noncommercial) |
| AIMOS | site+dataset | missing |  |  |  |  | missing | unidentified |

## AIC-4 proposals

| name | in CSV? | zenmetrics metric | n | median \|Δ\| | max \|Δ\| | max \|Δ\| in JND | status | note |
|---|---|---|---|---|---|---|---|---|
| proposal-Butteraugli | site+dataset | `butteraugli` (`butteraugli_pnorm3`), affine only | 61 | 1.08 | 3.42 | 1.22 | differs | raw med |Δ| 1.08; but pub ≈ 2.083·pnorm3 − 1.079 (Pearson 0.99994, residual med 0.004, max 0.098): pnorm-3 family, different scale/config |
| proposal-mDCTPSNR / mDCTpsnr-JND | site+dataset | `mdctpsnr` + the reference `-jnd` remap (not a CLI column) | 61 | 7.64e-08 | 1e-06 | 6.08e-07 | reproduced | JND = 0.000475997802·max(0, 80 − dB)^2.3182 applied to our dB |
| proposal-IDFIQA / IDFIQA | site+dataset | missing |  |  |  |  | missing | CNN features (Khakpour et al.); no licensed code |
| proposal-DVIFM-0.2 / DVIFM-0.2.0 | site+dataset | missing (dvifmish, outside zenmetrics) |  |  |  |  | missing | dvifmish (dev ~/work/dvifmish, publish=false) has port presets of the author's v0.2/v0.3 builds; not compared to these columns here |
| proposal-DVIFM-0.2-use_chroma / DVIFM-0.2.0-use_chroma | dataset | missing (dvifmish) |  |  |  |  | missing | dataset column (excluded on the site) |
| proposal-DVIFM-2d / DVIFM-2d | site | missing (dvifmish `port-2d-v0.2`, probably) |  |  |  |  | missing |  |
| proposal-DVIFM-min / DVIFM-min | site | missing (dvifmish `port-min-v0.2`, probably) |  |  |  |  | missing |  |
| proposal-DVIFM (dataset) | dataset | missing |  |  |  |  | missing | older version; ranks differently from -0.2 |
| proposal-DVIFM-min-v0.2 (excluded) | neither | missing in zenmetrics; dvifmish `port-min-v0.2` has the same name |  |  |  |  | missing | excluded; placeholder map (a=b=c=1), no values to compare |
| proposal-DVIFM-2d-v0.2 (excluded) | neither | missing in zenmetrics; dvifmish `port-2d-v0.2` has the same name |  |  |  |  | missing | excluded; placeholder map (a=b=c=1), no values to compare |
| proposal-DVIFM-lap-v0.2 (excluded) | neither | missing in zenmetrics; dvifmish `port-lap-v0.2` has the same name |  |  |  |  | missing | excluded; placeholder map (a=b=c=1), no values to compare |
| proposal-DVIFM-lap-v0.3 (excluded) | neither | missing in zenmetrics; dvifmish `port-lap-v0.3` has the same name |  |  |  |  | missing | excluded; placeholder map (a=b=c=1), no values to compare |

## Variants (no published values; parameter needed)

| name | in CSV? | zenmetrics metric | n | median \|Δ\| | max \|Δ\| | max \|Δ\| in JND | status | note |
|---|---|---|---|---|---|---|---|---|
| FSIM-k2 (pyiqa-FSIM-k2) | neither | `fsim` (`fsim_*`), auto factor only |  |  |  |  | variant-needed | = FSIM with a fixed pre-scoring downscale factor 2 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| FSIM-k3 (pyiqa-FSIM-k3) | neither | `fsim` (`fsim_*`), auto factor only |  |  |  |  | variant-needed | = FSIM with a fixed pre-scoring downscale factor 3 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. Its map equals the base FSIM map to 7 digits, i.e. the fitted values were these. |
| FSIM-k4 (pyiqa-FSIM-k4) | neither | `fsim` (`fsim_*`), auto factor only |  |  |  |  | variant-needed | = FSIM with a fixed pre-scoring downscale factor 4 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| FSIM-nodownscale (pyiqa-FSIM-nodownscale) | neither | `fsim` (`fsim_*`), auto factor only |  |  |  |  | variant-needed | = FSIM with a fixed pre-scoring downscale factor 1 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| FSIMc-k2 (pyiqa-FSIMc-k2) | neither | `fsim` (`fsimc_*`), auto factor only |  |  |  |  | variant-needed | = FSIMc with a fixed pre-scoring downscale factor 2 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| FSIMc-k3 (pyiqa-FSIMc-k3) | neither | `fsim` (`fsimc_*`), auto factor only |  |  |  |  | variant-needed | = FSIMc with a fixed pre-scoring downscale factor 3 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. Its map equals the base FSIMc map to 7 digits, i.e. the fitted values were these. |
| FSIMc-k4 (pyiqa-FSIMc-k4) | neither | `fsim` (`fsimc_*`), auto factor only |  |  |  |  | variant-needed | = FSIMc with a fixed pre-scoring downscale factor 4 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| FSIMc-nodownscale (pyiqa-FSIMc-nodownscale) | neither | `fsim` (`fsimc_*`), auto factor only |  |  |  |  | variant-needed | = FSIMc with a fixed pre-scoring downscale factor 1 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| VSI-k2 (pyiqa-VSI-k2) | neither | `vsi`, auto factor only |  |  |  |  | variant-needed | = VSI with a fixed pre-scoring downscale factor 2 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| VSI-k3 (pyiqa-VSI-k3) | neither | `vsi`, auto factor only |  |  |  |  | variant-needed | = VSI with a fixed pre-scoring downscale factor 3 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. Its map equals the base VSI map to 7 digits, i.e. the fitted values were these. |
| VSI-k4 (pyiqa-VSI-k4) | neither | `vsi`, auto factor only |  |  |  |  | variant-needed | = VSI with a fixed pre-scoring downscale factor 4 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| VSI-nodownscale (pyiqa-VSI-nodownscale) | neither | `vsi`, auto factor only |  |  |  |  | variant-needed | = VSI with a fixed pre-scoring downscale factor 1 instead of max(1, round(min(w,h)/256)); the crate computes the factor internally and exposes no override. |
| pyiqa-VSI-f32 | neither | `vsi` |  |  |  |  | variant-needed | not in site config; f32 = float input without u8 rounding (the f32 SSIM probes agree with this reading); no option exposed |
| SSIM-pyiqa-f32-k2 (pyiqa-SSIM-f32-k2) | neither | missing (no luma SSIM w/o downscale in CLI; no factor option) |  |  |  |  | variant-needed | = SSIM-pyiqa on unrounded luma after a 2×2 downscale. Test (variant_test.tsv): with the downscale, map_k2 of our numpy values sits within +0.01…+0.13 JND (median) of the base map; without it, +1.6…+10.5 |
| SSIM-pyiqa-f32-k3 (pyiqa-SSIM-f32-k3) | neither | missing (no luma SSIM w/o downscale in CLI; no factor option) |  |  |  |  | variant-needed | = SSIM-pyiqa on unrounded luma after a 3×3 downscale. Test (variant_test.tsv): with the downscale, map_k3 of our numpy values sits within +0.01…+0.13 JND (median) of the base map; without it, +1.6…+10.5 |
| SSIM-pyiqa-f32-k4 (pyiqa-SSIM-f32-k4) | neither | missing (no luma SSIM w/o downscale in CLI; no factor option) |  |  |  |  | variant-needed | = SSIM-pyiqa on unrounded luma after a 4×4 downscale. Test (variant_test.tsv): with the downscale, map_k4 of our numpy values sits within +0.01…+0.13 JND (median) of the base map; without it, +1.6…+10.5 |
| SSIM-pyiqa-f32-k5 (pyiqa-SSIM-f32-k5) | neither | missing (no luma SSIM w/o downscale in CLI; no factor option) |  |  |  |  | variant-needed | = SSIM-pyiqa on unrounded luma after a 5×5 downscale. Test (variant_test.tsv): with the downscale, map_k5 of our numpy values sits within +0.01…+0.13 JND (median) of the base map; without it, +1.6…+10.5 |
| SSIM-pyiqa-f32-k6 (pyiqa-SSIM-f32-k6) | neither | missing (no luma SSIM w/o downscale in CLI; no factor option) |  |  |  |  | variant-needed | = SSIM-pyiqa on unrounded luma after a 6×6 downscale. Test (variant_test.tsv): with the downscale, map_k6 of our numpy values sits within +0.01…+0.13 JND (median) of the base map; without it, +1.6…+10.5 |
| SSIMc-pyiqa-f32-k2 (pyiqa-SSIMc-f32-k2) | neither | missing (`ssim` has no factor option) |  |  |  |  | variant-needed | = per-RGB SSIM after a 2×2 downscale; same test: +0.02…+0.19 with, +1.3…+3.8 without |
| SSIMc-pyiqa-f32-k3 (pyiqa-SSIMc-f32-k3) | neither | missing (`ssim` has no factor option) |  |  |  |  | variant-needed | = per-RGB SSIM after a 3×3 downscale; same test: +0.02…+0.19 with, +1.3…+3.8 without |
| SSIMc-pyiqa-f32-k4 (pyiqa-SSIMc-f32-k4) | neither | missing (`ssim` has no factor option) |  |  |  |  | variant-needed | = per-RGB SSIM after a 4×4 downscale; same test: +0.02…+0.19 with, +1.3…+3.8 without |
| SSIMc-pyiqa-f32-k5 (pyiqa-SSIMc-f32-k5) | neither | missing (`ssim` has no factor option) |  |  |  |  | variant-needed | = per-RGB SSIM after a 5×5 downscale; same test: +0.02…+0.19 with, +1.3…+3.8 without |
| SSIMc-pyiqa-f32-k6 (pyiqa-SSIMc-f32-k6) | neither | missing (`ssim` has no factor option) |  |  |  |  | variant-needed | = per-RGB SSIM after a 6×6 downscale; same test: +0.02…+0.19 with, +1.3…+3.8 without |
| pyiqa-SSIM-f32, pyiqa-SSIMc-f32 | neither | missing |  |  |  |  | variant-needed | not in site config; unrounded-input SSIM without downscale |
| pyiqa-MS-SSIM-f32, -f32-k2…k6 | neither | `msssim` (no factor option) |  |  |  |  | variant-needed | not in site config; same downscale reading assumed |
| pyiqa-GMSD-materialized-y | neither | `gmsd` (probably) |  |  |  |  | unverified | not in site config. "materialized-y" reads as u8-rounded Y; our gmsd scores 8-bit BT.601 luma and matches the CSV GMSD to 5e-6, so the CSV column is likely this variant or indistinguishable from it |

## JND display columns (not metrics)

| name | in CSV? | zenmetrics metric | n | median \|Δ\| | max \|Δ\| | max \|Δ\| in JND | status | note |
|---|---|---|---|---|---|---|---|---|
| JND_CVVDP | site+dataset | fixed remap of the base column (docs/AIC2026_METRICS_AND_FITTING.md) |  |  |  |  | n/a | deterministic function of its metric; follows the base metric row |
| JND_SSIMULACRA2 | site+dataset | fixed remap of the base column (docs/AIC2026_METRICS_AND_FITTING.md) |  |  |  |  | n/a | deterministic function of its metric; follows the base metric row |
| JND_PSNR-Y | site+dataset | fixed remap of the base column (docs/AIC2026_METRICS_AND_FITTING.md) |  |  |  |  | n/a | deterministic function of its metric; follows the base metric row |
| JND_SSIM | site+dataset | fixed remap of the base column (docs/AIC2026_METRICS_AND_FITTING.md) |  |  |  |  | n/a | deterministic function of its metric; follows the base metric row |
| JND_MS-SSIM | site+dataset | fixed remap of the base column (docs/AIC2026_METRICS_AND_FITTING.md) |  |  |  |  | n/a | deterministic function of its metric; follows the base metric row |
| JND_PSNR-HVS-vmaf (= dataset JND_PSNR-HVS) | site | fixed remap of the base column (docs/AIC2026_METRICS_AND_FITTING.md) |  |  |  |  | n/a | deterministic function of its metric; follows the base metric row |
