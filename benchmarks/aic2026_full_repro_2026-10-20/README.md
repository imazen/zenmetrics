# AIC2026 full-set reproduction — cropped set, fast tier (2026-10-20)

All 9,618 distorted images of the July 2026 AIC2026 cropped set (840×944)
scored with zenmetrics and compared against the published oracle columns.
Same binary, same routes throughout; per-metric scores and the assembled
`metrics_cropped.zenmetrics.csv` (identical header + row order, nine
metadata columns copied, `JND_*` recomputed from our raw values via the
site-config mapping) live outside git — see `metrics.pointer.md`.

Binary: `zenmetrics` (zenmetrics-cli) v0.6.0 built from the CI-pinned
sibling snapshot of master `4d833195` + the `aic_vmaf_features` example
(ADM2 / VIF_vmaf columns, which `batch` does not expose).

Run: `zenmetrics batch --metric <m> --pairs pairs.tsv --output <m>.tsv
--jobs 16 --group-by-ref` (cvvdp adds `--display-model standard_fhd`).

## Result: 29 of 35 computed columns inside the 1e-3 median site-JND gate

`compare_cropped.tsv` has the full per-column table (n, exact-equal count
at published precision, median/max |Δraw|, median/max |Δ site-JND|,
Pearson, Spearman). JND deltas map the site-config fit onto BOTH our value
and the published raw value; the published `JND_*` columns use different
fits, so they are compared informationally only.

Exact-level columns (median |Δ| at float-representation scale):
`SSIMULACRA2` 2.5e-7, `FLIP` 2.5e-7, `mDCT-PSNR` 5e-7, `proposal-mDCTPSNR`
(derived) 7.8e-8, `SSIMc` 2.6e-7, `SSIM-pyiqa` 4.7e-7, `MS-SSIM-pyiqa`
2.8e-7, `CW-SSIM` 4.3e-7, `GMSD` 4.4e-7, `DSSIM` 2.6e-7.

## The six misses, diagnosed

Every miss is a small raw residual amplified by a near-vertical site-JND
curve, or a documented variant difference — none are route/option errors.
Worst-five rows per column are in `compare_cropped.worst.txt` next to the
scores (not committed; per-image values stay out of git).

| column | route | median \|dJND\| | diagnosis |
|---|---|---:|---|
| `VMAF` | `vmaf-studio601` | 0.00107 | Borderline (gate 1e-3). Raw median 6.5e-3/100, Pearson 1.00000 — libvmaf-version-level noise; worst rows are top PNG-ladder rungs |
| `NLPD` | `nlpd-iqa` | 0.0355 | Documented residual: `docs/AIC2026_REPRODUCTION_STATUS.md` records "median 2.6e-3, 2.6% relative (f32 order and bilinear edges)". Our 2.3e-3 matches that scale |
| `FSIM` | `fsim-y` | 0.0105 | Raw median 4.7e-5; site JND slope ≈ −172/unit at score 0.99 where the column clusters. All diffs at >0.96, ours systematically higher; doc noted "one JPEG tail pair" on the earlier probe |
| `FSIMc` | `fsim` | 0.0098 | Same mechanism, shared phase-congruency core |
| `VSI` | `vsi` | 0.0074 | Raw median 1.5e-5; slope ≈ −270/unit at 0.99. Previously verified "reproduced" at ≤3.4e-6 residual; over gate purely from amplification |
| `proposal-Butteraugli` | `butteraugli` (pnorm3) | 0.374 | Authors' own variant, not stock butteraugli: Pearson 0.99996 vs our pnorm3 but the pub/p3 ratio varies 0.005–2.03 and the log-log exponent is codec-dependent (PNG ≈1.18, JXL ≈2.41, Cool-Chic ≈2.53). Column map already flags it as family-level match only |

## Timing (dev, nice-19, 16 jobs per metric batch)

Fast metrics 20–200 s each for 9,618 pairs; heavy: `cw-ssim` 28 min,
`ciede2000` 18 min, `flip-v17` ~60 min, `butteraugli` ~8 min at 12 jobs,
`mdctpsnr` ~50 min. `aic_vmaf_features` (all three VMAF models + feature
dump) 30 s for the full set.

Full-resolution pass streams the 23 GB archive one source at a time
(extract → score → delete); see `compare_fullres.tsv` when complete.
