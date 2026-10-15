# zenmetrics-nn against the published JPEG AIC-4 (2026) columns (2026-10-15)

61 pairs from the AIC-4 **cropped** set (`PTC_*`, ~840×944), one row in every 160 of
`metrics_fullres.csv` (all 16 codecs), matched to `metrics_cropped.csv`'s published
values. Data on dev: `/mnt/v/datasets/aic2026/` (`AIC2026-dataset-cropped.zip`,
`metrics_cropped.csv`); the subset and its published rows were built into dev
`~/tmp/aic-sub-crop/`. Scored on i265 with `zenmetrics batch --use-legacy-scheduler`,
engine ort-cuda (CPU parity of every model is in
[`nn_lpips_phase0_2026-10-15.md`](nn_lpips_phase0_2026-10-15.md)). Our scores: the
`*.tsv` files in [`nn_aic2026_crosscheck_2026-10-15/`](nn_aic2026_crosscheck_2026-10-15/).

| our column | AIC-4 column | n | median \|Δ\| | max \|Δ\| |
|---|---|---|---|---|
| `lpips_alex` | `LPIPS-alex` | 61 | 2.3e-07 | 4.9e-07 |
| `lpips_vgg` | `LPIPS-VGG` | 61 | 3.1e-06 | 1.0e-04 |
| `dists` | `DISTS` | 61 | 2.8e-06 | 2.5e-05 |
| `stlpips_alex` | `STLPIPS` | 61 | 3.8e-06 | 9.3e-05 |
| `stlpips_vgg` | `STLPIPS-VGG` | 61 | 3.4e-06 | 6.0e-05 |
| `deepdc` | `DeepDC` | 61 | 1.2e-06 | 3.0e-05 |

- All six reproduce the published columns: AIC-4 computed them as the upstream packages
  do, at native resolution.
- **DeepDC needs ImageNet input normalisation.** With `DeepDC_PyTorch`'s default (no
  normalisation) our values were ~0.63× the column (median |Δ| 1.1e-2); with upstream's
  own `use_input_norm=True` they match (8-pair probe: `deepdc_variants2.py`; resizing
  to 224, upstream's demo default, does not, `deepdc_variants.py`). The upstream source
  comments "Correlation: use_norm = True", so `deepdc` is now defined with it.
- Full-resolution AIC-4 images (~8 MP) do not fit the 16 GB RTX 4080 on ort-cuda for the
  VGG-based models (lpips_vgg, dists, stlpips_vgg, deepdc: allocator failure); lpips_alex
  and stlpips_alex ran. The CLI reports the error instead of switching engines.
