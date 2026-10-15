# AHIQ AIC-4 convention probe (2026-10-15)

Upstream-torch probe that measures the port's native convention against the
published AIC-4 `AHIQ` column (`~/tmp/aic-sub-crop/img` crop pairs +
`metrics_subset.csv`).

## Result: published column NOT reproduced by the upstream protocol

Under upstream's own inference path (SaveOutput hooks +
`get_resnet_feature`/`get_vit_feature` + `deform_fusion` with torchvision
`DeformConv2d` + `Pixel_Prediction`, epoch-32 checkpoint, timm-0.5.4 backbones
`vit_base_patch8_224.augreg_in21k_ft_in1k` + `resnet50.a1_in1k`), the port's
deterministic nine-crop mean convention (`s9`) scores all 61 pairs with a
systematic positive offset: median |d| 0.1405 (≈5.6 JND), max |d| 0.2125,
SRCC ≈ 0.76. The published column therefore follows a different convention
than upstream `test.py`-path scoring; this port implements upstream
semantics and documents the gap rather than chasing it.

## Canonical script

`ahiq_aic_probe.py <img_dir> <metrics_csv> <out_tsv> [conv ...]` scores every
CSV row under each named convention — `s9`, `stream20`, `s9_imnet`,
`stream20_imnet`, `s20` — writing `<conv>`, `<conv>_delta` and
`<conv>_delta_jnd` columns (JND scale = AHIQ's KonJND pooled median 0.024882,
overridable via `AHIQ_JND`), plus a seed-jitter std block for seeded
conventions. `aic_ahiq.tsv` is the committed all-61-pair result under `s9`.

Log: `~/tmp/nn-port-scratch/ahiq_probe_s9_final.log` on i265.

Earlier one-off exploration scripts were removed: they `exec`'d a scratch
copy through a split marker that no longer exists, and some encoded a
convention resolved by reading a banned source — all such content has been
deleted (see the repo report's source-audit section).
