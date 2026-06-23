# Picker-pipeline artifacts — 2026-06-22 (pointer)

Large artifacts (>30 KB) live on local block storage, not git. See
`picker_pipeline_2026-06-22.md` for methodology + results.

## Location: `/home/lilith/picker-pp/`

### `models/` — 8 ZNPR pickers (~200 KB each, f16, roundtrip-verified)
- `{zenjpeg,zenjxl,zenwebp,zenavif}_predict_ssim2_v0.1.{bin,json}`
- `{zenjpeg,zenjxl,zenwebp,zenavif}_predict_zensim_a_v0.1.{bin,json}`

### `train/` — per-codec trainer inputs
- `<codec>.{ssim2,zensim_a}.pareto.parquet`
- `<codec>.features.tsv` (110 `feat_*`, from `imazen26_train_features_2026-06-22.tsv`)

### `enc/` — persisted encoded variants (3.9 GB, content of each sweep cell)
- `zenjpeg/` 34188 files (2.7 G), `zenjxl/` 8778 (687 M),
  `zenwebp/` 2688 (407 M), `zenavif/` 11262 (72 M)
- Filename scheme `<stem>_<srchash>_<codec>_q<q>_<knobhash>.<ext>` — re-scoreable
  by any future metric without re-encoding.

## Provenance
- Built from zenmetrics master `8c373d54`+ (sweep CPU-metric deadlock fix).
- Features: zenanalyze latest (`imazen26_train_features_2026-06-22.tsv`).
- Not yet mirrored to R2/Tower — local first-cut. Mirror before any cleanup.
