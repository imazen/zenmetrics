# Data pointer — avif/jpeg knob-ablation first-cut (2026-06-28)

Large artifacts (>30 KB) live in block storage, NOT git. Report:
`benchmarks/avif_jpeg_knobablation_firstcut_2026-06-28.md`.

## Locations
- **Local (canonical):** `/mnt/v/output/avif-jpeg-knobablation-firstcut-2026-06-28/`
- **Tower mirror:** `/mnt/tower/output/avif-jpeg-knobablation-firstcut-2026-06-28/`
- **zentrain (R2):** PENDING — local + Tower hold two durable copies; upload to
  `s3://zentrain/knobablation/avif-jpeg-firstcut-2026-06-28/` is the remaining persist
  step (run dir is ~786 MB, mostly the persisted encoded variants).

## Contents (786 MB)
- `avif.tsv` (7,614 rows) / `jpeg.tsv` (10,904 rows) — per-cell:
  `image_path / codec / q / knob_tuple_json({cell,fp,plan}) / encoded_bytes / encode_ms /
  encoded_filename / decode_ms / score_zensim / score_ssim2`.
- `avif_enc/` + `jpeg_enc/` — 18,518 persisted encoded variants (content-named).
  Re-scoring any new metric needs NO re-encode (persist-everything).
- `avif_features.tsv` / `jpeg_features.tsv` — zenanalyze features at the swept sizes.
- `*.rdanalysis_score_{ssim2,zensim}.json` — per-axis verdicts (robust iso-q metric).
- `*.tail.json` — oracle-gap overhead tails.
- `picks16.json` / `sources.json` — corpus selection + rendition provenance.
- `*.plan.json` — sweep audit manifests.
- `REPORT.md` + `knobablation_firstcut_*.py` — report copy + analysis suite.

## Provenance
- Corpus: imazen-26 train origins (`origin_split`, even last-digit), k-means K=16 on
  `/mnt/v/output/imazen-26-features/imazen26_features_2026-06-23.parquet` (94 feat cols),
  Lanczos downscale to {256,512,768}.
- Sweep: `zenmetrics sweep --plan modes_full --max-deviations 1 --q-grid 30,45,60,75,85,92`,
  codecs zenavif + zenjpeg, metrics zensim + ssim2 (CPU).
- Binary: `zenmetrics-cli` `--features sweep,png,jpeg,webp,avif,cpu-metrics,gpu-ssim2`.
  codec-corpus RO; outputs are derived training/analysis data (zentrain-class).
