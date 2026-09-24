# CVVDP SafeSyn fleet harvest — 2026-09-23

TRAIN-role synthetic pairs with metric labels only. This is a descriptive label sidecar, not a human-accuracy or model-selection result.

## Completion and provenance

- Run `cvvdp-safesyn-20260923`: **3,218 jobs**; 3,222 raw ledger rows, 3,218 latest `done` jobs, **3,218 blobs**. Missing ledger rows, missing blobs, failed/not-done jobs: **0 / 0 / 0**.
- **196,086 unique pair row IDs**, four nonnull metric columns, **784,344 score rows**. Error rows, missing pixel stamps, audit pixel mismatches, duplicate score disagreements: **0 / 0 / 0 / 0**.
- Source build: zenmetrics `9f36f88b8a23e645bd791c1193842d950eccf4be`; image `sha256:d110a3a79d98f06ade7c7dae3920c47b71c10362c959bd852341f6c5ff046050`. The full manifest contains all 13 sibling repo commits and dirty-diff hashes. Sidecar sha256 `775bdb8f95dbb3d116faaa646ec8dcafbb0810bc11dc30264997641e28701bc9`.
- Latest `done` jobs by worker tier: r5600g 709; r3500 1,080; tower 1,429.

## Independent keyed checks

- Fresh SSIM2 against the Sept. 14 `peer_ssim2.score`: 196,086/196,086 keyed rows checked; 70 differ; max |Δ| = 2.1316282072803006e-14. All raw SSIM2 values equal their harvested sidecar values bit for bit.
- Sept. 14 audit CVVDP 4K values available: **0**; cache CVVDP columns: **0**. The requested 4K-vs-Sept.-14 numeric check therefore has zero overlapping rows.

| Codec family | Worker tier | Rows | Nonexact SSIM2 | Max abs Δ |
|---|---|---:|---:|---:|
| `mozjpeg-rs-420-e4` | r3500 | 14,574 | 7 | 2.13e-14 |
| `mozjpeg-rs-420-e4` | r5600g | 9,546 | 1 | 1.78e-14 |
| `mozjpeg-rs-420-e4` | tower | 19,553 | 8 | 2.13e-14 |
| `zenavif-s5-e6` | r3500 | 11,467 | 3 | 1.78e-14 |
| `zenavif-s5-e6` | r5600g | 7,650 | 0 | 0 |
| `zenavif-s5-e6` | tower | 14,884 | 8 | 1.78e-14 |
| `zenjpeg-420-e2` | r3500 | 13,249 | 6 | 1.78e-14 |
| `zenjpeg-420-e2` | r5600g | 8,606 | 2 | 7.11e-15 |
| `zenjpeg-420-e2` | tower | 17,358 | 11 | 1.42e-14 |
| `zenjpeg-420-xyb-e2` | r3500 | 9,601 | 1 | 7.11e-15 |
| `zenjpeg-420-xyb-e2` | r5600g | 6,224 | 1 | 1.78e-14 |
| `zenjpeg-420-xyb-e2` | tower | 12,357 | 2 | 1.42e-14 |
| `zenjxl-e7` | r3500 | 8,783 | 4 | 1.78e-14 |
| `zenjxl-e7` | r5600g | 5,921 | 2 | 7.11e-15 |
| `zenjxl-e7` | tower | 11,658 | 3 | 1.42e-14 |
| `zenwebp-default-m4` | r3500 | 8,134 | 3 | 1.78e-14 |
| `zenwebp-default-m4` | r5600g | 5,561 | 3 | 1.42e-14 |
| `zenwebp-default-m4` | tower | 10,960 | 5 | 1.42e-14 |

## Descriptive rank agreement

SROCC uses `zen_stats.panel_batch_indexed` on the stored metric-label oracle. Its score is absolute; every displayed signed SROCC is positive. These are TRAIN metric-to-metric comparisons, not human validation.

| Metric | Overall SROCC vs original oracle | SROCC vs stored fresh SSIM2 |
|---|---:|---:|
| `cvvdp_jod_standard_4k` | 0.984479 | 0.984972 |
| `cvvdp_jod_standard_fhd` | 0.979473 | 0.979773 |
| `cvvdp_jod_sdr_fhd_24` | 0.981147 | 0.981433 |
| `ssim2_fresh` | 0.999679 | 1.000000 |

### By family

| Group | n | 4K | standard FHD | SDR FHD 24 | fresh SSIM2 |
|---|---:|---:|---:|---:|---:|
| `mozjpeg-rs-420-e4` | 43,673 | 0.988822 | 0.983555 | 0.985551 | 0.999963 |
| `zenavif-s5-e6` | 34,001 | 0.992397 | 0.989507 | 0.990542 | 0.999535 |
| `zenjpeg-420-e2` | 39,213 | 0.981209 | 0.965237 | 0.970484 | 0.999892 |
| `zenjpeg-420-xyb-e2` | 28,182 | 0.977967 | 0.968547 | 0.969771 | 0.999098 |
| `zenjxl-e7` | 26,362 | 0.988294 | 0.984227 | 0.984568 | 0.999864 |
| `zenwebp-default-m4` | 24,655 | 0.941296 | 0.950482 | 0.949170 | 0.999988 |

### By band

| Group | n | 4K | standard FHD | SDR FHD 24 | fresh SSIM2 |
|---|---:|---:|---:|---:|---:|
| `oracle_ge_m10` | 187,487 | 0.982322 | 0.976708 | 0.978588 | 0.999633 |
| `oracle_lt_m50` | 2,020 | 0.664738 | 0.473849 | 0.489185 | 0.999425 |
| `oracle_m25_to_m10` | 3,444 | 0.397134 | 0.302196 | 0.281220 | 0.999166 |
| `oracle_m50_to_m25` | 3,135 | 0.518690 | 0.295224 | 0.323896 | 0.999950 |

### Largest display disagreements

| row_id | family | 4K JOD | SDR FHD 24 JOD | abs Δ |
|---:|---|---:|---:|---:|
| 194861 | `zenwebp-default-m4` | 7.34392 | 9.57257 | 2.22865 |
| 12673 | `zenjpeg-420-xyb-e2` | 3.84160 | 5.61965 | 1.77805 |
| 183565 | `zenwebp-default-m4` | 6.83522 | 8.50259 | 1.66738 |
| 35758 | `zenjpeg-420-xyb-e2` | 2.91973 | 4.36776 | 1.44804 |
| 194862 | `zenwebp-default-m4` | 7.72583 | 9.15000 | 1.42417 |
| 183564 | `zenwebp-default-m4` | 8.03777 | 9.41223 | 1.37446 |
| 106918 | `zenjpeg-420-xyb-e2` | 4.23139 | 5.59737 | 1.36598 |
| 9696 | `zenjpeg-420-xyb-e2` | 5.22388 | 6.57397 | 1.35010 |
| 147172 | `zenjpeg-420-xyb-e2` | 5.22203 | 6.49912 | 1.27709 |
| 25209 | `zenavif-s5-e6` | 8.64190 | 7.36890 | 1.27300 |

Full row-key examples and all 48 canonical statistic results are in the adjacent JSON record. No scoring was rerun for this harvest.

## Worker teardown

- `zen-score-cvvdp` removed and confirmed absent on r3500 and tower.
- r5600g removal is unconfirmed: SSH had no route or timed out on four attempts; ping failed from dev and tower. The coordinator retains the host OS decision. No other host state was changed.
