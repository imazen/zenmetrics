# aic2026_rescore_full_2026-10-20 — pointer

Full-precision rescore of the AIC2026 published metric columns by
`scripts/complexity/aic_rescore.py` (`just aic-rescore`), per the 2026-10-20
addendum: one reference CSV, one scorer, full-precision output. Scores are
written per-pair into resumable checkpoint cells, then streamed into
`metrics_<view>.zenmetrics.csv` (committed gzipped, `gzip -n -9`, deterministic).
`scripts/aic2026/repro_check.py --reference <combined.csv>` verifies everything
against the sha256 values below.

## Where

| where | path | state |
|---|---|---|
| local | `/mnt/v/output/zenmetrics/aic2026_rescore/2026-10-20/` | cropped complete (9,618 pairs, 0 failures) |
| fullres images | `/mnt/tower/input/aic2026/extracted-fullres/` | extracted; campaign pending |
| Tower | `/mnt/tower/output/zenmetrics/aic2026_rescore/2026-10-20/` | mirrored at lane end |

## Files (sha256)

| file | bytes | sha256 |
|---|--:|---|
| `metrics_cropped.zenmetrics.csv` | 12354495 | `7daf6fc55d53265c3a1fe8761ab7a13b0182accf2ca28c52183ed7a58d77428e` |
| `metrics_cropped.zenmetrics.csv.gz` | 5284908 | `d2cb1ac3fa6c561747f9a09fcfaaba6f003733a1c129956440123b49756b14bb` |
| `plan_cropped.json` | 4190492 | `776c37002d91a5511a562c5a553461459d6e8c0794e7312fec4069839a5565a0` |
| `provenance_cropped.json` | 158 | `942831a558f26c3bb241fd13e5893d6331bb0a99ece04f74543581351300f2ae` |
| `compare_cropped.tsv` | 5087 | `638036e864a625bfd8594ae82457da1cb5fffbd8cbef5a31d12b26020e64741e` |
| `metrics_fullres.zenmetrics.csv` | — | pending fullres campaign |

`provenance_cropped.json` is the scorer's `summary.json`
(500,136 scored + 76,944 derived cells, `hashes_verified`, `complete`).
`plan_cropped.json` records the reference sha256, every per-image sha256 it
scored, and the routed CLI/options per published column.

Build: `just snap-cargo @ build --release -p zenmetrics-cli --bin zenmetrics
--example aic_vmaf_features --features nn,pyiqa-ports` at source commit
`8e4646f6aa4c` → binary sha256
`8dee74bbd6af4e51d10c399c8fdad200cc0b3732da9d4b59fa5a1d72cf255b7e`
(`zenmetrics`), `70f7c712403cc78b87ebd7916ef2aef727d6c9683b317e353cb20c6fc8fc1c44`
(`examples/aic_vmaf_features`), zenmetrics-cli 0.6.0.

## Reference table

Compared against the July/drop union (published scores only): reference
`combined.csv` sha256 `91d7c2df86f6ac973288f511b1bb625890d8b4a2c35fbe8b3199d28c76836866`
(s3://zentrain/aic2026/csv/july-drop-2026-10-20/, https://zentrain.r2.imazen.org/aic2026/csv/july-drop-2026-10-20/combined.csv;
[reconciliation](../aic_csv_reconcile_2026-10-20/README.md)). `--view cropped|fullres`
selects the rows; `ptc_subset` marks the PTC rows.

## Contents

The CSV keeps the published header and row order verbatim, copies the nine
metadata columns unchanged, fills every routed metric column with zenmetrics
values at full (repr) precision, and derives each `JND_<col>` from our raw
value with the site-config mapping `jnd(x) = a*|sign*(x-b)|^c` — columns
without a public route stay empty. `compare_<view>.tsv` comes from
`scripts/aic2026/july_diff.py --view <view> --plan plan_<view>.json`.

## Cropped verdicts (n = 9,618)

37 MATCH / 15 MISS / 8 info-only. All 15 misses are the same residual
families recorded in `docs/AIC2026_REPRODUCTION_STATUS.md`: VMAF-model
variants (`VMAF`, `VMAF v0`, `VMAF-neg v0`, `VMAF v1`), `VIF-vmaf-scale0..3`
(libvmaf feature internals), `HDR_VDP_2` (third-implementation residual, see
`hdrvdp2-investigation`), `HDR_VDP_3` (1.6e-3 Q), `NLPD`, `FSIM`, `FSIMc`,
`VSI`, `WS-HV`. `FLIP` reproduces all 9,618 published values exactly;
`CW-SSIM` med |d| = 5.6e-16 (machine epsilon).
