# aic2026_full_repro_2026-10-20 — pointer

Assembled reproduction CSVs for the July 2026 AIC2026 published metric
columns, 9,618 rows × 70 columns each. The cropped CSV is also committed here, gzipped
(`gzip -n -9`, deterministic), as `metrics_cropped.zenmetrics.csv.gz`; its provenance is
`provenance_cropped.json`. `scripts/aic2026/repro_check.py` verifies both against the sha256
values below.

## Where

| where | path | state |
|---|---|---|
| local | `/mnt/v/output/zenmetrics/aic2026_reproduction/2026-10-20/` | cropped complete; fullres streaming |
| Tower | `/mnt/tower/output/zenmetrics/aic2026_reproduction/2026-10-20/` | mirrored at lane end |

## Files (sha256)

| file | bytes | sha256 |
|---|--:|---|
| `metrics_cropped.zenmetrics.csv` | 6138816 | `373377ac3aa68feedf2f045e24b4b857a6703a382de1b0039eed65a681698e5e` |
| `provenance.json` (committed as `provenance_cropped.json`) | 8202 | `547957b4cc3ac3dff8fe437cc34524178e10798646250e8b01c2fc982804e490` |
| `metrics_cropped-online.zenmetrics.csv` | — | pending last two metric runs |
| `metrics_fullres.zenmetrics.csv` | — | pending fullres pass |

`provenance.json` records the binary sha256, source commit `4d833195`,
the CLI metric name + options used for every published column, and the
exact batch commands.

## Reference table

Compared against the July/drop union (published scores only), cropped view: reference
`combined.csv` sha256 `91d7c2df86f6ac973288f511b1bb625890d8b4a2c35fbe8b3199d28c76836866`
(s3://zentrain/aic2026/csv/july-drop-2026-10-20/, https://zentrain.r2.imazen.org/aic2026/csv/july-drop-2026-10-20/combined.csv;
[reconciliation](../aic_csv_reconcile_2026-10-20/README.md)). Its July columns agree numerically with the July
`metrics_cropped.csv` this set was first compared with.

## Contents

Each CSV keeps the published header and row order verbatim, copies the
nine metadata columns unchanged, fills the computed metric columns with
zenmetrics values (uncomputed columns left empty), and recomputes every
`JND_<col>` from our raw value with the site-config mapping
`jnd(x) = a*|sign*(x-b)|^c`.
