# aic2026_full_repro_2026-10-20 — pointer

Assembled reproduction CSVs for the July 2026 AIC2026 published metric
columns. Not in git: each CSV is 9,618 rows × 70 columns.

## Where

| where | path | state |
|---|---|---|
| local | `/mnt/v/output/zenmetrics/aic2026_reproduction/2026-10-20/` | cropped complete; fullres streaming |
| Tower | `/mnt/tower/output/zenmetrics/aic2026_reproduction/2026-10-20/` | mirrored at lane end |

## Files (sha256)

| file | bytes | sha256 |
|---|--:|---|
| `metrics_cropped.zenmetrics.csv` | 5614750 | `289e1a5c291d32549b802c18d9eadbfb2df351b445649459298eb1e66a455236` |
| `provenance.json` | 6941 | `76e0fa7c6a6a488d00231477f36f458eb6842a18e5235df75aae22cd357f8b10` |
| `metrics_fullres.zenmetrics.csv` | — | pending fullres pass |

`provenance.json` records the binary sha256, source commit `4d833195`,
the CLI metric name + options used for every published column, and the
exact batch commands.

## Contents

Each CSV keeps the published header and row order verbatim, copies the
nine metadata columns unchanged, fills the computed metric columns with
zenmetrics values (uncomputed columns left empty), and recomputes every
`JND_<col>` from our raw value with the site-config mapping
`jnd(x) = a*|sign*(x-b)|^c`.
