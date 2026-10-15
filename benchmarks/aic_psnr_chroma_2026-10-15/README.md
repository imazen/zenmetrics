# AIC-4 PSNR-Cb / PSNR-Cr convention search (2026-10-15)

61 JPEG AIC-4 full-resolution pairs (`~/tmp/aic-sub/pairs.tsv` on i265) against the
published `PSNR-Cb`, `PSNR-Cr` and `PSNR-YCbCr611` columns.

- `probe.py`: 4:4:4 vs 4:2:0, rounded vs float chroma ([probe.log](probe.log)). Full-resolution
  rounded studio-swing BT.601 is the only family within 0.1 dB; every 4:2:0 variant is off by 1.5–5 dB.
- `probe2.py`: coefficient precision (exact Kr/Kb, 3-decimal), f32, ties-to-even, truncation
  and 8–16-bit fixed point ([probe2.log](probe2.log)). Nothing beats the plain rounded float
  conversion; the residual (median 2e-3 dB, either sign) does not move.
- Two rows in the logs are probe bugs, not candidate conventions: `sws444` in `probe.log` and
  the `fixed*` Cr column in `probe2.log` (errors of 30–160 dB). The `fixed*` Cb column is valid.
- PSNR-Y on the same conversion is within 1e-4 dB on most pairs, so the pixels agree. The chroma
  residual probably comes from the authors' pipeline (inferred, not confirmed: chroma taken from
  the codecs' own YCbCr output instead of the RGB PNGs).
- [cli_compare.txt](cli_compare.txt): the `zenmetrics` CLI (`psnr-cb`, `psnr-cr`, `psnr-ycbcr611`)
  against the columns, in dB and in JND units via the site's `a·max(0, b − x)^c` mappings.

| metric | median \|Δ\| dB | max \|Δ\| dB | median \|Δ\| JND | max \|Δ\| JND |
|---|---|---|---|---|
| psnr-cb | 1.9e-3 | 1.7e-2 | 2.2e-4 | 2.0e-3 |
| psnr-cr | 2.1e-3 | 5.7e-2 | 3.3e-4 | 8.8e-3 |
| psnr-ycbcr611 | 3.1e-4 | 3.5e-3 | 5.2e-5 | 5.3e-4 |

Scripts read `~/tmp/aic-sub/metrics_subset.csv` and `~/tmp/aic-site/config.json` (site API
snapshot archived on dev at `/mnt/v/datasets/aic2026/site-api-2026-10-15/`).
