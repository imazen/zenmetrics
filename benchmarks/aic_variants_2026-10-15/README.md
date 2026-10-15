# AIC2026 fixed-factor and SSIM-convention validation

This panel implements the supplied AIC column conventions from the original
SSIM/MS-SSIM papers and the existing FSIM/VSI ports. No pyiqa code or package
was read. Sources: [dataset](https://doi.org/10.18419/DARUS-6156),
[site mapping configuration](https://merkur242.inf.uni-konstanz.de/aic2026/api/metric-eval/config),
[SSIM paper](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf),
[MS-SSIM paper](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf).

`fsim_summary.tsv` and `vsi_summary.tsv` report the named mapping's signed
median offset and maximum absolute offset against `JND_CVVDP`, including the
base metric. Each row covers 61 pairs. These observations are not tuning data.
The matching `.meta` files record commands, source and resource measurements.

Fixed SSIM factors use complete N×N block averages from the top-left, retaining
every N-th valid box-filter sample and discarding incomplete bottom/right blocks.
This matches the audit's box hypothesis; the published variant maps do not
establish the site's exact downscale kernel or boundary convention.

Five-scale MS-SSIM requires every scale to contain an 11×11 valid window.
Odd-size pooling adds a leading zero row or column on odd axes and divides
each 2×2 sum by four. Trailing bottom/right padding is never sampled. Inputs whose short side after
fixed downscaling is below 161 are rejected explicitly (owner-approved).
`panel.py --expect-small-ms-errors` tests those refusals, then scores the supported
subset and reports both counts. It never substitutes fewer scales.

Recipes in the local `justfile` serialize tests/builds/scoring and use the
CI-pinned snapshot. From this directory, `just cargo test -p fsim -p vsi` runs the
crate tests; `just panel <output-dir> <metric>...` runs a named scoring panel.
The two published SSIM base gates enforce maximum absolute difference ≤1e-5.

`ssim_gates.tsv` retains both 61-pair published comparisons and ten independent
audit-probe comparisons of SSIM/SSIMc fixed-factor scores. Max published errors
are 3.78e-6 (SSIM) and 1.88e-6 (MS-SSIM); the odd S53 pair differs by 2.35e-7.
`ssim_summary.tsv` includes the published RGB base beside the variants. MS-SSIM
k6 scores 27 pairs and rejects 34; the other nineteen names score all 61 pairs.
Missing site maps are recorded explicitly rather than replaced by another map.


## Round 2: rounded luma ties

Rounded luma retains **ties-to-even**. Fresh 61-pair runs differ at the CLI's
six-decimal precision; ties-to-even has lower mean absolute error for both
published columns. Half-away therefore does not qualify for the equal-output
fallback. These errors use the published values without refitting anything.

| Metric | Tie rule | Median absolute error | Maximum absolute error | Mean absolute error |
|---|---|---:|---:|---:|
| SSIM-pyiqa | ties-to-even | 4.12384e-7 | 3.77692e-6 | 6.78512e-7 |
| SSIM-pyiqa | half-away | 3.69388e-7 | 9.22308e-6 | 7.68571e-7 |
| MS-SSIM-pyiqa | ties-to-even | 2.90910e-7 | 1.88279e-6 | 3.30971e-7 |
| MS-SSIM-pyiqa | half-away | 3.16957e-7 | 1.69273e-6 | 3.72118e-7 |

Exact measurements and the comparison criterion are in `round2_rounding.tsv`
and `round2_rounding.json`; `rounding.py` reproduces the comparison from saved
score tables. Both rules satisfy the existing 1e-5 base gates.
