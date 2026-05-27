# Orchestrator vs legacy parity sweep — 2026-05-27

Binary: `/home/lilith/work/zen/zenmetrics--orch-phase771/target/release/zen-metrics`  
Total cells: 54  
PASS: 45  FAIL (value): 9  FAIL (column-name): 0  

| metric | size | q | legacy | orchestrator | abs_diff | tol | verdict |
|---|---|---|---|---|---|---|---|
| cvvdp | 256 | 20 | 9.296264 | 9.296264 | 0.000000e+00 | 1.000000e-03 | PASS-EXACT |
| cvvdp | 256 | 50 | 9.695044 | 9.695044 | 0.000000e+00 | 1.000000e-03 | PASS-EXACT |
| cvvdp | 256 | 80 | 9.831320 | 9.831320 | 0.000000e+00 | 1.000000e-03 | PASS-EXACT |
| cvvdp | 1024 | 20 | 8.741311 | 8.741311 | 0.000000e+00 | 1.000000e-03 | PASS-EXACT |
| cvvdp | 1024 | 50 | 9.602884 | 9.602884 | 0.000000e+00 | 1.000000e-03 | PASS-EXACT |
| cvvdp | 1024 | 80 | 9.828098 | 9.828098 | 0.000000e+00 | 1.000000e-03 | PASS-EXACT |
| cvvdp | 4096 | 20 | 8.824126 | 8.824267 | 1.411438e-04 | 1.000000e-03 | PASS |
| cvvdp | 4096 | 50 | 9.576204 | 9.576591 | 3.862381e-04 | 1.000000e-03 | PASS |
| cvvdp | 4096 | 80 | 9.825951 | 9.826287 | 3.366470e-04 | 1.000000e-03 | PASS |
| ssim2-gpu | 256 | 20 | 50.369539 | 50.369539 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 256 | 50 | 67.564850 | 67.564850 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 256 | 80 | 74.543556 | 74.543556 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 1024 | 20 | 16.577593 | 16.577593 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 1024 | 50 | 52.221685 | 52.221685 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 1024 | 80 | 66.478549 | 66.478549 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 4096 | 20 | 2.738577 | 2.738577 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 4096 | 50 | 42.531040 | 42.531040 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| ssim2-gpu | 4096 | 80 | 61.722500 | 61.722500 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| dssim-gpu | 256 | 20 | 0.006545 | 0.006545 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 256 | 50 | 0.002722 | 0.002722 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 256 | 80 | 0.001532 | 0.001532 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 1024 | 20 | 0.012714 | 0.012714 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 1024 | 50 | 0.004883 | 0.004883 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 1024 | 80 | 0.002543 | 0.002543 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 4096 | 20 | 0.014045 | 0.014045 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 4096 | 50 | 0.006032 | 0.006032 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| dssim-gpu | 4096 | 80 | 0.002970 | 0.002970 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| butteraugli-gpu | 256 | 20 | 7.290324 | 6.293411 | 9.969134e-01 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 256 | 50 | 3.207370 | 3.030164 | 1.772058e-01 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 256 | 80 | 2.765682 | 2.731609 | 3.407311e-02 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 1024 | 20 | 6.274173 | 5.149696 | 1.124477e+00 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 1024 | 50 | 3.533831 | 3.118569 | 4.152613e-01 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 1024 | 80 | 2.878555 | 2.766140 | 1.124144e-01 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 4096 | 20 | 5.670236 | 4.964922 | 7.053137e-01 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 4096 | 50 | 3.746134 | 3.387536 | 3.585978e-01 | 5.000000e-04 | FAIL |
| butteraugli-gpu | 4096 | 80 | 3.206090 | 3.213450 | 7.359743e-03 | 5.000000e-04 | FAIL |
| iwssim-gpu | 256 | 20 | 0.926276 | 0.926276 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 256 | 50 | 0.968235 | 0.968235 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 256 | 80 | 0.986055 | 0.986055 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 1024 | 20 | 0.858085 | 0.858085 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 1024 | 50 | 0.953837 | 0.953837 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 1024 | 80 | 0.983684 | 0.983684 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 4096 | 20 | 0.860366 | 0.860366 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 4096 | 50 | 0.951840 | 0.951840 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 4096 | 80 | 0.983449 | 0.983449 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| zensim-gpu | 256 | 20 | 49.207358 | 49.207358 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 256 | 50 | 64.038768 | 64.038768 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 256 | 80 | 74.299704 | 74.299704 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 1024 | 20 | 23.997321 | 23.997321 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 1024 | 50 | 50.256124 | 50.256124 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 1024 | 80 | 63.856136 | 63.856136 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 4096 | 20 | 13.983181 | 13.983181 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 4096 | 50 | 37.642540 | 37.642540 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |
| zensim-gpu | 4096 | 80 | 53.881700 | 53.881700 | 0.000000e+00 | 5.000000e-03 | PASS-EXACT |

## Failures

- **butteraugli-gpu size=256 q=20** (value): legacy=7.290324 orch=6.293411 diff=9.969134e-01 > tol=5.000000e-04. Notes: pnorm3 diverged by 4.817867e-01
- **butteraugli-gpu size=256 q=50** (value): legacy=3.207370 orch=3.030164 diff=1.772058e-01 > tol=5.000000e-04. Notes: pnorm3 diverged by 2.108327e-01
- **butteraugli-gpu size=256 q=80** (value): legacy=2.765682 orch=2.731609 diff=3.407311e-02 > tol=5.000000e-04. Notes: pnorm3 diverged by 1.104569e-01
- **butteraugli-gpu size=1024 q=20** (value): legacy=6.274173 orch=5.149696 diff=1.124477e+00 > tol=5.000000e-04. Notes: pnorm3 diverged by 4.710798e-01
- **butteraugli-gpu size=1024 q=50** (value): legacy=3.533831 orch=3.118569 diff=4.152613e-01 > tol=5.000000e-04. Notes: pnorm3 diverged by 2.195297e-01
- **butteraugli-gpu size=1024 q=80** (value): legacy=2.878555 orch=2.766140 diff=1.124144e-01 > tol=5.000000e-04. Notes: pnorm3 diverged by 1.156325e-01
- **butteraugli-gpu size=4096 q=20** (value): legacy=5.670236 orch=4.964922 diff=7.053137e-01 > tol=5.000000e-04. Notes: pnorm3 diverged by 4.342189e-01
- **butteraugli-gpu size=4096 q=50** (value): legacy=3.746134 orch=3.387536 diff=3.585978e-01 > tol=5.000000e-04. Notes: pnorm3 diverged by 2.295463e-01
- **butteraugli-gpu size=4096 q=80** (value): legacy=3.206090 orch=3.213450 diff=7.359743e-03 > tol=5.000000e-04. Notes: pnorm3 diverged by 1.209601e-01
