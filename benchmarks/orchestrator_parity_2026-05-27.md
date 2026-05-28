# Orchestrator vs legacy parity sweep — 2026-05-27

Binary: `/home/lilith/work/zen/zenmetrics--phase8c1-b/target/release/zen-metrics`  
Total cells: 54  
PASS: 45  FAIL (value): 9  FAIL (column-name): 0  

| metric | size | q | legacy | orchestrator | abs_diff | tol | verdict |
|---|---|---|---|---|---|---|---|
| cvvdp | 256 | 20 |  | 9.296264 | inf | 1.000000e-03 | FAIL |
| cvvdp | 256 | 50 |  | 9.695044 | inf | 1.000000e-03 | FAIL |
| cvvdp | 256 | 80 |  | 9.831320 | inf | 1.000000e-03 | FAIL |
| cvvdp | 1024 | 20 |  | 8.741311 | inf | 1.000000e-03 | FAIL |
| cvvdp | 1024 | 50 |  | 9.602884 | inf | 1.000000e-03 | FAIL |
| cvvdp | 1024 | 80 |  | 9.828098 | inf | 1.000000e-03 | FAIL |
| cvvdp | 4096 | 20 |  | 8.824267 | inf | 1.000000e-03 | FAIL |
| cvvdp | 4096 | 50 |  | 9.576591 | inf | 1.000000e-03 | FAIL |
| cvvdp | 4096 | 80 |  | 9.826287 | inf | 1.000000e-03 | FAIL |
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
| butteraugli-gpu | 256 | 20 | 7.290324 | 7.290324 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 256 | 50 | 3.207370 | 3.207370 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 256 | 80 | 2.765682 | 2.765682 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 1024 | 20 | 6.274173 | 6.274173 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 1024 | 50 | 3.533831 | 3.533831 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 1024 | 80 | 2.878555 | 2.878555 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 4096 | 20 | 5.670236 | 5.670236 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 4096 | 50 | 3.746134 | 3.746134 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
| butteraugli-gpu | 4096 | 80 | 3.206090 | 3.206090 | 0.000000e+00 | 5.000000e-04 | PASS-EXACT |
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

- **cvvdp size=256 q=20** (value): legacy= orch=9.296264 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=256 q=50** (value): legacy= orch=9.695044 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=256 q=80** (value): legacy= orch=9.831320 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=1024 q=20** (value): legacy= orch=8.741311 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=1024 q=50** (value): legacy= orch=9.602884 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=1024 q=80** (value): legacy= orch=9.828098 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=4096 q=20** (value): legacy= orch=8.824267 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=4096 q=50** (value): legacy= orch=9.576591 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

- **cvvdp size=4096 q=80** (value): legacy= orch=9.826287 diff=inf > tol=1.000000e-03. Notes: legacy_err= Ryzen 9 7950X 16-Core Processor cache=/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml
error: orchestrator: chooser: no feasible backend (considered 4 candidates) (backends tried: [])

