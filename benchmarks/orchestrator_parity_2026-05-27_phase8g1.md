# Orchestrator vs legacy parity sweep — 2026-05-27

Binary: `/home/lilith/work/zen/zenmetrics--phase8g1/target/release/zenmetrics`  
Total cells: 9  
PASS: 9  FAIL (value): 0  FAIL (column-name): 0  

| metric | size | q | legacy | orchestrator | abs_diff | tol | verdict |
|---|---|---|---|---|---|---|---|
| iwssim-gpu | 256 | 20 | 0.926276 | 0.926276 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 256 | 50 | 0.968235 | 0.968235 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 256 | 80 | 0.986055 | 0.986055 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 1024 | 20 | 0.858085 | 0.858085 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 1024 | 50 | 0.953837 | 0.953837 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 1024 | 80 | 0.983684 | 0.983684 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 4096 | 20 | 0.860366 | 0.860366 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 4096 | 50 | 0.951840 | 0.951840 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
| iwssim-gpu | 4096 | 80 | 0.983449 | 0.983449 | 0.000000e+00 | 5.000000e-05 | PASS-EXACT |
