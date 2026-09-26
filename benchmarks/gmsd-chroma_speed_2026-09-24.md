# gmsd-chroma speed record with the clean-room MDSI — 2026-09-25

Label: **CONTENDED**. Measured on the dev workstation's **AVX-512 tier** (`avx512` feature build, native dispatch reaches v4) while an unrelated training job shared the box (peak load 12), so the numbers are indicative, not clean. Descriptive only; not evidence for adoption and not a gate. Implementation commit `19b8c03c87934b1df2e078aa689534d17777713c`. This supersedes the earlier record, which measured the previous MDSI implementation on an AVX2 worker.

zenbench interleaved run of plain `gmsd`, `mdsi` and `ms_gmsdc` at 64², 256², 1024², 4096², 1 and 8 threads (24 cells). Raw: `/var/tmp/gmsd-chroma/speed/{zenbench,report}.json`; recompute: `python3 scripts/gmsd-chroma/recompute_headlines.py speed`. Fits are `total = α + β·pixels` per arm and thread count over the four sizes. MS-GMSDc's negative α indicates fit instability under contention. MDSI's β is small because the averaged planes stay about 256 samples on the short side at every size; its α (about 1 ms single-threaded) is the fixed work on that plane.

| arm | threads | α (ns) | β (ns/pixel) |
|---|---:|---:|---:|
| gmsd | 1 | 38260.2 | 0.7464 |
| gmsd | 8 | 39190.8 | 0.1483 |
| mdsi | 1 | 1019490.1 | 0.3230 |
| mdsi | 8 | 503329.2 | 0.1881 |
| ms_gmsdc | 1 | -1932580.8 | 24.2634 |
| ms_gmsdc | 8 | -1379263.8 | 21.0272 |

## Optimized path against the straight-line equations

`mdsi_vs_reference` asserts bit-identical scores, then reports best-of-N wall time (3 reps at 4096², 8 otherwise; reference single-threaded, optimized at the stated thread count). Raw: `/var/tmp/gmsd-chroma/mdsi_vs_reference.tsv`.

| size | threads | straight-line reference (ms) | optimized (ms) | speedup | plain GMSD (ms) |
|---:|---:|---:|---:|---:|---:|
| 64² | 1 | 0.245 | 0.080 | 3.06× | 0.007 |
| 64² | 8 | 0.248 | 0.080 | 3.11× | 0.007 |
| 256² | 1 | 4.829 | 1.469 | 3.29× | 0.062 |
| 256² | 8 | 4.938 | 0.483 | 10.23× | 0.031 |
| 1024² | 1 | 8.830 | 1.940 | 4.55× | 0.747 |
| 1024² | 8 | 8.823 | 0.743 | 11.87× | 0.168 |
| 4096² | 1 | 71.140 | 4.989 | 14.26× | 11.927 |
| 4096² | 8 | 71.927 | 2.201 | 32.67× | 2.469 |

## Cells

| arm | threads | size | mean ns | rounds |
|---|---:|---:|---:|---:|
| gmsd | 1 | 64² | 11076.0 | 30 |
| mdsi | 1 | 64² | 92873.9 | 30 |
| ms_gmsdc | 1 | 64² | 41173.6 | 30 |
| gmsd | 8 | 64² | 15711.7 | 30 |
| mdsi | 8 | 64² | 102768.8 | 30 |
| ms_gmsdc | 8 | 64² | 46069.8 | 30 |
| gmsd | 1 | 256² | 70179.2 | 30 |
| mdsi | 1 | 256² | 1395495.2 | 30 |
| ms_gmsdc | 1 | 256² | 613862.3 | 30 |
| gmsd | 8 | 256² | 48878.5 | 30 |
| mdsi | 8 | 256² | 673183.1 | 30 |
| ms_gmsdc | 8 | 256² | 540991.5 | 30 |
| gmsd | 1 | 1024² | 871184.9 | 30 |
| mdsi | 1 | 1024² | 1970691.8 | 30 |
| ms_gmsdc | 1 | 1024² | 20494564.2 | 30 |
| gmsd | 8 | 1024² | 220397.1 | 30 |
| mdsi | 8 | 1024² | 961130.5 | 30 |
| ms_gmsdc | 8 | 1024² | 18665096.2 | 30 |
| gmsd | 1 | 4096² | 12557082.0 | 10 |
| mdsi | 1 | 4096² | 6398575.5 | 10 |
| ms_gmsdc | 1 | 4096² | 405324771.2 | 10 |
| gmsd | 8 | 4096² | 2525400.9 | 10 |
| mdsi | 8 | 4096² | 3641636.6 | 10 |
| ms_gmsdc | 8 | 4096² | 351521219.7 | 10 |
