# IQA method-complexity tallies, 2026-10-17 (provisional, in progress)

Method: [docs/IQA_COMPLEXITY.md](../../docs/IQA_COMPLEXITY.md). Evaluator:
`python3 scripts/complexity/evaluate.py <tallies>`. Tallies are counts of
source code at the commit each file records, not measurements.

commit: 13aa694abd3380c5090831041a8551c77089f828
command: python3 scripts/complexity/evaluate.py benchmarks/iqa_complexity_2026-10-17/tallies/*.json

Coverage and reconciliation status are recorded below as the work lands.

## Coverage

<!-- coverage:begin -->
Tallied CLI metric names: **120 of 122** (shape 1024x1024; flops per original pixel, intrinsic + implementation classes; halo in original pixels). Two counters on one metric are independent counts; a difference between them is unresolved until listed in `reconcile/`.

| metric | counter | flops/px | intrinsic | implementation | halo px | work/span | barriers |
|---|---|---:|---:|---:|---:|---:|---:|
| afine | swe2 | 414768.5 | 414768.5 | 0.0 | 2 | 1.09e+11 | 0 |
| ahiq | swe2 | 1137591.4 | 1137591.4 | 0.0 | 0 | 2.68e+09 | 0 |
| ahiq-pyiqa | claude | 4781663.7 | 4781663.7 | 0.0 | 0 | 7.33e+09 | 1 |
| arniqa | swe2 | 39225.2 | 39225.2 | 0.0 | 0 | 2.08e+10 | 0 |
| brisque | swe2 | 303.3 | 303.3 | 0.0 | 6 | 1.98e+08 | 0 |
| butteraugli | swe | 3447.3 | 3447.3 | 0.0 | 75 | 6.57e+07 | 1 |
| butteraugli-gpu | swe2 | 3839.8 | 3839.8 | 0.0 | 16 | 1.96e+09 | 0 |
| butteraugli-original | swe2 | 2610.0 | 2610.0 | 0.0 | 82 | 7.45e+06 | 1 |
| cambi | swe | 32.3 | 24.3 | 8.0 | 228 | 1.51e+06 | 1 |
| ciede2000 | swe2 | 150.0 | 150.0 | 0.0 | 0 | 5.8e+07 | 1 |
| ckdn | swe2 | 10433.9 | 10433.9 | 0.0 | 2 | 2.01e+07 | 0 |
| cvvdp | swe | 680.0 | 680.0 | 0.0 | 1150 | 1.58e+07 | 1 |
| cvvdp | swe2-recount | 681.3 | 681.3 | 0.0 | 894 | 1.62e+07 | 0 |
| cvvdp-gpu | swe2 | 690.7 | 690.7 | 0.0 | 96 | 7.87e+08 | 0 |
| cw-ssim | swe2 | 502.0 | 502.0 | 0.0 | 24 | 2.4e+05 | 3 |
| cw-ssim-pyiqa | claude | 294.3 | 294.3 | 0.0 | 24 | 2.78e+05 | 4 |
| deepdc | swe2 | 1556054.0 | 1556054.0 | 0.0 | 0 | 7.98e+08 | 0 |
| dists | swe2 | 1227041.4 | 1227041.4 | 0.0 | 0 | 6.3e+08 | 0 |
| dmm | swe2 | 480362.8 | 480362.8 | 0.0 | 1 | 2.47e+08 | 0 |
| dmm-pyiqa | claude | 505464.4 | 505464.4 | 0.0 | 673 | 2.95e+08 | 1 |
| dreamsim | swe2 | 101222.2 | 101222.2 | 0.0 | 2 | 2.39e+08 | 0 |
| dssim | swe | 772.6 | 772.6 | 0.0 | 94 | 1.77e+07 | 1 |
| dssim-gpu | swe2 | 658.8 | 658.8 | 0.0 | 33 | 1.52e+07 | 0 |
| flip | swe2 | 1371.0 | 1371.0 | 0.0 | 20 | 1.49e+07 | 1 |
| flip-v17 | swe2 | 1371.0 | 1371.0 | 0.0 | 20 | 1.49e+07 | 1 |
| fsim | swe | 350.1 | 261.1 | 89.0 | 12 | 5.38e+06 | 5 |
| fsim | swe2 | 243.1 | 243.1 | 0.0 | 1 | 3.74e+03 | 0 |
| fsim-k2 | swe2 | 949.2 | 949.2 | 0.0 | 1 | 3.65e+03 | 0 |
| fsim-k3 | swe2 | 421.4 | 421.4 | 0.0 | 1 | 3.63e+03 | 0 |
| fsim-k4 | swe2 | 243.1 | 243.1 | 0.0 | 1 | 3.74e+03 | 0 |
| fsim-nodownscale | swe2 | 4029.0 | 4029.0 | 0.0 | 1 | 3.88e+03 | 0 |
| fsim-y | swe | 320.2 | 231.2 | 89.0 | 12 | 4.91e+06 | 5 |
| gmsd | swe | 31.9 | 31.5 | 0.4 | 3 | 1.12e+06 | 1 |
| haarpsi | swe2 | 85.3 | 85.3 | 0.0 | 16 | 2.04e+06 | 1 |
| haarpsi-y | swe-recount | 65.5 | 65.5 | 0.0 | 18 | 3.36e+06 | 1 |
| haarpsi-y | swe2 | 57.5 | 57.5 | 0.0 | 16 | 1.39e+06 | 1 |
| hdrvdp | swe | 6402.8 | 6402.8 | 0.0 | 1020 | 7.09e+07 | 3 |
| hdrvdp3 | swe | 4108.3 | 4108.3 | 0.0 | 1278 | 1.16e+08 | 1 |
| hyperiqa | swe2 | 205140.8 | 205140.8 | 0.0 | 0 | 1.08e+11 | 0 |
| iwssim | swe | 1005.3 | 998.7 | 6.6 | 126 | 1.08e+06 | 3 |
| iwssim-gpu | swe2 | 987.0 | 987.0 | 0.0 | 80 | 7.69e+08 | 0 |
| iwssim-piq | swe | 1005.3 | 998.7 | 6.6 | 126 | 1.08e+06 | 3 |
| liqe | swe2 | 126710.4 | 126710.4 | 0.0 | 0 | 6.67e+10 | 0 |
| lpips-alex | swe2 | 52471.4 | 52471.4 | 0.0 | 0 | 2.7e+07 | 0 |
| lpips-squeeze | swe2 | 21703.6 | 21703.6 | 0.0 | 0 | 1.17e+07 | 0 |
| lpips-vgg | swe2 | 1225954.0 | 1225954.0 | 0.0 | 0 | 6.3e+08 | 0 |
| mad | swe | 5211.3 | 5211.3 | 0.0 | 1024 | 1.95e+08 | 1 |
| maniqa | swe2 | 260146.4 | 260146.4 | 0.0 | 0 | 1.37e+11 | 0 |
| mdctpsnr | swe2 | 16186.2 | 16186.2 | 0.0 | 8 | 2.52e+08 | 1 |
| mdsi | swe2 | 8.9 | 8.9 | 0.0 | 12 | 2.76e+05 | 1 |
| ms-gmsd | swe2 | 127.0 | 127.0 | 0.0 | 23 | 3.56e+06 | 5 |
| ms-gmsdc | swe2 | 127.0 | 127.0 | 0.0 | 23 | 3.56e+06 | 5 |
| ms-ssim-pyiqa | swe2 | 564.5 | 564.5 | 0.0 | 95 | 2.13e+07 | 0 |
| ms-ssim-pyiqa-f32 | swe2 | 564.5 | 564.5 | 0.0 | 95 | 2.13e+07 | 0 |
| ms-ssim-pyiqa-f32-k2 | swe2 | 147.7 | 147.7 | 0.0 | 95 | 5.58e+06 | 0 |
| ms-ssim-pyiqa-f32-k3 | swe2 | 70.7 | 70.7 | 0.0 | 95 | 2.67e+06 | 0 |
| ms-ssim-pyiqa-f32-k4 | swe2 | 43.8 | 43.8 | 0.0 | 95 | 1.65e+06 | 0 |
| ms-ssim-pyiqa-f32-k5 | swe2 | 31.3 | 31.3 | 0.0 | 95 | 1.18e+06 | 0 |
| ms-ssim-pyiqa-f32-k6 | swe2 | 24.7 | 24.7 | 0.0 | 95 | 9.31e+05 | 0 |
| msssim | swe | 381.6 | 381.6 | 0.0 | 88 | 4.93e+07 | 1 |
| msssim-libvmaf | swe | 447.9 | 447.9 | 0.0 | 112 | 1.01e+07 | 1 |
| musiq | swe2 | 302.8 | 302.8 | 0.0 | 0 | 1.6e+08 | 0 |
| niqe | swe2 | 189.8 | 189.8 | 0.0 | 3 | 1.24e+08 | 0 |
| nlpd | swe | 397.9 | 397.9 | 0.0 | 318 | 7.11e+06 | 1 |
| nlpd-iqa | swe2 | 125.3 | 125.3 | 0.0 | 318 | 1.51e+06 | 1 |
| paq2piq | swe2 | 72732.8 | 72732.8 | 0.0 | 0 | 3.83e+10 | 0 |
| pieapp | swe2 | 765788887.7 | 765788887.7 | 0.0 | 0 | 3.92e+11 | 0 |
| pieapp-pyiqa | claude | 2336878.8 | 2336878.8 | 0.0 | 0 | 1.61e+10 | 1 |
| piqe | swe2 | 36.1 | 36.1 | 0.0 | 0 | 2.32e+07 | 0 |
| psnr | lead | 9.0 | 9.0 | 0.0 | 0 | 4.18e+05 | 1 |
| psnr-cb | swe2 | 42.0 | 42.0 | 0.0 | 0 | 2.41e+07 | 0 |
| psnr-cr | swe2 | 42.0 | 42.0 | 0.0 | 0 | 2.41e+07 | 0 |
| psnr-y | swe2 | 13.0 | 13.0 | 0.0 | 0 | 6.49e+05 | 1 |
| psnr-y-libvmaf | swe2 | 20.0 | 20.0 | 0.0 | 0 | 1.2e+06 | 1 |
| psnr-y-studio601 | swe2 | 16.0 | 16.0 | 0.0 | 0 | 9.99e+05 | 1 |
| psnr-y601 | swe2 | 13.0 | 13.0 | 0.0 | 0 | 6.49e+05 | 1 |
| psnr-ycbcr611 | swe2 | 96.0 | 96.0 | 0.0 | 0 | 5.35e+07 | 0 |
| psnrhvs | swe | 247.7 | 247.7 | 0.0 | 8 | 9.2e+06 | 1 |
| psnrhvs | swe2 | 288.6 | 288.6 | 0.0 | 4 | 4.3e+06 | 0 |
| psnrhvs-daala | swe2 | 262.5 | 262.5 | 0.0 | 4 | 3.9e+06 | 0 |
| psnrhvs-y | swe | 94.6 | 94.6 | 0.0 | 8 | 3.76e+06 | 1 |
| ssim | swe | 756.3 | 708.0 | 48.3 | 5 | 2.7e+07 | 1 |
| ssim-libvmaf | swe | 33.4 | 33.4 | 0.0 | 28 | 1.43e+06 | 1 |
| ssim-pyiqa | swe2 | 425.4 | 425.4 | 0.0 | 5 | 1.87e+07 | 0 |
| ssim-pyiqa-f32 | swe2 | 425.4 | 425.4 | 0.0 | 5 | 1.87e+07 | 0 |
| ssim-pyiqa-f32-k2 | swe2 | 113.5 | 113.5 | 0.0 | 5 | 4.98e+06 | 0 |
| ssim-pyiqa-f32-k3 | swe2 | 55.6 | 55.6 | 0.0 | 5 | 2.44e+06 | 0 |
| ssim-pyiqa-f32-k4 | swe2 | 35.6 | 35.6 | 0.0 | 5 | 1.56e+06 | 0 |
| ssim-pyiqa-f32-k5 | swe2 | 26.1 | 26.1 | 0.0 | 5 | 1.14e+06 | 0 |
| ssim-pyiqa-f32-k6 | swe2 | 21.1 | 21.1 | 0.0 | 5 | 9.25e+05 | 0 |
| ssim2 | swe | 1363.7 | 1363.7 | 0.0 | 223 | 5.03e+05 | 1 |
| ssim2 | swe2-recount | 1375.7 | 1375.7 | 0.0 | 160 | 1.04e+06 | 0 |
| ssim2-gpu | swe2 | 1047.8 | 1047.8 | 0.0 | 256 | 8.45e+08 | 0 |
| ssimc | swe2 | 1240.4 | 1240.4 | 0.0 | 5 | 2.99e+07 | 1 |
| ssimc-pyiqa-f32 | swe2 | 1240.4 | 1240.4 | 0.0 | 5 | 2.99e+07 | 1 |
| ssimc-pyiqa-f32-k2 | swe2 | 380.0 | 380.0 | 0.0 | 5 | 9.16e+06 | 1 |
| ssimc-pyiqa-f32-k3 | swe2 | 326.9 | 326.9 | 0.0 | 5 | 7.87e+06 | 1 |
| ssimc-pyiqa-f32-k4 | swe2 | 308.6 | 308.6 | 0.0 | 5 | 7.43e+06 | 1 |
| ssimc-pyiqa-f32-k5 | swe2 | 300.0 | 300.0 | 0.0 | 5 | 7.22e+06 | 1 |
| ssimc-pyiqa-f32-k6 | swe2 | 295.6 | 295.6 | 0.0 | 5 | 7.11e+06 | 1 |
| ssimulacra1 | swe2 | 3846.1 | 3846.1 | 0.0 | 480 | 4.84e+07 | 2 |
| stlpips-alex | swe2 | 88238.0 | 88238.0 | 0.0 | 0 | 4.52e+07 | 0 |
| stlpips-vgg | swe2 | 157236.5 | 157236.5 | 0.0 | 0 | 8.21e+07 | 0 |
| vif | swe2 | 1013.1 | 1013.1 | 0.0 | 36 | 1.83e+07 | 4 |
| vifvec | swe2 | 1657.4 | 1657.4 | 0.0 | 62 | 4.56e+06 | 4 |
| vmaf | swe | 17.6 | 17.6 | 0.0 | 128 | 1.01e+06 | 3 |
| vmaf-4k | swe2 | 43.7 | 43.7 | 0.0 | 16 | 5.42e+07 | 0 |
| vmaf-neg | swe | 17.6 | 17.6 | 0.0 | 128 | 1.01e+06 | 3 |
| vmaf-neg-studio601 | swe2 | 43.7 | 43.7 | 0.0 | 16 | 5.42e+07 | 0 |
| vmaf-studio601 | swe2 | 43.7 | 43.7 | 0.0 | 16 | 5.42e+07 | 0 |
| vmaf-v1 | swe2 | 162.6 | 162.6 | 0.0 | 16 | 3.84e+07 | 0 |
| vsi | swe-recount | 121.4 | 121.4 | 0.0 | 28 | 3.29e+06 | 1 |
| vsi | swe2 | 163.9 | 163.9 | 0.0 | 34 | 4.47e+06 | 2 |
| vsi-k2 | swe2 | 189.9 | 189.9 | 0.0 | 26 | 5.47e+06 | 2 |
| vsi-k3 | swe2 | 170.9 | 170.9 | 0.0 | 30 | 4.81e+06 | 2 |
| vsi-k4 | swe2 | 163.9 | 163.9 | 0.0 | 34 | 4.47e+06 | 2 |
| vsi-nodownscale | swe2 | 288.6 | 288.6 | 0.0 | 24 | 8.33e+06 | 2 |
| wadiqam-fr-live | swe2 | 140992.0 | 140992.0 | 0.0 | 0 | 7.24e+07 | 0 |
| wadiqam-fr-tid | swe2 | 140992.0 | 140992.0 | 0.0 | 0 | 7.24e+07 | 0 |
| wadiqam-nr-live | swe2 | 131825.0 | 131825.0 | 0.0 | 0 | 6.93e+10 | 0 |
| wadiqam-nr-tid | swe2 | 131825.0 | 131825.0 | 0.0 | 0 | 6.93e+10 | 0 |
| wadiqam-pyiqa | claude | 248540.0 | 248540.0 | 0.0 | 16 | 2.31e+09 | 1 |
| wd | swe2 | 1113586.0 | 1113586.0 | 0.0 | 0 | 5.71e+08 | 0 |
| ws-hv | swe2 | 170.5 | 170.5 | 0.0 | 71 | 3.89e+06 | 1 |
| zensim | swe2 | 1168.6 | 1168.6 | 0.0 | 40 | 6.19e+08 | 0 |
| zensim-gpu | swe2 | 950.6 | 950.6 | 0.0 | 40 | 8.59e+08 | 0 |

**Not yet tallied:** topiq-fr, topiq-fr-pipal.
<!-- coverage:end -->
