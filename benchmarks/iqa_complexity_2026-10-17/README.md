# IQA method-complexity tallies, 2026-10-17 (provisional, in progress)

Method: [docs/IQA_COMPLEXITY.md](../../docs/IQA_COMPLEXITY.md). Evaluator:
`python3 scripts/complexity/evaluate.py <tallies>`. Tallies are counts of
source code at the commit each file records, not measurements.

commit: 13aa694abd3380c5090831041a8551c77089f828
command: python3 scripts/complexity/evaluate.py benchmarks/iqa_complexity_2026-10-17/tallies/*.json

Coverage and reconciliation status are recorded below as the work lands.

## Coverage

<!-- coverage:begin -->
Tallied CLI metric names: **96 of 115** (shape 1024x1024; flops per original pixel, intrinsic + implementation classes; halo in original pixels). Two counters on one metric are independent counts; a difference between them is unresolved until listed in `reconcile/`.

| metric | counter | flops/px | intrinsic | implementation | halo px | work/span | barriers |
|---|---|---:|---:|---:|---:|---:|---:|
| afine | swe2 | 414756.3 | 414756.3 | 0.0 | 2 | 2.12e+08 | 0 |
| ahiq | swe2 | 1135087.1 | 1135087.1 | 0.0 | 0 | 2.67e+09 | 0 |
| arniqa | swe2 | 39168.7 | 39168.7 | 0.0 | 0 | 2.06e+10 | 0 |
| brisque | swe2 | 303.3 | 303.3 | 0.0 | 6 | 1.98e+08 | 0 |
| butteraugli | swe | 3447.3 | 3447.3 | 0.0 | 75 | 7.28e+07 | 1 |
| butteraugli-original | swe2 | 2610.0 | 2610.0 | 0.0 | 82 | 1.4e+07 | 1 |
| ciede2000 | swe2 | 150.0 | 150.0 | 0.0 | 0 | 5.8e+07 | 1 |
| ckdn | swe2 | 9906.0 | 9906.0 | 0.0 | 2 | 1.82e+07 | 0 |
| cvvdp | swe | 680.0 | 680.0 | 0.0 | 1150 | 1.89e+07 | 1 |
| cvvdp | swe2-recount | 1123.3 | 1123.3 | 0.0 | 1278 | 9.55e+06 | 1 |
| cw-ssim | swe2 | 502.0 | 502.0 | 0.0 | 24 | 2.4e+05 | 3 |
| deepdc | swe2 | 3112082.0 | 3112082.0 | 0.0 | 0 | 1.59e+09 | 0 |
| dists | swe2 | 2452085.4 | 2452085.4 | 0.0 | 0 | 1.26e+09 | 0 |
| dmm | swe2 | 480218.1 | 480218.1 | 0.0 | 1 | 2.46e+08 | 0 |
| dreamsim | swe2 | 101031.1 | 101031.1 | 0.0 | 2 | 2.38e+08 | 0 |
| flip | swe2 | 1371.0 | 1371.0 | 0.0 | 20 | 2.28e+07 | 1 |
| flip-v17 | swe2 | 1371.0 | 1371.0 | 0.0 | 20 | 2.28e+07 | 1 |
| fsim | swe | 350.1 | 261.1 | 89.0 | 12 | 5.38e+06 | 5 |
| fsim | swe2 | 243.1 | 243.1 | 0.0 | 1 | 3.74e+03 | 0 |
| fsim-k2 | swe2 | 949.2 | 949.2 | 0.0 | 1 | 3.65e+03 | 0 |
| fsim-k3 | swe2 | 421.4 | 421.4 | 0.0 | 1 | 3.63e+03 | 0 |
| fsim-k4 | swe2 | 243.1 | 243.1 | 0.0 | 1 | 3.74e+03 | 0 |
| fsim-nodownscale | swe2 | 4029.0 | 4029.0 | 0.0 | 1 | 3.88e+03 | 0 |
| fsim-y | swe | 320.2 | 231.2 | 89.0 | 12 | 4.91e+06 | 5 |
| gmsd | swe | 31.9 | 31.5 | 0.4 | 3 | 1.12e+06 | 1 |
| haarpsi | swe2 | 85.3 | 85.3 | 0.0 | 16 | 2.04e+06 | 1 |
| haarpsi-y | swe2 | 57.5 | 57.5 | 0.0 | 16 | 1.39e+06 | 1 |
| hyperiqa | swe2 | 205004.0 | 205004.0 | 0.0 | 0 | 1.08e+11 | 0 |
| iwssim | swe | 996.6 | 989.9 | 6.6 | 126 | 1.08e+06 | 3 |
| iwssim-piq | swe | 996.6 | 989.9 | 6.6 | 126 | 1.08e+06 | 3 |
| liqe | swe2 | 126528.4 | 126528.4 | 0.0 | 0 | 6.65e+10 | 0 |
| lpips-alex | swe2 | 104799.8 | 104799.8 | 0.0 | 0 | 5.38e+07 | 0 |
| lpips-squeeze | swe2 | 42978.9 | 42978.9 | 0.0 | 0 | 2.25e+07 | 0 |
| lpips-vgg | swe2 | 2450402.0 | 2450402.0 | 0.0 | 0 | 1.26e+09 | 0 |
| maniqa | swe2 | 259690.3 | 259690.3 | 0.0 | 0 | 1.36e+11 | 0 |
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
| musiq | swe2 | 301.5 | 301.5 | 0.0 | 0 | 1.59e+08 | 0 |
| niqe | swe2 | 189.8 | 189.8 | 0.0 | 3 | 1.24e+08 | 0 |
| nlpd-iqa | swe2 | 125.3 | 125.3 | 0.0 | 318 | 1.51e+06 | 1 |
| paq2piq | swe2 | 72768.5 | 72768.5 | 0.0 | 0 | 3.82e+10 | 0 |
| pieapp | swe2 | 765785647.4 | 765785647.4 | 0.0 | 0 | 3.92e+11 | 0 |
| piqe | swe2 | 36.1 | 36.1 | 0.0 | 0 | 2.32e+07 | 0 |
| psnr | lead | 9.0 | 9.0 | 0.0 | 0 | 4.18e+05 | 1 |
| psnr-cb | swe2 | 42.0 | 42.0 | 0.0 | 0 | 2.41e+07 | 0 |
| psnr-cr | swe2 | 42.0 | 42.0 | 0.0 | 0 | 2.41e+07 | 0 |
| psnr-y | swe2 | 13.0 | 13.0 | 0.0 | 0 | 6.49e+05 | 1 |
| psnr-ycbcr611 | swe2 | 96.0 | 96.0 | 0.0 | 0 | 5.35e+07 | 0 |
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
| ssim2 | swe2-recount | 1030.4 | 1030.4 | 0.0 | 383 | 4.28e+05 | 1 |
| ssimc | swe2 | 1240.4 | 1240.4 | 0.0 | 5 | 2.99e+07 | 1 |
| ssimc-pyiqa-f32 | swe2 | 1240.4 | 1240.4 | 0.0 | 5 | 2.99e+07 | 1 |
| ssimc-pyiqa-f32-k2 | swe2 | 380.0 | 380.0 | 0.0 | 5 | 9.16e+06 | 1 |
| ssimc-pyiqa-f32-k3 | swe2 | 326.9 | 326.9 | 0.0 | 5 | 7.87e+06 | 1 |
| ssimc-pyiqa-f32-k4 | swe2 | 308.6 | 308.6 | 0.0 | 5 | 7.43e+06 | 1 |
| ssimc-pyiqa-f32-k5 | swe2 | 300.0 | 300.0 | 0.0 | 5 | 7.22e+06 | 1 |
| ssimc-pyiqa-f32-k6 | swe2 | 295.6 | 295.6 | 0.0 | 5 | 7.11e+06 | 1 |
| ssimulacra1 | swe2 | 886.8 | 886.8 | 0.0 | 480 | 1.56e+07 | 2 |
| stlpips-alex | swe2 | 176449.7 | 176449.7 | 0.0 | 0 | 9.04e+07 | 0 |
| stlpips-vgg | swe2 | 313666.6 | 313666.6 | 0.0 | 0 | 1.62e+08 | 0 |
| vif | swe2 | 1013.1 | 1013.1 | 0.0 | 36 | 1.83e+07 | 4 |
| vifvec | swe2 | 1657.4 | 1657.4 | 0.0 | 62 | 4.56e+06 | 4 |
| vmaf | swe | 17.6 | 17.6 | 0.0 | 128 | 1.04e+06 | 3 |
| vmaf-neg | swe | 17.6 | 17.6 | 0.0 | 128 | 1.04e+06 | 3 |
| vsi | swe2 | 163.9 | 163.9 | 0.0 | 34 | 4.47e+06 | 2 |
| vsi-k2 | swe2 | 189.9 | 189.9 | 0.0 | 26 | 5.47e+06 | 2 |
| vsi-k3 | swe2 | 170.9 | 170.9 | 0.0 | 30 | 4.81e+06 | 2 |
| vsi-k4 | swe2 | 163.9 | 163.9 | 0.0 | 34 | 4.47e+06 | 2 |
| vsi-nodownscale | swe2 | 288.6 | 288.6 | 0.0 | 24 | 8.33e+06 | 2 |
| wadiqam-fr-live | swe2 | 141178.6 | 141178.6 | 0.0 | 0 | 7.24e+07 | 0 |
| wadiqam-fr-tid | swe2 | 141178.6 | 141178.6 | 0.0 | 0 | 7.24e+07 | 0 |
| wadiqam-nr-live | swe2 | 131965.5 | 131965.5 | 0.0 | 0 | 6.93e+10 | 0 |
| wadiqam-nr-tid | swe2 | 131965.5 | 131965.5 | 0.0 | 0 | 6.93e+10 | 0 |
| wd | swe2 | 1113586.0 | 1113586.0 | 0.0 | 0 | 5.71e+08 | 0 |
| ws-hv | swe2 | 170.5 | 170.5 | 0.0 | 71 | 3.89e+06 | 1 |
| zensim | swe2 | 1168.6 | 1168.6 | 0.0 | 40 | 1.24e+09 | 0 |

**Not yet tallied:** butteraugli-gpu, cambi, cvvdp-gpu, dssim, dssim-gpu, hdrvdp, hdrvdp3, iwssim-gpu, mad, nlpd, psnr-y-libvmaf, psnr-y-studio601, psnr-y601, ssim2-gpu, vmaf-4k, vmaf-neg-studio601, vmaf-studio601, vmaf-v1, zensim-gpu.
<!-- coverage:end -->
