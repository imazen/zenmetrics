# IQA method-complexity tallies, 2026-10-17 (provisional, in progress)

Method: [docs/IQA_COMPLEXITY.md](../../docs/IQA_COMPLEXITY.md). Evaluator:
`python3 scripts/complexity/evaluate.py <tallies>`. Tallies are counts of
source code at the commit each file records, not measurements.

commit: 13aa694abd3380c5090831041a8551c77089f828
command: python3 scripts/complexity/evaluate.py benchmarks/iqa_complexity_2026-10-17/tallies/*.json

Coverage and reconciliation status are recorded below as the work lands.

## Coverage

<!-- coverage:begin -->
Tallied CLI metric names: **15 of 115** (shape 1024x1024; flops per original pixel, intrinsic + implementation classes; halo in original pixels). Two counters on one metric are independent counts; a difference between them is unresolved until listed in `reconcile/`.

| metric | counter | flops/px | intrinsic | implementation | halo px | work/span | barriers |
|---|---|---:|---:|---:|---:|---:|---:|
| ciede2000 | swe2 | 150.0 | 150.0 | 0.0 | 0 | 5.8e+07 | 1 |
| cvvdp | swe | 680.0 | 680.0 | 0.0 | 1150 | 1.89e+07 | 1 |
| haarpsi | swe2 | 85.3 | 85.3 | 0.0 | 16 | 2.04e+06 | 1 |
| haarpsi-y | swe2 | 57.5 | 57.5 | 0.0 | 16 | 1.39e+06 | 1 |
| mdctpsnr | swe2 | 16186.2 | 16186.2 | 0.0 | 8 | 2.52e+08 | 1 |
| mdsi | swe2 | 8.9 | 8.9 | 0.0 | 12 | 2.76e+05 | 1 |
| ms-gmsd | swe2 | 127.0 | 127.0 | 0.0 | 23 | 3.56e+06 | 5 |
| ms-gmsdc | swe2 | 127.0 | 127.0 | 0.0 | 23 | 3.56e+06 | 5 |
| nlpd-iqa | swe2 | 125.3 | 125.3 | 0.0 | 318 | 1.51e+06 | 1 |
| psnr | lead | 9.0 | 9.0 | 0.0 | 0 | 4.18e+05 | 1 |
| psnr-y | swe2 | 13.0 | 13.0 | 0.0 | 0 | 6.49e+05 | 1 |
| ssimulacra1 | swe2 | 886.8 | 886.8 | 0.0 | 480 | 1.56e+07 | 2 |
| vif | swe2 | 1013.1 | 1013.1 | 0.0 | 36 | 1.83e+07 | 4 |
| vifvec | swe2 | 1657.4 | 1657.4 | 0.0 | 62 | 4.56e+06 | 4 |
| vsi | swe2 | 163.9 | 163.9 | 0.0 | 34 | 4.47e+06 | 2 |

**Not yet tallied:** afine, ahiq, arniqa, brisque, butteraugli, butteraugli-gpu, butteraugli-original, cambi, ckdn, cvvdp-gpu, cw-ssim, deepdc, dists, dmm, dreamsim, dssim, dssim-gpu, flip, flip-v17, fsim, fsim-k2, fsim-k3, fsim-k4, fsim-nodownscale, fsim-y, gmsd, hdrvdp, hdrvdp3, hyperiqa, iwssim, iwssim-gpu, iwssim-piq, liqe, lpips-alex, lpips-squeeze, lpips-vgg, mad, maniqa, ms-ssim-pyiqa, ms-ssim-pyiqa-f32, ms-ssim-pyiqa-f32-k2, ms-ssim-pyiqa-f32-k3, ms-ssim-pyiqa-f32-k4, ms-ssim-pyiqa-f32-k5, ms-ssim-pyiqa-f32-k6, msssim, msssim-libvmaf, musiq, niqe, nlpd, paq2piq, pieapp, piqe, psnr-cb, psnr-cr, psnr-y-libvmaf, psnr-y-studio601, psnr-y601, psnr-ycbcr611, psnrhvs, psnrhvs-daala, psnrhvs-y, ssim, ssim-libvmaf, ssim-pyiqa, ssim-pyiqa-f32, ssim-pyiqa-f32-k2, ssim-pyiqa-f32-k3, ssim-pyiqa-f32-k4, ssim-pyiqa-f32-k5, ssim-pyiqa-f32-k6, ssim2, ssim2-gpu, ssimc, ssimc-pyiqa-f32, ssimc-pyiqa-f32-k2, ssimc-pyiqa-f32-k3, ssimc-pyiqa-f32-k4, ssimc-pyiqa-f32-k5, ssimc-pyiqa-f32-k6, stlpips-alex, stlpips-vgg, vmaf, vmaf-4k, vmaf-neg, vmaf-neg-studio601, vmaf-studio601, vmaf-v1, vsi-k2, vsi-k3, vsi-k4, vsi-nodownscale, wadiqam-fr-live, wadiqam-fr-tid, wadiqam-nr-live, wadiqam-nr-tid, wd, ws-hv, zensim, zensim-gpu.
<!-- coverage:end -->
