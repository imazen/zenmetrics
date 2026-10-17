# IQA method-complexity tallies, 2026-10-17 (provisional, in progress)

Method: [docs/IQA_COMPLEXITY.md](../../docs/IQA_COMPLEXITY.md). Evaluator:
`python3 scripts/complexity/evaluate.py <tallies>`. Tallies are counts of
source code at the commit each file records, not measurements.

commit: 13aa694abd3380c5090831041a8551c77089f828
command: python3 scripts/complexity/evaluate.py benchmarks/iqa_complexity_2026-10-17/tallies/*.json

Coverage and reconciliation status are recorded below as the work lands.

## Coverage

<!-- coverage:begin -->
Tallied CLI metric names: **1 of 115** (shape 1024x1024; flops per original pixel, intrinsic + implementation classes; halo in original pixels). Two counters on one metric are independent counts; a difference between them is unresolved until listed in `reconcile/`.

| metric | counter | flops/px | intrinsic | implementation | halo px | work/span | barriers |
|---|---|---:|---:|---:|---:|---:|---:|
| psnr | lead | 9.0 | 9.0 | 0.0 | 0 | 4.18e+05 | 1 |

**Not yet tallied:** afine, ahiq, arniqa, brisque, butteraugli, butteraugli-gpu, butteraugli-original, cambi, ciede2000, ckdn, cvvdp, cvvdp-gpu, cw-ssim, deepdc, dists, dmm, dreamsim, dssim, dssim-gpu, flip, flip-v17, fsim, fsim-k2, fsim-k3, fsim-k4, fsim-nodownscale, fsim-y, gmsd, haarpsi, haarpsi-y, hdrvdp, hdrvdp3, hyperiqa, iwssim, iwssim-gpu, iwssim-piq, liqe, lpips-alex, lpips-squeeze, lpips-vgg, mad, maniqa, mdctpsnr, mdsi, ms-gmsd, ms-gmsdc, ms-ssim-pyiqa, ms-ssim-pyiqa-f32, ms-ssim-pyiqa-f32-k2, ms-ssim-pyiqa-f32-k3, ms-ssim-pyiqa-f32-k4, ms-ssim-pyiqa-f32-k5, ms-ssim-pyiqa-f32-k6, msssim, msssim-libvmaf, musiq, niqe, nlpd, nlpd-iqa, paq2piq, pieapp, piqe, psnr-cb, psnr-cr, psnr-y, psnr-y-libvmaf, psnr-y-studio601, psnr-y601, psnr-ycbcr611, psnrhvs, psnrhvs-daala, psnrhvs-y, ssim, ssim-libvmaf, ssim-pyiqa, ssim-pyiqa-f32, ssim-pyiqa-f32-k2, ssim-pyiqa-f32-k3, ssim-pyiqa-f32-k4, ssim-pyiqa-f32-k5, ssim-pyiqa-f32-k6, ssim2, ssim2-gpu, ssimc, ssimc-pyiqa-f32, ssimc-pyiqa-f32-k2, ssimc-pyiqa-f32-k3, ssimc-pyiqa-f32-k4, ssimc-pyiqa-f32-k5, ssimc-pyiqa-f32-k6, ssimulacra1, stlpips-alex, stlpips-vgg, vif, vifvec, vmaf, vmaf-4k, vmaf-neg, vmaf-neg-studio601, vmaf-studio601, vmaf-v1, vsi, vsi-k2, vsi-k3, vsi-k4, vsi-nodownscale, wadiqam-fr-live, wadiqam-fr-tid, wadiqam-nr-live, wadiqam-nr-tid, wd, ws-hv, zensim, zensim-gpu.
<!-- coverage:end -->
