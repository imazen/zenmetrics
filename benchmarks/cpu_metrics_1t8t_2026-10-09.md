# CPU metric cost at 1 and 8 threads, 1 MP and 8.4 MP — 2026-10-09

Every CPU metric the CLI scores from sRGB8 (29; vmaf at 8.4 MP is measured at 3354×2516, since it refuses the odd width), timed through the shipped
dispatch (`zenmetrics_cli::metrics::run_metric_display`, `GpuRuntime::Cpu`) on one
real photo pair: `zenmetrics-corpus` `source.png` against its q20 JPEG, mirror-tiled to
1024×1024 and to 3355×2516 (8.4 MP, odd width). Host r7900x (Ryzen 9 7900X, Zen 4).
Provenance, thread pinning and per-leg contention: `cpu_metrics_1t8t_2026-10-09/run.meta`.
Reproduce: `just cpu-metrics-1t8t <out_dir> 1024 3355x2516`.

**Precision.** 1 MP cells: zenbench, 3 rounds each. The median absolute deviation is
≤3% for most metrics but reaches 7–12% at 1t (cvvdp, ms-gmsd, gmsd, nlpd-iqa, psnrhvs,
mdctpsnr) and 5–17% at 8t (gmsd, psnrhvs, ms-gmsdc, dssim, zensim): treat differences
under ~15% as noise. 8.4 MP 8t: one call per metric. 8.4 MP 1t: the second (warm) of two
serial calls; the cold call is up to 13% slower (mdsi; iwssim, msssim, dssim ~7%). The
1t 1 MP leg shared the box with another agent's cargo build (15 of 44 load samples), so
its times may read slightly high and the 1 MP speedups slightly optimistic. The 8.4 MP
legs ran clean. One content pair only: data-dependent costs on other content are not
measured.

| metric | 1t 1MP ms | 8t 1MP ms | 8t speedup 1MP | 1t 8.4MP ms | 8t 8.4MP ms | 8t speedup 8.4MP | 1t ms/MP @8.4MP | 1t scaling vs linear | 1t 8.4MP maxRSS MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| psnr-y | 2.0 | 2.0 | 1.01 | 16.2 | 15.8 | 1.03 | 1.9 | 1.01 | 54 |
| mdsi | 2.0 | 1.1 | 1.81 | 6.9 | 2.4 | 2.90 | 0.8 | 0.42 | 62 |
| gmsd | 2.1 | 0.6 | 3.66 | 15.2 | 4.6 | 3.31 | 1.8 | 0.90 | 55 |
| psnr | 2.3 | 2.2 | 1.02 | 17.9 | 18.2 | 0.98 | 2.1 | 0.99 | 54 |
| psnrhvs | 10.5 | 9.2 | 1.13 | 143 | 133 | 1.08 | 17.0 | 1.70 | 248 |
| ssim-libvmaf | 11.0 | 11.3 | 0.98 | 117 | 118 | 0.99 | 13.9 | 1.33 | 167 |
| haarpsi | 14.9 | 14.5 | 1.02 | 165 | 164 | 1.01 | 19.6 | 1.38 | 185 |
| nlpd-iqa | 26.1 | 14.5 | 1.80 | 330 | 302 | 1.09 | 39.1 | 1.57 | 424 |
| ms-gmsd | 29.5 | 12.2 | 2.42 | 298 | 247 | 1.20 | 35.2 | 1.25 | 539 |
| ms-gmsdc | 30.3 | 12.2 | 2.49 | 297 | 258 | 1.15 | 35.2 | 1.22 | 539 |
| vmaf† | 33.0 | 32.4 | 1.02 | 273 | 312 | 0.87 | 32.3 | 1.03 | 373 |
| msssim | 39.9 | 28.7 | 1.39 | 417 | 402 | 1.04 | 49.4 | 1.30 | 471 |
| nlpd | 42.3 | 27.9 | 1.52 | 467 | 441 | 1.06 | 55.4 | 1.37 | 586 |
| psnrhvs-daala | 46.4 | 47.8 | 0.97 | 384 | 378 | 1.02 | 45.5 | 1.03 | 103 |
| fsim | 72.2 | 71.8 | 1.00 | 848 | 830 | 1.02 | 100 | 1.46 | 120 |
| vsi | 84.6 | 68.1 | 1.24 | 639 | 601 | 1.06 | 75.7 | 0.94 | 540 |
| iwssim-piq | 86.9 | 64.1 | 1.36 | 823 | 833 | 0.99 | 97.5 | 1.18 | 672 |
| iwssim | 87.4 | 69.6 | 1.26 | 834 | 832 | 1.00 | 98.8 | 1.19 | 672 |
| zensim | 91.4 | 16.9 | 5.41 | 495 | 111 | 4.48 | 58.7 | 0.67 | 255 |
| msssim-libvmaf | 101 | 87.6 | 1.15 | 915 | 901 | 1.02 | 108 | 1.13 | 476 |
| vif | 108 | 59.5 | 1.82 | 1,014 | 1,061 | 0.96 | 120 | 1.16 | 907 |
| ssim | 120 | 74.4 | 1.61 | 1,605 | 1,584 | 1.01 | 190 | 1.67 | 569 |
| ssim2 | 120 | 76.5 | 1.57 | 956 | 923 | 1.04 | 113 | 0.99 | 1,270 |
| butteraugli | 139 | 71.8 | 1.93 | 1,236 | 692 | 1.79 | 146 | 1.11 | 1,385 |
| dssim | 165 | 36.7 | 4.50 | 1,749 | 465 | 3.76 | 207 | 1.32 | 975 |
| cvvdp | 184 | 111 | 1.66 | 1,649 | 1,667 | 0.99 | 195 | 1.11 | 1,761 |
| vifvec | 489 | 470 | 1.04 | 3,828 | 3,819 | 1.00 | 454 | 0.97 | 569 |
| mad | 1,929 | 1,598 | 1.21 | 59,680 | 60,337 | 0.99 | 7,070 | 3.84 | 1,309 |
| mdctpsnr | 8,189 | 7,740 | 1.06 | 65,844 | 63,626 | 1.03 | 7,800 | 1.00 | 124 |

† vmaf's 8.4 MP columns are at 3354×2516 (8.44 MP); the 3355×2516 frame is refused.

`1t scaling vs linear` = (1t 8.4 MP time ÷ 1t 1 MP time) ÷ 8.05, the pixel ratio. 1.0
is linear; above 1 grows faster than pixel count. Max RSS includes ~54 MiB of decoded
inputs (psnr's peak is the floor).

## Findings

1. **mad and mdctpsnr dominate any all-metrics run.** At 8.4 MP, single-threaded, they
   take 59.7 s and 65.8 s; the other 26 metrics together take 19.3 s. At 1 MP mdctpsnr
   is 8.2 s, 4× mad and 44× cvvdp. Neither uses threads: 8t speedup ≤1.06 for
   mdctpsnr at both sizes, and 0.99 for mad at 8.4 MP.
2. **mad grows 3.8× faster than linear** from 1024² to 3355×2516 (1.9 s → 59.7 s, 1t).
   Not root-caused. Untested hypothesis: MAD's log-Gabor stage runs FFTs at the image
   size, and 3355 = 5·11·61, 2516 = 2²·17·37 have large prime factors. Two sizes
   cannot separate pixel count from factorization; a power-of-two 8 MP size would.
3. **Most metrics are single-threaded in this build, by configuration.** The
   default CLI build compiles eleven in-tree metric crates (cvvdp, iwssim, fsim,
   vsi, haarpsi, msssim, psnrhvs, vif, mad-iqa, vmaf, gmsd) without their
   `parallel` (rayon) feature, and fast-ssim2 without `rayon` (`cargo tree -e
   features -i <crate> -p zenmetrics-cli`). Only zensim, dssim and butteraugli
   are threaded there; gmsd's `parallel` was on here only because the
   benchmark crate enables it. So the 8t columns measure the shipped build,
   not what the algorithms can do. A follow-up run also found that this
   interleaved harness moves single-threaded metrics by up to 1.6× between legs
   at 1 MP (ssim 75 ms in one leg, 117 ms in another, identical code), so the
   1 MP speedup column is not reliable at that precision. The 8.4 MP 1t figures
   (one process per metric) are the trustworthy ones. Threaded-build numbers:
   see "Threading re-check" below.
4. **vmaf refuses the 3355×2516 frame:** "VMAF requires matching, even dimensions
   >= 32". Any odd-width or odd-height image cannot be VMAF-scored through the CLI.
   Whether libvmaf itself accepts odd sizes was not checked. Cropped to 3354×2516 it
   scores 89.24 and takes 273 ms at 1 thread and 312 ms at 8 threads (zenbench, 3
   rounds): no gain from threads, as at 1 MP (1.02×).
5. **Memory at 8.4 MP, 1t (max RSS):** cvvdp 1,761 MiB, butteraugli 1,385, mad 1,309,
   ssim2 1,270, dssim 975, vif 907. Lowest: psnrhvs-daala 103 MiB, fsim 120, mdctpsnr
   124.
6. **Cheapest useful signals:** psnr / psnr-y, gmsd and mdsi cost ≤2.3 ms at 1 MP and
   ≤18 ms at 8.4 MP single-threaded. mdsi grows sub-linearly (0.42) because it works on
   reduced planes.

Not covered: hdrvdp / hdrvdp3 refuse sRGB8 by design (absolute-nits input only;
`crates/hdrvdp/benches/pipeline.rs` times v2). Below 1024² was dropped from this pass;
iwssim refuses images under 176 px on the short side (5-level pyramid).

## Threading re-check (same day)

**Why.** The first pass suggested most metrics gain nothing from threads. The
cause was the build: see finding 3. This section turns each crate's threading
on and measures again.

**Build.** `cargo build --release -p cpu-profile --bin new-metrics-wall
--features metrics-parallel`. The feature enables `parallel` on cvvdp, iwssim,
fsim, vsi, haarpsi, msssim, psnrhvs, vif, mad-iqa and vmaf, and `rayon` on
fast-ssim2 (gmsd's `parallel` was already on in the benchmark crate).
Raw rows: `cpu_metrics_1t8t_2026-10-09/threading_recheck/`.

**8.4 MP, photo pair, 8 threads (taskset 0-7), one zenbench call per metric;
1t is the single-thread column from above.** Every score is bit-identical
between the default and threaded builds (also checked at 1 MP).
mdctpsnr was skipped: it has no threading code. vmaf (at 3354×2516) measured
315 ms threaded vs 312 ms default: its `parallel` feature splits work across
video frames, not within one image.

| metric | 1t | 8t, default build | 8t, threading on | threading on vs 1t |
|---|--:|--:|--:|--:|
| zensim | 495 ms | 111 ms | 101 ms | 4.90× |
| gmsd | 15 ms | 4.6 ms | 3.3 ms | 4.66× |
| dssim | 1.75 s | 465 ms | 443 ms | 3.95× |
| mdsi | 6.9 ms | 2.4 ms | 2.4 ms | 2.86× |
| cvvdp | 1.65 s | 1.67 s | 687 ms | 2.40× |
| fsim | 848 ms | 830 ms | 433 ms | 1.96× |
| iwssim | 834 ms | 832 ms | 455 ms | 1.83× |
| iwssim-piq | 823 ms | 833 ms | 450 ms | 1.83× |
| butteraugli | 1.24 s | 692 ms | 677 ms | 1.82× |
| mad | 59.68 s | 60.34 s | 33.15 s | 1.80× |
| psnrhvs | 143 ms | 133 ms | 85 ms | 1.68× |
| haarpsi | 165 ms | 164 ms | 114 ms | 1.45× |
| vif | 1.01 s | 1.06 s | 736 ms | 1.38× |
| msssim | 417 ms | 402 ms | 305 ms | 1.37× |
| vsi | 639 ms | 601 ms | 473 ms | 1.35× |
| ms-gmsdc | 297 ms | 258 ms | 246 ms | 1.21× |
| ms-gmsd | 298 ms | 247 ms | 260 ms | 1.14× |
| nlpd-iqa | 330 ms | 302 ms | 303 ms | 1.09× |
| ssim-libvmaf | 117 ms | 118 ms | 109 ms | 1.08× |
| ssim | 1.60 s | 1.58 s | 1.55 s | 1.03× |
| psnrhvs-daala | 384 ms | 378 ms | 372 ms | 1.03× |
| nlpd | 467 ms | 441 ms | 455 ms | 1.03× |
| ssim2 | 956 ms | 923 ms | 941 ms | 1.02× |
| msssim-libvmaf | 915 ms | 901 ms | 904 ms | 1.01× |
| vifvec | 3.83 s | 3.82 s | 3.83 s | 1.00× |
| psnr | 18 ms | 18 ms | 18 ms | 1.00× |
| psnr-y | 16 ms | 16 ms | 16 ms | 1.00× |

Reading it: the crates that were threaded all along (zensim, dssim, gmsd,
mdsi, butteraugli) keep their 1.8–4.9×. Turning the features on gives
cvvdp 2.40×, fsim 1.96×, iwssim 1.83×, mad 1.80×, psnrhvs 1.68×, haarpsi
1.45×, vif 1.38×, msssim 1.37×, vsi 1.35×. These implementations thread some
stages, not all, so 8 cores buy 1.4–2.4×. Little or no gain (≤1.21×)
with every feature on: ssim (the CLI's own), nlpd, nlpd-iqa, vifvec,
ssim-libvmaf, msssim-libvmaf, psnrhvs-daala, ms-gmsd, ms-gmsdc, and ssim2:
fast-ssim2's `rayon` takes it from 956 ms to 941 ms here on Zen 4, in line
with the earlier aarch64 finding that it does not parallelise as shipped.

**The 1 MP harness problem.** At 1 MP the interleaved run (29 metrics in one
zenbench group) moved single-threaded code by up to 2× between legs. Timed
alone, the same code is stable and pinning does not matter:

| single-thread @ 1 MP | ssim | nlpd | vif |
|---|--:|--:|--:|
| one serial call (`heap` mode), pinned to 1 core | 131 ms | 52 ms | 103 ms |
| one serial call, 8 cores available | 140 ms | 52 ms | 104 ms |
| zenbench, these three only, pinned / 8 cores | 143 / 144 ms | 55 / 56 ms | 107 / 106 ms |
| interleaved with all 29 metrics, across legs | 75–117 ms | 28–47 ms | 60–88 ms |

So interleaving many allocation-heavy metrics in one process changes the
cost of the ones that follow (allocator or cache state; not root-caused).
At 8.4 MP the interleaved and isolated figures agree within ~5%, which is
why the threading table above uses 8.4 MP. For 1 MP work, time each metric
in its own process or its own zenbench group.
