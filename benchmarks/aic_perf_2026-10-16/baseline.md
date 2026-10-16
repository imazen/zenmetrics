# AIC-4 CPU metric perf round — baseline, 2026-10-16

Host `i265` (Intel Core Ultra 7 265K, 20 logical CPUs, 30 GiB RAM).
Release build, no `target-cpu=native`. Source: `849ea8da` (master@origin) plus
the harness `crates/zenmetrics-cli/examples/aic_perf_wall.rs` (this commit).
Snapshot export `133f8c853eea` of the working tree; dependencies from the
CI-pinned lock.

Each observation scores the procedural RGB8 pair defined by
`crates/zenmetrics-cli/examples/aic_perf_wall.rs` five times and reports
`ms/call` (mean over the five calls) and process maximum RSS. One-thread runs
are pinned to CPU 2; sixteen-thread runs are unpinned and use
`RAYON_NUM_THREADS=16`. Wall observations are exploratory, without statistical
intervals. The metric list is the perf-round scope: SSIMULACRA1, WS-HV, FLIP,
FLIP-v17, CW-SSIM, CIEDE2000, PSNR-Cb, PSNR-Cr (CAMBI and Butteraugli-original
moved to another lane).

Command (one heavy-lock hold, 177 s, `run-heavy` peak RSS 1.06 GiB):

```
flock ~/tmp/i265-heavy.lock ~/work/zen/scripts/run-heavy --mem 16G -- \
  bash ~/tmp/aic-classic-scratch/baseline_run.sh \
       ~/tmp/aic-classic-scratch/bin/aic_perf_wall \
       ~/tmp/aic-classic-scratch/baseline
```

`aic_perf_wall <metric> <w> <h> <reps>` calls `run_metric(MetricKind, ..)`
with `GpuRuntime::Cpu`; per-run maximum RSS comes from `/usr/bin/time -v`.
Raw table: `baseline.tsv`; wrapper log: `baseline-run-heavy.log`.

## Ranked by one-thread wall (3355x2516)

| Rank | Metric | 1T ms/call | 16T ms/call | 1T RSS MiB | 16T RSS MiB | 1T bits |
|---:|---|---:|---:|---:|---:|---|
| 1 | ssimulacra1 | 4742.08 | 4747.97 | 601.8 | 601.5 | `3fb7496b12d654d0` |
| 2 | flip | 3154.31 | 3187.25 | 1082.6 | 1082.6 | `3fc70b7e7780773d` |
| 3 | flip-v17 | 3130.83 | 3267.88 | 1082.5 | 1082.4 | `3fc67296e0000000` |
| 4 | cw-ssim | 2871.21 | 2771.85 | 488.3 | 488.1 | `3feeb148a4c6b1ec` |
| 5 | ciede2000 | 1292.62 | 1289.08 | 52.2 | 52.3 | `40253adf2dc585bc` |
| 6 | ws-hv | 622.81 | 647.08 | 430.5 | 430.3 | `3fe5aa76ed8113d4` |
| 7 | psnr-cb | 84.92 | 85.43 | 52.1 | 52.1 | `4036fb037aeb73ae` |
| 8 | psnr-cr | 84.33 | 84.83 | 52.1 | 51.9 | `4033e3c56853e8fa` |

## 1024x1024

| Metric | 1T ms/call | 16T ms/call | 1T RSS MiB |
|---|---:|---:|---:|
| ssimulacra1 | 534.65 | 534.22 | 78.9 |
| flip | 343.80 | 357.78 | 145.8 |
| flip-v17 | 352.64 | 352.36 | 145.8 |
| cw-ssim | 116.96 | 116.48 | 65.9 |
| ciede2000 | 161.18 | 160.34 | 9.8 |
| ws-hv | 77.07 | 72.43 | 57.1 |
| psnr-cb | 10.59 | 10.43 | 9.8 |
| psnr-cr | 10.41 | 15.19 | 9.7 |

## Observations

- No target scales with threads: 1T and 16T are within run-to-run noise
  (the small 1024x1024 `psnr-cr` 16T outlier is noise). None of the target
  modules uses Rayon except `classical.rs`, and neither CIEDE2000 nor the
  PSNR-Cb/Cr path there is parallel.
- FLIP and FLIP-v17 share the same `error_map` and differ only in the final
  pooling (`mean` vs `mean_v17`), so they time identically and one change to
  the shared map computation moves both.
- RSS is dominated by full-resolution per-pixel planes: flip ~1.08 GiB
  (two 8.4 MP f32 RGB/Lab planes plus convolution scratch), cw-ssim ~0.49 GiB,
  ssimulacra1 ~0.60 GiB, ws-hv ~0.43 GiB.
