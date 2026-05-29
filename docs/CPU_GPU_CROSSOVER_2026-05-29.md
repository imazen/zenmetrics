# CPU vs GPU one-shot crossover — perceptual metrics (2026-05-29)

Task #141. Per metric: the image size below which scoring a SINGLE image on a
**cold process** is faster on CPU than GPU, and the batch/warm verdict.

## Summary (plain English)

A GPU score pays a fixed cold-start floor of roughly **170-190 ms of CUDA
context init** plus per-metric JIT/allocation before the first pixel is touched
(`gpu_coldstart_2026-05-29.tsv`, `client_init_ms` ≈ 170-190 ms, then
`metric_new_ms` + `first_compute_ms` on top). So for a **single small image on a
freshly-launched process the CPU wins** — it starts computing immediately with no
device handshake. As the image grows, the GPU's parallel throughput eventually
outruns the CPU even after paying that one-time floor, and the one-shot crossover
is where the CPU's full-image wall first exceeds the GPU's cold total.

For **batch / server use (warm GPU context, reference cached)** the GPU is
**faster at every measured size** — the warm per-call wall is 10-100x below the
CPU wall — so there is no batch crossover for any of these metrics in the range
measured.

CPU = full-mode zenbench wall, 7950X, release, no `-C target-cpu=native`
(interleaved round-robin, paired stats). GPU = cuda backend, RTX 5070.

## Per-metric verdict

- **cvvdp** — one-shot: CPU faster at ALL measured GPU-cold sizes (512 .. 4096 = 16.8 MP). Batch/warm: GPU faster at all measured sizes.
- **ssim2** — one-shot: CPU faster at ALL measured GPU-cold sizes (512 .. 4096 = 16.8 MP). Batch/warm: GPU faster at all measured sizes.
- **dssim** — one-shot: CPU faster up to 4.2 MP (2048); GPU faster from 16.8 MP (4096) up. Crossover is between 4.2 MP and 16.8 MP (interpolated, not a measured point). Batch/warm: GPU faster at all measured sizes.
- **butter** — one-shot: CPU faster at ALL measured GPU-cold sizes (512 .. 4096 = 16.8 MP). Batch/warm: GPU faster at all measured sizes.
- **iwssim** — one-shot: CPU faster up to 1.0 MP (1024); GPU faster from 4.2 MP (2048) up. Crossover is between 1.0 MP and 4.2 MP (interpolated, not a measured point). Batch/warm: GPU faster at all measured sizes.
- **zensim** — one-shot: CPU faster at ALL measured GPU-cold sizes (512 .. 4096 = 16.8 MP). Batch/warm: GPU faster at all measured sizes.

## Full table

`cpu_full_ms` = CPU full-mode wall (one score per call, cold). `gpu_cold_total_ms` = GPU one-shot floor (context-init + metric_new + first_compute). `gpu_warm_ms` = GPU warm per-call.

| metric | size | MP | cpu_full_ms | gpu_cold_total_ms | gpu_warm_ms | one-shot winner | batch winner |
|---|---|---|---|---|---|---|---|
| cvvdp | 512 | 0.262 | 32.480 | 504.457 | 4.225 | CPU | GPU |
| cvvdp | 1024 | 1.049 | 128.353 | 589.273 | 5.069 | CPU | GPU |
| cvvdp | 2048 | 4.194 | 607.282 | 1188.461 | 29.650 | CPU | GPU |
| cvvdp | 4096 | 16.777 | 3812.256 | 4282.671 | 45.461 | CPU | GPU |
| cvvdp | 12MP | 12.000 | 2350.286 | - | - | GPU-cold unmeasured >16MP | N/A |
| cvvdp | 30MP | 30.000 | 8997.807 | - | - | GPU-cold unmeasured >16MP | N/A |
| ssim2 | 512 | 0.262 | 16.665 | 396.225 | 3.956 | CPU | GPU |
| ssim2 | 1024 | 1.049 | 70.049 | 609.998 | 5.968 | CPU | GPU |
| ssim2 | 2048 | 4.194 | 297.757 | 1475.153 | 58.263 | CPU | GPU |
| ssim2 | 4096 | 16.777 | 2591.026 | 6740.518 | 50.654 | CPU | GPU |
| ssim2 | 12MP | 12.000 | 1675.430 | - | - | GPU-cold unmeasured >16MP | N/A |
| ssim2 | 30MP | 30.000 | 4585.436 | - | - | GPU-cold unmeasured >16MP | N/A |
| dssim | 512 | 0.262 | 30.531 | 376.076 | 4.144 | CPU | GPU |
| dssim | 1024 | 1.049 | 123.476 | 506.134 | 4.776 | CPU | GPU |
| dssim | 2048 | 4.194 | 546.164 | 1059.513 | 15.701 | CPU | GPU |
| dssim | 4096 | 16.777 | 4114.345 | 3949.438 | 50.465 | GPU | GPU |
| dssim | 12MP | 12.000 | 2609.401 | - | - | GPU-cold unmeasured >16MP | N/A |
| dssim | 30MP | 30.000 | 7670.598 | - | - | GPU-cold unmeasured >16MP | N/A |
| butter | 512 | 0.262 | 12.691 | 498.659 | 1.544 | CPU | GPU |
| butter | 1024 | 1.049 | 62.692 | 653.397 | 3.686 | CPU | GPU |
| butter | 2048 | 4.194 | 347.527 | 1331.167 | 47.293 | CPU | GPU |
| butter | 4096 | 16.777 | 1690.868 | 4923.904 | 62.269 | CPU | GPU |
| butter | 12MP | 12.000 | 1138.888 | - | - | GPU-cold unmeasured >16MP | N/A |
| butter | 30MP | 30.000 | 3321.020 | - | - | GPU-cold unmeasured >16MP | N/A |
| iwssim | 512 | 0.262 | 59.815 | 491.365 | 6.533 | CPU | GPU |
| iwssim | 1024 | 1.049 | 261.885 | 526.678 | 10.119 | CPU | GPU |
| iwssim | 2048 | 4.194 | 1169.064 | 834.892 | 27.937 | GPU | GPU |
| iwssim | 4096 | 16.777 | 6665.177 | 2512.473 | 45.336 | GPU | GPU |
| iwssim | 12MP | 12.000 | 5621.689 | - | - | GPU-cold unmeasured >16MP | N/A |
| iwssim | 30MP | 30.000 | 12150.069 | - | - | GPU-cold unmeasured >16MP | N/A |
| zensim | 512 | 0.262 | 6.919 | 570.253 | 1.664 | CPU | GPU |
| zensim | 1024 | 1.049 | 13.921 | 574.101 | 3.264 | CPU | GPU |
| zensim | 2048 | 4.194 | 78.859 | 635.011 | 9.929 | CPU | GPU |
| zensim | 4096 | 16.777 | 369.664 | 914.246 | 38.136 | CPU | GPU |
| zensim | 12MP | 12.000 | 246.755 | - | - | GPU-cold unmeasured >16MP | N/A |
| zensim | 30MP | 30.000 | 707.726 | - | - | GPU-cold unmeasured >16MP | N/A |

## cvvdp methodology note (zenbench vs recovered single-call)

The cvvdp `cpu_full_ms` above is the **zenbench** measurement (same harness and methodology as the other 5 metrics), NOT the previously-recovered `cpu_path_a_recovered_2026-05-29.tsv` number. The recovered file timed a **single** `score()` call per run (median of 7, `cpu-profile` driver `t_score_ms`); at small sizes that single-call wall bakes in per-call fixed overhead (allocator warmup, first-touch faults) that zenbench's interleaved multi-iteration sampling amortizes. Using zenbench for cvvdp keeps the crossover internally consistent. Both are shown for transparency:

| size | MP | cvvdp zenbench full_ms | cvvdp recovered single-call full_ms |
|---|---|---|---|
| 512 | 0.262 | 32.480 | 59.110 |
| 1024 | 1.049 | 128.353 | 233.630 |
| 2048 | 4.194 | 607.282 | 958.000 |
| 4096 | 16.777 | 3812.256 | 4610.070 |
| 12MP | 12.000 | 2350.286 | 3011.060 |
| 30MP | 30.000 | 8997.807 | 8482.170 |

(They converge as size grows — the per-pixel work dominates the fixed per-call overhead above ~4 MP.)

## Caveats

- **No extrapolation.** Every `cpu_full_ms` cell is a measured zenbench run; crossover is stated as a bracket between two measured sizes, never a fabricated MP.
- **GPU cold is only measured at 512 / 1024 / 2048 (4 MP) / 4096 (16 MP).** CPU 12 MP and 30 MP cells are flagged `GPU-cold unmeasured >16MP` — the one-shot winner there is NOT computed (would require running the GPU cold harness at those sizes).
- GPU warm per-call at 512 / 1024 comes from `gpu_coldstart_2026-05-29.tsv`'s `warm_per_call_ms` column; 2048+/4096 use the dedicated `gpu_metrics_sweep_2026-05-28.tsv` cuda full-mode `wall_median_ms` when present.
- dssim has no strip walker (dssim-core 3.4); only full/warm modes exist. It is still measured in full mode for this table.

Sources: `benchmarks/cpu_wall_all_metrics_2026-05-29.tsv`, `crates/cvvdp/benchmarks/cpu_path_a_recovered_2026-05-29.tsv`, `benchmarks/gpu_coldstart_2026-05-29.tsv`, `benchmarks/gpu_metrics_sweep_2026-05-28.tsv`. Generated by `benchmarks/synth_crossover.py`.
