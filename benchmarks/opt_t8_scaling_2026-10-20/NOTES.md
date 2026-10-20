# t1→t8 scaling fixes — 2026-10-20 (i270, 8.4 MP)

The fresh `cpu_metrics_1t8t_2026-10-20-i270` sweep flagged the metrics
whose 8-thread wall is ≈ the 1-thread wall. Four root causes were fixed;
scores are **bit-identical** across thread counts and equal to the
recorded sweep's photo-pair scores (verified below).

## Baseline → fixed (photo pair, 3355x2516, heap rep1)

Host i270 (Core Ultra 7 270K Plus; taskset 0-7; RAYON_NUM_THREADS=1|8).
"8t before" is the same-day sweep log (`cpu_metrics_1t8t_2026-10-20-i270`,
pre-change binary); 1t/8t "after" columns are the new binary.

| metric           | 1t before | 8t before | 8t after | speedup after |
|------------------|----------:|----------:|---------:|--------------:|
| vmaf             |     245.9 |     237.5 |    154.8 |         1.63x |
| vmaf-neg         |     242.5 |     235.1 |    152.9 |         1.62x |
| vmaf-4k          |     245.4 |     238.5 |    155.4 |         1.62x |
| vmaf-v1          |     160.8 |     147.3 |     55.3 |         3.04x |
| ssim-libvmaf     |      73.6 |      66.5 |     15.9 |         4.18x |
| msssim-libvmaf   |     330.3 |     158.4 |     75.4 |         4.24x |
| psnrhvs-daala    |     270.6 |     142.9 |     71.7 |         3.24x |

Synthetic-content legs showed similar ratios (vmaf×{,-neg,-4k} ~3.8x —
the photo vif finalize is ~2x the synthetic cost, so the real-content
ratio is lower).

## Root causes and fixes

1. **CLI ingress was serial** — every luma-plane metric's RGB→plane
   conversion ran scalar. `studio601_y_plane`+u16 widen fused into
   `studio601_y_plane_u16` (row-parallel ≥1MP); `studio601_yuv444` and
   `studio601_gray` row-parallel; `to_yuv420(_even)` block-row-parallel;
   `widen` par-mapped. Pure per-pixel maps → bit-exact.
2. **msssim decimate gate measured output size** — `for_each_row` gated
   on `out.len()`; a factor-10 libvmaf decimate writes ~2% of the input
   but reads the whole plane. New `for_each_row_work` gates on
   `out.len() * kw * kw` filter work. `plane_to_f32` also row-parallel.
3. **VMAF single-frame extraction was serial** — `extract_features`
   fanned out with `rayon::in_place_scope`: the heavy extractor (vif /
   adm3) stays inline on the calling thread so its **thread-local
   scratch pool stays warm**, while the smaller units spawn to workers.
   `rayon::join`/`scope` from a non-worker thread migrate *every* branch
   to pool workers with cold scratch (~100 ms of page faults at 8.4 MP).
4. **vif statistics was serial** — row loop extracted to `stat_rows`;
   row bands accumulate disjoint i64 partials (integer addition is
   order-free → bit-exact) via `in_place_scope` over ≤16 bands.
   Extractor-level vif: 138→30 ms at t8 (synthetic 8.4 MP).

## Verification

- `cargo test -p msssim --all-features`: 13 lib + 6 FFI parity tests pass
  (`float_ssim_parity`, `float_ms_ssim_parity`, `psnr_hvs_parity`).
- `cargo test -p vmaf --features simd,avx512,parallel`: 43 lib + 30 FFI
  fusion tests pass (`v0_fusion_matches_v321`, `v0_vif_scales_match_v321`,
  `bounded_parallel_frames_match_serial_bit_for_bit`, etc.).
- `cargo test -p zenmetrics-cli --features cpu-metrics studio601`: the
  studio-601 ingress tests pass (per-triple round parity).
- Photo-pair scores (heap logs) are bit-identical 1t vs 8t and equal to
  `cpu_metrics_1t8t_2026-10-20-i270` recorded scores.

## Also fixed

- `classical_wall` example build: `metrics/classical.rs` moved to
  `metrics/classical/mod.rs` so its `mod psnr_kernels;` child resolves
  under `metrics/classical/` in both the real crate and the `#[path]`
  splice (broken by the Oct-20 kernel move).
- `cpu_metrics_1t8t.sh`: `ZENBENCH_CPU_POLICY=unrestricted` (default,
  overridable) — zenbench's cgroup-cpuset certification otherwise fails
  under run-heavy's systemd scope while taskset affinity is correct.

## Remaining serial pieces (future work)

- `vmaf` v0 fan-out wall is ~adm2+validate on one worker (~55 ms) plus
  vif ~35 ms; adm2/validate could fan out internally for another ~20 ms.
- `psnr-y`/`psnr-y601` (`luma_sqerr`): the f64 left-to-right sum is
  parity-locked to scalar order — only two-phase (parallel pixel work +
  serial fold) is available, ~1.5x ceiling on a ~10 ms metric.
- `ssim-libvmaf` tail: `iqa_ssim` on the decimated 336×252 plane stays
  serial (small); remaining wall is mostly float_ssim ingress+decimate.
