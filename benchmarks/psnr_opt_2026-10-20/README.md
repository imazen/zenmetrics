# PSNR route optimization evidence (2026-10-20)

Eight classical PSNR routes in `crates/zenmetrics-cli/src/metrics/classical.rs`
re-pointed at attune-mode magetypes SIMD kernels
(`classical/psnr_kernels.rs`), bit-identical scores at every measured cell.

- `grid.tsv` — ms/call, baseline (`aic_perf_wall` at master e6b4e3b2) vs
  candidate, plus score `bits=` equality per cell. Host lilith (Ryzen 9
  7950X, dispatched tier v4x), single worker.
- `baseline_oracle.txt` — frozen baseline scores incl. `bits=` per route
  at 1024x1024 / 333x7 / 1x1.
- `source_hashes.txt` — sha256 of baseline source + baseline example
  executable and the candidate sources.
- `exhaustive_rgb8.txt` — all-16.7M-RGB8-triple conversion checks:
  `floor(x+0.5)` rounding equals `f32/f64::round` on every kernel path
  (0 mismatches); f32 candidate variants differ (68/16/4/160) and stay
  measurement-only.
- `f32_candidates.tsv` — timings of the rejected f32 variants
  (`psnr_opt_wall bench`).
