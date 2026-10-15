# vsi exact NEON srgb_to_lab — RESULT

- Commit pushed to master: 41f47c9f `perf(vsi): exact NEON srgb_to_lab via fdlibm pow64_libm_4x` (safe_push VERIFIED).
- Change: the f64x4 body macro is now token-generic (`srgb_to_lab_exact_body!`, cfg x86_64|aarch64);
  `srgb_to_lab_tier_neon_t` uses it with `NeonToken`/`pow64_libm_4x`. The f32 midp path is removed.
- Parity: wall 1024x1024 score bits `3fefb77a83788424` (matches x86 target). Approx baseline was `3fefb77a83dadd96`.
  Tests: `tier_parity` now has an aarch64 arm (bitwise); new `srgb_to_lab_dispatch_matches_scalar_bitwise`
  (note tier_parity alone shares `prepare`, so it never exercised Lab — the new test does).
  `cargo test -p vsi --features _dev` and `--release --features _dev,parallel`: 18/18 pass.
- Bench (M4 Pro, load ~1.5-2.0, `--features parallel`, interleaved A/B, 20 reps, 1024x1024):
  approx 6.88/6.07/6.02/6.07 ms; exact 7.53/7.44/7.50/7.19 ms  => exact ~1.22x slower (~+1.4 ms).
  256x256 (100 reps): approx 3.30 ms, exact 4.59 ms (~1.39x).
- Deviations: `run-heavy` does not work on macOS (needs ionice//proc); used plain `nice -n19`.
  Numbers taken under light shared load, not an idle box. Cargo.lock churn from local resolution was restored, not committed.
  Native f64x2 variant not tried (f64x4 delegation overhead not profiled).
