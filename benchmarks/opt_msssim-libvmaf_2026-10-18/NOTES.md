# opt_msssim-libvmaf_2026-10-18 — size-gate rayon dispatch

Flag source (measured independently on lilith-wsl vs the same
libvmaf-compute shim portcheck uses): ours-vs-libvmaf at 255x257 with a
32-thread pool was 21.9ms vs 8.9ms (float_ssim, 2.5x) and 92.7ms vs
11.8ms (float_ms_ssim, 7.9x) — far over the 1.5x flag threshold. Ours was
also ~21-53x slower than *itself at one thread* — a rayon dispatch storm,
not real work.

## What changed (crates/msssim/src/{kernel,libvmaf}.rs)

Every `parallel` rayon dispatch is now gated on plane size:

- `kernel::PAR_MIN = 1 << 21` (2M samples) — measured crossover on
  lilith-wsl: below ~1-2MP the work-stealing round-trip costs more than
  the per-plane op; at 4K the level-0 convolutions still win in parallel.
- `par_ok(n) = n >= PAR_MIN && current_num_threads() > 1` replaces bare
  `current_num_threads() > 1` in `for_each_row`, `maybe_join` (gained an
  `n` arg), `map2`, `fold_ranges`, the smap/cmap banded fill, and the
  `rd` elementwise product in libvmaf.rs.
- Bit-identical at every thread count: the serial path was already the
  bit-exact reference (the parallel splits only redistribute identical
  bands/folds — documented in the crate); the gate just picks serial
  below the crossover.
- `msssim` (the crate's own multi-scale metric) shares kernel.rs and
  gets the same fix: 255x257 t32 12.4ms -> 0.55ms.

## Parity

- score_bits identical on every measured cell (summary.tsv); the
  striped-fold comment in fold_ranges already asserted serial==parallel.
- `just snap-cargo @ test -p msssim --all-features`: 13 lib + 6
  ffi_libvmaf tests pass (FFI parity vs real libvmaf).

## Timing vs canonical (libvmaf-compute shim, vmaf-head-sys build, lilith-wsl)

| cell | ours before t32 | ours after t32 | canonical t32 | ours/canon after |
|---|---|---|---|---|
| ssim-libvmaf 255x257 | 21.9ms | 1.12ms | 8.9ms | 0.13x |
| msssim-libvmaf 255x257 | 92.7ms | 1.79ms | 11.8ms | 0.15x |
| msssim-libvmaf 3840x2160 | 227ms | 282ms | ~1700ms | 0.17x |

Canonical libvmaf itself degrades under 32 threads (it fans out feature
extractors); the flag fired because ours collapsed much harder.
