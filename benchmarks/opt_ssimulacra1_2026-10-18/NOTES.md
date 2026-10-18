# opt_ssimulacra1_2026-10-18 — SIMD blur + channel parallelism

Flag source: ours ~3x slower than the Cloudinary ssimulacra1 C++ reference
(commit 1fa99a4) at >=1MP on lilith-wsl — canonical process wall ~180ms at
1280x720 / ~1.6s at 3840x2160 vs ours 534ms / 4.84s (warm score, pre-change).

## What changed (crates/zenmetrics-cli/src/metrics/ssimulacra1.rs)

- `blur()` was fully scalar; every interior pixel emitted a libcall to
  `fmaf` (baseline x86-64 has no FMA target feature) — 33% of cycles in
  `libm::__powf_fma`-adjacent mul_add shims per `perf`.
- Replaced the inner loops with archmage-attuned kernels: `blur_passes`
  (f32x8 for v3/neon/wasm128/scalar tiers) + f32x16 overload behind the
  opt-in `avx512` crate feature. Every lane runs the same ordered
  fused-multiply-add chain as the scalar reference (`mul_add_portable` =>
  one rounding per tap on every backend) => bit-identical across tiers.
- Scalar `hrow_px`/`vcol_px` handle the BORDER_REFLECT_101 edges; the
  vector loops only cover the interior.
- `blur_into` reuses the `tmp` scratch across the ~90 blur calls per score.
- The three channels per scale are computed via rayon `into_par_iter`
  (size-gated at >=32Kpx); the f64 `total`/`norm` accumulation stays serial
  in channel order, so results are bit-identical at any thread count.
- New opt-in feature `avx512 = ["archmage/avx512", "magetypes/avx512"]`.

## Parity

- `score_bits` identical before/after on every measured cell (summary.tsv).
- `just snap-cargo @ test -p zenmetrics-cli ssimulacra1`: 4/4 pass incl.
  `blur_interior_fast_path_matches_ordered_reference` (SIMD vs ordered
  scalar reference) and `original_author_synthetic_pin` (1e-12 vs the
  committed canonical fixture).
- `scripts/ci/lock.sh --check`: PASS (lock regenerated via --regen).

## Timing vs canonical (lilith-wsl, warm median, iqa-compete `score`)

| size | ours before | ours after | canonical (process wall) | ratio after |
|---|---|---|---|---|
| 1280x720 t1 | 533.9ms | 138.3ms | ~180ms (incl. PNG decode) | 0.77x |
| 3840x2160 t1 | 4842ms | 1191ms | ~1600ms | 0.74x |

Post-change the port is faster than the canonical wall at every measured
size; the >=1.5x regression flag clears with margin even if canonical is
timed compute-only (~150ms at 1MP => ours ~0.92x).

Benchmarks above ran the default-feature build: `attuned!` dispatches to
`_v3` (f32x8) on this AVX-512-capable host (`simd_tier: v4-capable` in the
row logs is the CPU-capability report, not the active arm). Enabling the
opt-in `avx512` feature activates the f32x16 overload on such hosts.
