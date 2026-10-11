# IQA optimization push — conference 2026-10-13 — working plan

Status doc for the multi-lane optimization sprint. Author: devin lane (wP:p1).
Measured baseline: `benchmarks/cpu_metrics_1t8t_2026-10-09.md` (29 metrics,
1t/8t, 1 MP + 8.4 MP, r7900x Zen 4).

## Measured outliers (1t, 8.4 MP)

| metric | ms/MP | problem |
|---|---:|---|
| mdctpsnr | 7,800 | no threads; glibc-SVML `asm!` only |
| mad | 7,070 | superlinear 3.84× growth; threads only 1.80× |
| vifvec | 454 | steerable pyramid, scalar, no thread scaling |
| cvvdp | 195 | CSF stage scalar per-pixel (3×exp+LUT per px) |
| ssim (classical) | 190 | superlinear 1.67; ~1.0× threads |
| ssim2 | 113 | rayon doesn't parallelize as shipped (1.02×) |
| msssim-libvmaf | 108 | ~1.0× threads |
| iwssim | 99 | parallel unshipped; transcendentals v3-capped |
| nlpd / nlpd-iqa | 39–55 | mostly scalar; no rayon dep; alloc churn |
| hdrvdp3 | n/a | v3 module is scalar f64 (no f64xN SIMD exists) |

## Key structural findings

- `parallel` (rayon) features exist on 11 metric crates but were not enabled
  in shipped CLI/api builds. Enabling: cvvdp 2.40×, fsim 1.96×, iwssim 1.83×,
  mad 1.80×, psnrhvs 1.68×, haarpsi 1.45×, vif 1.38×, msssim 1.37×, vsi 1.35×
  at 8t (measured, Oct 9 threading re-check).
- `avx512` compiles on: gmsd, fsim, vsi, haarpsi, msssim, vif, mad-iqa,
  psnrhvs, hdrvdp. **Broken** on cvvdp, iwssim, nlpd — incant! emits v4 calls
  for `_v4` fns that were never written (f32x8 transcendentals have no
  F32x8Convert on V4; f64x4 has no V4 backend at all).
- magetypes already provides the transcendental suite: `exp/ln/log2/log10/
  pow/cbrt` at `lowp`/`midp` ± unchecked, `sigmoid_midp`, `silu_midp`, on
  f32x4/f32x8/f32x16; `F32x16Convert` exists for V4/V4x (w512+avx512).
  Missing: `I32x8Backend`+`F32x8Convert` on V4/V4x, `F64x4Backend` on V4,
  all f64xN transcendentals.
- magetypes backend traits + SimdToken are sealed (`pub(crate)` module):
  no external impls possible. The shared-math crate must be generic
  *consumers* — `fn foo<T: F32x8Convert>(token, ...)` + `incant!`, exactly
  the cvvdp/simd_math.rs pattern. Transcendental variants (bitexact
  backends) are a policy type-param in OUR crate, not new magetypes impls.
  Native width per tier (f32x16 on v4/v4x, f32x8 elsewhere) avoids ALL
  magetypes changes on the critical path.
- `par.rs` duplicated in cvvdp/iwssim/hdrvdp (n_bands + join2 + banding).
- zenbench has no affinity API → sol-zenbench lane.
- relaxed-simd FMA: already covered (`magetypes::nostd_math::fmaf`,
  `wasm_fma.rs`). No work needed.
- polyfit (lilith fork of rscarson): LSQ fitting, not minimax — use for
  domain-restricted bespoke polys (x^0.8, x^3.5, Weibull, PU21, logistic)
  feeding zenmetrics-math coefficient tables, gated by ULP harness; not for
  canonical exp/log (magetypes midp is better bounded).

## Lane assignments (herdr, all local on i265)

| lane | repo/checkout | work |
|---|---|---|
| `sol-math` (codex gpt-6.1-sol high) | `zenmetrics--solmath` (jj workspace) | new `crates/zenmetrics-math` (extract cvvdp simd_math + gather-lerp + par + policy seam, native-width tiers), migrate cvvdp, vectorize CSF |
| `sol-zenbench` (codex gpt-6.1-sol high) | `~/work/zen/zenbench` (fresh clone) | `AffinityGuard` scoped pinning + `.affinity()` bench API + `ZENBENCH_PIN` env; PR to lilith |
| `devin` (this lane) | `zenmetrics` primary | feature-edge flips (done), mdctpsnr threads, nlpd SIMD+threads |

Briefs: `~/tmp/handoff/iqa-math/brief.md`, `~/tmp/handoff/zenbench-affinity/brief.md`.

## Priority queue — status after the 04:30–05:10 push series

- [x] P0: enable `parallel`+`avx512` dep-edge features (cli+api) — avx512
      now ON for EVERY metric crate, iwssim included: archmage main
      dea3a461 landed `x86_v4_narrow_delegated.rs` (all 18 W128/W256
      f64/int backends + F32x*Convert + *Bitcast delegated to V3 for
      V4/V4x/FP16 tokens — spec-driven from the trait defs via a
      generalized xtask emitter); zenmetrics pins the three crates to
      that rev in [patch.crates-io] until the next release ships.
      iwssim's two missing arms (infow_map_into_v4/_v4x) added; verified
      dispatch on Zen4 hardware (r7900x).
- [x] P0: mdctpsnr — rayon errorline column-striping + push_row bands +
      hoisted env probes + `#[target_feature]` FMA clones of the fmaf
      libcall loops. 6.9s→1.63s @8t wall example; sweep 7740→2031 ms.
      `4ade8360` + `f1870dca`.
- [x] P0: nlpd — shared `for_each_row`/`for_each_row_init` striping all
      row filters + avx512 edge. 47.8→29.4 ms/call @8t (1.63x), identical
      scores, no 1t regression after the pool-size gate. `4a424978`.
- [x] P1: cvvdp CSF vectorization + zenmetrics-math crate — sol-math
      landed `01e28ca1` + `bd223f0c` (native-width kernels, policy seam,
      consolidated par; CSF 45.2→36.9 ms @8t; JOD delta ≤9.5e-7). cvvdp
      avx512 then enabled (`d97da263`).
- [x] P1: mad — root cause found: non-pow2 dims drop to serial scalar
      Bluestein chirp-z (~64 KiB alloc + 3 radix-2 FFTs of ~2n per row/
      col × ~42 transforms). Parallelized rows (par_chunks_mut +
      per-worker scratch), super-blocked columns mirroring the radix-2
      structure, all ical passes j-strip parallel, elementwise stages
      parallel, integral-input lum LUT — then batch8'd the Bluestein
      internals through fft_batch8 SoA (the m-length radix-2 stages
      become f32x8 ops). 3610→412 ms/call @8t at 1000×767 (8.77x),
      60,337→5,792 ms @8.4MP 8t (10.42x), identical scores.
      `ab65ee00` + `28144f17`.
- [x] P2: vifvec — 1.00x→~3.4x @8t via corr_dn rows, maybe_join
      pyramids, element-parallel mean, striped cu covariance, row-par
      ss/LU blocks + vifsub_est. `907b0f65`.
- [x] P2: ssim classical (CLI) — row-parallel gaussian passes. `01c9e8b4`.
- [x] P1: magetypes v4 fills — DONE on archmage main (dea3a461) via the
      generalized v4_delegation_gen emitter: all 18 W128/W256 f64/int
      backends + F32x4/F32x8Convert + all 10 Bitcast traits × 3 tokens.
      zenmetrics consumes it via a rev-pinned [patch.crates-io].
- [x] ssim classical — 1.03x→4.43x @8.4MP 8t (row-parallel gaussian
      passes, `01c9e8b4`); the Oct-9 "superlinear" note is resolved.
- [x] ssim2 — fast-ssim2 main 99b98cb9 parallelized the gaussian
      vertical column blocks under rayon (private per-block scratch +
      parallel row-scatter, bit-identical): whole-image 2.03x @8t
      (1024² 65.5→32.2ms, 4K 833.9→410.1ms). Reaches zenmetrics via the
      CLI's local-path 0.9.0 edge; the api + sweep driver still ride
      registry 0.8.2 — porting the api's ssim2 call sites to the 0.9.0
      PixelSlice API (3 call sites + cached_ref path) is deferred until
      the 0.9.0 release or an explicit port decision.
- [ ] P2: hdrvdp3 f64 — f64xN transcendentals (polyfit) or scoped f32
      under JOD parity gate.
- [ ] P2: msssim-libvmaf, vsi/fsim tails; vifvec's ~20% 1t regression
      (striped-cu setup); mdctpsnr residual serial ring work.
- [x] Serialize: `cpu-metrics-1t8t` re-run pinned on i265 —
      benchmarks/cpu_metrics_1t8t_2026-10-11-i265.md committed
      (mad 10.42x, vifvec 5.05x, mdctpsnr 4.60x, ssim 3.85x, cvvdp
      3.07x vs Oct-9 r7900x 8t numbers; post-sweep mdctpsnr commits
      push it to ~10.6x).

## Environment hazards found (i265)

- Local `zengif` checkout drifted past CI pin: wants `zencodec>=0.1.27`,
  crates.io has 0.1.26, and zengif holds live WIP (another session's
  fix/animation-decode-cancellation). Unblocked via gitignored
  `.cargo/config.toml` `[patch.crates-io] zencodec -> ~/tmp/zencodec-0.1.27`
  (git main, 0.1.27). Applied to BOTH zenmetrics checkouts. Resulting
  Cargo.lock churn is local-resolution noise — never commit it; restore
  Cargo.lock before pushing.
- Local `zenflate` drifted: `zenflate::png` unresolved breaks
  `zenpng`+default-feature builds. Unrelated to metrics edge; verify with
  `cargo check -p zenmetrics-cli --no-default-features --features cpu-metrics`.
- archmage repo has two other live lanes (codex + opencode) — do not edit
  magetypes from this effort; all work lands in zenmetrics-math.
