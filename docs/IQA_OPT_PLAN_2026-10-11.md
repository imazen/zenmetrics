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

## Priority queue

- [x] P0: enable `parallel`+`avx512` dep-edge features (cli+api) — avx512
      OFF for cvvdp/iwssim/nlpd until v4 arms exist (compile-verified).
- [ ] P0: mdctpsnr — rayon via errorline column-striping (per-element
      product order preserved → bit-identical) + parallel push_row bands.
- [ ] P0: nlpd — vectorize filter5_h/blur5/normalize, scratch reuse, add
      `parallel`+`avx512`-safe kernels.
- [ ] P1: cvvdp CSF vectorization (sol-math, gather+vexp).
- [ ] P1: zenmetrics-math crate lands; par.rs consolidation.
- [ ] P1: magetypes v4 fills (`I32x8Backend`+`F32x8Convert`+`F64x4Backend`)
      — upstream PR, NOT critical path (native-width route avoids them).
- [ ] P1: mad — root-cause 3.84× superlinear (FFT-vs-prime-factor dims?).
- [ ] P2: hdrvdp3 f64 — f64xN transcendentals (polyfit-assisted fitting) or
      scoped f32 conversion under JOD parity gate.
- [ ] P2: vifvec steerable pyramid; ssim classical cache-bound (halo/strip);
      ssim2 thread scaling; msssim-libvmaf.
- [ ] Serialize: full `cpu-metrics-1t8t` re-run pinned, all sizes + heaptrack
      RSS, after lanes merge (zenbench affinity or taskset).

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
