# BRISQUE canonical-speed rework, 2026-10-19

## What changed

`crates/brisque` was reworked for CPU speed with the archmage/magetypes
generic-SIMD pattern used by `crates/gmsd` (`src/kernels.rs`):

- Both MSCN 7×7 convs run as one `v3`/`v4x`/`neon` kernel over a 3-px
  zero-padded plane, x² read on the fly. Every output keeps the scalar
  oracle's sequential row-major FMA chain, so each tier is bitwise-identical
  (padded zeros contribute the same `k·0` products the bounds check picked;
  hardware FMA = `fmaf`). Scalar tier = the same `fmaf` chain.
- sRGB→Y luma is a `v3`/`v4x`/`neon` i32-extract kernel; `m01 = 0.299R +
  0.587G` stays two roundings, the `0.114B` term one FMA, then ties-even
  round — bitwise vs the scalar oracle.
- MSCN normalize and the vertical resize pass use only correctly-rounded
  ops and run on every tier incl. scalar/wasm128. The resize dot keeps
  `outer_sum`'s column tree (lazy leaves, no prod materialization; the
  `ks = 10` case is unrolled in the same order).
- The GGD/AGGD map reductions now accumulate in f64 fixed 8-lane sums
  (`summation::LaneSums8`, gmsd's lane convention): element `i` adds to
  lane `i & 7`, lanes combine `[a0+a4, a1+a5, a2+a6, a3+a7]` then `t0+t1+t2+t3`.
  Order is identical on every backend; f64 accumulation is near-exact.
  The roll/prod/sl2/sr2/abs/x2 materializations are gone — one fused pass.

## Correctness evidence

- Stage-1 (order-preserving changes only): **all 2016 konjnd-pairs images
  bitwise-identical** feature+score vs the previous implementation
  (`/tmp/brisque-ab`, base workspace `f5090db1`).
- Stage-2 (f64 lanes): 2016-image drift vs old impl — max |Δscore| 0.0117
  (0.774× JND/1000), driven by ~1e-8 feature-ulp changes through argmin
  grid flips; feature deltas ≤ 1e-3 (one gam step).
- Golden gate (oracle piq 0.8.0, JND/1000 = 0.015138428): the full
  170-row `tests/goldens_brisque.tsv` panel passes locally — 16 rows
  bitwise-exact, worst |Δ| = 8.301e-3 = 5.483e-4 JND (previous impl on the
  128 locally-verified konjnd rows: 0.0073). The missing `nn-phase0/ladder`
  corpus was regenerated with `benchmarks/nn_lpips_phase0_2026-10-15/scripts/
  ladder.py` from the public R2 imazen-26 originals: all 42 PNGs
  byte-identical to `ladder_manifest.tsv` sha256s.
- Cross-arch: 28 procedural cases on `zen-arm-xl` (NEON) produce **identical
  scores** to x86-64 AVX2 — the conv luma/normalize/vres paths are
  tier-consistent by construction.
- `cargo test -p brisque`: lib + behavior (packed=strided, cold=warm
  bitwise, degenerate/too-small kinds) + parity (170 goldens) all pass.
- Targets compile: x86_64, i686, aarch64, wasm32, `avx512`, `no_std`.

## Speed (same-process A/B, `pipeline_debug` core, r7900x, 1 thread)

| input | old | new | ratio |
|---|---|---|---|
| 640×480 | 94.48 ms | 3.86 ms | 24.5× |

End-to-end (`evaluate_image`, incl. luma + ingress) on r7900x:

| input | old (from 2026-10-19 baseline run) | new | ratio |
|---|---|---|---|
| 640×480 | ~99 ms | 6.32 ms | ~16× |
| 1024×1024 | ~335 ms | 21.7 ms | ~15× |
| 3840×2160 | ~4550 ms | 171.5 ms | ~27× |

ARM NEON (`zen-arm-xl`, Neoverse-N1, 1 thread, core path): 101.9 →
12.9 ms = 7.9× at 640×480.

Commands: `cargo build --release --example bench` then
`target/release/examples/bench <png> <reps> [WxH-mirror-tile]`; A/B driver
at `/tmp/brisque-ab` (`--gen` procedural cases, `--bench`, corpus list
mode). Host: r7900x (Ryzen 9 7900X), glibc, 1 thread.
