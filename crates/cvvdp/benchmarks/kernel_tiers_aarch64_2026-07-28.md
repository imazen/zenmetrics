# cvvdp transcendental kernels: per-tier NEON isolation — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/kernel_tiers.rs` (zenbench, interleaved arms), 1 M f32 per kernel

`vexp` / `vlog` / `vpow` are the hottest inner loops of the metric. Whole-metric benches
cannot reveal one of them being slower than its own scalar fallback — that failure mode was
real in this sweep (three zenfilters NEON kernels lost to their scalar tier), and
transcendentals are where it hides, because a hand-written polynomial competes against
whatever LLVM manages from the scalar body.

## Result: no losers, and the widest margins measured anywhere in this sweep

| kernel | NEON | scalar | speedup |
|---|---|---|---|
| vexp_into | 466 µs | 6764 µs | **14.5×** |
| vpow_into (p=0.42) | 889 µs | 11122 µs | **12.5×** |
| vpow_into (p=2.4) | 943 µs | 9941 µs | **10.5×** |
| vlog_into | 456 µs | 4455 µs | **9.8×** |

## Why these are so much larger than the rest of the sweep

Everywhere else on aarch64 the scalar arm is still autovectorized — NEON is baseline, so LLVM
widens ordinary slice loops and explicit SIMD often only matches it (the elementwise passes in
zenfilters correctly measure ~1.00× because they are at the memory-bandwidth wall).

Transcendentals are the exception. A scalar `f32::exp()` lowers to a **libm call per element**,
which LLVM cannot vectorize at all. So the comparison here is genuinely
"vectorized polynomial vs a million function calls", not "SIMD vs autovectorized SIMD". That
is why 10–15× is the honest number and why it should NOT be read as these kernels being
better-written than the 1.00× ones.

Inputs span 1e-3 to 40 (several decades, all positive — log and pow are undefined or
degenerate at/below zero). A narrow range would flatter a polynomial approximation.

## Note

The bench needs `archmage/testable_dispatch`, added as a dev-dependency so the baseline NEON
token can be disabled; feature unification keeps it to test/bench builds and consumers are
unaffected. Without it the bench skips loudly rather than reporting the SIMD path under both
labels.
