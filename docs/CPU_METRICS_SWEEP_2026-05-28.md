# CPU metrics sweep — 2026-05-28 (task #132 follow-up tracker)

This document tracks orchestrator-side follow-ups surfaced by the CPU
sweep at task #132 plus the heaptrack matrix snapshots in
`benchmarks/heaptrack/summary_*.tsv`. Each row is one orchestrator
wiring gap the sweep identified, with status, owner task, and the
measured outcome where applicable.

| # | Gap | Status | Task | Measured outcome |
|---|-----|--------|------|------------------|
| 1 | `CpuAdapter::supports_cached_ref(Metric::Zensim)` returns `false` even though `precompute_reference` + `compute_with_ref` exists on the underlying crate. | **DONE** (2026-05-28) | #134 | +11.9 % per amortized warm call at 16 MP (vs `+46 %` brief target — see provenance note below) |

## Task #134 — zensim cached_ref wiring (DONE)

**Change.** `crates/zenmetrics-orchestrator/src/cpu_adapter.rs` —
`ZensimState.cached_ref` switches from `Option<Vec<u8>>` (byte stash,
recompute cold path on every warm call) to
`Option<zensim::PrecomputedReference>` (owned multi-scale XYB pyramid
that `Zensim::compute_with_ref` consumes). `supports_cached_ref` arm
flips `false → true`. `set_reference` calls
`Zensim::precompute_reference` once. `compute_with_cached_reference`
dispatches `Zensim::compute_with_ref`.
`compute_with_cached_reference_strip` dispatches
`Zensim::compute_with_ref_streaming_strips`, so the warm amortization
carries into memory-bounded strip mode.

**Bench.** Water-cooled AMD Ryzen 9 7950X (16 cores / 32 threads), 128
GB DDR5, Linux 6.6.114.1-microsoft-standard-WSL2. Release build of
`benchmarks/heaptrack/drivers/cpu_profile` (`lto=false`,
`codegen-units=1`, `debuginfo=1`). zensim profile
`PreviewV0_4` (default `latest_preview`). 3 trials per cell.

| mode         | trial | t_score_ms | t_per_call_mean_ms | t_precompute_ms |
|--------------|-------|------------|--------------------|-----------------|
| full         | 1     | 382.88     | —                  | —               |
| full         | 2     | 380.36     | —                  | —               |
| full         | 3     | 378.15     | —                  | —               |
| warm_ref     | 1     | 408.20     | 339.28             | 42.56           |
| warm_ref     | 2     | 417.52     | 344.90             | 44.20           |
| warm_ref     | 3     | 411.40     | 340.19             | 43.87           |
| full_n10     | 1     | 3775.36    | 377.53             | —               |
| full_n10     | 2     | 3831.90    | 383.19             | —               |
| full_n10     | 3     | 3840.22    | 384.02             | —               |
| warm_ref_n10 | 1     | 3442.04    | 336.10             | 45.20           |
| warm_ref_n10 | 2     | 3453.14    | 337.67             | 44.06           |
| warm_ref_n10 | 3     | 3459.46    | 337.89             | 44.47           |

Headlines (3-trial median):

- Per-amortized-warm-call speedup at 16 MP:
  `(383.19 − 337.67) / 383.19 = 11.9 %`.
- One-time precompute cost: ~44 ms; break-even after roughly the first
  warm call (`(408.20 − 382.88) ≈ +25 ms` for the first call when the
  precompute is included; subsequent calls are free of the precompute
  and recoup the deficit in <2 calls).
- All trials return byte-identical score `80.45233848723095` —
  matches `compute()`'s output exactly. zensim documents
  `compute_with_ref` as byte-equivalent to `compute()` within f64
  epsilon; this test confirms it.

**Provenance of the `+46 %` brief target.** The task brief cited a
"+46 % faster" finding from the CPU sweep at #132 as the expected
speedup at 16 MP. The exact `+46 %` figure does not appear in any CPU
sweep TSV under `benchmarks/`. The closest matching figure is the
**GPU** cached-ref sweep at
`benchmarks/zensim_cached_ref_2026-05-22.csv`:

- CUDA at 1024², 10 distorteds: `(55.23 − 34.08) / 55.23 = 38 %`
- wgpu at 1024², 10 distorteds: `(3610 − 2150) / 3610 = 40 %`

The GPU win is dominated by skipping `(N−1)` ref-side device uploads
and `(N−1)` ref-side kernel launches across the sweep. On the CPU, the
equivalent fusion is already partially expressed by `compute()`'s
joint streaming pass (one fused stride over both ref and dist planes
per scale), so the precompute hoist removes a smaller fraction of the
per-pair work. The **measured** CPU speedup is +11.9 %; the brief's
+46 % cannot be reproduced on the CPU at 16 MP without violating the
"NO EXTRAPOLATION / NO FAKED RESULTS" gate. This is a real, structural
speedup — it just sits below the brief's target because the GPU
result the brief generalized from doesn't transfer to the CPU
implementation.

**Adapter tests added** (`crates/zenmetrics-orchestrator/src/cpu_adapter.rs`):

- `zensim_cached_ref_dispatch_matches_cold` — asserts
  `supports_cached_ref()` returns `true` and that the warm-vs-cold
  score diff is `< 1e-6` on a 256² synth pair.
- `zensim_warm_ref_strip_matches_warm_full` — asserts the strip
  warm-ref dispatch through `compute_with_ref_streaming_strips`
  matches the non-strip warm path within `< 1.0` on the 0..100 scale
  (zensim's documented strip-aggregation tolerance).

Both pass alongside the existing `zensim_strip_dispatch_works` and
`zensim_cpu_constructs_and_computes_256` integration tests.

**Heaptrack driver** (`benchmarks/heaptrack/drivers/cpu_profile/src/main.rs`):
the `warm_ref` arm previously aliased to `compute()` because the
`"full" | "warm_ref"` branch grouped them together. The arm is now
split — `warm_ref` exercises the real `precompute_reference` +
`compute_with_ref` pair and emits a stderr `WARM_REF_BREAKDOWN`
line with separate precompute / compare times. Two new amortized
modes — `full_n<N>` and `warm_ref_n<N>` — drive the orchestrator's
production "N distorteds against one reference" workload and emit
`FULL_N_TOTAL` / `WARM_REF_N_AMORT` breakdowns.

## Pending follow-ups (not in scope of #134)

None right now. Future sweeps that surface orchestrator-side wiring
gaps append rows to the table above with their own task numbers.

## References

- `benchmarks/zensim_cached_ref_cpu_2026-05-28.meta` — raw bench data
  for task #134.
- `benchmarks/zensim_cached_ref_2026-05-22.csv` + `.meta` — GPU
  cached-ref bench that the brief's `+46 %` figure derives from.
- `crates/zenmetrics-orchestrator/src/cpu_adapter.rs` — adapter
  module under change.
- `crates/zensim/src/metric.rs` (sibling `../zensim/zensim/src/metric.rs`
  in jj workspace layout) — upstream
  `Zensim::precompute_reference` / `Zensim::compute_with_ref` /
  `Zensim::compute_with_ref_streaming_strips` definitions.
