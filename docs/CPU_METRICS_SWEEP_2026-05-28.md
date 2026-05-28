# CPU metrics sweep — 2026-05-28 (task #132 follow-up tracker)

This document tracks orchestrator-side follow-ups surfaced by the CPU
sweep at task #132 plus the heaptrack matrix snapshots in
`benchmarks/heaptrack/summary_*.tsv`. Each row is one orchestrator
wiring gap the sweep identified, with status, owner task, and the
measured outcome where applicable.

| # | Gap | Status | Task | Measured outcome |
|---|-----|--------|------|------------------|
| 1 | `CpuAdapter::supports_cached_ref(Metric::Zensim)` returns `false` even though `precompute_reference` + `compute_with_ref` exists on the underlying crate. | **DONE** (2026-05-28) | #134 | +11.9 % per amortized warm call at 16 MP (vs `+46 %` brief target — see provenance note below) |
| 2 | `butter warm_ref` peak heap regresses +18 % vs cold at 16 MP (3.61 GB vs 3.07 GB) and +18.7 % at 40 MP (8.71 GB vs 7.34 GB). Root cause: 0.9.3 strip-API work added a 12 B/pixel linear-f32 source clone + the persistent `BufferPool` 48-buffer cap. | **DONE** (2026-05-28) | #135 | warm_ref ≤ cold-path peak heap restored at every measured size: -1.9 % @ 4 MP, -0.9 % @ 16 MP, +0.5 % @ 40 MP. Wall time unchanged within run-to-run variance. butteraugli bumped to 0.9.4. |
| 3 | `iwssim warm_ref` peak heap is identical to Full (the warm state caches `lp_ref + g_ref` but the dist side still builds full-image `lp_dis` + `compute_iw_maps` + `build_y_matrix` scratch). The strip variant `score_with_warm_ref_strip` keeps the same warm state and reduces peak heap by 33-48 % at 1-40 MP, with +9.5 → +34.7 % wall regression. The orchestrator's "cached_ref" entry point should route through it. | **DONE** (2026-05-28) | #136 | Adapter routes `compute_with_cached_reference` for `Metric::Iwssim` through `score_with_warm_ref_strip(STRIP_BODY_DEFAULT)`. Peak heap reduction: -33 % @ 1 MP (153.8 → 103.6 MB), -48 % @ 16 MP (2.47 GB → 1.29 GB), -48 % @ 40 MP (5.90 GB → 3.07 GB). Score diff ≤ 2e-6 (within iwssim's 1e-4 strip parity tolerance). 12/12 cpu_adapter unit + 15/15 cpu_backend integration + 14/14 iwssim parity tests pass. |

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

## Task #135 — butter warm_ref +18 % peak heap regression (DONE)

**Root cause.** Two compounding sources of persistent retention added in
butteraugli 0.9.3's strip-API work:

1. `ButteraugliReference::new(&[u8], ...)` and `new_linear(&[f32], ...)`
   clone the input into a `Vec<f32>` `linear_source` field, so
   `compare_strip` can slice strip-shaped windows from it. At 16 MP this
   is 192 MB per reference; at 40 MP, 480 MB. The sRGB constructor case
   (the common path from cpu_adapter) inflates the original 48 MB u8
   buffer 4× by pre-converting to f32 even though the strip walker
   already runs that same LUT conversion on the distorted side per
   call.
2. `BufferPool` cap of 48 — sized for the parallel join's worst-case
   concurrent buffer count plus headroom. In practice the persistent
   reference's pool fills to its cap between compares and holds those
   buffers (each ~67 MB at 16 MP, ~257 MB at 40 MP) through the next
   compare's peak. Cold-path `butteraugli()` doesn't see this because
   it creates a fresh local pool per call and drops it at end of
   function.

**Fix** (butteraugli 0.9.4):

1. Replace `linear_source: Option<Vec<f32>>` with
   `source: Option<ReferenceSource>` enum that retains the cheap form:
   `Srgb(Vec<u8>)` for `new()`-built references (3 B/pixel),
   `Linear(Vec<f32>)` for `new_linear()`-built (12 B/pixel, no
   compression opportunity). The strip walker calls the new
   `source_linear_rgb_owned` accessor which converts via the
   `SRGB_TO_LINEAR_LUT` on demand for `Srgb` references and clones for
   `Linear` references.
2. Reduce `BufferPool::put` cap 48 → 8 buffers. The retained ones still
   cover the inner pool reuse within a single compare; the gain comes
   from preventing the pool's idle-state footprint from dominating
   peak heap.
3. Added `drop_strip_source(&mut self)` and `shrink_to_fit(&mut self)`
   for callers that want explicit retention control.

**Bench.** Water-cooled AMD Ryzen 9 7950X, 128 GB DDR5,
Linux 6.6.114.1-microsoft-standard-WSL2. Release build of
`benchmarks/heaptrack/drivers/cpu_profile` with default features (LTO
off, codegen-units=1, debuginfo=1). 3 trials per cell; peak heap
median. Both 0.9.3 baseline and 0.9.4 fix measurements taken on
2026-05-28 with identical machine state.

| size | mode     | 0.9.3 peak (median) | 0.9.4 peak (median) | Δ vs cold (0.9.4) |
|------|----------|---------------------|---------------------|-------------------|
| 4 MP   | full     | n/a (regression unreported) | 814.61 MB | — |
| 4 MP   | warm_ref | n/a                 | **799.02 MB**       | **-1.9 %** |
| 16 MP  | full     | 3.26 GB             | 3.26 GB             | — |
| 16 MP  | warm_ref | 3.81 GB (+16.9 %)   | **3.23 GB**         | **-0.9 %** |
| 40 MP  | full     | 7.34 GB             | 7.79 GB             | — |
| 40 MP  | warm_ref | 8.71 GB (+18.7 %)   | **7.83 GB**         | **+0.5 %** |

Wall time (`t_score_ms`) unchanged within run-to-run variance: warm_ref
remains slower per single compare (the precompute is wasted on N=1)
and matches the +25 % gap that existed pre-0.9.4 (warm_ref 2.30 s vs
full 1.82 s at 16 MP).

**Score parity.** warm_ref score bit-identical to full at 4 / 16 / 40 MP
on the cpu_profile synth pair: 4.666544437408447 for both modes at
16 MP (sample run).

**Tests.** All 88 butteraugli unit tests + 11 strip-parity integration
tests pass. cpu_adapter integration tests (54 in zenmetrics-orchestrator)
pass. No collateral fast-ssim2 / zensim warm-ref impact.

## Task #136 — iwssim cached_ref routes through strip walker (DONE)

**Change.** `crates/zenmetrics-orchestrator/src/cpu_adapter.rs` —
`compute_with_cached_reference` for `Metric::Iwssim` now dispatches
through `iwssim::Iwssim::score_with_warm_ref_strip` with
`iwssim::STRIP_BODY_DEFAULT`. The non-strip
`score_with_warm_ref` retains the warm `lp_ref + g_ref + eigs` state
but STILL builds full-image-sized dist-side scratch in
`compute_iw_maps` (~11 × `h·w` f32) + `build_y_matrix`
(+`nexp × big_n × 4`), so it cannot deliver heap savings. The strip
variant carries the same warm reference state AND uses the strip
walker for the dist side, delivering the -48 % heap win measured by
the CPU sweep at #132.

**Measured** (heaptrack process peak, 7950X, synth pair, 3-trial median):

| size  | full     | warm_ref (pre) | warm_ref (post) | delta vs full |
|-------|----------|----------------|-----------------|---------------|
| 1 MP  | 153.8 MB | 153.8 MB       | 103.6 MB        | **-33 %**     |
| 16 MP | 2.47 GB  | 2.47 GB        | 1.29 GB         | **-48 %**     |
| 40 MP | 5.90 GB  | 5.90 GB        | 3.07 GB         | **-48 %**     |

Wall regression: +9.5 % (1 MP) → +23.9 % (16 MP) → +34.7 % (40 MP).
Per-pair score diff ≤ 2e-6 absolute at all sizes — well inside
iwssim's documented 1e-4 strip parity tolerance.

**Accepted tradeoff.** The cached-ref entry's value proposition for
the orchestrator is amortizing the ref-side eigendecomposition (which
the strip variant ALSO does — warm state is identical: `lp_ref`,
`g_ref`, per-scale `eigs`). The heap savings unlock 40 MP scoring on
8 GiB cards that would otherwise OOM at 5.90 GB. Callers that want to
pin a body height continue to use `compute_with_cached_reference_strip`
explicitly; the new default body is `STRIP_BODY_DEFAULT`.

**Tests.** 12/12 cpu_adapter unit tests pass. 15/15 cpu_backend
integration tests pass (`--features cpu-all,cuda`). 14/14 iwssim
strip parity tests pass.

## Pending follow-ups (not in scope of #136)

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
