# zensim strip mode: hoist `PrecomputedReference` outside the strip loop

**Phase:** 9.Y (CPU heaptrack follow-up, finding #5)
**Date:** 2026-05-27
**Status:** fix landed in `cpu-profile` driver (the only in-repo
consumer); production-caller pattern documented below.

This doc records the fix for the
[`CPU_HEAPTRACK_REPORT_2026-05-27.md`](CPU_HEAPTRACK_REPORT_2026-05-27.md)
finding #5: *zensim strip mode peaks +36 % HIGHER than full at 40 MP
(re-precomputes ref per strip)*.

---

## Symptom

The heaptrack matrix recorded, at the same input size (40 MP synthetic
pair), peak heap measured by heaptrack 1.3.0:

| mode             | peak heap @ 40 MP | vs. `full` |
|------------------|------------------:|-----------:|
| `full`           | 2.64 GB           | baseline   |
| `strip`          | **3.59 GB**       | **+36 %**  |
| `warm_ref_strip` | 2.99 GB           | +13 %      |

`strip` (zensim's `compute_streaming_strips_default`) was supposed to
*bound* memory at `O(strip_h × width)` per pair; instead it cost
MORE than the unconstrained `full` path. `warm_ref_strip` (where the
caller pre-builds a full `PrecomputedReference` and feeds it to
`compute_with_ref_streaming_strips_default`) achieved the expected
strip-style memory profile.

## Root cause

Inside zensim's `compute_multiscale_stats_streaming_strips`
(streaming.rs:2867-2889, zensim 0.3.0):

```rust
let process_strip = |dst_planes: &mut [Vec<f32>; 3],
                     (strip_y0, strip_y1, ...): (...)|
 -> (Vec<ScaleAccumulators>, [f64; 3], usize) {
    let src_strip = crate::source::SubsetView::new(source, strip_y0, ...);
    let dst_strip = crate::source::SubsetView::new(distorted, strip_y0, ...);
    // Per-strip PrecomputedReference allocation — this is the +36% culprit.
    let precomp = PrecomputedReference::new(&src_strip, num_scales, false);
    compute_multiscale_accums_streaming_with_ref_borrowed(
        &precomp, &dst_strip, dst_planes, ...
    )
};
```

Each strip rebuilds its `PrecomputedReference` independently — re-running
sRGB → XYB conversion + per-scale downscale for every strip's source
rows. With `n_strips ≈ 22` at 40 MP and 256-row body / 128-row margin
geometry, the total per-strip ref allocation work exceeds the cost of
materializing one full-image `PrecomputedReference` up front (≈213 MB
single allocation per heaptrack's top-allocator trace).

The code comment on line 2865 explicitly notes this is intentional
short-term scope and "Phase 4 of the plan eliminates that" — referring
to zensim's own internal streaming optimization plan. Until upstream
zensim fixes it, the avoidable +23 % is up to callers to dodge.

## Caveats — what's intrinsic vs. avoidable

- **Avoidable (≈+23 % at 40 MP, this fix):** repeated
  `PrecomputedReference::new` per strip. Hoisting it outside the strip
  loop reuses the full ref pyramid (zero-copy slicing per strip via
  `PrecomputedReference::slice_rows_view`).
- **Intrinsic (≈+13 % at 40 MP, remains after this fix):** the strip
  walker's per-worker dst XYB scratch (3 planes × padded_width ×
  strip_height × 4 bytes ≈ 250 MB at 80 MP geometry) plus the global
  reference pyramid held resident across the strip loop. This is the
  `warm_ref_strip` floor and cannot be hoisted further without a
  redesign of the strip-accumulator merge step.

So a one-shot `strip` caller can match `warm_ref_strip`'s memory
profile by pre-building the `PrecomputedReference` themselves — that's
exactly what this fix does at the driver level. It does NOT close the
remaining +13 %; that's a separate optimization on the strip-walker
internals (out of scope for this task, would require modifying the
external zensim repo).

## Fix shape (production-caller pattern)

zensim 0.3.0 lives at `~/work/zen/zensim/zensim/` (external sibling
repo; per repo CLAUDE.md must NOT be modified from inside zenmetrics).
Both APIs are already exposed publicly on `zensim::Zensim`:

```rust
pub fn precompute_reference(&self, source: &impl ImageSource)
    -> Result<PrecomputedReference, ZensimError>;

pub fn compute_with_ref_streaming_strips_default(
    &self,
    precomputed: &PrecomputedReference,
    distorted: &impl ImageSource,
) -> Result<ZensimResult, ZensimError>;
```

So any production caller that today calls
`compute_streaming_strips_default(&ref, &dist)` SHOULD instead spell:

```rust
// One-shot strip score with hoisted reference precompute.
// Score is bit-identical to compute_streaming_strips_default.
// Peak heap drops ~16 % at 40 MP, matching warm_ref_strip.
let pre = z.precompute_reference(&ref_img)?;
let result = z.compute_with_ref_streaming_strips_default(&pre, &dist_img)?;
```

The `PrecomputedReference` is a value-typed owned struct; for one-shot
usage it lives for the duration of one score call and drops naturally.
The naming "warm_ref" is misleading here — even WITHOUT cross-call
reference reuse, this pattern is strictly better for any single
`(ref, dist)` strip-score, because the cost we're paying once in
advance (one ref XYB conversion + downscale) is less than the cost we
were paying N times inside the strip loop.

If your caller already has a `PrecomputedReference` cached across many
distorted candidates (the encoder quantization loop pattern), the same
function call uses it; nothing else changes.

## Score parity

Verified bit-identical across the heaptrack matrix sizes after the fix:

| Size  | `full`              | `strip` (before)     | `strip` (after)     | `warm_ref_strip`    |
|-------|---------------------|----------------------|---------------------|---------------------|
| 1 MP  | 80.45223298546662   | 80.45223298546662    | 80.45223298546662   | 80.45223298546662   |
| 16 MP | 80.45277977209128   | 80.45277977209128    | 80.45277977209128   | 80.45277977209128   |
| 40 MP | 80.45113394427749   | 80.45113394427749    | 80.45113394427749   | 80.45113394427749   |

All five `strip` cells are bit-identical to their `full`/`warm_ref_strip`
counterparts. zensim's strip walker is already deterministic with
respect to the choice of reference precompute approach (full vs.
per-strip), because both produce the same XYB planes — the only
difference is heap traffic.

## Heap & wall-time delta

40 MP measurement (heaptrack 1.3.0, water-cooled 7950X, host idle):

|                  | peak heap | n_alloc | total runtime |
|------------------|----------:|--------:|--------------:|
| `strip` (before) | 3.53 GB   | 2675    | 1.75 s        |
| `strip` (after)  | 2.96 GB   | 2214    | 1.81 s        |
| `warm_ref_strip` | 2.96 GB   | 2214    | 1.72 s        |

Reduction: **−16.1 % peak heap (−580 MB)** at 40 MP. Allocation count
drops by 461 calls (one `PrecomputedReference` build per strip × 22
strips × a handful of inner allocs per build). Wall time is roughly
flat (the wall-time benefit of "less precompute work" is offset by the
slightly higher pyramid pressure when the full ref is held resident).

This brings the post-fix `strip` mode exactly onto
`warm_ref_strip`'s allocator profile, leaving the +13 % residual that
is intrinsic to the strip walker's accumulator + ref-pyramid
working set.

## Where the fix lives in this repo

- [`benchmarks/heaptrack/drivers/cpu_profile/src/main.rs`](../../../benchmarks/heaptrack/drivers/cpu_profile/src/main.rs)
  — the only in-repo caller of `compute_streaming_strips_default`. Now
  uses the hoisted-ref pattern for the `strip` mode and `warm_ref_strip`
  is kept as an alias (both call paths are identical post-fix; the two
  matrix cells exist for symmetry with the other metrics' four-mode
  surface).

- No production CPU adapter in `zenmetrics-orchestrator::cpu_adapter`
  currently calls the strip API directly — the CPU OOM fallback ladder
  uses `Zensim::compute` (full) only. When that adapter grows a strip
  fallback path (Phase 9.X / 9.Z), it MUST use the hoisted-ref pattern
  documented here.

## Upstream note

The fix this doc describes lives at the *caller* boundary because
the repo-CLAUDE.md hard-constraint forbids modifying external repos
from inside zenmetrics. The root-cause fix in the zensim
`compute_multiscale_stats_streaming_strips` function would be to share
a single `PrecomputedReference` across the strip loop internally —
which is what the source comment on `streaming.rs:2865` records as
"Phase 4 of the plan eliminates that". When that lands in zensim, this
driver-level wrapper becomes a no-op and the production-caller
guidance in this doc collapses to "just call `compute_streaming_strips_default`".
