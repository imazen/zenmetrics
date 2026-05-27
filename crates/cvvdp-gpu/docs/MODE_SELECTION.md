# Mode selection — keep Full and Strip side-by-side

## Rule

**The Full-mode dispatch path is load-bearing.** Mode E and Mode B exist
*alongside* it, not in place of it. Every change to the strip walker
or per-strip buffer allocator MUST preserve Full's wall-time at every
production size. The Auto resolver picks the variant that's both
memory-fitting AND fastest — never blindly Strip just because Strip
exists.

The user instruction that locked this in (2026-05-26):

> make sure that if there are full to strip perf regressions we keep
> side by side for smart runtime selection

## What "side by side" means concretely

1. **The level-major dispatch (Full) stays in the source tree.**
   Strip-major Mode B is a NEW code path inside the
   `MemoryMode::StripPair { .. }` branch. `MemoryMode::Full` continues
   to call the existing level-major path, unchanged.

2. **Per-strip buffer fields are additive.** The proven `dc41d713`
   template adds `Option<...>_strip` siblings beside the existing
   full-image fields. Full mode keeps reading the full-image fields;
   Strip mode reads the `*_strip` siblings. Neither path's hot loop
   gains a runtime branch on mode (the branch is at allocation time).

3. **Buffer shrinks NEVER remove a full-image allocation that Full
   mode reads.** Skip-full-alloc only fires when
   `StripMode::Pair && cap_levels.is_none()`. Both Full and Mode E
   (`StripMode::CachedRef`) keep allocating the full buffer.

4. **JOD bit-identical across all three paths** at every production
   size (128² / 1024² / 4096² minimum, ideally up to 24 MP). Any
   shrink that disturbs Full or Mode E JOD is a regression, not a
   feature.

## Wall-time benchmark — required for any new shrink

Every Phase 2 buffer-shrink commit MUST include a wall-time
measurement comparing Full vs StripPair at production sizes. Add a row
per shrink to `benchmarks/cvvdp_mode_b_wallclock_<date>.csv` with
columns:

| size | mode | h_body | wall_ms_p50 | wall_ms_p25 | wall_ms_p75 | nvidia_mb |
|---|---|---|---|---|---|---|
| 128 | Full | - | ... | ... | ... | ... |
| 128 | StripPair | 64 | ... | ... | ... | ... |
| 1024 | Full | - | ... | ... | ... | ... |
| 1024 | StripPair | 512 | ... | ... | ... | ... |
| 4096 | Full | - | ... | ... | ... | ... |
| 4096 | StripPair | 512 | ... | ... | ... | ... |

Use the existing `examples/mem_mode_b_vs_full.rs` subprocess-per-cell
harness extended with wall-time capture (`std::time::Instant` around
`compute_dkl_jod_with_warm_ref` calls, n=20 runs per cell, report
percentiles).

### Perf budget — Strip-vs-Full wall-time

**StripPair may be up to 20% slower than Full at sizes where both
modes fit.** This budget is the memory-savings-vs-perf tradeoff the
user accepted (locked in 2026-05-26: "under 20pct regression is ok").

- StripPair `wall_ms_p50 / Full wall_ms_p50 ≤ 1.20` at each size:
  shrink lands.
- Beyond 1.20 at a size where Full fits: still lands, but the Auto
  resolver MUST prefer Full at that size (perf-aware selection per
  the resolver rules below). The shrink commit must include the
  resolver change in the same PR.
- Beyond 1.20 at a size where Full does NOT fit: lands as-is. There's
  no Full alternative; StripPair-slower-than-Full is still
  StripPair-faster-than-OOM.

**Full's own baseline must not regress more than 1.5%** from the
parent commit. The 20% budget is for StripPair-vs-Full at the same
commit, NOT for Full degrading over time. If a shrink commit pushes
Full's `wall_ms_p50` past the 1.5% gate, the shrink goes behind a
feature flag or the code path stays unmerged until the regression is
understood — Full mode is the warm-path correctness baseline and must
not silently slow down.

## Auto resolver — perf-aware, not just memory-aware

Today's `resolve_auto` (in `memory_mode.rs`):

```rust
if full_bytes <= cap { return Ok(ResolvedMode::Full); }
// else fall back to Strip
```

This is correct as a memory policy: Full when it fits, Strip when it
doesn't. The perf-aware extension layers on top — when Full DOESN'T
fit the cap and Strip DOES, pick Strip (existing behaviour). When BOTH
fit, prefer Full (existing behaviour, which is also the fastest at
sizes where Full fits since cache locality + zero halo overhead).

**Smart selection (perf-aware) — landing rules**

When both Full and Strip fit the cap, the resolver consults a
static lookup table seeded from the committed wall-time benchmark:

- If `strip_p50 / full_p50 ≤ 1.20` at the resolved `(width, height)`
  bucket: either choice is acceptable. Default to Full for warmpath
  cache locality, but a caller asking for `MemoryMode::StripPair`
  explicitly is honored as-is.
- If `strip_p50 / full_p50 > 1.20` at that bucket: the Auto resolver
  picks Full even though Strip also fits. The 20% budget is the line
  past which "smart selection" kicks in to protect throughput.
- If Full does NOT fit: pick Strip regardless of perf ratio. There's
  no alternative.

The lookup table lives in `pipeline::strip_perf_ratio_for_size` (to
be added by Phase 2 alongside the wall-time benchmark commits). It's
indexed by log2 width and log2 height bucket (8 = 256, 10 = 1024,
12 = 4096), populated from the committed CSV, and queried by
`resolve_auto`. Never measure perf inside `new()` — the selection
must be deterministic and zero-cost.

## Mode-E vs Mode-B distinction

Both `MemoryMode::Strip` (Mode E) and `MemoryMode::StripPair` (Mode B)
walk the dist side as strips, but only Mode B walks ref as strips too.
Mode E reads the cached full ref state via `RefFullState`. The two
modes coexist with Full mode as three distinct dispatch paths, all
preserved.

When `warm_reference` is called and `MemoryMode::StripPair` is set,
Mode B mode SHOULD fall back to Mode E (it's strictly better when the
caller already paid for ref-full state). The shrink commits must keep
this fallback intact.

## What Phase 2 shrink agents MUST verify before pushing

1. `cargo test --release -p cvvdp-gpu --features cuda` — all tests pass.
2. JOD parity test at 128² / 1024² / 4096² — `|diff| ≤ 1e-4` (target
   bit-identical `|diff| = 0.0`).
3. Wall-time bench Full vs StripPair — Full unchanged within 1.5%
   from parent commit, StripPair within 20% of Full at sizes where
   both fit (else update the perf-aware resolver lookup table in the
   same commit).
4. `nvidia-smi` delta — StripPair memory dropped by the predicted
   amount; Full memory unchanged.
5. CappedPyramid still works (smoke test at 4096² with `levels=8`).
6. Mode E still works (smoke test at 4096² with cached ref).

Push only when all 6 pass. Honest-stop if any regress.
