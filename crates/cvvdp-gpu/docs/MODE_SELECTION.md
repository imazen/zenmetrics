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

### Perf budget — feasibility trumps perf

User direction locked 2026-05-26:

> "full can be 20pct slower and strip could be 3x slower to simply
> not fail"

The principle: memory-feasibility is the goal. Slow-but-running beats
OOM. We have generous headroom on both modes.

**StripPair vs Full wall-time at the same commit:**
- `strip_p50 / full_p50 ≤ 3.0` at sizes where both fit — shrink
  lands as-is.
- Beyond 3x at a size where Full fits: still lands, but the perf-
  aware Auto resolver MUST prefer Full at that size bucket. Same-
  commit resolver lookup-table update required.
- Beyond 3x at a size where Full does NOT fit: lands. There's no
  alternative; 3x-slow-Strip is still infinitely-faster-than-OOM.

**Full vs parent-commit Full wall-time:**
- `full_p50_new / full_p50_parent ≤ 1.20` at each size — shrink
  lands as-is.
- Beyond 1.20: investigate first. The 20% headroom is for cases
  where the dispatch-order change inherently costs Full a launch
  or two; if you find an avoidable regression hiding inside that
  budget (a bounds-check that LLVM can't optimize, a kernel-launch
  refactor that affected the warm path), fix it. The budget is a
  ceiling, not a target.
- Beyond 20%: honest-stop. Document what regressed and why before
  pushing.

The earlier 1.5% / 20% thresholds were tightened-too-aggressive. The
real tradeoffs:
1. Memory savings are large (-88% estimator delta at 4096²) — worth
   significant per-pixel overhead.
2. Feasibility (running at all on a 6 GB card) is binary — a Strip
   path that's 3x slower is infinitely better than a Full path that
   OOMs.
3. Full mode is the warm-path baseline but the dispatch-order
   refactor will inherently affect launch count; 20% headroom buys
   the structural changes Phase 2 needs without micro-optimizing.

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

- If `strip_p50 / full_p50 ≤ 3.0` at the resolved `(width, height)`
  bucket: either choice is acceptable. Default to Full for warmpath
  cache locality, but a caller asking for `MemoryMode::StripPair`
  explicitly is honored as-is.
- If `strip_p50 / full_p50 > 3.0` at that bucket: the Auto resolver
  picks Full even though Strip also fits. The 3x budget is the line
  past which "smart selection" kicks in to protect throughput.
- If Full does NOT fit: pick Strip regardless of perf ratio. There's
  no alternative — 3x or 5x or 10x slower Strip still beats OOM.

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
3. Wall-time bench Full vs StripPair — Full within 20% of parent
   commit (investigate before pushing if regression > 5%, hard-stop
   if > 20%); StripPair within 3x of Full at sizes where both fit
   (else update the perf-aware resolver lookup table in the same
   commit).
4. `nvidia-smi` delta — StripPair memory dropped by the predicted
   amount; Full memory unchanged.
5. CappedPyramid still works (smoke test at 4096² with `levels=8`).
6. Mode E still works (smoke test at 4096² with cached ref).

Push only when all 6 pass. Honest-stop if any regress.
