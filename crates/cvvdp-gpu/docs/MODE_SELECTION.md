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
measurement showing Full performance is unchanged. Add a row per shrink
to `benchmarks/cvvdp_mode_b_wallclock_<date>.csv` with columns:

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

**Regression gate:** Full's `wall_ms_p50` at any size must not
increase by more than 1.5% from the pre-shrink baseline measured at
parent commit. If a shrink commit pushes Full past that, the shrink
goes behind a feature flag or the code path stays unmerged until the
regression is understood.

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

**Do NOT change the resolver to prefer Strip when both fit** unless a
benchmark establishes that Strip is meaningfully faster at that size.
The current preference for Full is the conservative correct choice.

If a future Phase 2 shrink discovers Strip IS faster at some size
range (e.g. cache effects at very large images), add a perf-aware
branch using a static lookup table indexed by `(width, height)` from
the committed wall-time benchmark. Never measure inside `new()` — the
selection must be deterministic and zero-cost.

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
3. Wall-time bench Full vs StripPair — Full unchanged within 1.5%,
   StripPair documented (may be faster, slower, or equal).
4. `nvidia-smi` delta — StripPair memory dropped by the predicted
   amount; Full memory unchanged.
5. CappedPyramid still works (smoke test at 4096² with `levels=8`).
6. Mode E still works (smoke test at 4096² with cached ref).

Push only when all 6 pass. Honest-stop if any regress.
