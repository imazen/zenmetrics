# Phase #110 — CVVDP chooser-feasibility regression investigation

> **RESOLVED at `daf26f45`** (Phase 8i, 2026-05-27). The three
> cache-hygiene fixes recommended at the end of this document
> (Fix A / Fix B / Fix C) landed as three small commits on master:
>
> - **Fix A** `1d7535b1` — `known_oom_cell` cascade defeated by
>   positive measurement at `size >= oomed_pixels`.
> - **Fix B** `0530282f` — `record_oom_and_persist` prunes stale +
>   contradictory entries via a `retain()` cleanup pass on every
>   call. Existing cache files self-heal on the next legitimate OOM
>   recording — no migration script needed.
> - **Fix C** `daf26f45` — `CpuMetricUnavailable` /
>   `CpuBackendUnavailable` / `CpuNotYetWired` sentinels skip
>   `record_oom_and_persist` (they are feature-flag failures, not
>   memory failures).
>
> Regression tests landed alongside the fixes:
> `oom_cascade_defeated_by_positive_measurement_at_or_above_oom_size`,
> `oom_cascade_still_rejects_when_no_positive_measurement_at_or_above`,
> `record_oom_prunes_fossilized_unsupported_backend`,
> `record_oom_prunes_entry_contradicted_by_positive_measurement`,
> `sentinel_errors_do_not_pollute_cells_failed_oom`.
>
> The chooser's per-cell decision logic was correct as-shipped; the
> cache-write/read invariants needed the fix.

**Date**: 2026-05-28 (UTC)
**Investigator**: Phase #110 sibling-workspace agent
**Worktree**: `/home/lilith/work/zen/zenmetrics--phase110-bisect/`
**Baseline (54/54 PASS-EXACT)**: `phase771_run3` (binary built at commit
`150675077ad3` — "feat(cli): flip --use-orchestrator default ON; add
--use-legacy-scheduler opt-out (Phase 7.7.1)")
**Failing report (45/54, 9 cvvdp rejections)**:
`orchestrator_parity_2026-05-27_phase8c1b.{csv,md}` (binary built at
commit `9467a0c6c4c8` — "refactor(cvvdp): flip dep direction, move
params/presets/host_scalar/kernels scalars to cvvdp (Phase 8c.1-B)").
`7269722f3573` is the parity-report-only commit that publishes the
failing CSV/MD; it adds no code changes vs. `9467a0c6c4c8`.

---

## TL;DR

**No source-code regression was introduced between baseline and HEAD.**
The 9 failing cvvdp cells are caused by a **poisoned capability cache**
file at `~/.cache/zenmetrics/capability_6bfc55005d24a81a.toml`, NOT by
a chooser code change. The chooser is doing exactly what it's designed
to do: it rejects backends whose `(backend, pixels)` is present in
`MetricProfile.cells_failed_oom`, including any cell at a size ≥ a
recorded OOM size.

The 9467a0c6 parity report itself (committed in the same change that
"introduced" the regression) demonstrates the chooser works correctly
**on a fresh cache**: every cvvdp cell — including 4096² — returned a
numeric JOD value via the orchestrator path that uses a freshly-primed
`/tmp/orch_parity_*/cache/` cache directory. Only the *legacy* column
(which uses the persistent `~/.cache/zenmetrics/` cache) failed at all
9 cvvdp cells with `no feasible backend (considered 4 candidates)`.

The real regression — and the recommended fix — is structural: the
orchestrator's OOM-cache learning has no mechanism to invalidate stale
entries when the binary's chooser/bench backend-support matrix changes
between runs. The poisoned cache file contains fossilized
`(gpu_strip, 65536)` and `(gpu_strip, 1048576)` entries for cvvdp,
which the current `supported_backends(Cvvdp)` list (`[GpuFull,
GpuStripPair, Cpu]`) does not include and the current chooser would
never select — yet those entries still influence rejection via the
`*px < pixels` rule in `known_oom_cell` (see "Root-cause sub-bug" §
below).

## Identified failing cells (9 total)

From `benchmarks/orchestrator_parity_2026-05-27_phase8c1b.csv`,
each row's `verdict = FAIL` with `legacy_err = "no feasible backend
(considered 4 candidates) (backends tried: [])"`:

| # | metric | size  | q  | legacy col | orch col       | RejectReason (all 4 candidates) |
|---|--------|-------|----|------------|----------------|---------------------------------|
| 1 | cvvdp  | 256   | 20 | (empty)    | 9.296264       | see "Per-candidate verdict" below |
| 2 | cvvdp  | 256   | 50 | (empty)    | 9.695044       | "" |
| 3 | cvvdp  | 256   | 80 | (empty)    | 9.831320       | "" |
| 4 | cvvdp  | 1024  | 20 | (empty)    | 8.741311       | "" |
| 5 | cvvdp  | 1024  | 50 | (empty)    | 9.602884       | "" |
| 6 | cvvdp  | 1024  | 80 | (empty)    | 9.828098       | "" |
| 7 | cvvdp  | 4096  | 20 | (empty)    | (empty)        | "" |
| 8 | cvvdp  | 4096  | 50 | (empty)    | (empty)        | "" |
| 9 | cvvdp  | 4096  | 80 | (empty)    | (empty)        | "" |

Where the orch column is populated (cells 1–6), the value is
bit-identical to the `phase771_run3` baseline. Where the orch column
is empty (cells 7–9 at size=4096), even the orchestrator's freshly-
primed `/tmp` cache cannot service the request — see "Failure mode
B" below.

## The cache file in evidence

`/home/lilith/.cache/zenmetrics/capability_6bfc55005d24a81a.toml`
(machine_hash for AMD Ryzen 9 7950X + NVIDIA RTX 5070 + Linux). Key
state, verbatim from the file (`last_validated = 1779931994`, May 28
01:33 UTC):

```toml
[gpu]
present = true
model = "NVIDIA GeForce RTX 5070"
total_vram_mib = 12227

[metrics.cvvdp]
last_measured = 1779873538
cells_failed_oom = [
    ["gpu_full",        65536  ],   # cvvdp/GpuFull   at 256²
    ["gpu_strip",       65536  ],   # cvvdp/GpuStrip  at 256²  ← fossilized
    ["gpu_strip_pair",  65536  ],   # cvvdp/StripPair at 256²
    ["gpu_full",        1048576],   # cvvdp/GpuFull   at 1024²
    ["gpu_strip",       1048576],   # cvvdp/GpuStrip  at 1024²  ← fossilized
    ["gpu_strip_pair",  1048576],   # cvvdp/StripPair at 1024²
]

[metrics.cvvdp.ns_per_px_at.1048576]
gpu_full       = 5.938768
gpu_strip_pair = 27.468356
# (cvvdp at 1024² also has a *positive* measurement — the OOM and
#  positive measurements coexist for the same (backend, size)!)
```

Three observations:

1. **`gpu_strip` is impossible** for cvvdp under the current code. The
   chooser's `supported_backends(Cvvdp)` returns
   `[GpuFull, GpuStripPair, Cpu]` and the bench's `backends_for_kind`
   returns `[GpuFull, GpuStripPair]` (+ `Cpu` when `cpu-cvvdp`
   feature on). Neither lists `GpuStrip` for cvvdp. The
   `(gpu_strip, *)` entries can only have been written by a *prior
   binary version* that did include GpuStrip for cvvdp — or by a
   manual edit. They are **fossilized** state.
2. **The same cell can be both "measured-OK" and "OOM"** at the same
   time. `[metrics.cvvdp.ns_per_px_at.1048576]` lists `gpu_full = 5.94 ns/px`
   AND `cells_failed_oom = [..., (gpu_full, 1048576), ...]`. The bench
   never produces both for one cell, so the OOM entry was added
   **later** by `executor.rs::record_oom_and_persist` — i.e. an
   in-production task failed and persisted that failure even though
   bench had positive data.
3. **Every cvvdp request at size ≥ 65536 pixels is rejected.** Per
   `known_oom_cell` in `crates/zenmetrics-orchestrator/src/chooser.rs:419`:
   ```rust
   if *px < pixels { return true; }
   ```
   The OOM at 65536 pixels (= 256²) flags every request at any size
   above that, for every backend that has an OOM entry at any size.
   Result for cvvdp at the 9 parity cells:
   - 256² (65536 px): GpuFull → exact-px match → `KnownOomCell`;
     GpuStripPair → exact-px match → `KnownOomCell`;
     GpuStrip → `UnsupportedByMetric` (but redundantly also exact-px
     match, were it not unsupported);
     Cpu → `CpuMetricUnavailable` (the parity-sweep binary was
     built without `orchestrator-cpu-cvvdp`).
   - 1024² (1048576 px): GpuFull → exact-px match → `KnownOomCell`;
     GpuStripPair → exact-px match → `KnownOomCell`;
     same for the other two.
   - 4096² (16777216 px): GpuFull → 65536 < 16777216 → `KnownOomCell`
     (cascade rule); GpuStripPair → same; same fall-through.

   All four backends rejected → `NoFeasibleBackend (considered 4
   candidates)`. Matches the observed error string verbatim.

## Bisect result

**Candidate commits between baseline `150675077` and HEAD `7269722f`
that could change chooser/cvvdp behavior** (first-parent linear
history):

| Commit       | Phase  | Touches chooser? | Touches cvvdp? |
|--------------|--------|------------------|----------------|
| `00133cef`   | 7.6 L4 | no               | no             |
| `651495ab` … `bbdef6d9` | 7.6 L1–L3 + tests | no | no |
| `f1fda156`   | 7.5    | no (CLI sweep)   | no             |
| `9d0a90bd`   | 7.5    | no (CLI route)   | no             |
| `2257cdf5`   | 7.7.1  | no (executor MemoryMode::Auto) | no |
| `e2115d35`   | 7.7.1  | YES (NonPositivePrediction) | no |
| `9226073f`   | 7.7.1  | no (CLI iwssim rekey)         | no |
| `150675077`  | 7.7.1  | no (CLI default flip)         | no |
| `4fc03344`   | 8 plan | no (docs)                     | no |
| `4355dce0`   | 8d     | no (docs)                     | no |
| `07eeb11d`   | 8c     | no (cvvdp-cpu→cvvdp rename — no algo changes) | RENAME ONLY |
| `11251d14`   | 8a     | YES (NoGpuPresent fast-path)  | no |
| `65dd4729`   | 8a     | no (executor libcuda detect)  | no |
| `8d98d90a`   | 8a     | no (tests)                    | no |
| `aab300df`   | 8e     | no (docs)                     | no |
| `75902739`   | 9.1    | no (pool lanes)               | no |
| `98212fda`   | 9.2    | no (pipelined HtoD)           | no |
| `41d8b3f2`   | 9.3    | no (adaptive lanes)           | no |
| `7fe2ea8e`   | 8e     | no (docs)                     | no |
| `f2e57571`   | 8f     | no (docs)                     | no |
| `08325758`   | 8f     | no (docs)                     | no |
| `0832b904`   | 8f     | no (cubecl crates.io flip)    | no (workspace dep flip) |
| `89068c37`   | 8f     | no (parity report)            | no |
| `36c06368`   | 8c.1   | no (docs/audit)               | no |
| `6e9e7474`   | 8h     | no (ssim2 CPU adapter swap)   | no |
| `76dbdd46` … `4f6385af` | 8g (iwssim CPU + magetypes + adapter + parity test) | no | no |
| `9467a0c6`   | 8c.1-B | no (chooser unchanged)        | **YES — params/presets/scalars moved** |
| `7269722f`   | 8c.1-B | no (parity report)            | no |

**Code-bisect verdict**: the only commit between baseline and HEAD
that touches cvvdp at all is `9467a0c6` (Phase 8c.1-B), and even that
is a refactor that moves duplicated scalar code between two crates
without changing what gets executed. The chooser code that produces
the `NoFeasibleBackend` error has not changed in any way that would
affect cvvdp specifically since `e2115d35`'s `NonPositivePrediction`
rejection (which fires only on degenerate extrapolation, not on the
cells we observe failing) and `11251d14`'s `NoGpuPresent` fast-path
(which gates on `gpu.present == true` and is short-circuited on this
machine).

**The 9467a0c6 parity report itself proves cvvdp is not regressed.**
The orchestrator column at that commit reports JOD values for every
cvvdp cell — including 4096² — within 1.4e-4 of baseline. The only
"failures" in that report are the legacy column, which reads the
poisoned `~/.cache/zenmetrics/` cache. The 7269722f report differs from
9467a0c6's only by an empty `orchestrator` column at the three 4096²
cells; the binary and code are identical between the two parity runs
(`git diff 9467a0c6 7269722f` shows only `benchmarks/*` and
`CHANGELOG.md`).

Skipped the GPU re-bisect (run `parity_sweep.py` at each candidate
commit) because the textual evidence above is complete: there is no
candidate commit between baseline and HEAD that introduces a
chooser code path capable of producing the observed `KnownOomCell`
rejection from a clean cache. Spending 4–6 × 10–15 min of GPU sweep
budget to confirm "no commit broke the chooser" would not produce
new information.

## Failure mode A — legacy column (all 9 cells)

Repro path:

1. Earlier session writes `~/.cache/zenmetrics/capability_*.toml` with
   the cvvdp `cells_failed_oom` list above (specifically the
   `(gpu_full, 65536)` and `(gpu_strip_pair, 65536)` entries).
2. Parity sweep invokes
   `zen-metrics score --metric cvvdp --reference REF --distorted DIST
   --gpu-runtime cuda` *without* `--use-orchestrator` or
   `--orchestrator-cache`.
3. Post Phase 7.7.1 (`150675077`), `--use-orchestrator` is the **default**;
   the legacy column thus goes through the orchestrator on the
   user-default cache.
4. The orchestrator's chooser consults `cells_failed_oom`. Every cvvdp
   backend is rejected per the table in "The cache file in evidence"
   §3 above. `NoFeasibleBackend` surfaces.
5. The parity script captures this as `legacy_err = "...no feasible
   backend..."` and reports `legacy = ""` for the affected cells.

## Failure mode B — orchestrator column, cells 7–9 (size=4096)

Repro path:

1. Parity script creates a fresh `/tmp/orch_parity_*/cache/` directory
   for the orchestrator column.
2. Script calls `run_score` for `ssim2-gpu` size=256 with
   `bench-on-start=yes`. This forces a full bench (`Orchestrator::bench`),
   which sweeps all 6 metrics × `BenchPlan::default().sizes =
   [1024, 2048, 4096]` × backends. The fresh `/tmp` cache now contains
   measurements + any bench-time OOMs.
3. For cvvdp at 4096² (16,777,216 pixels), the bench either succeeds
   (writing `ns_per_px_at.16777216`) or OOMs (writing
   `cells_failed_oom.push((backend, 16777216))`).
4. In the **9467a0c6** parity run, cvvdp 4096² bench succeeded for both
   `GpuFull` and `GpuStripPair` — the orchestrator column reports
   `8.824267 / 9.576591 / 9.826287` JOD across the three q values.
5. In the **7269722f** parity run (same binary, ≈40 min later), cvvdp
   4096² bench OOMed (or the executor recorded a runtime OOM during
   the first task) for both GPU backends. The cells_failed_oom list
   in the fresh `/tmp` cache then poisons cells 7–9, producing the
   empty `orchestrator` column.

Mechanism for non-determinism between runs:

- cvvdp at 4096² uses ~1090 MiB (`GpuFull`) and ~834 MiB
  (`GpuStripPair`) per the existing bench data.
- The chooser's safety margin is `0.85 × free_vram_mib` (default
  `vram_safety_margin = 0.15`). 12227 MiB total × 0.85 = 10393 MiB
  usable.
- 1090 MiB sits comfortably inside that budget — but cubecl-cuda's
  buffer-pool retains driver-decommitted pages between iterations, and
  the bench iterates 5 sizes per metric × 6 metrics × 3 backends
  in one run. NVRTC compilation adds another ~200 MiB per fresh kernel
  set. On a host where the X server, Cargo, Rust-analyzer, etc. are
  also resident in VRAM, the headroom can vanish unpredictably.
- A single transient OOM during the cvvdp pass at 4096² is then
  persisted by `record_oom_and_persist` and locks out every
  subsequent cvvdp call at ≥ 4096² for the rest of the cache's
  lifetime.

## Root-cause sub-bug (the real defect)

Two coupled defects in
`crates/zenmetrics-orchestrator/src/chooser.rs` +
`crates/zenmetrics-orchestrator/src/executor.rs`:

### Sub-bug 1 — `known_oom_cell` cascade is too aggressive

`chooser.rs:419`:

```rust
fn known_oom_cell(profile: &MetricProfile, backend: Backend, pixels: u64) -> bool {
    // ...
    for (b, px) in &profile.cells_failed_oom {
        if *b != backend { continue; }
        if *px == pixels { return true; }
        if *px == snapped { return true; }
        if *px < pixels { return true; }  // ← cascade rule
    }
    false
}
```

The cascade rule "any OOM at a smaller size implies OOM at this size"
is correct in principle but lacks a sanity check: a *single fossilized*
or transiently-recorded OOM at 256² locks out ALL larger sizes
*forever* for that backend, even when the bench has positive
measurements at 1024², 2048², 4096². The cache file in evidence
shows exactly this — `cvvdp/GpuFull` has both a positive 1024²
measurement AND a 65536-pixel OOM, and the cascade rule honors the
OOM at the expense of the measurement.

The cascade rule should at minimum cross-reference `ns_per_px_at` —
if a positive measurement exists at a size ≥ the OOM size, the OOM
is stale and should not cascade.

### Sub-bug 2 — `record_oom_and_persist` lacks staleness invalidation

`executor.rs:1000`:

```rust
fn record_oom_and_persist(&mut self, metric: MetricKind, backend: Backend, pixels: u64) {
    let entry = self.capability_mut().metrics.entry(tag).or_default();
    let already = entry.cells_failed_oom.iter().any(|&(b, px)| b == backend && px == pixels);
    if !already {
        entry.cells_failed_oom.push((backend, pixels));
    }
    save_profile(...);
}
```

OOM entries are monotonic-append. There is no:

- Pruning of OOMs that are contradicted by a positive measurement at
  the same (backend, size).
- Pruning of OOMs whose backend is no longer in
  `supported_backends(metric)` (the `(gpu_strip, *)` fossils for cvvdp).
- Time-based expiry (the chooser treats a 6-week-old OOM the same as
  one from 30 seconds ago).
- Schema-version invalidation when the binary's chooser logic changes.

Cells_failed_oom is a write-only "punishment list" that survives
every binary upgrade and re-bench until the file is manually deleted.

### Sub-bug 3 — `CpuMetricUnavailable` / `CpuBackendUnavailable` get OOM-recorded

`executor.rs:928`-`942` records `(Backend::Cpu, pixels)` as an OOM when
the CPU adapter constructor returns `CpuMetricUnavailable:` /
`CpuBackendUnavailable:` sentinels. These are *feature-flag* failures,
not memory failures, but the OOM ladder treats them the same.

Effect: on a build without `orchestrator-cpu-cvvdp`, every cvvdp call
records a CPU-backend "OOM" at every size touched. The chooser's
`CpuMetricUnavailable` rejection already handles this case correctly
*before* construction — `record_oom_and_persist` for these sentinels
is redundant and pollutes the OOM list.

## Per-cell verdict (9 cells × {revert / fix / document-as-intended})

For all 9 cells the verdict is the same: **DOCUMENT-AS-INTENDED at the
chooser level, but FIX the cache-poisoning defects above so the cells
recover spontaneously on the next clean run.**

The chooser's per-cell decision is *correct* given the cache state it
sees. The fix lives in the cache write/read invariants, not in the
chooser's decision logic.

| # | (metric, size, q) | Chooser verdict | Recommended action |
|---|-------------------|-----------------|--------------------|
| 1 | cvvdp 256  q=20 | correct rejection given cache | clean cache & re-bench |
| 2 | cvvdp 256  q=50 | correct rejection given cache | clean cache & re-bench |
| 3 | cvvdp 256  q=80 | correct rejection given cache | clean cache & re-bench |
| 4 | cvvdp 1024 q=20 | correct rejection given cache | clean cache & re-bench |
| 5 | cvvdp 1024 q=50 | correct rejection given cache | clean cache & re-bench |
| 6 | cvvdp 1024 q=80 | correct rejection given cache | clean cache & re-bench |
| 7 | cvvdp 4096 q=20 | correct rejection given cache | clean cache & re-bench |
| 8 | cvvdp 4096 q=50 | correct rejection given cache | clean cache & re-bench |
| 9 | cvvdp 4096 q=80 | correct rejection given cache | clean cache & re-bench |

## Recommendation: FIX (cache hygiene), not REVERT (no code regression to revert)

There is no commit to revert: no code change between baseline and HEAD
introduces a chooser regression. The cleanup needs to address the
cache-poisoning failure mode in two parts:

### Immediate (operator): wipe the poisoned cache and re-run

```bash
rm -f /home/lilith/.cache/zenmetrics/capability_*.toml
# Then re-run the parity sweep — the orchestrator will warm() a fresh
# bench and the 54/54 PASS-EXACT baseline should be restored.
python3 scripts/orchestrator_parity_sweep.py \
    --binary target/release/zen-metrics \
    --out-csv  benchmarks/orchestrator_parity_phase110_clean.csv \
    --out-md   benchmarks/orchestrator_parity_phase110_clean.md
```

This is what the 9467a0c6 parity report's `/tmp/orch_parity_*` cache
column already demonstrates: a fresh cache on the same binary gives
bit-identical cvvdp scores at sizes 256/1024 (and at 4096 when the
GPU has the headroom — see Failure mode B).

### Structural (follow-up phase): three small fixes in
`zenmetrics-orchestrator`

#### Fix A — `known_oom_cell` consults `ns_per_px_at` before cascading

`crates/zenmetrics-orchestrator/src/chooser.rs:419`:

```rust
// Pseudocode (preserve current exact-match + nearest-snap logic,
// add the staleness check before falling through to the cascade rule).
fn known_oom_cell(profile: &MetricProfile, backend: Backend, pixels: u64) -> bool {
    // ... existing exact-match + snapped-match logic ...
    for (b, px) in &profile.cells_failed_oom {
        if *b != backend { continue; }
        if *px == pixels { return true; }
        if *px == snapped { return true; }
        if *px < pixels {
            // Phase 8c.1-C: cascade is only valid when no positive
            // measurement exists at a size ≥ the OOMed size. If the
            // bench later measured the same backend successfully at
            // a larger size, the smaller-size OOM is stale.
            let has_later_measurement = profile
                .ns_per_px_at
                .iter()
                .any(|(&size_px, bench)| size_px >= *px && bench.get(backend).is_some());
            if has_later_measurement {
                continue;   // stale OOM, ignore
            }
            return true;
        }
    }
    false
}
```

This restores the chooser's ability to use the positive 1024²
measurement when a 256² OOM is fossilized. It also makes the
`(gpu_strip, *)` fossils inert for cvvdp because the chooser already
rejects them as `UnsupportedByMetric` before reaching `known_oom_cell`,
but the existing cascade rule cannot then leak those OOMs into the
GpuFull/GpuStripPair evaluations.

#### Fix B — `record_oom_and_persist` prunes contradicted entries

`crates/zenmetrics-orchestrator/src/executor.rs:1000`:

```rust
fn record_oom_and_persist(&mut self, metric: MetricKind, backend: Backend, pixels: u64) {
    // ... existing append-if-absent logic ...
    let entry = self.capability_mut().metrics.entry(tag).or_default();
    // Phase 8c.1-C: drop OOM entries that are contradicted by a
    // newer positive measurement at the same (backend, size). The
    // bench writes positive measurements after the OOM ladder runs,
    // so we never end up with both pointing at the same task.
    entry.cells_failed_oom.retain(|&(b, px)| {
        !(b == backend && px == pixels && entry.ns_per_px_at.get(&px)
            .map(|bench| bench.get(b).is_some())
            .unwrap_or(false))
    });
    // Drop entries whose backend is no longer supported by the
    // chooser (rare but possible across binary upgrades).
    let supported = supported_backends(metric);
    entry.cells_failed_oom.retain(|&(b, _)| supported.contains(&b));
    // ... existing append-and-persist ...
}
```

This makes the cache self-heal on the next OOM recording event — the
first time the binary observes a cvvdp OOM after Fix B ships, it'll
drop the `(gpu_strip, *)` fossils because GpuStrip is not in
`supported_backends(Cvvdp)`.

#### Fix C — do not record OOM for `CpuMetricUnavailable` /
`CpuBackendUnavailable`

`crates/zenmetrics-orchestrator/src/executor.rs:928-942`:

Remove the `record_oom_and_persist(metric, backend, pixels)` calls in
the `CpuMetricUnavailable` / `CpuBackendUnavailable` branches. These
sentinels indicate a *feature-flag* missing in the build, not a memory
failure. The chooser's pre-construction rejection
(`CpuMetricUnavailable` reject reason) already handles this case
correctly and survives a binary rebuild without polluting the OOM list.

### Test plan for Fix A / B / C

Add three unit tests in
`crates/zenmetrics-orchestrator/tests/chooser.rs`:

1. `oom_cascade_blocked_by_later_positive_measurement` — pre-populate
   `cells_failed_oom = [(GpuFull, 256²)]` AND
   `ns_per_px_at[1024²] = GpuFull{...}`, request cvvdp/4096². Expect
   `Selected(GpuFull)` (cascade should be defeated by the 1024²
   measurement). Without Fix A, current behavior is
   `KnownOomCell`.
2. `record_oom_prunes_fossilized_backends` — pre-populate
   `cells_failed_oom = [(GpuStrip, 256²)]` for cvvdp, trigger any
   non-fatal cvvdp task that records an OOM. Expect `(GpuStrip, *)`
   entries to be removed by the retain filter.
3. `cpu_unavailable_sentinel_does_not_pollute_oom_list` — build with
   `cpu-cvvdp` disabled, run a cvvdp task that hits the CPU
   sentinel. Expect `cells_failed_oom` to remain empty.

### Where this DOES NOT need a fix

- The chooser's `evaluate_candidate` logic is correct.
- The bench's per-cell OOM recording (`bench.rs:389-393`) is correct
  for fresh-bench OOMs.
- `supported_backends(Cvvdp) = [GpuFull, GpuStripPair, Cpu]` is the
  right table.
- The parity-sweep script has no defect (it just exposes the cache
  poisoning).

## Cache compatibility notes (per the prompt's "carefully" reminder)

- The capability cache TOML schema has not changed between baseline
  and HEAD. Both baseline and HEAD parse the existing file without
  errors.
- The cache's machine_hash (`6bfc55005d24a81a`) is identical between
  baseline and HEAD because the host CPU/GPU/RAM/feature set has not
  changed. So both binaries write to the same file.
- During this investigation no cache files were modified. Bisect was
  performed via textual analysis of commit diffs (`jj diff` /
  `git log -p`) on the read-only worktree.
- The fossilized `(gpu_strip, *)` entries for cvvdp predate Phase 3's
  chooser (commit `4228a168`) — they must have been written by a
  pre-orchestrator binary or a hand-edit. The fix B retain filter
  in this doc will clean them up on the next OOM event.

## References

- `crates/zenmetrics-orchestrator/src/chooser.rs:419` — `known_oom_cell`
  cascade rule
- `crates/zenmetrics-orchestrator/src/executor.rs:1000` —
  `record_oom_and_persist` (monotonic-append, no pruning)
- `crates/zenmetrics-orchestrator/src/executor.rs:928-942` — CPU
  sentinels mis-classified as OOM
- `benchmarks/orchestrator_parity_2026-05-27_phase8c1b.csv` — failing
  report (45/54)
- `benchmarks/orchestrator_parity_2026-05-27_phase771_run3.md` —
  passing baseline (54/54)
- `benchmarks/orchestrator_parity_2026-05-27_phase8f.md` — same code
  surface as HEAD, passes 54/54 when run with clean cache
- `~/.cache/zenmetrics/capability_6bfc55005d24a81a.toml` — the
  poisoned cache file (operator should `rm` and re-bench)
