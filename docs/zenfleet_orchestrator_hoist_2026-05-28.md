# zenfleet-orchestrator hoist + speculative execution (2026-05-28)

## What landed

Commit `f080564d` ships **`crates/zenfleet-orchestrator/`** — a new
provider-generic crate carrying the launcher-side fleet orchestration
logic that had been inlined in `zen-cloud-salad/src/bin/zen-salad-sweep.rs`.

The hoist is structural-first: the **logic functions** (replicas
overshoot, TTL re-dispatch, class filter, speculative scheduler) live
in `zenfleet-orchestrator` and the Salad bin calls into them. The
full provider-trait extraction (`FleetSweep<P: ProviderHandle>`) is
queued for the next iter — it requires teasing apart the Salad-API
mint/provision/teardown plumbing into trait method shapes, which is
its own commit.

## Public API (skeletal — see lib.rs for the full doc)

```rust
// Replica provisioning
pub fn compute_provisioned_replicas(replicas: u32, overshoot: f64, quota: u32) -> u32;

// TTL re-dispatch decision
pub fn ttl_redispatch_decisions(
    elapsed_secs: f64,
    cfg: &SweepConfig,
    all_chunk_ids: &[String],
    completed: &HashSet<String>,
    already_redispatched: &mut HashSet<String>,
) -> Vec<String>;

// Class-aware filter
pub fn filter_classes(
    candidates: &[String],
    prior: &HashMap<String, PriorClassStats>,
    cfg: &SweepConfig,
) -> ClassFilterOutcome;

// Speculative execution scheduler
pub struct SpeculativeState { /* ... */ }
impl SpeculativeState {
    pub fn record_dispatched(&mut self, chunk_id: &str, now_secs: f64);
    pub fn record_completed(&mut self, chunk_id: &str, now_secs: f64);
    pub fn p95_completion_secs(&self) -> Option<f64>;
    pub fn decide_speculative(
        &self,
        chunk_id: &str,
        now_secs: f64,
        cfg: &SpeculativeConfig,
    ) -> Option<f64>;
    pub fn record_speculative_dispatched(&mut self, chunk_id: &str);
    pub fn total_speculative_dispatches(&self) -> u32;
}

pub struct SweepConfig { /* TTL, overshoot, filter, speculative */ }
pub struct SpeculativeConfig { /* factor, min_completed, cap */ }
pub struct PriorClassStats { name, median_warmup_secs, mean_chunks_processed }
pub struct ClassFilterOutcome { keep, dropped }
```

Defaults match the iter2.5 tuning (TTL=360 s, overshoot=1.7,
max_warmup=60 s, min_productive=2.0) + the speculative defaults from
Dean & Ghemawat 2004 (straggler_factor=1.5, min_completed=3,
cap_per_chunk=1).

## Speculative execution algorithm

The scheduler classifies in-flight chunks as "stragglers" using the
classic MapReduce backup-task rule (Dean & Ghemawat 2004 §3.6):

1. Track per-chunk first-dispatch time (`record_dispatched`).
2. As each sidecar lands in R2, register the completion duration
   (`record_completed`) and append to the in-memory completion-time
   distribution.
3. Once `n_completed >= min_completed_for_stats` (default 3), compute
   the nearest-rank p95 of that distribution.
4. For each in-flight chunk: if
   `elapsed > p95 × straggler_factor` AND we have NOT already
   speculatively re-dispatched it `cap_per_chunk` times,
   `decide_speculative` returns `Some(elapsed)` and the launcher
   re-pushes the chunk JSON.
5. Worker-side idempotency reconciles duplicates: a worker that claims
   a chunk whose omni sidecar already exists exits early (the
   `inline.rs::process_chunk_inline` HEAD pre-check in
   `crates/zen-cloud-vastai`). If two workers race past that gate, the
   fleet_summary stitch keeps the row with the OLDEST
   `worker_chunk_start_unix`.

The gates are deliberately conservative — at small n (3-5 samples),
nearest-rank p95 equals the max sample, so `factor × max` is the
effective threshold. As n grows, p95 starts diverging from the max
and the threshold tightens around the actual long-tail boundary.

## Salad-bin wiring

`zen-salad-sweep.rs` now calls into `zenfleet-orchestrator` for:

* `compute_provisioned_replicas(args.replicas, args.replicas_overshoot, 10)`
  — replaces the inline ceil/clamp.
* `ttl_redispatch_decisions(...)` — replaces the inline missing-chunk
  scan inside the poll loop.
* `SpeculativeState` — new state created at poll-loop entry, fed via
  `record_dispatched` (once per chunk after the initial push) and
  `record_completed` (each tick from the omni sidecar list).

New CLI knobs:

* `--no-speculative` (default false → speculative is ON).
* `--speculative-straggler-factor` (default 1.5).
* `--speculative-min-completed` (default 3).
* `--speculative-cap-per-chunk` (default 1).

New Summary field: `chunks_speculatively_dispatched: u32`.

## Tests

`cargo test -p zenfleet-orchestrator` covers:

* `provisioned_replicas_overshoots_then_clamps` — overshoot math.
* `speculative_disabled_returns_none` — master switch.
* `speculative_waits_for_min_samples` — n<min guards.
* `speculative_fires_after_threshold` — fires at p95 × factor.
* `speculative_respects_per_chunk_cap` — caps at 1 dispatch / chunk.
* `ttl_redispatch_returns_missing_chunks_once` — TTL fires once per
  chunk, skips completed and already-redispatched.
* `filter_drops_slow_warmup_keeps_unknown` — class filter logic.
* `filter_falls_back_when_everything_drops` — empty-result fallback.

`zen-cloud-salad`'s existing queue-roundtrip tests still pass; the
binary builds and `--help` exposes the new flags.

## What's NOT in this commit (queued for next iter)

* Full `FleetSweep<P: ProviderHandle>` trait extraction. The Salad
  bin still owns the `SaladApi` calls (mint scoped cred, POST
  container group, push jobs, poll instances, stop group). The
  next iter wraps these in a `SaladProviderHandle: ProviderHandle`
  trait impl and reduces the bin to a config-parser plus a
  `FleetSweep::run()` call.
* Vast.ai / RunPod launcher migrations to the same trait.
* RuntimeQuota provider trait (Salad's 10-replica quota is currently
  a literal in the bin).

## References

* Dean & Ghemawat 2004 §3.6 (MapReduce backup tasks):
  https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/
* iter2.5 baseline run:
  `s3://zen-tuning-ephemeral/runs/iter2.5-validate-20260528T121949/`
  (t_done=388 s, replicas_provisioned=10, no speculative execution).
* N=10 validation run (this iter): see
  `benchmarks/zenfleet_spec_validate_2026-05-28.md` (added after
  the run completes).
