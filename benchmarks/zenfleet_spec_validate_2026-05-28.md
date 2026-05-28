# Speculative execution validation — 2026-05-28

**Commit**: `f080564d` (zenfleet-orchestrator crate + Salad bin wire-up).

**Build**:
`cargo build --release -p zenfleet-orchestrator --example speculative_validate`
and
`cargo build --release -p zenfleet-orchestrator --example spec_factor_sweep`.

## Live N=10 Salad validation — BLOCKED on Salad API outage

Three sweep launches attempted at 17:39 / 17:40 / 17:43 UTC. All three
progressed identically through the orchestrator path:

* Resolved 19 GPU classes ≤ $0.10/hr via `gpu_classes_under_price`.
* Loaded prior fleet_summary from
  `s3://zen-tuning-ephemeral/runs/iter2.5-validate-20260528T121949/fleet_summary.json`
  (4 unique classes observed in priors; none of those 4 overlap with
  the 19 candidate cheap classes from the catalog at time of probe →
  `kept=19 dropped=0` from `filter_classes` — by-design, since the
  filter only drops classes for which it has negative prior signal).
* Minted scoped R2 cred (bucket=zen-tuning-ephemeral, prefixes
  `runs/<sweep>/` + `salad-smoke-2026-05-28-24cell/`, TTL 3600 s).
* Preflighted both R2 input HEADs successfully.
* Uploaded `chunks.jsonl` (30 chunks × 12 cells/chunk = 360 cells).
* **Salad `POST /queues` returned HTTP 500 Internal Server Error** on
  every attempt. Direct `curl` to the same endpoint returns the same
  500 (`traceId: 9309bec299c4e235e7f5488df44abd92`) and a separate
  10s timeout against the org-project containers endpoint. The first
  sweep launch failed at the gpu-classes step with HTTP 504 Gateway
  Timeout, suggesting a wider Salad API incident in progress.

The orchestrator code is exercised through the entire pre-provisioning
launcher path. The blockage is downstream of any code this commit
touches.

## Local algorithm validation (concrete numbers)

Run via:
```
./target/release/examples/speculative_validate iter2.5
./target/release/examples/speculative_validate mapreduce-shape
```

Two scenarios, both with N=30 chunks and the default
`SpeculativeConfig { straggler_factor: 1.5, min_completed_for_stats: 3,
speculation_cap_per_chunk: 1 }`. Simulated baseline `t_done` matches
the iter2.5 baseline (388 s) for scenario 1.

### Scenario 1 — `iter2.5` (realistic sweep shape)

Distribution: 28 fast completions in 130-280 s, 2 stragglers at
380-395 s. Straggler/p95 ratio ≈ 1.4×.

| Field | Value |
|---|---|
| `n_speculative_dispatches` | **0** |
| `t_first_speculative` | n/a |
| `p95_completion_secs` | 380.0 |
| Simulated `t_done` with speculative | 395.0 s |
| Baseline `t_done` | 388.0 s |
| Reduction | **−1.8 %** (slightly worse — the simulator's last
  completion is the slower of the two stragglers; speculative didn't
  fire because the distribution was too tight) |
| Duplicate compute overhead | 0.00 % |

**Honest finding**: on a workload where the straggler tail is only
~1.4× the p95 of fast completions, `factor=1.5` speculative is
correctly *conservative* — it does not fire, and TTL=360 s is the
right mechanism. This is the falsification of "speculative always
wins" — it correctly enables only when the straggler ratio justifies
the duplicate compute.

### Scenario 2 — `mapreduce-shape` (canonical win)

Distribution: 28 fast completions in 95-275 s, 2 stragglers at
800-900 s. Straggler/p95 ratio ≈ 3.4×. Matches the Dean & Ghemawat
2004 MapReduce regime.

| Field | Value |
|---|---|
| `n_speculative_dispatches` | **2** (c28 and c29, both at t=400 s) |
| `t_first_speculative` | 400.0 s |
| `p95_completion_secs` at dispatch | 265.0 s |
| Threshold | 265.0 × 1.5 = 397.5 s |
| Simulated `t_done` with speculative | 460.0 s |
| Baseline `t_done` | 900.0 s |
| Reduction | **+48.9 %** (matches D&G 2004's ~44 % in production) |
| Duplicate compute overhead | 2.61 % |

### Straggler-factor sweep on the iter2.5 shape

`./target/release/examples/spec_factor_sweep`:

| factor | n_dispatches | t_first_speculative | sim_t_done | reduction |
|---:|---:|---:|---:|---:|
| 1.50 (default) | 0 | n/a | 395.0 | −1.8 % |
| 1.40 | 0 | n/a | 395.0 | −1.8 % |
| 1.30 | 2 | 360.0 | 395.0 | −1.8 % (fires too late, before pickup) |
| 1.25 | 2 | 350.0 | 395.0 | −1.8 % (same) |
| 1.20 | 2 | 330.0 | 390.0 | −0.5 % |
| 1.15 | 2 | 320.0 | 380.0 | +2.1 % |
| 1.10 | 2 | 300.0 | 360.0 | **+7.2 %** |

Pattern: on a tight bimodal distribution, dropping the factor toward
1.1 unlocks a modest speculative win, but the gain saturates around
+7 % because the synthetic re-dispatch still takes 60 s pickup time.
TTL=360 with the existing iter2.5 default ALREADY handles this case
well; speculative adds little here.

The default `factor=1.5` is the right ship value: it stays out of the
way on tight distributions (matches iter2.5 baseline behavior) and
fires correctly on classic MapReduce stragglers (+49 % reduction).

## What the live N=10 validation would have measured (queued)

When Salad's queue API recovers, re-run with:
```
target/release/zen-salad-sweep \
    --replicas 10 \
    --cells-per-chunk 12 \
    --max-price-per-hour 0.10 \
    --prior-fleet-summary s3://zen-tuning-ephemeral/runs/iter2.5-validate-20260528T121949/fleet_summary.json \
    --source-dir-r2 s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/sources \
    --input-parquet-r2 s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/input/smoke.parquet \
    --max-wall-secs 600 --poll-secs 10
```

Expected (per simulation + iter2.5 baseline):

* `t_first_sidecar`: ~130-170 s (boot-included first chunk).
* `t_all_N_sidecars`: ~200-260 s.
* `t_done`: 280-395 s (baseline 388 s; speculative should produce a
  modest improvement IF the day's distribution has any straggler tail
  beyond p95×1.5; otherwise speculative stays at 0 and t_done matches
  baseline).
* `chunks_redispatched_ttl`: 0-2 (TTL=360; only fires if any chunk
  hasn't landed by t=360).
* `chunks_speculatively_dispatched`: 0-2 (see above; conservative
  default).
* Per-replica boot mean/p50/p90: 90-180 s (iter2.5: 144 s mean).
* GPU class diversity: filter is currently a no-op for these
  candidate classes (filter doesn't drop a class without negative
  prior signal). The 19 cheap classes get nominated; Salad picks.

## Unit-test coverage

`cargo test -p zenfleet-orchestrator` — 8 tests pass:

* `provisioned_replicas_overshoots_then_clamps`
* `speculative_disabled_returns_none`
* `speculative_waits_for_min_samples`
* `speculative_fires_after_threshold`
* `speculative_respects_per_chunk_cap`
* `ttl_redispatch_returns_missing_chunks_once`
* `filter_drops_slow_warmup_keeps_unknown`
* `filter_falls_back_when_everything_drops`

`cargo test -p zen-cloud-salad --features launcher` — 2 tests pass:

* `sidecar_post_roundtrips_through_next_chunk_and_ack`
* `failed_outcome_makes_receiver_return_5xx`

## Files

* `crates/zenfleet-orchestrator/src/lib.rs` — 8 public types/functions
  + 8 unit tests.
* `crates/zenfleet-orchestrator/examples/speculative_validate.rs` —
  this report's data source (scenarios `iter2.5` + `mapreduce-shape`).
* `crates/zenfleet-orchestrator/examples/spec_factor_sweep.rs` —
  straggler-factor sweep on the iter2.5 shape.
* `crates/zen-cloud-salad/src/bin/zen-salad-sweep.rs` — Salad bin
  with the new CLI knobs (`--no-speculative`,
  `--speculative-straggler-factor`, `--speculative-min-completed`,
  `--speculative-cap-per-chunk`) and the orchestrator wire-through.
* `/tmp/zenfleet_hoist_retry_2026-05-28.log` — Salad-side outage log
  from the third attempt.
* `/tmp/zenfleet_spec_iter25.log` and `/tmp/zenfleet_spec_mapreduce.log`
  — local validation output.
