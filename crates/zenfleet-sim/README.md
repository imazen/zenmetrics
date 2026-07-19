# zenfleet-sim

A deterministic fault-injection simulator for the zenfleet job system. It runs the
**real** `zenfleet-core` logic (reconcile, lease, idle detection, retry/poison
classification) against a **breakable** in-memory cloud substrate, so the fleet's
hardest-to-test failure modes become fast, reproducible tests instead of live-R2
bash demos that never ran in CI.

No cloud. No network. No threads. Every test is deterministic from a seed.

## Why

The fleet inventory found that the load-bearing reliability code — the R2
claim/steal race, the reconcile gap loop, idle detection, teardown — had
essentially no automated tests, because in production it is coupled to a real
object store (R2 via s5cmd, with all its eventual-consistency and partial-failure
quirks) and to real paid boxes that die, fail to start, and fail to tear down.
This crate decouples that: the failure modes are injected, and the production
logic is asserted against them.

## What it models

**`FaultStore`** — an in-memory `zenfleet-cloud::BlobStorage` that misbehaves the
way R2/s5cmd actually does:

| Fault | Models |
|---|---|
| `consistency_delay_secs` (+ `read_your_writes`) | read-after-write / list-after-write eventual consistency — why the vast claim sleeps 1.5 s |
| `creds_invalid` / `creds_expire_at` | bad or expired scoped credentials (403) |
| `partial_write_rate` | a silent partial upload (s5cmd killed mid-PUT → truncated "success") |
| `transient_rate` | R2 503 / throttle |
| `op_latency_secs` | per-request latency (makes tiny-file cost visible) |
| `delete_fail_rate` | a claim/box that refuses to be cleaned up |
| `list_drop_rate` | LIST omitting a visible key |

Plus the two strongly-consistent conditional ops R2 exposes: `put_if_absent`
(`If-None-Match: *`) and `cas` (`If-Match`).

**`claim`** — the token-race claim (mirrors `zenfleet-vastai::try_claim`) and the
conditional-PUT claim (models `zenfleet-worker::claim_or_steal_r2`) side by side,
with staleness decided by the real `zenfleet_core::Lease`.

**`fleet`** — simulated paid boxes (`SimBox`/`SimFleet`) that bill by the hour and
can never boot, die mid-run, sit idle, or refuse to tear down. They emit the real
`zenfleet_core::WorkerReport`, and a reaper built on the real
`zenfleet_core::detect_idle` tears down flagged boxes.

**`converge`** — the declare → reconcile → execute loop to convergence, over the
fault store, with per-job outcome scripts (succeed / transient-fail / poison).

## What the tests prove (`tests/`)

- **`chaos_claim`** — the conditional claim is exactly-once under an
  eventual-consistency window; the token-race **double-acquires** under the same
  window (the evidence for porting the vast claim to conditional PUT); bad/expiring
  creds never falsely acquire; a dead box's claim is reclaimed after it goes stale;
  a partial upload is caught by content hash.
- **`chaos_fleet`** — a crashed box is flagged `StaleHeartbeat`; a never-booted box
  is caught as a startup failure; an idle GPU box is reaped and its billing stops;
  a box whose teardown **fails** keeps billing until a retry succeeds; a reaper
  bounds spend >4× vs launch-and-forget.
- **`chaos_convergence`** — transient failures retry and recover; deterministic
  failures poison without endless retry; a transient job poisons at the attempt
  cap; a mixed workload converges to correct terminal states even under a store
  throwing 20% transient errors (self-heal).

## Run

```bash
cargo test -p zenfleet-sim
```

It runs in the CI `fleet-orchestration tests` step alongside the other pure-logic
fleet crates.

## Add a scenario

1. Pick the faults: build a `FaultSpec` (or `FaultSpec::eventual_consistency(n)` /
   `::bad_credentials()` / `::flaky(rate)`), and a `SimClock`.
2. Drive the real logic: call `claim_conditional` / `claim_token_race`,
   `SimFleet::reap`, or `run_to_convergence` against a `FaultStore`.
3. Assert the safety property (exactly-once, converged, box torn down, spend
   bounded). Advance the clock explicitly; never `sleep`.

## Note

This crate depends only on `zenfleet-core` + `zenfleet-cloud` and is verifiable in
isolation. If the full workspace fails to resolve locally on a `zenjpeg` feature
mismatch, that is pre-existing sibling-checkout drift unrelated to this crate; CI
clones the pinned siblings and builds it as a normal workspace member.
