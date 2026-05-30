# zen job system — live demos

Reproducible, isolated demonstrations that the job system's guarantees hold **live** (against real
R2), not just in unit tests.

## `demo_e2e_r2.sh` — goals A, E, I + foundations

Runs the full declare → reconcile → execute → coverage loop against an **isolated R2 prefix**
(`jobsys-demo-<ts>/`, deleted at the end unless `KEEP=1`). Synthetic metric jobs; the executor is
`/bin/cat`, so it needs no encoder/GPU and costs a handful of tiny R2 objects.

```bash
# needs R2_* env, aws (v1.44+), s5cmd, and built zen-jobworker + zen-jobctl
cargo build -p zen-jobworker -p zen-jobctl
bash scripts/jobsys/demo_e2e_r2.sh        # KEEP=1 to retain the R2 prefix
```

### What it proves (verified run 2026-05-30, prefix `jobsys-demo-20260530-025731`)

- **A — declare + idempotent enqueue.** `zen-jobctl declare` expanded a 2-item × 2-metric spec into
  4 `DesiredJob`s. `gap` before any work = **4**; `gap` after pass 1 = **0** — re-declaring done work
  is a structural no-op (content-addressed `JobId`).
- **E — convergence + restartable + lease.** Worker pass 1 converged the gap (`done=4`). A 2nd pass
  folding in the R2 ledger did **0** (`skipped` all) — fully restartable, ledger is truth. The R2
  conditional-write lease admits exactly one claim per job (a second `put-object --if-none-match '*'`
  on a held claim returns `PreconditionFailed`).
- **I — coverage from the ledger.** `zen-jobctl catalog` reported 4 done / 0 gap per codec×metric,
  derived purely from the R2 Parquet ledger (same source the dashboard reads).
- **Foundations.** Pass 1 wrote 4 **content-addressed** blobs (`<prefix>/blobs/<sha256>`), a columnar
  **Parquet** ledger (`pass1.parquet`, 4.8 KB), and claim objects — all in R2.

## `demo_spot_reclaim_r2.sh` — goal F (spot reclaim is a non-event)

Reproduces spot preemption with a local `kill -TERM` (a SIGTERM is the same whether it comes from
vast.ai/spot or a local kill). Starts a worker on a deliberately-slow job, sends SIGTERM mid-execution,
and shows the in-flight R2 claim is **released** (deleted) so the job requeues immediately instead of
waiting out the claim TTL.

```bash
bash scripts/jobsys/demo_spot_reclaim_r2.sh        # KEEP=1 to retain the R2 prefix
```

### What it proves (verified run 2026-05-30)

- A worker claims the job (1 claim object in R2) and begins executing.
- `kill -TERM` → the signal-hook thread releases the claim and exits 130
  (`zen-jobworker: spot preemption — released claim <id> for fast requeue`).
- Claims in R2 drop to **0**; `gap` still shows the job (**requeued, not lost**); a fresh worker then
  completes it. Best-effort: if the release misses, TTL stale-reclaim (goal E) still requeues it, so
  correctness never depends on the signal landing.

## `demo_pause_drain_r2.sh` — goal C (pause / resume / drain)

A `RunControl` object in R2 (`{"paused":bool,"drain":bool}`) gates whether a worker pulls new work.
The demo sets paused → runs a pass (does 0) → sets draining → runs a pass (does 0) → clears it → runs
a pass (completes the job). The ledger is never touched, so resume continues exactly where it left
off. The dashboard's Pause/Drain/Resume buttons write this same object (`ZEN_CONTROL_R2`); workers
read it via `--control-r2-key`.

```bash
bash scripts/jobsys/demo_pause_drain_r2.sh        # KEEP=1 to retain the R2 prefix
```

Verified run 2026-05-30: PAUSED/DRAINING passes did `done=0`; RESUME did `done=1`.

## `demo_notify_local.sh` — goal D (notifications)

Points the dashboard's `ZEN_NOTIFY_WEBHOOK` at a local HTTP receiver and feeds it a worker fixture
that crosses a budget cap, proving the full detect → format → POST path fires with a deep link — no
external service or real channel needed. The only thing a production channel adds is the destination
(set `ZEN_NOTIFY_WEBHOOK` to your Slack/Discord/ntfy URL).

```bash
bash scripts/jobsys/demo_notify_local.sh
```

Verified run 2026-05-30: received
`{"text":"budget crossed: $0.35 >= cap $0.10 - paid tiers tearing down — https://…/#cost"}`.

## `demo_speculative_r2.sh` — goal E (speculative execution)

Worker A claims a job and runs it slowly (a straggler); worker B, with `--spec-threshold-secs`, sees
the aged primary claim and takes a separate `claims/spec/<job_id>` claim to co-run it fast — bounding
the long tail. The ledger's latest-wins on `job_id` makes the loser harmless. `/api/speculative`
surfaces the active spec-claim count.

```bash
bash scripts/jobsys/demo_speculative_r2.sh        # KEEP=1 to retain the R2 prefix
```

Verified run 2026-05-30: primary claim aged past threshold → speculator `done=1`, 1 spec claim, job
converged (gap=0) while the slow primary was still running.

## `demo_gc_r2.sh` — goal G (safe garbage collection)

Runs the self-asserting `gc_live` example: a reachability GC over a synthetic blob set proving
referenced blobs are kept, the cheap-regenerable LRU tail is evicted (with a tombstone) under a byte
cap, the newest cheap blob is kept, and an unreferenced **irreplaceable** blob is refused (surfaced,
never deleted). The `zen-jobgc` CLI runs the same over a real Parquet blob index + ledger (dry-run by
default; `--execute` to delete; `verify_mirror` gates any non-regenerable delete).

```bash
bash scripts/jobsys/demo_gc_r2.sh
# real use:
zen-jobgc --blob-index s3://b/.../blob_index.parquet --ledger s3://b/.../ledger.parquet \
  --blobs-r2 s3://b/blobs --tombstones-r2 s3://b/jobsys/tombstones \
  --r2-endpoint "$EP" --cheap-cap-bytes 1000000   # add --execute to delete
```

## `launch_fleet.sh` / `watch_fleet.sh` / `teardown_fleet.sh` — goal H (heterogeneous fleet)

Bring up ≥3 interchangeable tiers (local + Hetzner + vast) on ONE R2 lease-queue, all running the same
**baked** `ghcr.io/imazen/zen-jobworker` image (binary + aws-cli + s5cmd + keep-alive entrypoint —
zero boot-time installs, per the bake-everything rule; image built by `.github/workflows/jobworker-image.yml`).
Scoped temp R2 creds per run; teardown by `group=<run>` label (or the dashboard Kill controls).

```bash
# one-time: ensure CI pushed the image and the ghcr package is public
bash scripts/jobsys/launch_fleet.sh 200 1 1   # 200 jobs, 1 Hetzner box, 1 vast box  (SPENDS MONEY)
bash scripts/jobsys/watch_fleet.sh  <RUN>      # ledger DONE rows by provider — proves concurrent tiers
bash scripts/jobsys/teardown_fleet.sh <RUN>    # delete every box for this run
```

A 2026-05-30 ad-hoc test (before this image existed) already had a Hetzner box do **60 real jobs** on
the shared queue and tore it down via the **dashboard's Kill** — the image makes the full clean 3-tier
launch reliable + repeatable.

The dashboard side of these guarantees (coverage/catalog, progress + speculative count, cost, kill,
stop-spend, pause/drain/resume, result peek + thumbnails + ad-hoc query, GC dry-run preview) is live at
the Railway deployment; see `crates/zen-jobdash`. Notifications go to ntfy (`ZEN_NOTIFY_WEBHOOK` +
`ZEN_NOTIFY_TOKEN`); `demo_notify_local.sh` proves the mechanism without an external channel.
