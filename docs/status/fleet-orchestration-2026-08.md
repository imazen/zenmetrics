# Fleet orchestration — Nomad decision + LAN-store next generation (2026-08)

Status/ADR doc, written 2026-08-22 from an operator-requested review. The private
companion (concrete topology, addresses, per-box roles, runbooks) lives in the private
fleet repo: `homefleet zenmetrics/ORCHESTRATION-2026-08.md`. Neutral node IDs only here.

## Decision

- **Pilot HashiCorp Nomad (CE 1.11.x; BUSL — fine for internal, non-competing use) as
  the BOX-LIFECYCLE layer for the LAN fleet only**: node membership + liveness, worker
  deployment/restart, drain (before dual-boot OS flips), image rolls, resource caps,
  periodic plumbing jobs, and node metadata (arch/SIMD-vendor/GPU/VRAM as node attrs).
- **zenfleet stays the work-distribution data plane, unchanged**: content-addressed
  `JobId`s, declare→gap→reconcile, S3 conditional-PUT claim leases, epoch-sharded
  claiming + registered handicaps, the Parquet ledger, retry-vs-poison error classes,
  blob GC. Nomad has no equivalent for any of these — it schedules *processes*, not
  *cells*, and its allocation history is not a queryable record.
- **Cloud burst is out of scope for Nomad** (vast/Hetzner/Salad keep the existing
  zenfleet launchers + R2 + self-destruct): ephemeral paid spot boxes joining a LAN
  raft quorum is the wrong shape, and there is no inbound path to the LAN store anyway.
- Windows sides of dual-boot boxes: out of scope (idle-only native workers unchanged).
- **Relationship to the 2026-08-21 orchestration evaluation** (private fleet repo):
  that review concluded "not yet — nothing in the current service set wants to be
  scheduled", while withdrawing its licence objection and naming Nomad the
  front-runner over k3s for this fleet. This ADR supersedes the "not yet" **for the
  worker fleet specifically**, on operator direction (2026-08-22): the schedulable
  service is the worker lifecycle itself — one service class across ~8 heterogeneous,
  partly-intermittent boxes, whose hand-rolled management layer is what recent
  campaign incident logs kept paying for. The licence analysis and Nomad-vs-k8s
  reasoning there remain the operative record.

## What Nomad replaces vs what stays

Replaces (today's mechanism → Nomad):

| today | under Nomad |
|---|---|
| per-box `zen-worker` systemd unit, `Restart=always` (`enroll_running_node.sh`) | `system`/long-lived job + restart/reschedule stanzas |
| `worker.env` pushed over SSH, weekly cred-rotation cron (`onboard_node.sh`) | job template + Nomad Variables (biggest ergonomic win) |
| per-box `docker pull` + unit edits for image rolls / GPU variants | jobspec `image` bump = fleet-wide roll |
| hand-assigned SIMD-tier pools per box (silent bitwise-gate breakage risk) | node attributes + constraints (enforced, not advisory) |
| `ZEN_CORE_OVERSUBSCRIBE` / `RAYON_NUM_THREADS` caps by convention | `resources { cpu, memory, memory_max }` cgroups |
| snapshot-refresh + crash-triage crons pinned to the operator box | `periodic` batch jobs on any always-on client (scoped creds only) |
| `fleet watch` polling + no worker heartbeat in lease mode | real node/alloc state + telemetry |
| manual drain before `to-windows` flips | `nomad node drain` wired into the flip helper |

Stays in zenfleet (no Nomad equivalent): job identity/dedup (`ids.rs`), the ledger as
truth, exactly-once at cell granularity (conditional-PUT lease), gap/reconcile,
retry-vs-poison `ErrorClass`, `ResourceHint` in-process admission, epoch sharding +
`fleet/handicaps.toml`, `RunControl` (campaign-scoped pause/drain spans tiers Nomad
will never manage), coverage/catalog, blob GC.

## Preconditions — fix in zenfleet BEFORE Nomad-managed churn

1. **#38 `JobId` preserve_order sensitivity** — until `JobId::of` serializes through an
   explicitly-sorted structure, any Nomad-era build/CI pipeline MUST keep per-crate
   binary builds (a unified workspace build silently forks every job identity).
   Identity code stays frozen while live campaigns run on current ids.
2. **SIGTERM chunk-claim release on the chunked path** — today a drain/reschedule
   strands the in-flight chunk's claim for the full TTL (release exists only on the
   serial path). Nomad makes drains routine; this becomes a per-drain stall otherwise.
3. **`ZEN_MAX_MIN` honored in single-run mode** (the wsl-gate 5-6 h unbudgeted-worker
   incident) — Nomad kill_timeout is a backstop, not a substitute.
4. **VRAM admission dimension in `BoxBudget`** (the avifgen OOM-storm follow-up).
   Nomad's device plugin reports VRAM; admission decisions stay zenfleet's.
5. **Worker task shape decision**: long-lived task with internal idle backoff
   (preferred under Nomad) vs today's drain-exit loop + `Restart=always` poll churn.

## Phases

- **P0** — preconditions 1–3; pick the 3-server quorum (always-on boxes only).
- **P1 pilot** — servers up; two client boxes run the standard worker image as a Nomad
  job against one pool run; compare ledger convergence, re-work tax, and idle
  detection against their systemd twins. No other box changes.
- **P2 enrollment/creds** — jobspec replaces `enroll_running_node.sh --start`; creds
  land via Nomad Variables → template → `worker.env` (LAN-store static cred; scoped
  7-day R2 mints where a wave needs R2).
- **P3 rollout** — all Linux LAN boxes + the mac (raw_exec driver, keeps its idle-only
  gate); `fleet-pxe` OS-flip helper drains the node first; systemd units stay
  installed-but-disabled as instant rollback.
- **P4 plumbing** — snapshot refresh + crash triage as `periodic` jobs off the
  operator box; never place the CF master token on a client — scoped creds only.
- **P5 re-evaluate** — epoch-shard roster vs Nomad node state; retire duplicated
  monitoring paths; decide whether POOL mode's lease claiming can drop to
  epoch-sharded now that membership is observable.

## LAN store: SeaweedFS is LIVE (switched 2026-08-21); 4.44 re-gated on a second box

Correction over this doc's first revision, which pre-dated a fetch: **the LAN store
already switched from MinIO to SeaweedFS 4.43 on 2026-08-21** (MinIO's community
edition is archived/EOL with unfixable advisories; endpoint unchanged, migration
manifest byte-identical, gate re-run at cutover — record in the private fleet repo's
LANSTORE doc; MinIO parked as rollback). What THIS review adds is the forward story:
**capacity is the binding constraint** (`zentrain` ~1.85 TB growing ~18 GB/day vs
~500 GiB free on the NAS hot tier), and SeaweedFS's native multi-box volume servers +
per-collection replication + R2 cloud-tiering are the designed answer once the new
box's 2×2 TB disks land.

Independent re-gate = the mandatory two-writer `PUT If-None-Match: *` race (harness
committed at [`scripts/lanstore/condput_gate.py`](../../scripts/lanstore/condput_gate.py);
A/B driver [`scripts/lanstore/store_ab_gate.sh`](../../scripts/lanstore/store_ab_gate.sh)),
run on a SECOND LAN worker box, loopback docker, stock config, 2026-08-22 — this
pre-validates the **next pin bump** (the live store runs 4.43):

| store | sequential | 8-way race ×5 | 1000×4KiB PUT (median of 3, s5cmd) | GET |
|---|---|---|---|---|
| **seaweedfs 4.44** | 200 → 412 | 1 win / 7 `PreconditionFailed`, 5/5 | **0.09 s** | 0.11 s |
| minio `2025-09-07` (control) | 200 → 412 | 1 win / 7, 5/5 | 0.14 s | 0.10 s |

**Verdict: 4.44 pre-validated for the next pin bump** — passes the exactly-once gate
at PUT-faster/GET-parity vs the retired MinIO pin on our tiny-object shape (sub-0.2 s
runs are near timing floor: read parity-or-better, not a precise ratio). Standing
constraints:

- **Non-versioned buckets ONLY.** Upstream recently fixed conditional writes under
  versioning+locking, but that path is outside our gate — never enable versioning or
  object-lock on job-system buckets without re-running the gate.
- **Pin the version; re-run the gate at the exact pin before any consumer moves and
  after EVERY upgrade** (the Garage lesson: silent `If-None-Match` no-ops are the
  worst failure mode and produce no log line).
- Buckets/roles/cred pattern unchanged (`zentrain`/`codec-corpus`/`zenfuzz`;
  `allow_http` already fixed in 55f8a339 + e7e04994).
- Expansion (storage v2, when the new box lands): volume servers on its 2×2 TB SATA
  as the bulk tier + optional volumes on two always-on boxes; canonical collections
  move to replication `010` once ≥2 volume hosts are up; filer metadata backed up
  nightly like a ledger; `weed filer.remote.sync` → R2 for new canonical data.
  Storage stays OUTSIDE Nomad at every phase. Topology in the private companion doc.

## zenfleet defect register (consolidated 2026-08-22)

Open, in priority order: **#38** JobId preserve_order (dual-encoding ledgers measured
on bf944); **VRAM admission dimension missing** in BoxBudget; **chunked-path SIGTERM
claim release missing**; **claim CAS shells out to the aws CLI** (the mac skip-all
root cause; port to native s3io); **`ZEN_MAX_MIN` ignored in single-run mode**;
capability routing is dead code (`metric_class()` returns `Gpu` unconditionally);
`WorkerReport`/dash inputs produced by nothing in production; `Lease::renew` has no
caller (claims age out from original timestamp only); verify the warm-exec `/tmp`
source-cache sweep reached every fleet box (the 620 GB ENOSPC class). Landed this
month and load-bearing: error-class fidelity (`SourceFetch`/`DiskFull` + markers),
ledger-snapshot `--ledger-in` contract in both modes, epoch-sharded claiming +
weighted handicaps, warm exec pool + ScoreFile warm-ref.
