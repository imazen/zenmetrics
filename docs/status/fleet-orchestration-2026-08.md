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

**Status (2026-08-24): all five landed on `master`**, per the HANDOFF's directive to land
them before P1 pilot churn. Verified (test suites + one item live against the real LAN
store — see each entry); the ledger snapshot decision below stayed conservative anyway.

1. **#38 `JobId` preserve_order sensitivity** — ~~until `JobId::of` serializes through an
   explicitly-sorted structure, any Nomad-era build/CI pipeline MUST keep per-crate
   binary builds~~ **FIXED, `1b2a1452`**: `JobId::of` now hashes a plain
   `#[derive(Serialize)]` struct (field order fixed at compile time by declaration order)
   instead of a `serde_json::Value` built via `json!` (whose key order is sensitive to the
   `preserve_order` cargo feature). Reproduces the pre-fix byte output exactly — verified
   via the golden test under both a plain build and one co-compiled with `zenmetrics-cli`
   (the combination that used to fork identities). **Combined per-crate binary builds are
   no longer required for identity correctness** — a unified Nomad-era build is safe on
   this axis. (Same commit also fixed two related bugs found auditing this: `metric_class`
   hardcoded every metric job to `ResourceClass::Gpu` regardless of name, and
   `epoch::any_gpu_metric` checked the wrong suffix — see the commit message for detail.)
2. **SIGTERM chunk-claim release on the chunked path** — ~~today a drain/reschedule
   strands the in-flight chunk's claim for the full TTL~~ **FIXED, `0de119d6` +
   `45c57bd3`**: `run_chunked`'s lease-claiming arm now tracks its in-flight chunk claim
   and releases it on SIGTERM/SIGINT (`spawn_spot_reclaim_chunk`, mirroring the existing
   per-cell `spawn_spot_reclaim`). A companion fix in `fleet-entrypoint.sh` was necessary
   for this to be reachable at all under real container orchestration: the old
   `out=$(timeout ... zenfleet-worker ...)` blocked the shell inside a command
   substitution, which does not act on signals until the command finishes — refactored to
   background the pass + `wait`, with a `trap` that forwards SIGTERM/SIGINT to it. Verified
   with a fake-tool harness: signal reaches the child in ~0.03s (not the pass's full
   duration), no leaked temp files.
3. **`ZEN_MAX_MIN` honored in single-run mode** — ~~the wsl-gate 5-6 h unbudgeted-worker
   incident~~ **FIXED, `45c57bd3`**: wired identically to `pool_mode`'s existing pattern.
   Verified end-to-end: `ZEN_MAX_MIN=1` exits at exactly 60s wall-clock. Nomad
   `kill_timeout` remains the backstop, not the primary mechanism.
4. **VRAM admission dimension in `BoxBudget`** — ~~the avifgen OOM-storm follow-up~~
   **FIXED, `58c5db95`**: `BoxBudget` gained an optional third `vram_budget_bytes` axis
   (`None` = don't gate, unchanged behavior for CPU-only/unprobed boxes);
   `host_box_budget()` probes real GPU VRAM via `nvidia-smi` and sets a 90%-of-probed
   budget. Nomad's device plugin can report VRAM later; the admission decision itself
   stays zenfleet's regardless, per the ADR's original framing.
5. **Worker task shape decision** — ~~long-lived task with internal idle backoff
   (preferred under Nomad) vs today's drain-exit loop + `Restart=always` poll churn~~
   **DECIDED AND LANDED, `45c57bd3`**: long-lived, opt-in via `ZEN_LONG_LIVED=1` (defaults
   to today's drain-exit behavior, since paid cloud/vast.ai/Hetzner boxes rely on it to
   self-destroy and stop billing — this flag is LAN-fleet-only, set by the Nomad jobspec
   when P1 lands it). Idle passes back off exponentially (capped) instead of exiting.
   Verified: stayed alive through 35 consecutive idle passes over 3.5s (would have exited
   at 5 passes / ~0.5s under default behavior); default behavior itself regression-tested
   byte-identical to pre-change.

Also ported off the `aws`-CLI-spawn per the defect register (not one of the five above,
but blocking the same "mac tier loses every claim race" symptom): `claim_or_steal_r2_key`
(the claim-with-steal CAS) and `fetch_control_r2` now use native `object_store`-backed
`s3io` calls instead of shelling out to `aws s3api` per attempt — **FIXED, `a916d24d`**,
verified LIVE against the real LAN store (SeaweedFS, `zentrain` bucket, scratch-prefixed
and cleaned up) via the crate's existing `examples/lease_live.rs`.

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

## zenfleet defect register (consolidated 2026-08-22; updated 2026-08-24)

**Fixed 2026-08-24** (see "Preconditions" above for commits + verification): #38 JobId
preserve_order; VRAM admission dimension missing in BoxBudget; chunked-path SIGTERM claim
release missing; claim CAS shells out to the aws CLI; `ZEN_MAX_MIN` ignored in single-run
mode; worker task shape (long-lived + idle backoff, opt-in `ZEN_LONG_LIVED=1`); capability
routing dead code (`metric_class()` returned `Gpu` unconditionally — fixed alongside #38
in the same commit, `1b2a1452`, along with a same-root-cause bug in
`epoch::any_gpu_metric`'s GPU-suffix check).

Still open, in priority order: `WorkerReport`/dash inputs produced by nothing in
production; `Lease::renew` has no caller (claims age out from original timestamp only —
relevant again now that the worker can be long-lived: a long `ZEN_LONG_LIVED=1` idle-poll
cycle holding a claim across many minutes should renew it, not just let it ride the
original timestamp toward staleness); verify the warm-exec `/tmp` source-cache sweep
reached every fleet box (the 620 GB ENOSPC class); `crates/zenfleet-vastai`'s chunk worker
has the same SIGTERM-release gap as the now-fixed zenfleet-worker chunked path (out of
scope for the LAN/Nomad precondition, since cloud burst stays zenfleet+R2 unchanged — left
as a separate, lower-priority item). Landed this month and load-bearing: error-class
fidelity (`SourceFetch`/`DiskFull` + markers), ledger-snapshot `--ledger-in` contract in
both modes, epoch-sharded claiming + weighted handicaps, warm exec pool + ScoreFile
warm-ref.
