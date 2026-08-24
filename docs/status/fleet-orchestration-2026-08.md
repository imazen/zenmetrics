# Fleet orchestration — Nomad decision + LAN-store next generation (2026-08)

Status/ADR doc, written 2026-08-22 from an operator-requested review. The private
companion (concrete topology, addresses, per-box roles, runbooks) lives in the private
fleet repo: `homefleet zenmetrics/ORCHESTRATION-2026-08.md`. Neutral node IDs only here.

## Decision

- **Pilot HashiCorp Nomad (CE **2.0.5**, corrected 2026-08-24 from this doc's original
  "1.11.x" — see the private topology doc's "Nomad specifics" for the full version
  analysis: the free/CE line moved to the 2.x series after 1.11.3, 1.11.4+ is
  Enterprise-only and never got 2.0.4's two Docker-driver CVE fixes, licence terms are
  unaffected; BUSL — fine for internal, non-competing use) as the BOX-LIFECYCLE layer
  for the LAN fleet only**: node membership + liveness, worker deployment/restart,
  drain (before dual-boot OS flips), image rolls, resource caps, periodic plumbing
  jobs, and node metadata (arch/SIMD-vendor/GPU/VRAM as node attrs).
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

**Status (2026-08-24, revised twice same day): three of five hold up; #2 was
REOPENED then RESOLVED; #4 is REOPENED and NOT yet resolved** — item 2's original
verification (a synthetic/mocked harness) did not catch that the real release
never fired in a real end-to-end test (found during G-P2 gate testing, same day)
— **then RESOLVED later the same day** after adding diagnostics + rebuilding the
executor image (see item 2 and `benchmarks/fleetbench_2026-08-24.md`); treat it
as re-verified-but-not-fully-understood (2/2 real repro, mechanism not
conclusively pinned down). Item 4 (VRAM admission) hit the SAME pattern during
the G-T1 ladder pass (later still, same day) — code + unit tests said done, a
real 500-job GPU-score run measured 16 concurrent CUDA contexts against a
predicted ~2-3 ceiling — and is **NOT yet resolved** (root cause not isolated;
see item 4 and CLAUDE.md's Known Bugs). Items 1/3/5 (test suites + one item live
against the real LAN store — see each entry) held throughout; the ledger
snapshot decision below stayed conservative anyway.

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
2. **SIGTERM chunk-claim release on the chunked path** — ⚠️ **REOPENED then
   RESOLVED same day, 2026-08-24 (G-P2 gate testing) — see CLAUDE.md's
   `### Resolved` entry and `benchmarks/fleetbench_2026-08-24.md` for the full
   arc.** The `0de119d6`/`45c57bd3` fix had only been verified against a
   synthetic/mocked harness; a REAL test (real `zenfleet-worker:exec` image,
   real Nomad allocation, a confirmed live chunk claim read directly off the
   LAN store, a real `nomad node drain -force`) showed the container exiting
   promptly and cleanly with the bash trap correctly firing — but the S3 claim
   object unchanged afterward, meaning the release itself never happened. A
   raw (no Docker/Nomad) reproduction proved the release LOGIC was correct in
   isolation, narrowing the bug to the container/Nomad/entrypoint layering.
   **Fix:** added an immediate marker + explicit stderr flushes to
   `spawn_spot_reclaim_chunk` and sub-second timestamps around the entrypoint's
   `wait`, rebuilt the executor image (musl), and re-tested — **2/2 real repro
   then showed the claim genuinely released.** Honest caveat: the new
   diagnostic lines STILL never appeared in captured logs even in the 2
   successful runs (the claim's deletion is what proves the code ran, not a
   log line) — so treat this as re-verified-but-not-fully-understood, not
   bulletproof. A separate exactly-once check was attempted but confounded by
   the lease-mode heterogeneous-chunking bug (item below) — needs a dedicated
   epoch-sharded rerun to isolate cleanly.
3. **`ZEN_MAX_MIN` honored in single-run mode** — ~~the wsl-gate 5-6 h unbudgeted-worker
   incident~~ **FIXED, `45c57bd3`**: wired identically to `pool_mode`'s existing pattern.
   Verified end-to-end: `ZEN_MAX_MIN=1` exits at exactly 60s wall-clock. Nomad
   `kill_timeout` remains the backstop, not the primary mechanism.
4. **VRAM admission dimension in `BoxBudget`** — ~~the avifgen OOM-storm follow-up~~
   believed **FIXED, `58c5db95`** (`BoxBudget` gained an optional third
   `vram_budget_bytes` axis, `None` = don't gate; `host_box_budget()` probes real
   GPU VRAM via `nvidia-smi` and sets a 90%-of-probed budget) — **⚠️ REOPENED
   2026-08-24 (G-T1 ladder pass), NOT yet resolved.** A live 500-job GPU-score
   run (`fleetbench-gpuscore-vram.nomad.hcl`, r7900x, 6 GB GTX 1060, watched via
   `nvidia-smi --query-compute-apps` every 5s) hit **16 concurrent CUDA contexts**
   — the admission math (`DEFAULT_GPU_JOB_VRAM_BYTES` 2 GiB/job vs. a
   correctly-probed ~5.8 GB budget, probe itself directly verified working
   inside the exact container image) predicts a ceiling of ~2-3. Individually
   ruled out: probe failure, GPU-metric misclassification, a stale/populated
   hint bypassing the fallback estimate, and a broken `InFlight` counter — each
   checked against source, none explains the gap. Root cause NOT isolated
   (needs dedicated timestamped tracing of `can_admit`'s runtime `cand_vram`
   argument). No OOM occurred in the observed run only because real per-job
   VRAM usage (8-256 MiB) stayed far under the 2 GiB reservation — the
   documented "large tier" (~1.3 GB real/job) would put 16-way concurrency at
   ≈21 GB, a genuine OOM on this card. **Do not treat VRAM admission as a
   working OOM guard until root-caused.** Full writeup:
   `benchmarks/fleetbench_2026-08-24.md`'s G-T1 VRAM-admission section;
   CLAUDE.md Known Bugs. Nomad's device plugin can report VRAM later; the
   admission decision itself stays zenfleet's regardless, per the ADR's
   original framing — that framing is unaffected by this defect.
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

- **P0** — all five preconditions **DONE**, though precondition 2 (SIGTERM chunk-claim
  release) went REOPENED→RESOLVED same day 2026-08-24 (re-verified-but-not-fully-
  understood — see "Preconditions" above). 3-server quorum **LIVE** (dev +
  tower(container) + r7900x, CE 2.0.5, raft healthy, dev leader).
- **P1 pilot** — servers up **DONE**. First client (i265) up and MECHANISM PROVEN
  2026-08-24 (not yet the full gate — see below): a real `service` job (Docker driver,
  the freshly-rebuilt `:exec` image, `ZEN_LONG_LIVED=1`) claimed + executed a 3-job
  smoke manifest end-to-end (`done=3`), stayed alive through 10+ idle passes with
  correctly growing/capped backoff (proving the P0 idle-backoff precondition works
  under real Nomad, not just the fake-tool harness), and a real `nomad node drain`
  delivered SIGTERM through Docker to the script's own PID-1 trap and the container
  exited gracefully in ~3s under an
  artificially-widened 20s `kill_timeout` (ruling out "it just happened to be faster
  than the forced-kill timeout" as an explanation) — **this proves the container-level
  signal delivery, NOT that the claim actually gets released; that turned out to be a
  separate, still-broken step — see precondition 2's reopening above.** Jobspec:
  `homefleet zenmetrics/ubuntu-node/nomad/jobs/zenfleet-worker-pilot.nomad.hcl`.
  Two real findings along the way, both fixed: Nomad's docker driver does not
  re-pull an already-cached tag by default (`force_pull = true` now required in
  every jobspec using a moving tag), and the executor image's binaries must be built
  with the musl target or a build-box-glibc-drift silently produces a binary that
  can't even start in the (older-glibc) base image.
  **Update 2026-08-24 (later same day):** the second pilot client is **r3500**, not
  i134 — both of the ADR's original picks turned out genuinely blocked (r5900xt never
  answers WoL at all; i134 was found mid-active-session in Windows and correctly left
  alone) and r3500 (freshly installed, no shared-use concerns) substituted cleanly.
  **G-P1 (WoL round-trip) PASSED 3/3 on both i265 (8s/8s/8s) and r3500 (7s/7s/7s)** —
  harness: homefleet `ubuntu-node/nomad/wol/{arm_wol.sh,wol_roundtrip_test.sh}` (built
  from scratch, nothing existed before). **fleetbench-2026-08-24 is now frozen and
  declared** (8,415 real encode DesiredJobs — zenjpeg/zenavif/zenjxl/zenpng, see
  `benchmarks/fleetbench_2026-08-24.md`) and **G-N1's real-workload throughput
  comparison is in progress**: the G-P0 systemd baseline is running against it on
  r7900x/i265/r3500 as this line is written, with a significant early finding —
  lease-mode claiming's genuine duplicate-execution rate climbed from 62% to 84%+ as
  the gap neared completion (a severe, LAN-latency-amplified case of the documented
  §5b lease-race, not a defect in the new deployment path — see the benchmarks doc for
  the full readout once G-P0 exits and G-N1 has run the identical manifest).
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
preserve_order; claim CAS shells out to
the aws CLI; `ZEN_MAX_MIN` ignored in single-run mode; worker task shape (long-lived +
idle backoff, opt-in `ZEN_LONG_LIVED=1`); capability routing dead code (`metric_class()`
returned `Gpu` unconditionally — fixed alongside #38 in the same commit, `1b2a1452`,
along with a same-root-cause bug in `epoch::any_gpu_metric`'s GPU-suffix check).

**REOPENED then RESOLVED 2026-08-24, same day (G-P2 gate testing):** chunked-path
SIGTERM claim release — reported fixed above (`0de119d6`/`45c57bd3`) on the strength
of a synthetic/mocked harness only; a real end-to-end test first showed the claim
was NOT actually released, then a diagnostics-plus-rebuild fix made it work
reliably (2/2 real repro) — see "Preconditions" item 2 above, CLAUDE.md's
`### Resolved` entry, and `benchmarks/fleetbench_2026-08-24.md` for the full
investigation, including the honest residual puzzle about why the new diagnostic
output still doesn't appear in captured logs despite the fix working.

**REOPENED 2026-08-24, same day, NOT yet resolved (G-T1 ladder pass):** VRAM
admission dimension in `BoxBudget` — reported fixed above (`58c5db95`) on the
strength of code review + unit tests; a real 500-job GPU-score run measured 16
concurrent CUDA contexts on a 6 GB card against a predicted ~2-3 ceiling. Probe
failure, metric misclassification, a stale hint, and a broken `InFlight` counter
were each individually checked against source and ruled out — root cause not
isolated. No OOM occurred in the observed run only because this workload's real
per-job VRAM usage stayed far under the flat 2 GiB/job reservation estimate; a
larger-tier GPU workload would likely OOM for real. See "Preconditions" item 4
above, CLAUDE.md Known Bugs, and `benchmarks/fleetbench_2026-08-24.md`'s G-T1
VRAM-admission section for the full investigation.

Also found + fixed the same day, in `scripts/jobsys/fleet_power.py` (chasing the
above): `suspend()`'s drain lacked `-force` (a plain `-enable -deadline 2m` doesn't
promptly interrupt a running allocation — measured ~70s, not "seconds"); `-force` and
`-deadline` are mutually exclusive Nomad CLI flags (the naive fix errors at runtime,
silently, since the call site uses `check=False` — caught by testing live before
shipping); `cmd_apply` passed `node_id=None` to `suspend()`, so the drain branch never
actually ran; a drained node stays `ineligible` forever after the drain completes
(separate flag from `Drain`) with no code anywhere to re-enable it, so a box would wake
from one sleep cycle and then never receive Nomad work again — all four fixed.

Still open, in priority order: the reopened SIGTERM release item above (now top
priority); `WorkerReport`/dash inputs produced by nothing in
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

**New 2026-08-24 (fleetbench G-P0 baseline, high priority — directly relevant to this
mission's fleet shape):** chunk-mode's per-chunk lease claim provides **zero**
cross-worker dedup on a heterogeneous-core-count fleet — measured 100%
duplicate-execution rate on a real 3-box run (r7900x 24C / i265 20C / r3500 6C).
Root cause: chunk-id hashes chunk MEMBERSHIP, and membership comes from
`BoxBudget::pack_chunks_lpt`, which takes the calling box's OWN core/RAM budget —
different-core boxes partition the identical gap into non-matching chunks, so
leases never collide. Every household LAN fleet is heterogeneous by construction,
making this maximally relevant here. Full writeup + likely fixes (canonical
box-independent chunk partitioning, or verify epoch-sharded claiming sidesteps it
since it shards by cell hash, not local budget): `benchmarks/fleetbench_2026-08-24.md`,
CLAUDE.md Known Bugs.
