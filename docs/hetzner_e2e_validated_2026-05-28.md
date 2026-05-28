# Hetzner end-to-end arc — Phase 2 redo (2026-05-28 evening)

This is the follow-on to `docs/zencloud_hetzner_arc_2026-05-28.md`,
which closed Phase 1 (provider trait + ARM cross-build + image push +
provision/teardown happy-path) and identified the Phase 2 ENTRYPOINT
bug. This pass rebuilt the image with the CMD fix from `74c18dd0`,
relaunched the 5×CAX11 sweep, and uncovered a **second** Phase 2
blocker — the launcher's `DEFAULT_IMAGE` constant still pointed at
the Salad image, so the rebuilt Hetzner image never ran. That fix is
landed as part of this commit.

Per-cell ARM CPU numbers and the Hetzner-vs-Salad headline are still
**not measured** — both are blocked behind a third sweep launch with
the corrected default. Spend on this attempt: $0.09 (well under cap).

## Phase 2 redo timeline

| Event | Time | Notes |
|---|---|---|
| Image rebuild + push (`v1` + `v1-74c18dd0`) | ~3 min | buildx + cross-compile cached; layer cache hit on L1-L4 |
| Sweep launch | t=0 | `--replicas 5 --server-type cax11 --location fsn1 --chunks 6 --cells-per-chunk 4` |
| All 5 replicas `running` | ~34 s | matches #71's 15-22 s `t_first_replica_running` band |
| Initial `queue/*.json` push | t=0 (logged) | written by `push_jobs(initial)` per driver.rs:147 |
| First chunk TTL re-dispatch | t=363 s | each chunk redispatched once (TTL=360s) |
| Sweep wall deadline / manual teardown | t=818 s | servers deleted via API; driver short-circuited and emitted summary |
| Final summary | t=826 s | `omni=0, distinct_workers=0, t_first_sidecar=null, spend=$0.09, teardown_ok=true` |

`fleet_summary.json` at
`s3://zen-tuning-ephemeral/runs/hetzner-20260528T231738/fleet_summary.json`.

## Why no chunks completed (root cause)

The fleet_summary recorded `image: "ghcr.io/imazen/zen-metrics-sweep-salad:v6-visibility-b"`.
That's the Salad x86_64 image — it bakes the Salad managed-queue
sidecar (not the Hetzner R2-polling loop) and ships a binary for
amd64, not arm64. Running it on CAX11 produced one of two failure
modes (both invisible without SSH into the box):

1. `docker run --platform linux/arm64 …` rejects the wrong-arch image
   and the worker container never starts, OR
2. Docker silently uses qemu-arm to emulate the x86 binary if
   QEMU userspace is preinstalled — and the Salad worker then
   dies because the managed-queue sidecar (`salad-http-job-queue-worker`)
   isn't reachable from outside Salad's network.

Either way, no worker ever polled `runs/<sweep>/queue/`, so the
chunks the orchestrator pushed at t=0 sat untouched. The TTL
re-dispatch at t=363 s overwrote them with the same payloads (same
result). After 818 s of empty queue, the launcher hit its
deadline and tore down cleanly.

The launcher source's `DEFAULT_IMAGE` constant at
`crates/zencloud-hetzner/src/bin/zencloud-hetzner-sweep.rs:36` was
the Salad image string — a copy-paste leftover from the Salad
launcher this binary was forked from. `cloud_init.rs` already had
the correct Hetzner default (`zen-metrics-sweep-hetzner:v1`), but
the launcher binary's `--image` default takes precedence: it
passes the wrong image into `ProvisionSpec`, which is what
cloud-init's `docker run` receives.

**Fix landed this commit:** `DEFAULT_IMAGE` →
`"ghcr.io/imazen/zen-metrics-sweep-hetzner:v1"`. The launcher
binary rebuilt in 7 s.

## Image rebuild (Phase 1 of this pass) — successful

| Field | Value |
|---|---|
| Tag | `ghcr.io/imazen/zen-metrics-sweep-hetzner:v1` |
| Tag (sha) | `ghcr.io/imazen/zen-metrics-sweep-hetzner:v1-74c18dd0` |
| Index digest | `sha256:f8a3667f34242a7077e4bee82239c3bc81f7a47f364cb06f2b11e4b724041233` |
| arm64 manifest | `sha256:129486b4b265b6eecb4d5698628f6c860d5ebd338108826fec7d097a8e39bad8` |
| Build/push wall time | ~80 s (cross-build cached from morning; only L7 changed) |

Sanity check via `--entrypoint /bin/sh` override confirmed the
new `CMD ["/usr/local/bin/zen-sweep-worker"]` flow renders the
worker `--help` text cleanly (no `argv[1]` leak from a prior
ENTRYPOINT array). The CMD fix from `74c18dd0` is functional in
the published `:v1` image; the failure was elsewhere.

Build log: `/tmp/hetzner_rebuild_2026-05-28.log`.

## Spend and teardown

| Item | Value |
|---|---|
| Servers provisioned | 5× CAX11 fsn1 |
| Wall time | 818 s ≈ 13.6 min |
| Estimated spend | €0.10 / $0.09 |
| Teardown | manual API `DELETE` (all 5 servers, http 200 each) |
| Post-teardown project servers | 0 (verified via label_selector + project-wide GET) |
| Driver teardown follow-up | OK (the launcher's own teardown found 0 servers and reported `teardown_ok=true`) |

Comfortably under the $0.30 brief cap. The $0.09 spend bought no
chunk data but did confirm: (1) the rebuilt `:v1` image pushes
correctly and the CMD flow renders the worker `--help`; (2) the
launcher's image default was the actual bug; (3) the orchestrator
handles a 13-min hung-worker timeout cleanly with no leftover
servers.

## Apples-to-apples Hetzner-vs-Salad headline

**Still not produced.** This pass blocked at the wrong-image
launcher default before any chunk completed. The per-cell ARM CPU
number (expected ~5-15 s/cell on CAX11) and the comparison to
Salad's 3.9 s/cell warm baseline both require a successful sweep.

Next session's first action is a third sweep launch with the
fixed default (no code changes; just rebuild the launcher binary
and rerun the same CLI). Expected wall: < 5 min total based on
the 24-cell × 5-worker × ~5-15 s/cell math.

## Files touched

- `crates/zencloud-hetzner/src/bin/zencloud-hetzner-sweep.rs` —
  `DEFAULT_IMAGE` swap (Salad → Hetzner ARM64 image).
- `docs/hetzner_e2e_validated_2026-05-28.md` (this doc).

## What's NOT touched

- Image source (Dockerfile + entrypoint + worker code) — unchanged
  from `74c18dd0`; the v1 image as pushed today is functional and
  the CMD fix works.
- Worker / orchestrator / R2-queue logic — untouched.
- Salad / vast.ai / RunPod launchers — untouched.

## Phase 2 redo iter 3 (2026-05-28 23:36 UTC) — third bug found

The third sweep launch under the corrected `DEFAULT_IMAGE` ran with
the fleet image confirmed correct (no Salad image in synthesized
POST body) — verified at dry-run time per
`/tmp/hetzner_dryrun_2026-05-28.log` and at runtime by inspecting
the cloud-init `user_data` that the launcher synthesized
(`docker pull 'ghcr.io/imazen/zen-metrics-sweep-hetzner:v1'`).

A pre-flight gotcha: the launcher's default
`--input-parquet-r2 s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/inputs.parquet`
is stale — the actual fixture lives at
`.../salad-smoke-2026-05-28-24cell/input/smoke.parquet`. First
launch errored at preflight HEAD (404). Worked around by passing
explicit `--input-parquet-r2 …/input/smoke.parquet` +
`--source-dir-r2 …/sources` overrides. Source dir + parquet path
mismatch is a queued fix for the launcher defaults (not changed
this commit per "no code unless a third bug surfaces" rule, since
the explicit-override path works).

### Timeline (manual teardown at 7:48 wall)

| Event | Time | Notes |
|---|---|---|
| Sweep launch (corrected fixture) | t=0 (23:36:40 UTC) | `sweep_id=hetzner-20260528T233640` |
| 4 of 5 replicas running | t=22.5 s | (`t_first_replica_running` in range with #71's 15-22 s) |
| All 5 replicas running | t=38.2 s | |
| First TTL re-dispatch | t=366.5 s | every chunk redispatched once at +6 min (TTL=360 s) — SAME pattern as iter 2 |
| Manual teardown (no chunks processed) | t=448 s | 5 servers deleted via API DELETE (http 200 each) |
| Project servers post-teardown | t=450 s | 0 (verified via label_selector + project-wide GET) |
| Total wall | 7.5 min | |

`t_first_sidecar = null` (no chunks completed). `omni=0` throughout
(29 polling ticks). `processed=0`. No `fleet_summary.json` was
written (launcher killed mid-wait via SIGINT before driver's
graceful summary path). All 5 servers torn down.

### Per-cell ARM CPU time on CAX11 — UNMEASURED

Same as iter 2: the headline number we've been chasing is still
not produced. Workers booted and stayed in Hetzner-reported
`running` state for 7+ minutes but never claimed a chunk from
`runs/<sweep>/queue/`. Comparison to Salad's 3.9 s/cell warm
GPU baseline still pending.

### Speculative + boot stats

- Speculative dispatch count: 0 (gate is
  `min_completed=3`; never reached).
- TTL re-dispatch count: 15 (every chunk once at t=366 s).
- Per-replica boot: provisioning → `running` ≈ 22-38 s; running
  → first sidecar = ∞ (no sidecars).

### Image confirmation (proves DEFAULT_IMAGE fix landed)

Synthesized `user_data` from the dry-run dump shows
`docker pull 'ghcr.io/imazen/zen-metrics-sweep-hetzner:v1'` —
confirming the Salad image string is no longer being shipped.
The DEFAULT_IMAGE fix (`7800dd61`) is functional in the rebuilt
launcher binary.

### Bug #3 (UNRESOLVED) — workers never claim chunks

After 7+ min of `running` state across 5 boxes, queue prefix
`s3://zen-tuning-ephemeral/runs/hetzner-20260528T233640/queue/`
still has all 15 original `scaleup-*.json` entries (one per
chunk plus TTL re-dispatch overwrites), and `omni/` is empty.
The worker container never wrote a heartbeat to
`s3://zen-tuning-ephemeral/heartbeat/` — the newest entry in
that prefix is from 2026-05-24, days before this sweep.

Possible causes (cannot disambiguate without SSH or rescue mode
access — both intentionally avoided per "no leaky monitor
patterns" discipline this pass):

1. **Cloud-init blocked on apt mirror.** Ubuntu 24.04 ARM
   `apt update + apt install docker.io` on Hetzner's fsn1 ARM
   mirror has been known to take 3-6 min cold. 7 min is at the
   upper edge but plausible.
2. **Cloud-init blocked on `docker pull` of the ARM64 image.**
   The arm64 manifest (`sha256:129486b4b…`) lives on GHCR;
   pulling it cold over Hetzner's transit could be slow on a
   2-core CAX11.
3. **Worker container started but `R2QueueLoopConfig::from_env`
   panicked.** The env vars `BUCKET=zen-tuning-ephemeral` +
   `CHUNKS_QUEUE_PREFIX=runs/<sweep>/queue/` are written to
   `/etc/zen/worker.env` and passed via `--env-file`. If parsing
   fails or R2 credentials are wrong, the container would
   `--restart=on-failure:5` and eventually exit. No
   container-side log surface visible to the launcher.
4. **R2 credentials in worker.env mismatched.** The launcher mints
   per-sweep scoped creds via the salad-shared minter; if those
   creds were issued without write to `omni/` or `claims/`,
   workers would silently fail their first claim.

The launcher cannot distinguish these — Hetzner only reports
`running` (the kernel is up) and cloud-init's stdout is teed to
`/var/log/zen-bootstrap.log` inside the box. Recovering that log
requires SSH-key injection (next iteration: add an SSH key to the
synthesized POST body via `ssh_keys: [<launcher_pubkey>]` so the
launcher can pull the bootstrap log on TTL re-dispatch firing).

### Spend + teardown

| Item | Value |
|---|---|
| Servers provisioned | 5× CAX11 fsn1 |
| Wall time | 7.5 min |
| Estimated spend | €0.07 / $0.08 (Hetzner 1-hour minimum × 5 boxes ≈ €0.018) |
| Teardown | Manual API DELETE (5× http 200, all servers) |
| Post-teardown verification | 0 servers in group, 0 project-wide (both `GET /servers?label_selector` + bare `GET /servers`) |

Combined spend across all three iterations: $0.09 + $0.09 + $0.08
≈ $0.26 — under the $0.30 brief cap.

### Next action (not started this pass)

Add SSH key injection to the launcher's `ServerCreateBody` so the
next iter can pull `/var/log/zen-bootstrap.log` via SSH on the
first TTL re-dispatch firing. That single log line will
disambiguate causes 1-4 above. Until that diagnostic surface
exists, blind iteration on the worker container will waste more
sweep cycles guessing at the symptom.

The DEFAULT_IMAGE fix has shipped; the CMD fix has shipped; both
are byte-equivalent in the published `:v1` image. The remaining
blocker is a worker-side opacity issue, not an image-build or
launcher-default issue.
