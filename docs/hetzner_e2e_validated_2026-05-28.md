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
