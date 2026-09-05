# Fleet nodes

The household fleet roster (node identities, hardware, network, provisioning,
PXE server, per-OS worker setup) lives in the **private** repo
[`imazen-private/homefleet`](https://github.com/imazen-private/homefleet)
(checked out locally at `~/work/zen/homefleet/zenmetrics/NODES.md`).

Public docs and benchmarks refer to nodes only by neutral IDs:
`node-2`, `node-3`, `tower`, `r7900x`, `mac`, `r5900xt`, `i265`.

## Standing rule (REVISED 2026-08-06): `node-2` is a permanent Ubuntu worker; `node-3` defaults to Windows

**User directive, 2026-08-06 (supersedes the 2026-08-05 "both permanently Ubuntu" rule below
for `node-3` ONLY): `node-2` stays a permanent Ubuntu fleet worker; `node-3` returns to
Windows as its default.** Executed 2026-08-06 ~21:30Z: `node-3` drained (worker unit stopped
+ disabled; in-flight chunks lease-expired and re-claimed by the rest of the fleet — the
content-addressed ledger makes an interrupted box cost only its un-sidecar'd in-flight
cells, which other workers redo), PXE worker flag CLEARED, rebooted, verified up in Windows.

- Do NOT re-flip `node-3` to Ubuntu on the strength of the 2026-08-05 rule — that rule is
  superseded for `node-3`. Borrowing `node-3` again is a user-approval action.
- Everything below about `node-2` still stands: permanent Ubuntu worker, one of the two GPU
  metric-scoring nodes (with `r7900x`); if found in its other OS, repair it back to Ubuntu.

## Superseded (for `node-3`) — standing rule of 2026-08-05: `node-2` and `node-3` are permanent Ubuntu workers

**User directive, 2026-08-05: these two boxes stay flipped to Ubuntu, always.**
*(2026-08-06: still in force for `node-2`; superseded for `node-3` — see above.)*

`node-2` and `node-3` are dual-boot boxes that used to default to their other OS,
with the fleet borrowing them opportunistically and handing them back afterwards.
That arrangement is **revoked**. Both now boot into the Ubuntu fleet worker by
default, on every boot.

Practical consequences for anyone driving the fleet:

- Treat `node-2` and `node-3` as **always-Ubuntu workers**, in the same class as
  `r7900x` and `i265` — not as borrowed capacity with a return deadline.
- **Do not flip them back as routine hygiene** at the end of a job or sweep.
  Returning a box to its other OS is now an explicit, deliberate, user-requested
  action only.
- If you find one of them booted into the other OS, that is drift (a hand-cleared
  flag or lost firmware boot order), not the intended default — repair it back to
  Ubuntu rather than leaving it.
- They can still be powered off or interrupted at any time. That remains normal
  and costs only the in-flight cell: the job ledger is resumable, so an
  interrupted cell requeues and is re-scored in the next window.

Both boxes carry GPUs; `node-2` is one of the two GPU metric-scoring nodes
(with `r7900x`). The provisioning commands, addressing, and per-box hardware
detail live in the private repo linked above.

### Yank rule: check session state, then reboot (2026-08-05)

User directive: **if the screen is locked or off, you may reboot/flip the box
immediately** — no waiting, no scheduling. Only an **active, unlocked, in-use**
session defers. Detection to run before each yank:

- **Linux:** `loginctl list-sessions`, then `loginctl show-session <id> -p Active
  -p LockedHint -p Class -p Remote -p Type`. Ignore `Class=manager` (that is the
  systemd user manager, not a person) and `Remote=yes` (that is your own SSH).
  Cross-check with `who` and `pgrep -x Xorg` / `pgrep -x gnome-shell`.
- **Other OS:** session/lock state through the admin SSH account — active-vs-
  disconnected session list, plus a locked-desktop check via the logon-UI
  process.
- **Inconclusive:** treat "no interactive session at all" as safe-to-yank and
  say so in the record; treat "active + unlocked" as defer.

Observed at the 2026-08-05 flips: **node-3 had no interactive session** (only a
`manager`-class session and the operator's own remote SSH; no display server
running) → yanked. **node-2** likewise had no interactive session at flip time.

### Boot-order note: PXE-first is intact on both (verified 2026-08-05)

A report that `node-3` "woke into its other OS because firmware boot order
regressed" was **investigated and not reproduced**. Both boxes read
`BootOrder` with **PXE IPv4 first**, and both showed `BootCurrent` = the PXE
entry — i.e. they genuinely booted via PXE.

The actual cause was that **the per-box worker flags were all cleared**, which
is the pre-directive default: PXE runs, consults the flag, finds none, and
chainloads the other OS exactly as designed. The fix is the standing rule above
(keep the flag set), not a firmware change.

Verified by a full reboot cycle on `node-3`: boot ID changed, `BootCurrent`
stayed the PXE entry, and it returned to Ubuntu unattended. **Before concluding
"firmware regression", check the worker flag first** — it is the far more likely
cause and costs one command to rule out.

## Nomad client stuck on `Node.Register: Permission denied` after a PARTIAL identity wipe (found + fixed 2026-08-25)

A LAN Nomad client (`r3500`) could not register on the 3-server cluster
(`dev` / `tower` / `r7900x`, all healthy, no ACLs, matching config, matching
clocks, matching root keyring across all three) — every `Node.Register` RPC
came back `rpc error: Permission denied`, and re-minting fresh
`nomad node intro create` tokens (including ones with an explicit `-ttl`)
never helped, no matter how quickly the token was deployed after minting.

**Root cause, confirmed via server-side `nomad monitor -log-level=DEBUG
-server-id=<raft-leader>`** (client-side logs never show the real reason —
only the generic `Permission denied`; you have to watch the leader's own
log at the moment of the RPC): the server logged, on every attempt,

```
[ERROR] nomad.client: node registration introduction authentication failure:
enforcement_level=warn node_id=<old-node-id> node_pool=default node_name=r3500
error="invalid claims: go-jose/go-jose/jwt: validation failed, token is expired (exp)"
```

— even immediately (single-digit seconds) after minting a token with a
30-minute TTL, and even after **deleting the intro-token file from disk
entirely**. That ruled out "the token itself is stale" outright: the client
was not authenticating with the file on disk at all.

The actual culprit was `state.db` in the client's data dir
(`<data_dir>/client/state.db`). A prior session's "wipe the client identity
for a fresh node_id" step had deleted `client-id` and `secret-id` but **left
`state.db` untouched**. `state.db` had a cached, previously-issued **node
identity token** (a separate, longer-lived JWT from the one-shot
introduction token — proof-of-identity used for ongoing RPCs after
registration) tied to the *old*, already-superseded `node_id`, and that
cached token had genuinely expired hours earlier. The client kept presenting
that stale cached identity instead of using the fresh introduction token,
so the "expired (exp)" error was **completely accurate** — just about the
wrong credential, one that no amount of re-minting could touch.

**Fix — wipe ALL FOUR client-identity files together, never just
`client-id`/`secret-id`:**

```
systemctl stop nomad
rm -f <data_dir>/client/{client-id,secret-id,state.db,intro_token.jwt}
nomad node intro create -node-name=<name> -node-pool=default -ttl=30m -json   # run on any server
# deploy the returned JWT to <data_dir>/client/intro_token.jwt (chmod 600), then:
systemctl start nomad
```

Verified: `node registration complete` on first start, and the node stayed
`ready`/`eligible` across 2 additional full `systemctl restart` cycles plus
60s of idle heartbeating with zero further RPC errors.

**Side finding on the "is registration just flaky here" question:** a
similarly-affected node (`i265`) that already had a long-lived successful
registration was separately seen throwing `Permission denied` on
`Node.UpdateStatus`/`Node.GetClientAllocs` (heartbeat RPCs, not
`Node.Register`) for about 30 seconds during the same investigation window,
then recovering on its own. That is a different, self-healing symptom
(an already-valid, already-registered session hitting a transient auth
hiccup) — don't conflate it with the `Node.Register` failure above, which
does **not** self-heal no matter how many times the client retries, because
the bad credential is cached on disk, not in a race.

Nomad's default `client_introduction` server config (no `client_introduction`
block was set on any of the 3 servers, so all defaults) is
`enforcement = "warn"`, `default_identity_ttl = "5m"`, `max_identity_ttl =
"30m"` — confirmed by requesting `-ttl=2h` and observing the server log
`node introduction identity TTL request exceeds server maximum, using
server maximum: requested_ttl=2h0m0s server_max_ttl=30m0s`. `warn`
enforcement only soft-allows a client that presents **no** introduction
token at all; a client that presents one and fails validation (expired,
wrong node, malformed) is hard-denied regardless of enforcement level.

## Fleet /tmp: disk-backed only (ban RAM-backed tmp everywhere, 2026-09-05)

**Standing rule**: no host or container in this fleet may serve worker scratch (pass output,
pooled manifests/runlists, ledger snapshots, jobexec's source cache) off a RAM-backed `tmpfs`.
`fleet-entrypoint.sh` now refuses to boot without a disk-backed `TMPDIR` — see
`docs/RUNNING_JOBS.md` §9c "TMPDIR discipline" for the mechanism and the launcher-side fix
(`lan_score_launch.sh`, `enroll_running_node.sh`). This table records what was audited directly
on 2026-09-05; hosts not listed are covered by the same launcher fix going forward but were not
individually re-verified this pass.

| Host | `/tmp` before | Action | `/tmp` after | Effective |
|---|---|---|---|---|
| `r7900x` | `tmpfs 15G` (systemd `tmp.mount`, static-loaded, no `/etc/fstab` entry) | `sudo systemctl mask tmp.mount` (no live job disturbed — a training process was running; mount left live, no reboot) | still `tmpfs 15G` live | **next reboot** (masked unit can no longer start) |
| `tower` (Unraid host) | host rootfs is RAM-booted by design — out of scope, never changed | none (host untouched, per the Docker-only rule) | unchanged | n/a |
| `tower` (`zen*` containers) | container `/tmp` is the overlay writable layer on `/var/lib/docker` (`btrfs` on a cache-pool loop device — disk-backed already, not RAM); no running `zen*` compute container had a `TMPDIR`/scratch mount at audit time | created `/mnt/user/coefficient/scratch` (array-backed) for launcher use; `lan_score_launch.sh` now bind-mounts it at `/scratch` + sets `TMPDIR=/scratch` on every future launch | disk-backed via bind mount | **next launch** of any `zen-score-*` container (no live compute worker was running to disturb — only `zen-lanstore`, a storage container, was up) |
| any LAN node enrolled via `enroll_running_node.sh` (the always-on pool workers) | depends on the box's own `tmp.mount` default | `enroll_running_node.sh` now bind-mounts `$HOME/tmp/zfw-scratch` (override `ZEN_TMPDIR_HOST_DIR`) at `/scratch` + sets `TMPDIR=/scratch` in the systemd unit's `ExecStart` | disk-backed via bind mount | **next `enroll_running_node.sh` run** (re-enroll to pick it up; not retroactive to an already-running unit) |

Per-host `tmp.mount` state elsewhere in the fleet (node-2/node-3/mac/other Ubuntu boxes) was not
re-audited this pass — check `findmnt -no FSTYPE,SIZE /tmp` before trusting any of them, and mask
`tmp.mount` (Linux) the same way if it comes back `tmpfs`.
