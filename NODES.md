# Fleet nodes

The household fleet roster (node identities, hardware, network, provisioning,
PXE server, per-OS worker setup) lives in the **private** repo
[`imazen-private/homefleet`](https://github.com/imazen-private/homefleet)
(checked out locally at `~/work/zen/homefleet/zenmetrics/NODES.md`).

Public docs and benchmarks refer to nodes only by neutral IDs:
`node-2`, `node-3`, `tower`, `lianli`, `mac`, `ryzen5800xt`, `i265`.

## Standing rule: `node-2` and `node-3` are permanent Ubuntu workers (2026-08-05)

**User directive, 2026-08-05: these two boxes stay flipped to Ubuntu, always.**

`node-2` and `node-3` are dual-boot boxes that used to default to their other OS,
with the fleet borrowing them opportunistically and handing them back afterwards.
That arrangement is **revoked**. Both now boot into the Ubuntu fleet worker by
default, on every boot.

Practical consequences for anyone driving the fleet:

- Treat `node-2` and `node-3` as **always-Ubuntu workers**, in the same class as
  `lianli` and `i265` — not as borrowed capacity with a return deadline.
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
(with `lianli`). The provisioning commands, addressing, and per-box hardware
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
