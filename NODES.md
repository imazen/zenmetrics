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
