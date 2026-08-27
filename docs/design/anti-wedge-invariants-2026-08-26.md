# Anti-wedge invariants — making wedged workers structurally impossible

Operator question (2026-08-26): "how can we fundamentally make it impossible to
wedge workers." Written against the day's live incident taxonomy; every invariant
below maps to a real wedge observed in the last 48 h, cited inline. The framing:
**you cannot forbid a process from misbehaving — you can make every misbehavior
lose its work-claim, leave a record, and be indistinguishable from a clean crash.**
A "wedge" is then impossible to *persist*: the system reaps it and moves on.

## The observed wedge taxonomy (2026-08-25/26)

| # | mode | example |
|---|---|---|
| W1 | invisible progress (healthy worker, no evidence) | LPT giant chunks flush only at completion — hours of results in RAM; operator killed productive passes twice; `docker stop` discarded an unflushed batch |
| W2 | liveness proxy kills healthy work | `ZEN_PASS_TIMEOUT=7200` murdered a productive pass, labeled "worker hung" (rc=124) |
| W3 | silent degraded mode | snapshot fetch fails `|| rm -f` → empty view → the documented 2.0-3.4× re-work treadmill (measured 2.44×); same pattern class as the fleet-writes-to-R2 incident |
| W4 | fast-fail retry treadmill | classing hole → VRAM-ungated CUDA storms; 46,680 failures; with done-only snapshots, attempts recompute as 1 forever — the poison ladder is inert |
| W5 | stale-code worker | lilith's zombie on an old image grinding gm-decode failures (1,259 fails/10 min) with no version/capability gate to exclude it |
| W6 | self-overload | `host_box_budget()` probes 32 host cores inside a 24-core cpuset → oversubscribed slice, tower load 42 |
| W7 | monitoring wedged, not worker | in-flight footerless chunk killed every ledger reader |
| W8 | dead box mid-work | WSL layer failure with claims held (TTL steal already covers this — the one mode the current design handles) |

## The invariants (priority order)

1. **Progress-conditioned lease renewal (kills W1+W8 persistence).** A chunk
   claim's refresh MUST carry a progress delta (`cells_done/total` in the claim
   body, already `<ts> <worker>` — extend it). Renewal without progress after K
   consecutive refreshes = voluntary release. A wedged worker then *loses the
   lease by construction* and the cell re-enters the gap; a healthy-but-slow one
   keeps it by showing deltas. Progress becomes observable from the store alone
   — no ssh, no docker logs, no guessing.
2. **Per-cell watchdog in the executor parent (kills W2's reason to exist).**
   Every cell child gets a kind-specific deadline (encode/score/diffmap × size
   tier; the per-metric `estimate_score_time_ms` × safety factor is already the
   right oracle). Parent kills at deadline → `Failed` row, `error_class:
   cell_timeout`, stderr tail attached. No cell can hold a slot forever, so the
   pass never needs a kill-timer: `ZEN_PASS_TIMEOUT` demotes to a 10×-expected
   absolute backstop labeled "budget exceeded", never "hung".
3. **Fail-loud preconditions, no silent degraded modes (kills W3).** The
   entrypoint verifies snapshot-fetch, creds, store reachability, and the
   image's compiled feature set; ANY miss = nonzero exit with a named cause.
   Degraded modes exist only behind explicit opt-in envs
   (`ZEN_ALLOW_NO_SNAPSHOT=1`). This is the bake-everything rule's runtime
   counterpart: first boot is production, and production never limps silently.
4. **Views that carry failure history (kills W4).** Snapshots today are
   done-rows-only, so `attempts` always recomputes to 1 and Poison is
   unreachable — pathological cells retry forever. Snapshot = done rows + the
   latest failed row per job_id. Then the existing retry-vs-poison ladder
   actually quarantines repeat offenders, and the treadmill self-limits. Also:
   fold chunk sidecars newer than the snapshot into the view at pass start
   (bounded work), so a stale snapshot degrades to a small tax, not a treadmill.
5. **Capability/version gating at claim time (kills W5).** The run manifest
   declares required executor capabilities (e.g. `hdr-gainmap`, min build sha);
   a worker compares its own capability set (the zenmetrics-orchestrator
   capability cache is exactly this substrate) and self-excludes from runs it
   cannot serve. A stale-image worker then claims nothing instead of grinding
   failures.
6. **cgroup-truthful budgets (kills W6).** `host_box_budget()` reads
   cpuset/cpu.max and memory.max from the cgroup before falling back to host
   probes. A capped container then admits within its slice.
7. **Tolerant readers everywhere (kills W7).** Per-file try/skip with loud
   counts in every ledger reader (pool report done 2026-08-26; port to
   zenfleet-ledger's dir reads + writeback + ad-hoc probes).
8. **The box-lifecycle layer is already decided — use it (bounds everything).**
   The Nomad ADR (docs/status/fleet-orchestration-2026-08.md) exists precisely
   because "the hand-rolled worker management layer is what recent campaign
   incident logs kept paying for" — tonight was another installment. Nomad owns
   process liveness/restart/drain + node health; zenfleet keeps cell-granular
   exactly-once. A worker that stops heartbeating gets restarted by the layer
   whose whole job that is, instead of by an operator reading `ps` at midnight.

## What NOT to build

- More timeouts on outer layers (W2 shows they kill healthy work — deadlines
  belong at the smallest unit, the cell, where "expected duration" is knowable).
- Operator-side watchers as the primary mechanism (they watch signals W1 makes
  meaningless; fix observability at the source instead).
- Process/PID-count health checks (`nvidia-smi`/`ps` counts are documented-
  unreliable — fleetbench G-T1; ground truth is `can_admit`'s own accounting
  and, after invariant 1, the claim bodies).

## Sequencing

Invariants 3, 6, 7 are small and immediate. 1, 2, 4 are one focused zenfleet
wave (claim-body schema + watchdog + snapshot schema, each with a live-store
test per the preconditions pattern). 5 rides the orchestrator capability cache.
8 is the ADR's P2/P3, already in motion.

## Addendum (2026-08-27, bitten live): W9 — the hanging-GET wedge

A store serving objects whose chunks are GONE (damaged/quarantined volumes)
can HANG the transfer instead of erroring. Measured: 67 dead ledger sidecars
wedged every reader in the system at once — worker pass-start sidecar folds
(invariant 4's own machinery!), `jobctl report`'s per-file fallback, and an
operator verify task — while direct probes of healthy objects stayed green,
making the store look fine from outside. The tolerant-reader contract
(invariant 7) is not enough when "unreadable" presents as "never returns".

**Invariant 9b: every remote transfer is BOUNDED.** zenfleet-ledger's s5cmd
transfers now carry a timeout (default 120 s, `ZEN_S5CMD_TIMEOUT_SEC`,
0 disables) so a hanging object becomes a counted skip, never a wedge
(`294a6944`). Mitigation verb for the incident class: identify dead objects
with short-timeout parallel scans and DELETE them — their data is already
lost with the volumes, and the job system's latest-wins + gap semantics
re-run the affected cells (the 2026-08-27 recovery: 67 objects, report went
from wedged to 2 s, zero completed-work loss confirmed by `audit-blobs`).
