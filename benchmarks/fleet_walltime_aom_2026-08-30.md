# Fleet wall-time forensics + the claim-exclusivity A/B — `avifaom-enc-20260830`

**Date** 2026-08-30 · **Fleet** r3500 / r5900xt / r7900x (LAN, heterogeneous core counts)
**Store** tower SeaweedFS · **Tools** `scripts/jobsys/fleet_walltime.py`,
`scripts/jobsys/fleet_schedule_sim.py` (both new, committed with this record)
**Code under test** `zenfleet-core` `BoxBudget::pack_chunks_lpt_uniform` +
`fleet-entrypoint.sh` drain rule (commit `d7982e9b`)

## 1. Where the 12 hours went (reconstructed from the live ledger)

`fleet_walltime.py` reads the run's ledger sidecars and claim objects. **Timing semantics,
learned the hard way:** a ledger ROW's `ts` is the pass's injected clock (`WorkerCtx::now`) —
identical for every row in a pass, never a cell completion time. The chunk SIDECAR's object
mtime *is* the completion instant. A CLAIM object's body ts is the last progress renewal
(invariant 1 overwrites it on every completion), not the claim instant.

| metric | value |
|---|---|
| wall clock | 12.09 h (2026-08-30 05:27:31Z → 17:33:05Z) |
| chunk sidecars / cell-rows | 4,553 / 204,691 |
| fleet box-time idle (10-min buckets) | **18.17 of 37.00 box-h = 49.1 %** |
| — 0 boxes active (whole fleet down between operator generations) | 12.50 box-h |
| — 1 box active (2 idle) | 5.33 box-h |
| — 2 boxes active (1 idle) | 0.33 box-h |
| **scheduler-addressable idle** | **5.67 box-h** |
| per-worker gaps ≥ 10 min | r3500 10 (6.30 h) · r5900xt 4 (5.91 h) · r7900x 5 (6.27 h) |
| worst single gap | 3.13 h (r5900xt 14:25:04Z → 17:32:57Z) |

The largest single loss is **not** the scheduler: from 15:29Z to 17:32Z (2.06 h) the run had
drained on all three boxes and nothing ran until the operator's next manual requeue. The
generation boundaries (09:42→10:21, 10:54→11:52, 12:11→13:20) are the rebuild → `requeue` →
`compact` → relaunch cycle plus a 10-minute poll interval, ×5 rounds.

## 2. The duplicate-work defect

`BoxBudget::pack_chunks_lpt` closed a chunk at `Σcost / self.max_concurrent(mem, threads)`,
and `max_concurrent` is a property of **the box doing the packing**. Its own doc claimed
"every worker forms identical chunks and the per-chunk claim remains exclusive" — true of the
LPT *order*, false of the *boundaries*.

Measured chunk sizes on the three boxes: **19 / 41 / 82 cells**. No two workers ever computed
the same `chunk_id`, so `claim_or_steal_r2_key` always succeeded and boxes re-encoded each
other's in-flight cells.

| | executions | distinct cells | redundancy |
|---|---|---|---|
| whole wave | 184,302 | 126,000 | **1.463×** |
| — concurrent duplicates (2nd box ≤ 900 s apart, different worker) | 36,068 over 24,695 cells | | **19.6 % of all work** |
| — cross-generation retries after a pardon (legitimate) | 22,234 | | |
| 312-cell remainder round (18:14Z) | 917 | 312 | **2.939×** ¹ |

¹ lower bound: chunk sidecar names (`pass-<worker>-<pass>.chunk-<cid>`) recur across
generations, so some earlier rows had already been overwritten when this was counted. The
uploaded snapshot is unaffected (`distinct_done` stable at 125,941) — this is exactly why
`compact --upload` is the durable done-set — but sidecar history is not a durable archive.

**It gets worse as the gap shrinks**: a remainder that fits inside one box's chunk is packed
whole by all three. 2.939× on 312 cells vs 1.463× on 126,000.

## 3. Predicted (simulated before writing the code)

```
whole wave — 176,332 cells, 22.54 core-h, ideal 7.51 h on 3 boxes
  per_box_chunks (shipped)  redundancy 1.463  makespan 10.99 h
  uniform_chunks            redundancy 1.000  makespan  7.51 h   −3.48 h (31.6 %)
  uniform_spread (ships)    redundancy 1.000  makespan  7.51 h   −3.48 h (31.6 %)

312-cell remainder — 5.78 core-h, ideal 1.93 h
  per_box_chunks (shipped)                 makespan 2.921 h  (1.51× ideal)
  uniform_chunks                           makespan 1.997 h  (1.04× ideal)
  uniform_spread (cap = gap/64, floor 4)   makespan 1.934 h  (1.00× ideal)
```

## 4. Measured A/B on the fleet (same three boxes, back to back)

| | BEFORE — per-box packing (`exec-zensim944hdr-7e4695b3`) | AFTER — fleet-uniform (`…-d7982e9b`) |
|---|---|---|
| gap | 312 cells | 59 cells |
| executions / distinct | 917 / 312 = **2.939×** | 59 / 59 = **1.000×** |
| work split per box | 312 / 312 / 293 — *every box did the whole gap* | 24 / 8 / 27 — disjoint, sums to exactly 59 |
| chunk sizes | 19 / 41 / 82 (differs per box) | **4 ×14 + 3 ×1, identical on every box** (`max(59/64, 4)` cap ⇒ 15 chunks) |
| poison-row redundancy | 483 / 312 = 1.55× | 71 / 59 = 1.20× |
| contended pass behaviour | box drains out (`done=0` counted as idle) | `skipped=8` → keeps polling (r7900x reached pass 38) |

Exactly-once held: 59 executions for 59 cells, no cell executed twice, all three boxes
contributed. The prediction was **conservative** — the real per-box redundancy on a small gap
(2.939×) was worse than the whole-wave average the simulation assumed (1.463×).

### The drain rule, caught live on an unfixed image

While scoring, the GPU box (`exec-gpu-cuda13-6d4f9963`, pre-fix entrypoint) was relaunched
over 23 cells whose claims were still held by a killed worker's 10-minute lease. It ran **8
passes in 1 second**, every one `done=0 … skipped=23`, counted them all as idle and exited —
twice. Overlaying the fixed entrypoint (`exec-gpu-cuda13-d7982e9b`) fixed it; the run then
completed 69/69.

## 5. What this does NOT fix

12.50 of the 18.17 idle box-hours were the whole fleet down between operator generations. No
claim scheduler recovers that; it needs auto pardon-and-relaunch on drain and a completion
beacon instead of a poll. Tracked separately.
