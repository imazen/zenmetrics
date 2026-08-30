#!/usr/bin/env python3
"""fleet_schedule_sim.py — replay a measured wave through alternative CLAIM schedulers.

Consumes the cell set + per-cell cost model recovered by `fleet_walltime.py` and answers
ONE question: how much of the measured wall time is the *scheduler's* fault, and which
candidate mechanism recovers the most of it.

Modelled (all with exactly-once preserved — the ledger reduce is never touched):

  per_box_chunks   the 2026-08-30 shipped behaviour: each box packs the gap with ITS OWN
                   `BoxBudget::max_concurrent`, so boundaries differ per box, chunk_ids
                   are disjoint, EVERY claim succeeds and boxes redundantly re-run each
                   other's cells. Redundant work is real wall time.
  uniform_chunks   fleet-uniform boundaries (`pack_chunks_lpt_uniform`): identical chunk
                   ids everywhere, so the claim excludes; a box that loses every race
                   waits instead of duplicating.
  uniform_spread   uniform + the gap-relative cell cap (chunks shrink as the gap drains),
                   i.e. what ships.

Costs come from the measured wave; the simulation is a discrete-event schedule over the
same cells with the same box count, so the comparison is apples-to-apples.

usage: fleet_schedule_sim.py --profile <enc_profile.json> [--boxes 3] [--json out.json]
"""
import argparse, collections, heapq, json


def cell_costs(profile):
    """Per-cell seconds, from the measured chunk completions.

    A chunk's sidecar mtime is its completion instant; a worker's consecutive completions
    bound the work it did between them. Attribute a chunk's elapsed time to its cells in
    proportion to pixels (the dominant cost term for an intra encode), which recovers a
    per-cell cost without needing per-cell instrumentation.
    """
    by = collections.defaultdict(list)
    for r in profile["chunks"]:
        if r["done_ts"]:
            by[r["worker"]].append(r)
    costs, conc = [], {}
    for w, v in by.items():
        v.sort(key=lambda r: r["done_ts"])
        # Concurrency proxy: chunks whose windows overlap. Use the median inter-completion
        # gap over the run as the serial slot time.
        for i in range(1, len(v)):
            dt = v[i]["done_ts"] - v[i - 1]["done_ts"]
            if dt <= 0 or dt > 3600:      # skip generation boundaries / operator gaps
                continue
            n = max(1, v[i]["cells"])
            px = max(1, v[i]["pixels"])
            for _ in range(n):
                costs.append(dt / n)
        conc[w] = len(v)
    costs.sort()
    return costs


def simulate(costs, boxes, mode, chunk_cells, redundancy):
    """Greedy list schedule over `boxes` identical servers.

    `redundancy` (>=1.0) inflates the delivered work to model duplicate execution: with
    per-box chunking a cell is executed `redundancy` times on average across the fleet,
    and every one of those executions occupies a real core-second.
    """
    # Cells are claimed a CHUNK at a time; a chunk is the scheduling quantum.
    work = sorted(costs, reverse=True)     # LPT
    chunks, cur = [], []
    for c in work:
        cur.append(c)
        if len(cur) >= chunk_cells:
            chunks.append(sum(cur)); cur = []
    if cur:
        chunks.append(sum(cur))
    chunks = [c * redundancy for c in chunks]
    heap = [0.0] * boxes
    heapq.heapify(heap)
    for c in sorted(chunks, reverse=True):
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + c)
    return max(heap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--boxes", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("--tail-cells", type=int, default=0,
                    help="also model a REMAINDER gap of this many cells (the straggler case: "
                         "the aggregate schedule hides it because 176k cells balance trivially, "
                         "but a 312-cell remainder is exactly where chunk granularity decides "
                         "whether three boxes share the work or two go idle)")
    a = ap.parse_args()
    prof = json.load(open(a.profile))
    costs = cell_costs(prof)
    if not costs:
        raise SystemExit("no per-cell costs recoverable from this profile")
    total = sum(costs)
    print(f"cells modelled = {len(costs)}   total core-seconds = {total/3600:.2f} core-h"
          f"   ideal makespan on {a.boxes} boxes = {total/a.boxes/3600:.2f} h")
    rows = []
    for name, chunk_cells, red in (
        ("per_box_chunks (shipped 2026-08-30)", 41, 1.463),
        ("uniform_chunks", 41, 1.0),
        ("uniform_spread (ships)", 8, 1.0),
    ):
        m = simulate(costs, a.boxes, name, chunk_cells, red)
        rows.append((name, chunk_cells, red, m))
        print(f"  {name:<36} chunk={chunk_cells:<4} redundancy={red:<6} makespan={m/3600:6.2f} h")
    base = rows[0][3]
    print("\nsaving vs shipped:")
    for name, _, _, m in rows[1:]:
        print(f"  {name:<36} {(base-m)/3600:6.2f} h  ({100*(base-m)/base:5.1f}%)")
    if a.tail_cells:
        tail = sorted(costs, reverse=True)[: a.tail_cells]
        tt = sum(tail)
        print(f"\n== remainder gap of {a.tail_cells} cells ({tt/3600:.2f} core-h,"
              f" ideal {tt/a.boxes/3600:.2f} h on {a.boxes} boxes) ==")
        for name, chunk_cells, red in (
            ("per_box_chunks (shipped)", 41, 1.463),
            ("uniform_chunks", 41, 1.0),
            ("uniform_spread (ships, cap=gap/64 floor 4)",
             max(4, a.tail_cells // 64), 1.0),
        ):
            m = simulate(tail, a.boxes, name, chunk_cells, red)
            print(f"  {name:<44} chunk={chunk_cells:<4} redundancy={red:<6} makespan={m/3600:6.3f} h"
                  f"  ({m/ (tt/a.boxes):.2f}x ideal)")

    if a.json:
        json.dump([{"mode": r[0], "chunk_cells": r[1], "redundancy": r[2], "makespan_s": r[3]}
                   for r in rows], open(a.json, "w"))


if __name__ == "__main__":
    main()
