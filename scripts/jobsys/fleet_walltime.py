#!/usr/bin/env python3
"""fleet_walltime.py — reconstruct a run's WALL-TIME profile from its live ledger.

Answers, from the ledger + claim objects alone (no instrumentation, no re-run):

  * per-worker busy/idle timeline (cells finished per bucket, and the gaps),
  * chunk size + chunk duration distribution,
  * the STRAGGLER TAIL: which cells/boxes held the last X% of wall time, and how
    much box-time the others burned idle underneath it,
  * the cost model (seconds per cell as a function of pixels x speed) that a
    schedule simulation needs.

WHY THIS EXISTS (2026-08-30): the aom-rs encode wave took 12.09 h of wall clock on
three boxes and nobody could say where the time went, because the only signal the
fleet emits is `done=N` per pass. The ledger already carries it: every chunk sidecar's
mtime is that chunk's completion instant, its rows name the cells, and the claim
object's body carries `<last-renewal-ts> <worker> <done>/<total>`. This turns those
into a schedule.

TIMING SEMANTICS — read before trusting a number:
  * A ledger ROW's `ts` is the PASS's injected clock (`WorkerCtx::now`), NOT the cell's
    completion instant. Every row in a pass shares it. Do not use it for durations.
  * A chunk SIDECAR's object mtime IS the chunk's completion instant (the worker flushes
    the sidecar as the chunk finishes). This is the load-bearing timestamp here.
  * A CLAIM object's body ts is the LAST progress renewal (~chunk end), not the claim
    instant — the claim is overwritten on every completion (invariant 1). So
    claim->done duration is NOT directly available; chunk duration is estimated from
    the per-worker completion process and the worker's concurrency.

usage:
  fleet_walltime.py --ledger-dir <dir of sidecar parquets> --ledger-ls <ls tsv>
                    [--claims-dir <dir>] [--manifest <manifest.json>]
                    [--tail-pct 10] [--bucket-secs 600] [--out <report.json>]

`--ledger-ls` is `aws s3 ls` output for the ledger prefix (LOCAL time), one line per
object: `YYYY-MM-DD HH:MM:SS  <size>  <name>` — optionally with a leading run column.
"""
import argparse, collections, datetime, glob, json, os, re, sys

CHUNK_RE = re.compile(r"^pass-(.+?)-(\d+)\.chunk-([0-9a-f]+|poison)\.parquet$")
DIM_RE = re.compile(r"\.scale(\d+)x(\d+)\.")


def parse_ls(path, tz_offset_hours):
    """`aws s3 ls` lines -> {name: epoch_secs}. Tolerates a leading run column."""
    tz = datetime.timezone(datetime.timedelta(hours=tz_offset_hours))
    out = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:            # run \t 'date time' \t size \t name
            _, ts, _size, name = parts[:4]
        else:                          # raw `aws s3 ls`
            f = line.split()
            if len(f) < 4:
                continue
            ts, name = f[0] + " " + f[1], f[3]
        try:
            t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
        except ValueError:
            continue
        out[name] = int(t.timestamp())
    return out


def load_chunks(ledger_dir, mtimes):
    """One record per chunk sidecar: completion time, worker, cells, statuses, pixels."""
    import pyarrow.parquet as pq
    recs, unreadable = [], 0
    for f in sorted(glob.glob(os.path.join(ledger_dir, "*.parquet"))):
        base = os.path.basename(f)
        m = CHUNK_RE.match(base)
        if not m:
            continue                     # pardon sidecars, pass-level rollups
        try:
            t = pq.read_table(f)
        except Exception:
            unreadable += 1
            continue
        d = t.to_pydict()
        px = 0
        for p in d.get("image_path", []):
            dm = DIM_RE.search(p or "")
            if dm:
                px += int(dm.group(1)) * int(dm.group(2))
        recs.append({
            "file": base,
            "worker": m.group(1),
            "pass": int(m.group(2)),
            "chunk": m.group(3),
            "done_ts": mtimes.get(base),
            "cells": t.num_rows,
            "pixels": px,
            "status": collections.Counter(d.get("status", [])),
        })
    return recs, unreadable


def timeline(recs, bucket):
    """Per-worker cells finished per time bucket + the idle gaps between flushes."""
    by = collections.defaultdict(list)
    for r in recs:
        if r["done_ts"] is not None:
            by[r["worker"]].append(r)
    series, gaps = {}, {}
    for w, v in by.items():
        v.sort(key=lambda r: r["done_ts"])
        b = collections.Counter()
        for r in v:
            b[r["done_ts"] // bucket * bucket] += r["cells"]
        series[w] = dict(sorted(b.items()))
        gaps[w] = sorted(
            ((v[i + 1]["done_ts"] - v[i]["done_ts"]), v[i]["done_ts"], v[i + 1]["file"])
            for i in range(len(v) - 1)
        )
    return series, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-dir", required=True)
    ap.add_argument("--ledger-ls", required=True)
    ap.add_argument("--tz-offset-hours", type=float, default=-6.0,
                    help="tz of the `aws s3 ls` timestamps (default MDT)")
    ap.add_argument("--tail-pct", type=float, default=10.0)
    ap.add_argument("--bucket-secs", type=int, default=600)
    ap.add_argument("--out")
    a = ap.parse_args()

    mtimes = parse_ls(a.ledger_ls, a.tz_offset_hours)
    recs, unreadable = load_chunks(a.ledger_dir, mtimes)
    if not recs:
        sys.exit("no chunk sidecars matched — wrong --ledger-dir?")
    ts = [r["done_ts"] for r in recs if r["done_ts"] is not None]
    t0, t1 = min(ts), max(ts)
    span = t1 - t0
    workers = sorted({r["worker"] for r in recs})
    print(f"run window {datetime.datetime.fromtimestamp(t0, datetime.timezone.utc):%Y-%m-%d %H:%M:%SZ}"
          f" -> {datetime.datetime.fromtimestamp(t1, datetime.timezone.utc):%H:%M:%SZ}"
          f"  = {span/3600:.2f} h wall,  {len(workers)} workers,"
          f"  {len(recs)} chunks, {sum(r['cells'] for r in recs)} cell-rows"
          f"{f', {unreadable} unreadable' if unreadable else ''}")

    series, gaps = timeline(recs, a.bucket_secs)
    print("\n== per-worker ==")
    for w in workers:
        v = sorted((r for r in recs if r["worker"] == w), key=lambda r: r["done_ts"])
        busy_first, busy_last = v[0]["done_ts"], v[-1]["done_ts"]
        g = [x[0] for x in gaps.get(w, [])]
        big = [x for x in gaps.get(w, []) if x[0] >= 600]
        print(f"  {w:<16} chunks={len(v):<5} cells={sum(r['cells'] for r in v):<7} "
              f"active={busy_first-t0:>6}s..{busy_last-t0:>6}s  "
              f"gap p50={sorted(g)[len(g)//2] if g else 0:>4}s p99={sorted(g)[int(len(g)*.99)] if g else 0:>5}s "
              f"max={max(g) if g else 0:>6}s  gaps>=10min: {len(big)} ({sum(x[0] for x in big)/3600:.2f} h)")

    # STRAGGLER TAIL — the last tail-pct of wall time.
    cut = t1 - span * a.tail_pct / 100.0
    tail = [r for r in recs if r["done_ts"] is not None and r["done_ts"] >= cut]
    tail_by_w = collections.Counter(r["worker"] for r in tail)
    tail_cells = sum(r["cells"] for r in tail)
    total_cells = sum(r["cells"] for r in recs)
    print(f"\n== straggler tail (last {a.tail_pct:.0f}% of wall = {span*a.tail_pct/100/3600:.2f} h,"
          f" from {datetime.datetime.fromtimestamp(cut, datetime.timezone.utc):%H:%M:%SZ}) ==")
    print(f"  cells finished in the tail: {tail_cells} / {total_cells} = {100.0*tail_cells/total_cells:.2f}%")
    for w in workers:
        v = [r for r in recs if r["worker"] == w and r["done_ts"] is not None]
        last = max(r["done_ts"] for r in v)
        in_tail = [r for r in v if r["done_ts"] >= cut]
        idle = (t1 - last) if not in_tail else 0
        print(f"  {w:<16} chunks_in_tail={tail_by_w.get(w,0):<5} cells={sum(r['cells'] for r in in_tail):<6}"
              f" last_flush=-{t1-last:>6}s from end")
    # Box-time burned idle underneath the tail: sum over workers of (tail window minus
    # the part of it in which that worker flushed anything).
    idle_box_secs = 0
    for w in workers:
        v = sorted((r["done_ts"] for r in recs if r["worker"] == w and r["done_ts"] is not None))
        last = v[-1]
        idle_box_secs += max(0, t1 - max(last, cut)) if last < t1 else 0
    print(f"  box-seconds idle after each box's own last flush, inside the tail: {idle_box_secs}s"
          f" = {idle_box_secs/3600:.2f} box-h")

    print("\n== chunk shape ==")
    cs = sorted(r["cells"] for r in recs)
    ps = sorted(r["pixels"] for r in recs)
    def pct(x, p): return x[min(len(x) - 1, int(len(x) * p))]
    print(f"  cells/chunk   p10={pct(cs,.1)} p50={pct(cs,.5)} p90={pct(cs,.9)} max={cs[-1]}")
    print(f"  MPix/chunk    p10={pct(ps,.1)/1e6:.1f} p50={pct(ps,.5)/1e6:.1f} p90={pct(ps,.9)/1e6:.1f} max={ps[-1]/1e6:.1f}")

    if a.out:
        json.dump({"t0": t0, "t1": t1, "span": span, "workers": workers,
                   "series": {w: series[w] for w in series},
                   "chunks": [{k: (dict(v) if isinstance(v, collections.Counter) else v)
                               for k, v in r.items()} for r in recs]},
                  open(a.out, "w"))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
