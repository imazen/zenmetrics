#!/usr/bin/env python3
"""sweep_diag — aggregate per-worker stats from R2 to find waste patterns.

Pulls every `s3://coefficient/heartbeats/<RUN_ID>/stats/<worker>.tsv`,
computes per-worker:

  * cells/sec instantaneous (last window)
  * cells/sec lifetime
  * CPU% recent vs lifetime (idle workers vs saturated)
  * total rows emitted
  * wall clock since first heartbeat
  * inferred idle/working ratio (CPU<20% counts as idle)

Then prints a sorted table (slowest-throughput first) and aggregate
fleet stats. Run on demand or wire into a watch loop.

Usage:
    python3 sweep_diag.py [run_id]                  # human-readable
    python3 sweep_diag.py [run_id] --json           # machine-readable
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN_ID = sys.argv[1] if (len(sys.argv) > 1 and not sys.argv[1].startswith("--")) else "sweep-v15r-2026-05-06"
JSON_OUT = "--json" in sys.argv

R2_ENDPOINT = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
R2_ENV = {
    "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
    "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
    "PATH": os.environ.get("PATH", ""),
}

STATS_PREFIX = f"s3://coefficient/heartbeats/{RUN_ID}/stats/"


def s3_ls(prefix: str) -> list[tuple[str, int]]:
    out = subprocess.run(
        ["aws", "s3", "ls", prefix, "--endpoint-url", R2_ENDPOINT],
        env=R2_ENV, capture_output=True, text=True,
    )
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            rows.append((parts[-1], int(parts[2])))
    return rows


def s3_get(key: str, dest: Path) -> bool:
    r = subprocess.run(
        ["aws", "s3", "cp", key, str(dest), "--endpoint-url", R2_ENDPOINT],
        env=R2_ENV, capture_output=True, text=True,
    )
    return r.returncode == 0


def parse_worker_stats(path: Path) -> dict | None:
    try:
        with open(path) as f:
            lines = f.read().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split("\t")
        rows = [dict(zip(header, l.split("\t"))) for l in lines[1:] if l]
        if not rows:
            return None
        first = rows[0]; last = rows[-1]
        # Recent window — last 5 minutes of wall_min if available
        last_wm = float(last.get("wall_min", 0) or 0)
        recent = [r for r in rows if (last_wm - float(r.get("wall_min", 0) or 0)) <= 5.0]
        if len(recent) < 2 and len(rows) >= 2:
            recent = rows[-min(10, len(rows)):]
        rows_done_total = int(last.get("rows_done", 0) or 0)
        wall_min_total = float(last.get("wall_min", 0) or 0)
        # Recent-window throughput.
        if len(recent) >= 2:
            r0, r1 = recent[0], recent[-1]
            rd_recent = int(r1["rows_done"]) - int(r0["rows_done"])
            wm_recent = float(r1["wall_min"]) - float(r0["wall_min"])
            cells_min_recent = rd_recent / wm_recent if wm_recent > 0 else 0
            cpu_recent = sum(float(r.get("cpu_pct", 0) or 0) for r in recent) / len(recent)
        else:
            cells_min_recent = 0
            cpu_recent = float(last.get("cpu_pct", 0) or 0)
        cells_min_life = rows_done_total / wall_min_total if wall_min_total > 0 else 0
        cpu_life = sum(float(r.get("cpu_pct", 0) or 0) for r in rows) / len(rows)
        idle_fraction = sum(1 for r in rows if float(r.get("cpu_pct", 0) or 0) < 20) / len(rows)
        return {
            "worker": path.stem,
            "rows_done": rows_done_total,
            "wall_min": wall_min_total,
            "cells_min_recent": cells_min_recent,
            "cells_min_life": cells_min_life,
            "cpu_recent": cpu_recent,
            "cpu_life": cpu_life,
            "idle_fraction": idle_fraction,
            "n_samples": len(rows),
        }
    except Exception as e:
        return None


def main() -> int:
    listing = s3_ls(STATS_PREFIX)
    listing = [(k, sz) for (k, sz) in listing if k.endswith(".tsv")]
    if not listing:
        print(f"no stats files under {STATS_PREFIX}", file=sys.stderr)
        return 1
    print(f"[diag] {len(listing)} workers, RUN_ID={RUN_ID}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        def fetch(rec):
            name, _ = rec
            dest = tmp / name
            if s3_get(STATS_PREFIX + name, dest):
                return parse_worker_stats(dest)
            return None
        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(fetch, listing))
    workers = [r for r in results if r]

    if not workers:
        print("no parseable stats", file=sys.stderr)
        return 1

    # Aggregate.
    fleet = {
        "n_workers": len(workers),
        "rows_total": sum(w["rows_done"] for w in workers),
        "rows_per_min_recent": sum(w["cells_min_recent"] for w in workers),
        "rows_per_min_life": sum(w["cells_min_life"] for w in workers),
        "cpu_avg_recent": sum(w["cpu_recent"] for w in workers) / len(workers),
        "cpu_avg_life": sum(w["cpu_life"] for w in workers) / len(workers),
        "idle_workers_recent": sum(1 for w in workers if w["cpu_recent"] < 20),
    }
    # Estimate "wasted" core-hours: for each worker, idle_fraction × wall_min.
    waste_min = sum(w["idle_fraction"] * w["wall_min"] for w in workers)
    work_min = sum((1 - w["idle_fraction"]) * w["wall_min"] for w in workers)
    fleet["work_min_total"] = work_min
    fleet["waste_min_total"] = waste_min
    fleet["waste_pct"] = (waste_min / (waste_min + work_min) * 100) if (waste_min + work_min) > 0 else 0

    if JSON_OUT:
        out = {"fleet": fleet, "workers": workers}
        print(json.dumps(out, indent=2))
        return 0

    # Human table — sort by recent throughput ascending (slowest first).
    workers_sorted = sorted(workers, key=lambda w: w["cells_min_recent"])
    print()
    print(f"=== fleet aggregate ({fleet['n_workers']} workers) ===")
    print(f"  rows_total      : {fleet['rows_total']:,}")
    print(f"  rows/min recent : {fleet['rows_per_min_recent']:,.0f}")
    print(f"  rows/min life   : {fleet['rows_per_min_life']:,.0f}")
    print(f"  cpu_avg recent  : {fleet['cpu_avg_recent']:.1f}%")
    print(f"  cpu_avg life    : {fleet['cpu_avg_life']:.1f}%")
    print(f"  idle workers    : {fleet['idle_workers_recent']}/{fleet['n_workers']}  (cpu<20% in recent window)")
    print(f"  waste %         : {fleet['waste_pct']:.1f}% of fleet wall-time")
    print(f"  total wall-time : work={work_min/60:.1f}h waste={waste_min/60:.1f}h")
    print()
    print(f"{'worker':<40} {'rows':>8} {'wall_min':>8} {'cells/min recent':>18} {'cells/min life':>15} {'cpu rec':>8} {'cpu life':>8} {'idle%':>6}")
    print("-" * 130)
    for w in workers_sorted:
        print(f"{w['worker']:<40} {w['rows_done']:>8,} {w['wall_min']:>8.1f} {w['cells_min_recent']:>18,.0f} {w['cells_min_life']:>15,.0f} {w['cpu_recent']:>7.1f}% {w['cpu_life']:>7.1f}% {w['idle_fraction']*100:>5.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
