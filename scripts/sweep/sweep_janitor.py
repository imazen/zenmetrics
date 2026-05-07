#!/usr/bin/env python3
"""sweep_janitor — periodic check-in + idle-worker reaper.

For each worker in `s3://coefficient/heartbeats/<RUN_ID>/stats/<worker>.tsv`,
compute lifetime cells/min and recent cpu_pct. If a worker is dragging the
fleet (lifetime cells/min < THRESH and run_time > GRACE), look up its
vast.ai instance id from `<instances_file>` and destroy it.

When the sweep is complete (TSV count >= TARGET), destroy the entire fleet.

Usage:
    python3 sweep_janitor.py <run_id> <instances_file> <target_tsv_count> [--once]

Default behaviour: prints status + reaps offenders, then exits.
Wrap in a Monitor (or cron / loop) for periodic check-ins.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --- Tunables --------------------------------------------------------------
THRESH_CELLS_MIN = 100       # below this lifetime cells/min => reap
GRACE_MIN = 8                # don't reap workers younger than this (in minutes)
IDLE_CPU_PCT = 5             # < this cpu% averaged over recent samples => reap
IDLE_RECENT_SAMPLES = 6      # how many recent samples to average for idle test
# ---------------------------------------------------------------------------

R2_ENDPOINT = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
R2_ENV = {
    "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
    "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
    "PATH": os.environ.get("PATH", ""),
}


def s3_ls(prefix: str) -> list[str]:
    out = subprocess.run(
        ["aws", "s3", "ls", prefix, "--endpoint-url", R2_ENDPOINT],
        env=R2_ENV, capture_output=True, text=True,
    )
    return [line.split()[-1] for line in out.stdout.splitlines() if line.strip()]


def s3_get(key: str, dest: Path) -> bool:
    r = subprocess.run(
        ["aws", "s3", "cp", key, str(dest), "--endpoint-url", R2_ENDPOINT],
        env=R2_ENV, capture_output=True, text=True,
    )
    return r.returncode == 0


def parse_stats(path: Path) -> dict | None:
    try:
        lines = path.read_text().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split("\t")
        rows = [dict(zip(header, l.split("\t"))) for l in lines[1:] if l]
        if not rows:
            return None
        last = rows[-1]
        recent = rows[-IDLE_RECENT_SAMPLES:] if len(rows) >= IDLE_RECENT_SAMPLES else rows
        rows_done = int(last.get("rows_done", 0) or 0)
        wall_min = float(last.get("wall_min", 0) or 0)
        cells_min = rows_done / wall_min if wall_min > 0 else 0
        cpu_recent = sum(float(r.get("cpu_pct", 0) or 0) for r in recent) / max(1, len(recent))
        return {
            "worker": path.stem,
            "rows_done": rows_done,
            "wall_min": wall_min,
            "cells_min": cells_min,
            "cpu_recent": cpu_recent,
        }
    except Exception:
        return None


def load_instance_map(path: Path) -> dict[str, str]:
    """Read `INSTANCE_ID OFFER_ID WORKER_ID` lines → {worker_id: instance_id}."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            instance_id, _offer_id, worker_id = parts[0], parts[1], parts[2]
            out[worker_id] = instance_id
    return out


def vastai_destroy(instance_id: str) -> bool:
    r = subprocess.run(
        ["vastai", "destroy", "instance", instance_id, "-y"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def count_tsvs(run_id: str) -> int:
    rows = s3_ls(f"s3://zentrain/{run_id}/zenjpeg/")
    return sum(1 for r in rows if r.endswith(".tsv"))


def reap_one(run_id: str, instances_file: Path, target: int) -> int:
    tsv_count = count_tsvs(run_id)
    print(f"[janitor] tsv={tsv_count}/{target}", file=sys.stderr)

    # Sweep complete? destroy everything.
    if tsv_count >= target:
        print(f"[janitor] sweep complete; destroying fleet", file=sys.stderr)
        wmap = load_instance_map(instances_file)
        for worker_id, instance_id in wmap.items():
            ok = vastai_destroy(instance_id)
            print(f"[janitor] destroy {instance_id} ({worker_id}): {'OK' if ok else 'fail'}", file=sys.stderr)
        return 0

    # Else: reap idle workers
    stats_prefix = f"s3://coefficient/heartbeats/{run_id}/stats/"
    listing = s3_ls(stats_prefix)
    listing = [k for k in listing if k.endswith(".tsv")]
    if not listing:
        print(f"[janitor] no stats yet under {stats_prefix}", file=sys.stderr)
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        def fetch(name):
            dest = tmp / name
            if s3_get(stats_prefix + name, dest):
                return parse_stats(dest)
            return None
        with ThreadPoolExecutor(max_workers=16) as ex:
            workers = [w for w in ex.map(fetch, listing) if w]

    wmap = load_instance_map(instances_file)
    reaped = 0
    for w in workers:
        if w["wall_min"] < GRACE_MIN:
            continue  # bootstrap grace
        is_slow = w["cells_min"] < THRESH_CELLS_MIN
        is_idle = w["cpu_recent"] < IDLE_CPU_PCT
        if not (is_slow or is_idle):
            continue
        instance_id = wmap.get(w["worker"])
        if not instance_id:
            print(f"[janitor] no instance map for {w['worker']}", file=sys.stderr)
            continue
        reason = []
        if is_slow: reason.append(f"slow({w['cells_min']:.0f} cells/min)")
        if is_idle: reason.append(f"idle({w['cpu_recent']:.1f}% cpu)")
        ok = vastai_destroy(instance_id)
        msg = "destroyed" if ok else "destroy-failed"
        print(f"[janitor] {msg} {instance_id} {w['worker']}: {' '.join(reason)} wall={w['wall_min']:.1f}min", file=sys.stderr)
        if ok:
            reaped += 1
    print(f"[janitor] reaped {reaped} workers", file=sys.stderr)
    return reaped


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    run_id = sys.argv[1]
    instances_file = Path(sys.argv[2])
    target = int(sys.argv[3])
    reap_one(run_id, instances_file, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
