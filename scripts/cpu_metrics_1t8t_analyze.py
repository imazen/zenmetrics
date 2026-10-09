#!/usr/bin/env python3
"""Summarize a scripts/cpu_metrics_1t8t.sh run (new-metrics-wall, photo content).

Usage: cpu_metrics_1t8t_analyze.py <out_dir> [big_size]   (big_size default 3355x2516)

Inputs in <out_dir>:
  st_photo_1024.tsv                   zenbench, 1 thread (taskset 1 core, RAYON=1)
  mt8_photo_1024.tsv                  zenbench, 8 threads (taskset 0-7, RAYON=8)
  mt8_photo_<big>.tsv                 zenbench, 8 threads
  logs/heap_st_<big>_<metric>.log     1 thread, serial cold+warm call, /usr/bin/time -v

Prints a TSV table sorted by 1-thread 1 MP cost. The 1t big-size column is the
warm (second) call.
"""
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
BIG = sys.argv[2] if len(sys.argv) > 2 else "3355x2516"
PIX_1M = 1024 * 1024
PIX_BIG = int(BIG.split("x")[0]) * int(BIG.split("x")[1])
BIG_MP = f"{PIX_BIG / 1e6:.1f}MP"


def load(name):
    p = root / f"{name}.tsv"
    rows = {}
    if not p.exists():
        return rows
    with p.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["mode"] == "error":
                rows[r["metric"]] = ("error", r["score"])
            else:
                rows[r["metric"]] = (float(r["mean_ms"]), int(r["n_rounds"]))
    return rows


st1k = load("st_photo_1024")
mt1k = load("mt8_photo_1024")
mtbig = load(f"mt8_photo_{BIG}")

heap = {}
for p in (root / "logs").glob(f"heap_st_{BIG}_*.log"):
    m = p.stem.removeprefix(f"heap_st_{BIG}_")
    txt = p.read_text()
    reps = re.findall(r"rep(\d): ([0-9.]+) ms", txt)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    err = "ERROR" in txt
    heap[m] = {
        "cold": float(reps[0][1]) if len(reps) > 0 else None,
        "warm": float(reps[1][1]) if len(reps) > 1 else None,
        "rss_mb": int(rss.group(1)) / 1024 if rss else None,
        "err": err,
    }


def val(d, m):
    v = d.get(m)
    return v[0] if v and v[0] != "error" else None


def f(x, nd=1):
    if x is None:
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.{nd}f}"


metrics = list(dict.fromkeys(list(st1k) + list(mt1k) + list(mtbig) + list(heap)))
table = []
for m in metrics:
    a, b = val(st1k, m), val(mt1k, m)
    h = heap.get(m, {})
    c = h.get("warm")
    dd = val(mtbig, m)
    table.append(
        (
            a if a is not None else float("inf"),
            [
                m,
                f(a),
                f(b),
                f(a / b, 2) if a and b else "-",
                f(c),
                f(dd),
                f(c / dd, 2) if c and dd else "-",
                f(c / (PIX_BIG / 1e6), 1) if c else "-",
                f((c / a) / (PIX_BIG / PIX_1M), 2) if c and a else "-",
                f(h.get("rss_mb"), 0),
                "ERR" if h.get("err") else "",
            ],
        )
    )
table.sort(key=lambda t: t[0])
print(
    "\t".join(
        [
            "metric",
            "1t 1MP ms",
            "8t 1MP ms",
            "8t speedup 1MP",
            f"1t {BIG_MP} ms",
            f"8t {BIG_MP} ms",
            f"8t speedup {BIG_MP}",
            f"1t ms/MP @{BIG_MP}",
            "1t scaling vs linear",
            f"1t {BIG_MP} maxRSS MiB",
            "",
        ]
    )
)
for _, row in table:
    print("\t".join(row))
print()
for name, d in [("st_photo_1024", st1k), ("mt8_photo_1024", mt1k), (f"mt8_photo_{BIG}", mtbig)]:
    rounds = [v[1] for v in d.values() if v[0] != "error"]
    errs = [k for k, v in d.items() if v[0] == "error"]
    print(f"{name}: rounds min={min(rounds) if rounds else '-'} errors={errs}")
