#!/usr/bin/env python3
"""Summarise pool_determinism TSVs.

For each (backend, pair, size, mode): distinct JOD bit patterns within one process (max over
processes) and across all processes, the same for Full mode's band-score hash, and where the
new kernel's JOD sits in the old kernel's run-to-run band.

usage: python3 benchmarks/cvvdp_pool_determinism_2026-09-26.analyze.py <dir>   (reads <dir>/{old,new}_<backend>_p<k>.tsv)
"""
import glob, os, re, statistics, sys
from collections import defaultdict

d = sys.argv[1]
cells = defaultdict(lambda: defaultdict(list))  # key -> kernel -> [(proc, jod, bits, qhash)]
for path in sorted(glob.glob(os.path.join(d, "*_p*.tsv"))):
    m = re.match(r"(old|new)_([a-z0-9]+)_p(\d+)\.tsv$", os.path.basename(path))
    if not m:
        continue
    kernel, backend, proc = m.group(1), m.group(2), int(m.group(3))
    for line in open(path):
        pair, size, mode, rep, jod, bits, qh = line.rstrip("\n").split("\t")
        cells[(backend, pair, size, mode)][kernel].append((proc, float(jod), bits, qh))


def stats(v):
    procs = sorted({p for p, *_ in v})
    per_proc = max(len({b for p, _, b, _ in v if p == q}) for q in procs)
    allb = len({b for _, _, b, _ in v})
    qs = [q for *_, q in v if q != "-"]
    if qs:
        qper = max(len({q for p, _, _, q in v if p == pp}) for pp in procs)
        qall = len(set(qs))
        qtxt = f"q_in_proc={qper} q_all={qall}"
    else:
        qtxt = "q=-"
    return f"procs={len(procs)} reps={len(v)} jod_in_proc={per_proc} jod_all={allb} {qtxt}"


for key in sorted(cells):
    row = cells[key]
    line = [" ".join(key)]
    for kernel in ("old", "new"):
        if row.get(kernel):
            line.append(f"{kernel}[{stats(row[kernel])}]")
    if row.get("old") and row.get("new"):
        o = [j for _, j, _, _ in row["old"]]
        nv = row["new"][0][1]
        omin, omax, omed = min(o), max(o), statistics.median(o)
        line.append(
            f"old_band=[{omin:.9f},{omax:.9f}] width={omax - omin:.3e} median={omed:.9f} "
            f"new={nv:.9f} new-median={nv - omed:+.3e} inside={omin <= nv <= omax}"
        )
    print(" | ".join(line))
