#!/usr/bin/env python3
"""Report for the hdrvdp UPIQ run: SROCC/PLCC of our res.Q vs JOD truth and
vs UPIQ's own per-condition HDRVDP2_2 column (per-image parity on real
content). Reads the TSV emitted by `crates/hdrvdp/examples/upiq_score.rs`.

usage: upiq_hdrvdp_report.py <scores.tsv>"""
import csv, math, sys

def plcc(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs)); dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx*dy else float('nan')

def srocc(xs, ys):
    n = len(xs)
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i]); rk = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2.0+1
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    return plcc(ranks(xs), ranks(ys))

rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t') if r['hdrvdp'] not in ('', 'nan')]
our  = [float(r['hdrvdp']) for r in rows]
off  = [float(r['official_hdrvdp2_2']) for r in rows]
jod  = [float(r['jod']) for r in rows]
dl   = [o-f for o, f in zip(our, off)]
n = len(rows)
sd = math.sqrt(sum((d-sum(dl)/n)**2 for d in dl)/n)

print(f"n = {n}")
print(f"ours vs JOD        : SROCC {srocc(our, jod):+.4f}  PLCC {plcc(our, jod):+.4f}")
print(f"official vs JOD    : SROCC {srocc(off, jod):+.4f}  PLCC {plcc(off, jod):+.4f}  (published 0.812)")
print(f"ours vs official   : SROCC {srocc(our, off):+.4f}  PLCC {plcc(our, off):+.4f}")
print(f"per-image delta    : mean {sum(dl)/n:+.3f}  sd {sd:.3f}  min {min(dl):+.3f}  max {max(dl):+.3f}")
worst = sorted(range(n), key=lambda i: -abs(dl[i]))[:8]
for i in worst:
    print(f"  {rows[i]['condition_id']:20} off {off[i]:8.3f}  ours {our[i]:8.3f}  d {dl[i]:+7.3f}  jod {jod[i]:+.3f}")
