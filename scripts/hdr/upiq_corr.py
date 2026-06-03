#!/usr/bin/env python3
"""Correlate UPIQ-HDR metric scores (per feeding) against subjective JOD.
Reports |SRCC| + |PLCC| so feedings are directly comparable. Reference bars:
PU_SSIM 0.740, HDR-VDP-2.2 0.812 SRCC (computed earlier from the dataset CSV)."""
import csv, math, glob, os, sys

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

# which column is the metric score in each output file
COL = {'ssim2':'ssim2','dssim':'dssim','cvvdp':'cvvdp_imazen_v0_0_1',
       'buttergpu':'butteraugli_pnorm3_gpu','butter':'butteraugli_pnorm3'}

outdir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/upiq_val'
rows = []
for fn in sorted(glob.glob(f'{outdir}/*.tsv')):
    name = os.path.basename(fn)[:-4]
    base = name.split('_')[0]
    col = COL.get(base)
    if not col: continue
    with open(fn) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        if col not in rdr.fieldnames:
            # fall back to last numeric col
            col = rdr.fieldnames[-1]
        ms, jod = [], []
        for r in rdr:
            try:
                m = float(r[col]); j = float(r['jod'])
                if math.isfinite(m): ms.append(m); jod.append(j)
            except: pass
    if len(ms) < 10:
        print(f"{name:<22} n={len(ms)} (too few)"); continue
    s, p = srocc(ms, jod), plcc(ms, jod)
    rows.append((name, len(ms), abs(s), abs(p), s))
print(f"{'config':<22} {'n':>4} {'|SRCC|':>7} {'|PLCC|':>7}   (vs JOD; ref: PU_SSIM 0.740, HDR-VDP-2.2 0.812)")
for name, n, s, p, sgn in sorted(rows, key=lambda r: -r[2]):
    print(f"{name:<22} {n:>4} {s:>7.4f} {p:>7.4f}")
