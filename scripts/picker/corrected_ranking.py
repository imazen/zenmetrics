#!/usr/bin/env python3
"""Principled, confound-corrected lossy codec ranking from the EXISTING data (no re-sweep):
  - RD measure = bytes_to_reach(target): cheapest encode achieving >= target zq (clean, no interp).
  - AVIF uses its BEST swept speed (incl s2, near RD-optimal; speed effect measured ~3%/step).
  - PAIRED pairwise: compare two codecs only on images where BOTH reach the target (coverage-robust,
    no missing-not-at-random oracle bias).
  - STRATIFIED by image size + target quality, then REWEIGHTED uniformly over strata (kills the
    small-image corpus skew that flattered WebP).
Outputs: pairwise win matrix, win-rate by (size x quality), and the reweighted overall order."""
import pandas as pd, collections, statistics as st, re
import numpy as np
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
FAMS = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
NAMES = [f for f, _ in FAMS]

def curves(d):
    df = pd.read_parquet(f'{BASE}/{d}/train.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
    c = collections.defaultdict(list)
    for v, z, b in zip(df.variant_name, df.score_zensim, df.encoded_bytes):
        c[v].append((float(z), float(b)))
    return c
C = {f: curves(d) for f, d in FAMS}

def reach(points, t):
    cand = [b for z, b in points if z >= t]
    return min(cand) if cand else None

def px(v):  # parse pixels from o_<id>.png.scale<W>x<H>
    m = re.search(r'scale(\d+)x(\d+)', v)
    return int(m.group(1)) * int(m.group(2)) if m else None

TARGETS = [50, 60, 70, 80, 88]
SIZE_BINS = [('tiny <64k', 0, 64_000), ('small 64-260k', 64_000, 260_000), ('med 260k-1M', 260_000, 1_000_000), ('large >1M', 1_000_000, 9e12)]
allv = set(C['jxl'])
cells = []  # (size_label, target, {fam: bytes})
for v in allv:
    p = px(v)
    if not p: continue
    sl = next(s for s, lo, hi in SIZE_BINS if lo <= p < hi)
    for t in TARGETS:
        bb = {f: reach(C[f].get(v, []), t) for f in NAMES}
        bb = {f: b for f, b in bb.items() if b is not None}
        if bb: cells.append((sl, t, bb))

# Paired pairwise win-rate (only cells where both reach), reweighted uniformly over (size,target) strata
strata = sorted(set((s, t) for s, t, _ in cells))
def pair_winrate(a, b):  # P(a cheaper than b), averaged over strata (equal weight)
    per = []
    for s, t in strata:
        w = [(bb[a], bb[b]) for sl, tt, bb in cells if sl == s and tt == t and a in bb and b in bb]
        if w: per.append(sum(1 for x, y in w if x < y) / len(w))
    return st.mean(per) if per else float('nan')
print('PAIRED pairwise win-rate P(row cheaper than col), reweighted over size x quality strata:')
print('        ' + ''.join(f'{c:>7}' for c in NAMES))
score = {f: 0.0 for f in NAMES}
for a in NAMES:
    row = []
    for b in NAMES:
        if a == b: row.append('  -  '); continue
        wr = pair_winrate(a, b); row.append(f'{wr:6.0%}'); score[a] += wr
    print(f'  {a:>5} ' + ''.join(f'{x:>7}' for x in row))
print('\nreweighted ranking (sum of pairwise win-rates, higher=better):')
for f in sorted(NAMES, key=lambda f: -score[f]):
    print(f'  {f:>5}: {score[f]:.2f}')

# oracle win-rate by (size x quality) — the content-adaptive signal
print('\noracle win-rate (argmin bytes) by size x quality — where each codec actually wins:')
for s, lo, hi in SIZE_BINS:
    line = f'  {s:>14}: '
    for t in TARGETS:
        sub = [bb for sl, tt, bb in cells if sl == s and tt == t and len(bb) >= 2]
        if not sub:
            line += f'zq{t}:--  '; continue
        win = collections.Counter(min(bb, key=bb.get) for bb in sub)
        top = win.most_common(1)[0]
        line += f'zq{t}:{top[0][:4]}{100*top[1]//len(sub):>2}% '
    print(line)
