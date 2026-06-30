#!/usr/bin/env python3
"""Statistically correct the AVIF speed-4 understatement using the s4/s6/s8 data we ALREADY have.
At matched quality, AVIF bytes fall as speed drops; the per-step factor (measured from s8->s6->s4)
extrapolates to the RD-optimal s2/s0. Then re-compare AVIF-vs-WebP with the corrected AVIF — no
re-encoding. RD measure = bytes_to_reach(target): cheapest encode achieving >= target zq."""
import pandas as pd, json, collections, statistics as st
import numpy as np
BASE = '/mnt/v/output/canonical-picker-2026-06-27'

def load(d):
    df = pd.read_parquet(f'{BASE}/{d}/train.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes', 'knob_tuple_json']).dropna()
    return df

av = load('zenavif_lossy'); wb = load('zenwebp_lossy')
def speed(j):
    c = json.loads(j).get('cell', '')
    tok = c.split('-')[0]
    return int(tok[1:]) if tok[:1] == 's' and tok[1:].isdigit() else None
av['speed'] = av.knob_tuple_json.apply(speed)
print('avif speeds present:', sorted(av.speed.dropna().unique()))

# points[variant][speed] = [(zq,bytes)]
pts = collections.defaultdict(lambda: collections.defaultdict(list))
for v, z, b, s in zip(av.variant_name, av.score_zensim, av.encoded_bytes, av.speed):
    if s is not None:
        pts[v][int(s)].append((float(z), float(b)))
wbp = collections.defaultdict(list)
for v, z, b in zip(wb.variant_name, wb.score_zensim, wb.encoded_bytes):
    wbp[v].append((float(z), float(b)))

def reach(points, t):  # cheapest encode reaching >= target zq
    c = [b for z, b in points if z >= t]
    return min(c) if c else None

TARGETS = [60, 70, 75, 80, 85]
# Step 1: measure the per-step speed factor (bytes(slower)/bytes(faster)) at matched quality
step = collections.defaultdict(list)  # (fast,slow) -> ratios bytes_fast/bytes_slow (<1 => slower is better)
for v in pts:
    for t in TARGETS:
        bs = {s: reach(pts[v][s], t) for s in (4, 6, 8) if pts[v].get(s)}
        if 4 in bs and 6 in bs and bs[6]: step[(4, 6)].append(bs[4] / bs[6])
        if 6 in bs and 8 in bs and bs[8]: step[(6, 8)].append(bs[6] / bs[8])
f46 = st.median(step[(4, 6)]); f68 = st.median(step[(6, 8)])
print(f'per-2-speed-step byte factor (median): s4/s6={f46:.3f}  s6/s8={f68:.3f}  (consistent => extrapolable)')
fac = st.median(step[(4, 6)] + step[(6, 8)])  # pooled per-2-step factor
print(f'pooled per-step factor f={fac:.3f}; extrapolate s4 -> s2 (x f), s2 -> s0 (x f^2)')

# Step 2: AVIF-vs-WebP, raw (s4 = best swept) vs speed-corrected (s2, s0)
print('\nAVIF/WebP byte ratio (>1 = AVIF worse). raw=s4, corr2=s4*f (~speed2), corr0=s4*f^2 (~speed0):')
for t in TARGETS:
    raw, c2, c0 = [], [], []
    for v in pts:
        a4 = reach(pts[v].get(4, []), t); w = reach(wbp.get(v, []), t)
        if a4 and w:
            raw.append(a4 / w); c2.append(a4 * fac / w); c0.append(a4 * fac * fac / w)
    if raw:
        wr = lambda r: sum(1 for x in r if x < 1) / len(r)
        print(f'  zq{t}: raw med={st.median(raw):.2f} (avif-wins {wr(raw):.0%}) | corr2 med={st.median(c2):.2f} ({wr(c2):.0%}) | corr0 med={st.median(c0):.2f} ({wr(c0):.0%})  n={len(raw)}')
