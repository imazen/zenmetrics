#!/usr/bin/env python3
"""Dump the per-router TEST sets (raw input vector + true label) so a Rust scorer can load
the real .bin and measure i8-vs-f32 family-acc on the actual ZNPR forward. One line per row:
  <label>\\t<v0>\\t<v1>...  (raw, unscaled — the .bin's scaler runs inside predict)."""
import numpy as np, pandas as pd, collections
SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
OUT = '/mnt/v/output/router-features-2026-06-30'
FAM_IDX = {'jpeg': 0, 'webp': 1, 'jxl': 2, 'avif': 3, 'png': 4, 'gif': 5}
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
LL = [('png', 'zenpng_lossless'), ('webp', 'zenwebp_lossless'), ('jxl', 'zenjxl_lossless')]
side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]
side_np = side[QCOLS].to_numpy(dtype=np.float64)
vidx = {v: i for i, v in enumerate(side.index)}

def bytes_at(pts, zq):
    pts = sorted(pts)
    for i in range(1, len(pts)):
        z0, b0 = pts[i-1]; z1, b1 = pts[i]
        if z0 <= zq <= z1 and z1 > z0: return b0 + (b1-b0)*(zq-z0)/(z1-z0)
    return None

def lossy_rd(s):
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in LOSSY:
        df = pd.read_parquet(f'{BASE}/{d}/{s}.parquet', columns=['variant_name','score_zensim','encoded_bytes']).dropna()
        for v,z,b in zip(df.variant_name.values, df.score_zensim.values, df.encoded_bytes.values): rd[v][fam].append((float(z),float(b)))
    return rd

def ll_min(s):
    mb = collections.defaultdict(dict)
    for fam, d in LL:
        df = pd.read_parquet(f'{BASE}/{d}/{s}.parquet', columns=['variant_name','score_zensim','encoded_bytes']).dropna()
        for v,b in df[df.score_zensim>=99.999].groupby('variant_name')['encoded_bytes'].min().items(): mb[v][fam]=float(b)
    return mb

def write(kind, rows):
    with open(f'{OUT}/router_{kind}_test.tsv','w') as f:
        for label, x in rows:
            f.write(f'{label}\t' + '\t'.join(f'{v:.6g}' for v in x) + '\n')
    print(f'{kind}: {len(rows)} test rows')

rd = lossy_rd('test'); rows = []
for v in rd:
    if v not in vidx: continue
    base = side_np[vidx[v]]
    for zq in np.arange(45,91,3.0):
        bb = {f: bytes_at(rd[v][f], zq) for f in rd[v]}; bb = {f:b for f,b in bb.items() if b is not None}
        if len(bb) >= 2: rows.append((FAM_IDX[min(bb,key=bb.get)], list(base)+[zq]))
write('lossy', rows)

mb = ll_min('test'); rows = []
for v, bb in mb.items():
    if v in vidx and len(bb) >= 2: rows.append((FAM_IDX[min(bb,key=bb.get)], list(side_np[vidx[v]])))
write('lossless', rows)

rd = lossy_rd('test'); mb = ll_min('test'); rows = []
for v in rd:
    if v not in vidx or not mb.get(v): continue
    llb = min(mb[v].values()); base = side_np[vidx[v]]
    for zq in np.arange(45,99,3.0):
        lb = [bytes_at(rd[v][f], zq) for f in rd[v]]; lb = [x for x in lb if x is not None]
        rows.append((1 if llb < (min(lb) if lb else float('inf')) else 0, list(base)+[zq]))
write('gate', rows)
