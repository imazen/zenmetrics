#!/usr/bin/env python3
"""Lossy router retrained on the CLEAN sidecar features: 101 qualified source-only
zenanalyze features (no zensim pair-feature leak, experimental-complete, NaN-free) + dims
+ target_zq -> best lossy family. Labels (RD curves) from the canonical parquets; split from
the canonical train/test files. Compare to the old leaky 469-feature result (75.5%)."""
import pandas as pd, numpy as np, collections, statistics
from sklearn.ensemble import HistGradientBoostingClassifier
SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
FAMS = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
FAMIDX = {f: i for i, (f, _) in enumerate(FAMS)}
NAMES = ['jpeg', 'webp', 'jxl', 'avif']
side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]
FEAT = QCOLS + ['width', 'height']
side_np = side[FEAT].to_numpy(dtype=float)
vidx = {v: i for i, v in enumerate(side.index)}
print(f'sidecar: {len(side)} variants, {len(QCOLS)} qualified source features (+dims)')

def bytes_at(pts, zq):
    pts = sorted(pts)
    for i in range(1, len(pts)):
        z0, b0 = pts[i - 1]; z1, b1 = pts[i]
        if z0 <= zq <= z1 and z1 > z0:
            return b0 + (b1 - b0) * (zq - z0) / (z1 - z0)
    return None

def load_split(split):
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in FAMS:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        for v, z, b in zip(df.variant_name.values, df.score_zensim.values, df.encoded_bytes.values):
            rd[v][fam].append((float(z), float(b)))
    X, y, info = [], [], []
    for v in rd:
        if v not in vidx:
            continue
        base = side_np[vidx[v]]
        for zq in np.arange(45, 91, 3.0):
            bb = {f: bytes_at(rd[v][f], zq) for f in rd[v]}
            bb = {f: b for f, b in bb.items() if b is not None}
            if len(bb) >= 2:
                X.append(np.append(base, zq)); y.append(FAMIDX[min(bb, key=bb.get)]); info.append(bb)
    return np.asarray(X), np.asarray(y), info

Xtr, ytr, _ = load_split('train')
Xte, yte, infote = load_split('test')
clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr, ytr)
pred = clf.predict(Xte)
ohs = []; cant = 0
for i, bb in enumerate(infote):
    oracle = min(bb.values()); pf = NAMES[pred[i]]
    if pf in bb: ohs.append(bb[pf] / oracle - 1.0)
    else: cant += 1
ohs.sort()
print(f'CLEAN lossy router (source-only): rows train={len(Xtr)} test={len(Xte)}')
print(f'  family-acc = {(pred==yte).mean():.1%}  (old leaky 469-feat was 75.5%)')
print(f'  RD overhead vs oracle: mean={statistics.mean(ohs)*100:.2f}% median={ohs[len(ohs)//2]*100:.2f}% '
      f'p90={ohs[int(len(ohs)*0.9)]*100:.2f}% | cant-reach={cant/len(infote):.1%}')
c2 = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr[:, :-1], ytr)
print(f'  WITHOUT target_zq: acc={(c2.predict(Xte[:, :-1])==yte).mean():.1%}')
# Does dims (width,height) matter? cols 0..100=qual feats, 101=w, 102=h, 103=target.
keep = list(range(len(QCOLS))) + [Xtr.shape[1] - 1]  # 101 qualified + target, DROP w/h
c3 = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr[:, keep], ytr)
pred3 = c3.predict(Xte[:, keep])
oh3 = []
for i, bb in enumerate(infote):
    pf = NAMES[pred3[i]]
    if pf in bb: oh3.append(bb[pf] / min(bb.values()) - 1.0)
oh3.sort()
print(f'  WITHOUT dims (101 qual + target, NO w/h): acc={(pred3==yte).mean():.1%} '
      f'RD-overhead mean={statistics.mean(oh3)*100:.2f}% p90={oh3[int(len(oh3)*0.9)]*100:.2f}%')
