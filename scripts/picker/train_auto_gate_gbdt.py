#!/usr/bin/env python3
"""Auto-gate: the lossy-vs-lossless decision. At a target quality (zq), is the best LOSSY
encode cheaper than the best TRUE-lossless encode (or can lossy even reach that quality)?
-> route lossy or lossless. Reveals the crossover: low/mid zq -> lossy wins; near-lossless
zq (or a zq lossy can't reach) -> lossless wins. GBDT over (features + dims + target_zq)."""
import pandas as pd, numpy as np, collections
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
LL = [('png', 'zenpng_lossless'), ('webp', 'zenwebp_lossless'), ('jxl', 'zenjxl_lossless')]
FEATCOLS = [c for c in pq.read_schema(f'{BASE}/zenjpeg_lossy/train.parquet').names if c.startswith('feat_')]
ZQ = list(range(45, 90, 5)) + list(range(90, 99, 1))  # dense at the high end where the gate flips

def bytes_at(pts, zq):
    pts = sorted(pts)
    for i in range(1, len(pts)):
        z0, b0 = pts[i - 1]; z1, b1 = pts[i]
        if z0 <= zq <= z1 and z1 > z0:
            return b0 + (b1 - b0) * (zq - z0) / (z1 - z0)
    return None  # zq outside the swept lossy range (e.g. above the lossy ceiling)

def load_split(split):
    fv = (pd.read_parquet(f'{BASE}/zenjpeg_lossy/{split}.parquet')
          .sort_values('q').drop_duplicates('variant_name').set_index('variant_name'))
    feat_np = fv[FEATCOLS + ['width', 'height']].to_numpy(dtype=float)
    vidx = {v: i for i, v in enumerate(fv.index)}
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in LOSSY:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        for v, z, b in zip(df.variant_name.values, df.score_zensim.values, df.encoded_bytes.values):
            rd[v][fam].append((float(z), float(b)))
    llmin = collections.defaultdict(dict)
    for fam, d in LL:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        ll = df[df.score_zensim >= 99.999]  # true-lossless only
        for v, b in ll.groupby('variant_name')['encoded_bytes'].min().items():
            llmin[v][fam] = float(b)
    X, y = [], []
    for v in fv.index:
        if not llmin.get(v):
            continue
        llb = min(llmin[v].values())
        base = feat_np[vidx[v]]
        for zq in ZQ:
            lb = [bytes_at(rd[v][f], zq) for f in rd[v]]
            lb = [x for x in lb if x is not None]
            lossy_b = min(lb) if lb else float('inf')  # inf = no lossy family reaches this quality
            X.append(np.append(base, zq)); y.append(0 if lossy_b < llb else 1)  # 0=lossy 1=lossless
    return np.asarray(X), np.asarray(y)

Xtr, ytr = load_split('train'); Xte, yte = load_split('test')
clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f'rows train={len(Xtr)} test={len(Xte)} | AUTO-GATE test acc={(pred==yte).mean():.1%} | lossless-frac train={ytr.mean():.1%}')
zq = Xte[:, -1]
print('crossover — fraction where TRUE-lossless beats best-lossy, by target zq:')
for q in ZQ:
    m = zq == q
    if m.sum():
        print(f'  zq={q:>2}: lossless-better(oracle)={yte[m].mean():5.1%}  router-acc={ (pred[m]==yte[m]).mean():5.1%}  (n={int(m.sum())})')
