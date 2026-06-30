#!/usr/bin/env python3
"""Image-AWARE order from LINEAR PROJECTIONS of zenanalyze, fit on confound-CORRECTED data.
Per family, a linear projection of (101 qualified zenanalyze features + log_pixels + target_zq)
predicts log(bytes_to_reach target); the per-image order = ascending predicted bytes; take the
best supported. Trained on the corrected oracle (bytes_to_reach, AVIF best-speed incl s2, paired
support). Compares to image-blind fixed orders. The per-family weight vectors ARE the projections
(interpretable). Held-out train->test."""
import pandas as pd, collections, statistics as st, re
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
FAMS = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
NAMES = [f for f, _ in FAMS]; FI = {f: i for i, f in enumerate(NAMES)}
side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]
snp = side[QCOLS].to_numpy(float); vix = {v: i for i, v in enumerate(side.index)}
COLS = [c.split('@')[0] for c in QCOLS] + ['log_pixels', 'target_zq']
TARGETS = list(range(48, 89, 4))

def reach(points, t):
    c = [b for z, b in points if z >= t]
    return min(c) if c else None

def logpx(v):
    m = re.search(r'scale(\d+)x(\d+)', v)
    return np.log(int(m.group(1)) * int(m.group(2))) if m else None

def build(split):
    C = {}
    for f, d in FAMS:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        c = collections.defaultdict(list)
        for v, z, b in zip(df.variant_name, df.score_zensim, df.encoded_bytes):
            c[v].append((float(z), float(b)))
        C[f] = c
    X, B = [], []  # X: features; B: per-cell {fam: bytes} (only supported)
    for v in C['jxl']:
        if v not in vix:
            continue
        lp = logpx(v)
        if lp is None:
            continue
        base = snp[vix[v]]
        for t in TARGETS:
            bb = {f: reach(C[f].get(v, []), t) for f in NAMES}
            bb = {f: b for f, b in bb.items() if b is not None}
            if len(bb) >= 2:
                X.append(np.concatenate([base, [lp, t]])); B.append(bb)
    return np.array(X), B

Xtr, Btr = build('train'); Xte, Bte = build('test')
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
# per-family linear projection: predict log(bytes_to_reach), trained on cells where family supported
models = {}
for f in NAMES:
    idx = [i for i, bb in enumerate(Btr) if f in bb]
    y = np.log([Btr[i][f] for i in idx])
    models[f] = Ridge(alpha=2.0).fit(Ztr[idx], y)
pred = {f: models[f].predict(Zte) for f in NAMES}

def overhead(picker):
    o = []
    for i, bb in enumerate(Bte):
        pk = picker(i, bb)
        o.append(bb[pk] / min(bb.values()) - 1.0)
    o.sort()
    return f"mean={st.mean(o)*100:5.2f}% median={o[len(o)//2]*100:4.2f}% p90={o[int(len(o)*.9)]*100:5.2f}%"

# linear-projection order: among SUPPORTED families, pick lowest predicted bytes
def proj_pick(i, bb):
    return min(bb, key=lambda f: pred[f][i])
def fixed(order):
    return lambda i, bb: next(f for f in order if f in bb)
print(f"{len(Bte)} held-out cells. RD overhead vs corrected oracle (bytes_to_reach, best-speed AVIF):")
print(f"  image-blind  always-jxl              : {overhead(fixed(['jxl','avif','webp','jpeg']))}")
print(f"  image-blind  always-avif             : {overhead(fixed(['avif','jxl','webp','jpeg']))}")
print(f"  image-AWARE  linear projection (this) : {overhead(proj_pick)}")
# interpretability: each family's top projection weights
for f in NAMES:
    w = models[f].coef_; top = np.argsort(-np.abs(w))[:5]
    print(f"  {f:>4} bytes ~ " + ", ".join(f"{'+' if w[i]>0 else '-'}{COLS[i]}" for i in top))
