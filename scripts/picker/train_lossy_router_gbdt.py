#!/usr/bin/env python3
"""First (most-obvious) lossy codec-router: GBDT over (zenanalyze features + dims +
target_zq) -> best lossy family (min encoded_bytes at the target zensim). Held-out
train.parquet vs test.parquet. Confirms quality is the dominant input."""
import pandas as pd, numpy as np, collections
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
BASE = "/mnt/v/output/canonical-picker-2026-06-27"
FAMS = [("jpeg", "zenjpeg_lossy"), ("webp", "zenwebp_lossy"), ("jxl", "zenjxl_lossy"), ("avif", "zenavif_lossy")]
FAMIDX = {f: i for i, (f, _) in enumerate(FAMS)}
NAMES = ["jpeg", "webp", "jxl", "avif"]
FEATCOLS = [c for c in pq.read_schema(f"{BASE}/zenjpeg_lossy/train.parquet").names if c.startswith("feat_")]

def bytes_at(pts, zq):
    pts = sorted(pts)
    for i in range(1, len(pts)):
        z0, b0 = pts[i - 1]; z1, b1 = pts[i]
        if z0 <= zq <= z1 and z1 > z0:
            return b0 + (b1 - b0) * (zq - z0) / (z1 - z0)
    return None

def load_split(split):
    # source features (identical across codecs) from one family
    fv = (pd.read_parquet(f"{BASE}/zenjpeg_lossy/{split}.parquet")
          .sort_values("q").drop_duplicates("variant_name").set_index("variant_name"))
    feat_np = fv[FEATCOLS + ["width", "height"]].to_numpy(dtype=float)
    vidx = {v: i for i, v in enumerate(fv.index)}
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in FAMS:
        df = pd.read_parquet(f"{BASE}/{d}/{split}.parquet",
                             columns=["variant_name", "score_zensim", "encoded_bytes"]).dropna()
        for v, z, b in zip(df.variant_name.values, df.score_zensim.values, df.encoded_bytes.values):
            rd[v][fam].append((float(z), float(b)))
    X, y, info = [], [], []
    for v in fv.index:
        base = feat_np[vidx[v]]
        for zq in np.arange(45, 91, 3.0):
            bb = {f: bytes_at(rd[v][f], zq) for f in rd[v]}
            bb = {f: b for f, b in bb.items() if b is not None}
            if len(bb) >= 2:
                X.append(np.append(base, zq)); y.append(FAMIDX[min(bb, key=bb.get)]); info.append(bb)
    return np.asarray(X), np.asarray(y), info

Xtr, ytr, _ = load_split("train")
Xte, yte, infote = load_split("test")
clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"rows train={len(Xtr)} test={len(Xte)} | LOSSY-ROUTER test family-acc = {(pred==yte).mean():.1%}")
import statistics
ohs = []; cant = 0
for i, bb in enumerate(infote):
    oracle = min(bb.values()); pf = NAMES[pred[i]]
    if pf in bb: ohs.append(bb[pf] / oracle - 1.0)
    else: cant += 1
ohs.sort()
print(f"  RD overhead vs oracle: mean={statistics.mean(ohs)*100:.2f}% median={ohs[len(ohs)//2]*100:.2f}% "
      f"p90={ohs[int(len(ohs)*0.9)]*100:.2f}% | predicted-cant-reach={cant/len(infote):.1%}")
# baseline: always the single most-common family
mode = np.bincount(ytr).argmax()
print(f"  baseline (always {NAMES[mode]}): {(yte==mode).mean():.1%}")
# per target-zq band
zq = Xte[:, -1]
for lo, hi in [(45, 60), (60, 75), (75, 91)]:
    m = (zq >= lo) & (zq < hi)
    if m.sum(): print(f"  zq[{lo},{hi}) acc={(pred[m]==yte[m]).mean():.1%} (n={int(m.sum())})")
# quality's contribution: retrain WITHOUT the target_zq column
c2 = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr[:, :-1], ytr)
print(f"  WITHOUT target_zq input: acc={(c2.predict(Xte[:, :-1])==yte).mean():.1%}  (drop = quality's contribution)")
