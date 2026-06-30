#!/usr/bin/env python3
"""'Try multiple model shapes' for the lossy router: GBDT (the obvious) vs MLP (the shape
that bakes to ZNPR / zenpredict in production). Same data, same held-out test. Tells us
whether a distilled MLP student can match the GBDT teacher before we bake."""
import pandas as pd, numpy as np, collections, statistics
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
    fv = (pd.read_parquet(f"{BASE}/zenjpeg_lossy/{split}.parquet")
          .sort_values("q").drop_duplicates("variant_name").set_index("variant_name"))
    feat_np = fv[FEATCOLS + ["width", "height"]].to_numpy(dtype=float)
    vidx = {v: i for i, v in enumerate(fv.index)}
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in FAMS:
        df = pd.read_parquet(f"{BASE}/{d}/{split}.parquet", columns=["variant_name", "score_zensim", "encoded_bytes"]).dropna()
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

def report(name, pred, yte, infote):
    ohs = []; cant = 0
    for i, bb in enumerate(infote):
        oracle = min(bb.values()); pf = NAMES[pred[i]]
        if pf in bb: ohs.append(bb[pf] / oracle - 1.0)
        else: cant += 1
    ohs.sort()
    print(f"{name:>22}: family-acc={(pred==yte).mean():5.1%} | RD overhead mean={statistics.mean(ohs)*100:.2f}% "
          f"median={ohs[len(ohs)//2]*100:.2f}% p90={ohs[int(len(ohs)*0.9)]*100:.2f}% cant-reach={cant/len(infote):.1%}")

Xtr, ytr, _ = load_split("train")
Xte, yte, infote = load_split("test")
# GBDT (teacher / the obvious shape)
gb = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr, ytr)
report("GBDT", gb.predict(Xte), yte, infote)
# MLP (the ZNPR/zenpredict production shape) — needs NaN impute (tiny-cell percentile
# features are NaN; GBDT eats NaN natively, MLP can't) + feature scaling, as the bake does.
imp = SimpleImputer(strategy="median").fit(Xtr)
sc = StandardScaler().fit(imp.transform(Xtr))
prep = lambda A: sc.transform(imp.transform(A))
for hls in [(128, 64), (256, 128, 64)]:
    mlp = MLPClassifier(hidden_layer_sizes=hls, max_iter=400, early_stopping=True,
                        alpha=1e-4, learning_rate_init=1e-3, random_state=0).fit(prep(Xtr), ytr)
    report(f"MLP{hls}", mlp.predict(prep(Xte)), yte, infote)
# GBDT-teacher -> MLP-student distillation (train MLP on GBDT's predicted labels)
soft = gb.predict(Xtr)
mlp_d = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=400, early_stopping=True,
                      alpha=1e-4, random_state=0).fit(prep(Xtr), soft)
report("MLP-distilled(256,128,64)", mlp_d.predict(prep(Xte)), yte, infote)
