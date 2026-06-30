#!/usr/bin/env python3
"""Lossless codec-router: GBDT over (zenanalyze features + dims) -> best lossless family
(min encoded_bytes). NO quality target — lossless has no quality dial; the only objective
is fewest bytes. Held-out train.parquet vs test.parquet. Families: png/webp/jxl-modular."""
import pandas as pd, numpy as np, collections, statistics
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
BASE = "/mnt/v/output/canonical-picker-2026-06-27"
FAMS = [("png", "zenpng_lossless"), ("webp", "zenwebp_lossless"), ("jxl", "zenjxl_lossless")]
FAMIDX = {f: i for i, (f, _) in enumerate(FAMS)}
NAMES = [f for f, _ in FAMS]
FEATCOLS = [c for c in pq.read_schema(f"{BASE}/zenpng_lossless/train.parquet").names if c.startswith("feat_")]

def load_split(split):
    # features (source-content, identical across codecs for the same variant): union over
    # families, dedup on variant_name. min-bytes per family per variant = codec's best effort.
    feats = []
    minb = collections.defaultdict(dict)
    for fam, d in FAMS:
        df = pd.read_parquet(f"{BASE}/{d}/{split}.parquet",
                             columns=["variant_name", "encoded_bytes", "score_zensim"] + FEATCOLS + ["width", "height"]
                             ).dropna(subset=["encoded_bytes", "score_zensim"])
        feats.append(df[["variant_name"] + FEATCOLS + ["width", "height"]])
        # CRITICAL: png's modes_full sweep mixes in LOSSY palette-quantized encodes (only
        # 53.8% of png rows are truly lossless). Filter to score==100 before min-bytes, or
        # the router compares png's lossy small files against jxl/webp's true-lossless ones.
        ll = df[df.score_zensim >= 99.999]
        for v, b in ll.groupby("variant_name")["encoded_bytes"].min().items():
            minb[v][fam] = float(b)
    fv = pd.concat(feats).drop_duplicates("variant_name").set_index("variant_name")
    feat_np = fv[FEATCOLS + ["width", "height"]].to_numpy(dtype=float)
    vidx = {v: i for i, v in enumerate(fv.index)}
    X, y, info = [], [], []
    for v in fv.index:
        bb = minb.get(v, {})
        if len(bb) >= 2:  # need a choice
            X.append(feat_np[vidx[v]]); y.append(FAMIDX[min(bb, key=bb.get)]); info.append(bb)
    return np.asarray(X), np.asarray(y), info

Xtr, ytr, _ = load_split("train")
Xte, yte, infote = load_split("test")
clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"rows train={len(Xtr)} test={len(Xte)} | LOSSLESS-ROUTER test family-acc = {(pred==yte).mean():.1%}")
ohs = []; cant = 0
for i, bb in enumerate(infote):
    oracle = min(bb.values()); pf = NAMES[pred[i]]
    if pf in bb: ohs.append(bb[pf] / oracle - 1.0)
    else: cant += 1
ohs.sort()
print(f"  RD overhead vs oracle: mean={statistics.mean(ohs)*100:.2f}% median={ohs[len(ohs)//2]*100:.2f}% "
      f"p90={ohs[int(len(ohs)*0.9)]*100:.2f}% | predicted-cant-reach={cant/len(infote):.1%}")
mode = np.bincount(ytr).argmax()
print(f"  baseline (always {NAMES[mode]}): {(yte == mode).mean():.1%}  | family mix train={np.bincount(ytr)/len(ytr)}")
