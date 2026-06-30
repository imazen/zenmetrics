#!/usr/bin/env python3
"""Export a READABLE decision tree on raw, human-named features (size/detail/chroma/quality)
for the lossy family pick, with real thresholds + the family at each leaf + RD overhead. This
is the human-friendly 'tree of choices' form (vs the abstract LDA composite)."""
import sys, statistics as st
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts/picker')
from picker_data import load_rd, oracle_rows
from sklearn.tree import DecisionTreeClassifier, export_text
SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
NAMES = ['jpeg', 'webp', 'jxl', 'avif']; FAM_IDX = {n: i for i, n in enumerate(NAMES)}
side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]; snp = side[QCOLS].to_numpy(float)
vix = {v: i for i, v in enumerate(side.index)}
names = [c.split('@')[0] for c in QCOLS] + ['target_zq']

def build(split):
    rd = load_rd(BASE, LOSSY, split); rows, _ = oracle_rows(rd, LOSSY, list(np.arange(45, 91, 3.0)), require='all')
    X, y, B = [], [], []
    for r in rows:
        if r['variant'] in vix:
            X.append(np.append(snp[vix[r['variant']]], r['target'])); y.append(FAM_IDX[r['oracle']]); B.append(r['bytes'])
    return np.array(X), np.array(y), B

Xtr, ytr, _ = build('train'); Xte, yte, Bte = build('test')

def overhead(pred):
    o = sorted((Bte[i][NAMES[p]] / min(Bte[i].values()) - 1.0) for i, p in enumerate(pred))
    return f"mean={st.mean(o)*100:.2f}% p90={o[int(len(o)*.9)]*100:.1f}%"

for leaves in [6, 8, 16]:
    t = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(Xtr, ytr)
    print(f"\n===== {leaves}-leaf tree  ({overhead(t.predict(Xte))}) =====")
    txt = export_text(t, feature_names=names, class_names=NAMES, max_depth=8)
    print(txt)
