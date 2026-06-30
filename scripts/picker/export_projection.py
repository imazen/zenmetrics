#!/usr/bin/env python3
"""Re-fit the per-family linear projection with inputs = [101 qualified zenanalyze feats + target_zq]
(NO raw log_pixels — size is already carried by qualified feats like log_padded_pixels_8, so the
runtime needs only the Offer + the quality target). Confirm overhead ~ the log_pixels version, then
FOLD the StandardScaler into the Ridge weights -> a single affine per family on RAW inputs, and
export Rust consts + a round-trip test vector."""
import pandas as pd, collections, statistics as st, re, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
FAMS = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
NAMES = [f for f, _ in FAMS]
side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]
snp = side[QCOLS].to_numpy(float); vix = {v: i for i, v in enumerate(side.index)}
TARGETS = list(range(48, 89, 4))

def reach(points, t):
    c = [b for z, b in points if z >= t]
    return min(c) if c else None

def build(split, with_lp):
    C = {}
    for f, d in FAMS:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        c = collections.defaultdict(list)
        for v, z, b in zip(df.variant_name, df.score_zensim, df.encoded_bytes):
            c[v].append((float(z), float(b)))
        C[f] = c
    X, B = [], []
    for v in C['jxl']:
        if v not in vix:
            continue
        m = re.search(r'scale(\d+)x(\d+)', v)
        if not m:
            continue
        lp = np.log(int(m.group(1)) * int(m.group(2)))
        base = snp[vix[v]]
        for t in TARGETS:
            bb = {f: reach(C[f].get(v, []), t) for f in NAMES}
            bb = {f: b for f, b in bb.items() if b is not None}
            if len(bb) >= 2:
                tail = [lp, t] if with_lp else [t]
                X.append(np.concatenate([base, tail])); B.append(bb)
    return np.array(X), B

def fit_eval(with_lp):
    Xtr, Btr = build('train', with_lp); Xte, Bte = build('test', with_lp)
    sc = StandardScaler().fit(Xtr)
    models = {}
    for f in NAMES:
        idx = [i for i, bb in enumerate(Btr) if f in bb]
        y = np.log([Btr[i][f] for i in idx])
        models[f] = Ridge(alpha=2.0).fit(sc.transform(Xtr[idx]), y)
    Zte = sc.transform(Xte); pred = {f: models[f].predict(Zte) for f in NAMES}
    o = []
    for i, bb in enumerate(Bte):
        pk = min(bb, key=lambda f: pred[f][i]); o.append(bb[pk] / min(bb.values()) - 1.0)
    o.sort()
    return st.mean(o) * 100, o[int(len(o) * .9)] * 100, sc, models, Xte, Bte

for wlp in (True, False):
    m, p90, *_ = fit_eval(wlp)
    print(f"inputs {'WITH' if wlp else 'WITHOUT'} log_pixels: mean={m:.2f}% p90={p90:.2f}%")

# Export the WITHOUT-log_pixels model (cleaner runtime)
m, p90, sc, models, Xte, Bte = fit_eval(False)
mean, scale = sc.mean_, sc.scale_
COLS = QCOLS + ['target_zq']
rows = {}
for f in NAMES:
    co, ic = models[f].coef_, models[f].intercept_
    raw_w = co / scale
    raw_b = float(ic - np.sum(co * mean / scale))
    rows[f] = {'w': raw_w.tolist(), 'b': raw_b}
# round-trip check: folded affine on RAW Xte == sklearn pipeline
for f in NAMES:
    raw = Xte @ np.array(rows[f]['w']) + rows[f]['b']
    skl = models[f].predict(sc.transform(Xte))
    assert np.allclose(raw, skl, atol=1e-4), f'fold mismatch {f}: {np.abs(raw-skl).max()}'
print('folded affine == sklearn pipeline (max err ok)')
out = {'cols': COLS, 'families': NAMES, 'weights': rows,
       'test_vector': Xte[0].tolist(),
       'test_scores': {f: float(Xte[0] @ np.array(rows[f]['w']) + rows[f]['b']) for f in NAMES}}
P = '/tmp/claude-1000/-home-lilith-work-zen-zenmetrics/51b72165-bbdf-44d4-9b34-be022d2f50f5/scratchpad/projection.json'
json.dump(out, open(P, 'w'))
print(f'exported {len(COLS)} inputs x {len(NAMES)} families -> {P}')
print('test_scores (lower=fewer bytes=better):', {f: round(v, 3) for f, v in out['test_scores'].items()})
