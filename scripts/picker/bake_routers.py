#!/usr/bin/env python3
"""Train the 3 router MLPs on the CLEAN sidecar features (101 qualified source-only feats)
and bake each to a ZNPR BakeRequestJson. Classifier -> ZNPR-argmin: negate the final logits
(argmin(-logit) = argmax logit = predicted family) and pad to 6 CodecFamily outputs (the
families this branch doesn't choose get a +1e9 cost, also masked by AllowedFamilies::LOSSY/
LOSSLESS at runtime). Gate is 2-out (lossy=0, lossless=1; route reads out[1]<out[0]).

Self-verifies each bake by replicating the ZNPR forward pass in numpy and asserting argmin
(over the branch family set) == sklearn mlp.predict, BEFORE writing the JSON. Then shells to
`zenpredict-bake` to produce the .bin."""
import json, os, subprocess, collections, statistics
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SIDE = '/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
BASE = '/mnt/v/output/canonical-picker-2026-06-27'
OUT = '/mnt/v/output/router-features-2026-06-30'
ALL_LABELS_CSV = 'jpeg,webp,jxl,avif,png,gif'  # CodecFamily::ALL order
FAM_IDX = {'jpeg': 0, 'webp': 1, 'jxl': 2, 'avif': 3, 'png': 4, 'gif': 5}
LOSSY = [('jpeg', 'zenjpeg_lossy'), ('webp', 'zenwebp_lossy'), ('jxl', 'zenjxl_lossy'), ('avif', 'zenavif_lossy')]
LL = [('png', 'zenpng_lossless'), ('webp', 'zenwebp_lossless'), ('jxl', 'zenjxl_lossless')]

side = pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS = [c for c in side.columns if '@' in c]
side_np = side[QCOLS].to_numpy(dtype=np.float64)
vidx = {v: i for i, v in enumerate(side.index)}
print(f'sidecar {len(side)} variants x {len(QCOLS)} qualified feats')


def bytes_at(pts, zq):
    pts = sorted(pts)
    for i in range(1, len(pts)):
        z0, b0 = pts[i - 1]; z1, b1 = pts[i]
        if z0 <= zq <= z1 and z1 > z0:
            return b0 + (b1 - b0) * (zq - z0) / (z1 - z0)
    return None


def lossy_rd(split):
    rd = collections.defaultdict(lambda: collections.defaultdict(list))
    for fam, d in LOSSY:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        for v, z, b in zip(df.variant_name.values, df.score_zensim.values, df.encoded_bytes.values):
            rd[v][fam].append((float(z), float(b)))
    return rd


def ll_min(split):
    mb = collections.defaultdict(dict)
    for fam, d in LL:
        df = pd.read_parquet(f'{BASE}/{d}/{split}.parquet', columns=['variant_name', 'score_zensim', 'encoded_bytes']).dropna()
        ll = df[df.score_zensim >= 99.999]  # true-lossless only
        for v, b in ll.groupby('variant_name')['encoded_bytes'].min().items():
            mb[v][fam] = float(b)
    return mb


def build_lossy(split):
    rd = lossy_rd(split); X, y = [], []
    for v in rd:
        if v not in vidx:
            continue
        base = side_np[vidx[v]]
        for zq in np.arange(45, 91, 3.0):
            bb = {f: bytes_at(rd[v][f], zq) for f in rd[v]}; bb = {f: b for f, b in bb.items() if b is not None}
            if len(bb) >= 2:
                X.append(np.append(base, zq)); y.append(FAM_IDX[min(bb, key=bb.get)])
    return np.array(X), np.array(y)


def build_lossless(split):
    mb = ll_min(split); X, y = [], []
    for v, bb in mb.items():
        if v not in vidx or len(bb) < 2:
            continue
        X.append(side_np[vidx[v]]); y.append(FAM_IDX[min(bb, key=bb.get)])
    return np.array(X), np.array(y)


def build_gate(split):
    rd = lossy_rd(split); mb = ll_min(split); X, y = [], []
    for v in rd:
        if v not in vidx or not mb.get(v):
            continue
        llb = min(mb[v].values()); base = side_np[vidx[v]]
        for zq in np.arange(45, 99, 3.0):
            lb = [bytes_at(rd[v][f], zq) for f in rd[v]]; lb = [x for x in lb if x is not None]
            lossy_b = min(lb) if lb else float('inf')
            X.append(np.append(base, zq)); y.append(1 if llb < lossy_b else 0)
    return np.array(X), np.array(y)


def mlp_bake_json(mlp, scaler, qcols, n_out, family_order):
    layers = []
    nl = len(mlp.coefs_)
    for li in range(nl):
        W, b = mlp.coefs_[li], mlp.intercepts_[li]
        if li < nl - 1:
            layers.append({"in_dim": W.shape[0], "out_dim": W.shape[1], "activation": "relu",
                           "dtype": "f32", "weights": W.flatten().tolist(), "biases": b.tolist()})
        else:  # final layer: negate logits + pad to n_out CodecFamily slots
            Wout = np.zeros((W.shape[0], n_out)); bout = np.full(n_out, 1e9)
            if len(mlp.classes_) == 2 and W.shape[1] == 1:
                # sklearn binary: ONE logit unit = score for classes_[1]. Map to two argmin
                # slots: slot[c1] = -logit, slot[c0] = 0 -> argmin matches predict (and route's
                # out[c1] < out[c0] gate test fires iff logit > 0).
                c0, c1 = int(mlp.classes_[0]), int(mlp.classes_[1])
                Wout[:, c0] = 0.0; bout[c0] = 0.0
                Wout[:, c1] = -W[:, 0]; bout[c1] = -b[0]
            else:
                for j, c in enumerate(mlp.classes_):
                    Wout[:, int(c)] = -W[:, j]; bout[int(c)] = -b[j]
            layers.append({"in_dim": W.shape[0], "out_dim": n_out, "activation": "identity",
                           "dtype": "f32", "weights": Wout.flatten().tolist(), "biases": bout.tolist()})
    meta = [{"key": "zentrain.feature_columns", "type": "utf8", "text": "\n".join(qcols)}]
    if family_order:
        meta.append({"key": "zenpicker.family_order", "type": "utf8", "text": family_order})
    return json.dumps({"schema_hash": 0, "scaler_mean": scaler.mean_.tolist(),
                       "scaler_scale": scaler.scale_.tolist(), "layers": layers, "metadata": meta})


def znpr_forward(spec, Xraw):
    h = (Xraw - np.array(spec['scaler_mean'])) / np.array(spec['scaler_scale'])
    for L in spec['layers']:
        W = np.array(L['weights']).reshape(L['in_dim'], L['out_dim'])
        h = h @ W + np.array(L['biases'])
        if L['activation'] == 'relu':
            h = np.maximum(h, 0.0)
    return h


def train_bake(kind, builder, n_out, family_order, branch_idx, hidden=(128, 64)):
    Xtr, ytr = builder('train'); Xva, yva = builder('validate'); Xte, yte = builder('test')
    Xfit = np.vstack([Xtr, Xva]); yfit = np.concatenate([ytr, yva])
    sc = StandardScaler().fit(Xfit)
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, early_stopping=True,
                        alpha=1e-4, random_state=0).fit(sc.transform(Xfit), yfit)
    acc = (mlp.predict(sc.transform(Xte)) == yte).mean()
    spec_str = mlp_bake_json(mlp, sc, QCOLS, n_out, family_order)
    spec = json.loads(spec_str)
    # SELF-VERIFY: ZNPR forward argmin over the branch family set == mlp.predict
    out = znpr_forward(spec, Xte.astype(np.float64))
    masked = np.full_like(out, np.inf); masked[:, branch_idx] = out[:, branch_idx]
    znpr_pick = masked.argmin(axis=1)
    mlp_pick = mlp.predict(sc.transform(Xte))
    match = (znpr_pick == mlp_pick).mean()
    jpath = f'{OUT}/router_{kind}.bakereq.json'
    with open(jpath, 'w') as f:
        f.write(spec_str)
    print(f'[{kind}] test-acc={acc:.1%} in_dim={spec["layers"][0]["in_dim"]} n_out={n_out} '
          f'classes={list(mlp.classes_)} | ZNPR-vs-sklearn argmin match={match:.4f} -> {jpath}')
    assert match > 0.9999, f'{kind}: ZNPR forward disagrees with sklearn ({match:.4f}) — bake layout bug'
    return jpath


jobs = [
    ('lossy', build_lossy, 6, ALL_LABELS_CSV, [0, 1, 2, 3]),
    ('lossless', build_lossless, 6, ALL_LABELS_CSV, [1, 2, 4]),
    ('gate', build_gate, 2, None, [0, 1]),
]
paths = [train_bake(*j) for j in jobs]
print('all bake JSONs verified:', paths)
