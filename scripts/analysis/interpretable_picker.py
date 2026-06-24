#!/usr/bin/env python3
# Human-comprehensible picker (parallel to the black-box MLP): a shallow decision tree whose rules are
# directly readable, plus feature importances. Reports val overhead on a held-out IMAGE split so it's
# comparable to train_hybrid's MLP number. Labeling: per (image, quality-target Q), rd-optimal config =
# min-bytes config whose RD curve reaches quality >= Q (the size_optimal objective).
#   usage: interpretable_picker.py <pareto.parquet> <features.tsv> [metric=ssim2] [--gbdt]
import sys, csv, os
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, export_text

pareto_p, feat_tsv = sys.argv[1], sys.argv[2]
METRIC = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "ssim2"
USE_GBDT = "--gbdt" in sys.argv

# features per image basename
feats = {}
with open(feat_tsv) as f:
    r = csv.DictReader(f, delimiter="\t")
    meta = {"image_path", "image_sha", "split", "content_class", "source", "size_class", "width", "height"}
    feat_cols = [c for c in r.fieldnames if c not in meta]
    for row in r:
        feats[os.path.basename(row["image_path"])] = np.array(
            [float(row[c]) if row[c] and row[c].strip() else 0.0 for c in feat_cols], np.float32)

t = pq.read_table(pareto_p, columns=["image_path", "config_name", "bytes", METRIC]).to_pydict()
configs = sorted(set(t["config_name"]))
cfgidx = {c: i for i, c in enumerate(configs)}
NC = len(configs)
by = defaultdict(lambda: defaultdict(list))
for ip, cn, b, q in zip(t["image_path"], t["config_name"], t["bytes"], t[METRIC]):
    by[ip][cn].append((q, b))

QT = list(range(10, 100, 5))
X, Y, bpc, rmeta = [], [], [], []
for ip, cfgs in by.items():
    bn = os.path.basename(ip)
    if bn not in feats:
        continue
    fv = feats[bn]
    maxq = max(q for pts in cfgs.values() for (q, b) in pts)  # image's best achievable quality
    for Q in QT:
        if Q > maxq:
            continue  # unreachable target for this image — train_hybrid's ceiling logic skips these
        cell = np.full(NC, np.nan)
        for cn, pts in cfgs.items():
            ok = [b for (q, b) in pts if q >= Q]
            if ok:
                cell[cfgidx[cn]] = min(ok)
        if np.all(np.isnan(cell)):
            continue
        X.append(np.concatenate([fv, [Q]]))
        Y.append(int(np.nanargmin(cell)))
        bpc.append(cell)
        rmeta.append(bn)
X = np.array(X, np.float32); Y = np.array(Y); bpc = np.array(bpc); cols = feat_cols + ["quality_target"]
print(f"rows={len(X)}  configs={NC}  features={len(cols)}  metric={METRIC}")

imgs = sorted(set(rmeta))
rng = np.random.default_rng(12345)
val = set(rng.choice(imgs, size=max(1, len(imgs) // 5), replace=False))
tr = np.array([m not in val for m in rmeta]); va = ~tr

# The picker is per-cell BYTES REGRESSION + argmin-over-reachable (NOT classification — a tree that picks
# one config blind ignores reachability + the RD structure). Match that: regress log-bytes per cell;
# unreachable cells get a high sentinel so argmin avoids them (mirrors train_hybrid's reach head).
rowmax = np.nanmax(np.where(np.isnan(bpc), -np.inf, bpc), axis=1)
Yreg = np.where(np.isnan(bpc), np.log(rowmax * 4.0)[:, None], np.log(np.where(np.isnan(bpc), 1.0, bpc)))


def overhead_from_pick(pick):
    o = []
    for i, p in enumerate(pick):
        row = bpc[i]; bm = np.nanmin(row); b = row[p]
        o.append((np.nanmax(row) - bm) / bm if np.isnan(b) else (b - bm) / bm)
    return float(np.mean(o)) * 100


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def eval_percell(make_reg, name):
    preds = np.zeros((int(va.sum()), NC))
    for c in range(NC):
        preds[:, c] = make_reg().fit(X[tr], Yreg[tr, c]).predict(X[va])
    print(f"  {name}: val overhead {overhead_from_pick(preds.argmin(1)):.2f}%")


print(f"\n=== interpretable per-cell regression pickers (held-out {len(val)}/{len(imgs)} images) ===")
print(f"  (compare: MLP student 5.87%, GBDT teacher 6.38%)")
eval_percell(lambda: HistGradientBoostingRegressor(max_depth=6, max_iter=400, learning_rate=0.08, random_state=12345), "per-cell GBDT depth6 x400 (interpretable)")
eval_percell(lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.1, random_state=12345), "per-cell GBDT depth3 x150")
eval_percell(lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=30, random_state=12345), "per-cell tree depth6 (readable)")

# feature importances aggregated across the 24 per-cell GBDTs
imp = np.zeros(len(cols))
for c in range(NC):
    g = HistGradientBoostingRegressor(max_depth=3, max_iter=80, random_state=12345).fit(X[tr], Yreg[tr, c])
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(g, X[va], Yreg[va, c], n_repeats=2, random_state=12345, n_jobs=4)
    imp += pi.importances_mean
top = sorted(zip(cols, imp), key=lambda x: -x[1])[:12]
print("\n=== top features (summed permutation importance across 24 per-cell GBDTs) ===")
for n, v in top:
    print(f"  {v:8.3f}  {n}")
