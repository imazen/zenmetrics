#!/usr/bin/env python3
"""Pick K content-diverse representative ORIGIN images from imazen-26 via
k-means on zenanalyze features (centroid-nearest per cluster), train-origin only
(canonical origin_split, even last digit). Per the dense-sampling rule."""
import sys, json
sys.path.insert(0, '/home/lilith/work/zen/zenmetrics/scripts/picker')
import origin_split
import numpy as np
import pyarrow.parquet as pq, pyarrow.compute as pc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

K = 16
FEAT = '/mnt/v/output/imazen-26-features/imazen26_features_2026-06-23.parquet'

t = pq.read_table(FEAT)
names = t.schema.names
feat_cols = names[9:]  # after the 9 meta cols
# full + native rows only (one canonical row per origin)
m = pc.and_(pc.equal(t['crop_label'], 'full'), pc.equal(t['size_class'], 'native'))
ft = t.filter(m)
paths = ft['image_path'].to_pylist()
classes = ft['content_class'].to_pylist()
ws = ft['width'].to_pylist(); hs = ft['height'].to_pylist()
# canonical train filter (re-derive split from the path; don't trust the column)
keep = [i for i, p in enumerate(paths) if origin_split.split_of(p) == 'train']
print(f'full+native rows: {ft.num_rows}; canonical-train origins: {len(keep)}', file=sys.stderr)

X = np.column_stack([np.asarray(ft[c].to_pylist(), dtype=float) for c in feat_cols])
X = X[keep]
paths = [paths[i] for i in keep]; classes = [classes[i] for i in keep]
mx = [max(ws[i], hs[i]) for i in keep]

# drop all-NaN / constant cols; median-impute residual NaN
colmed = np.nanmedian(X, axis=0)
good = ~np.isnan(colmed)
X = X[:, good]; fcols = [feat_cols[i] for i in range(len(feat_cols)) if good[i]]
colmed = colmed[good]
inds = np.where(np.isnan(X))
X[inds] = np.take(colmed, inds[1])
# drop zero-variance cols
var = X.var(axis=0)
nz = var > 1e-12
X = X[:, nz]; fcols = [fcols[i] for i in range(len(fcols)) if nz[i]]
print(f'features used for kmeans: {X.shape[1]}', file=sys.stderr)

Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xs)
picks = []
for c in range(K):
    members = np.where(km.labels_ == c)[0]
    d = np.linalg.norm(Xs[members] - km.cluster_centers_[c], axis=1)
    best = members[np.argmin(d)]
    picks.append({'cluster': c, 'cluster_size': int(len(members)),
                  'image_path': paths[best], 'content_class': classes[best],
                  'native_longedge': int(mx[best]),
                  'origin_id': origin_split.origin_id(paths[best])})

picks.sort(key=lambda r: r['cluster'])
with open('/tmp/claude-1000/-home-lilith-work-zen-zenmetrics/51b72165-bbdf-44d4-9b34-be022d2f50f5/scratchpad/picks16.json', 'w') as f:
    json.dump(picks, f, indent=2)
print(f'{"cl":>2} {"size":>5} {"longE":>6} {"class":35s} path')
for r in picks:
    print(f'{r["cluster"]:>2} {r["cluster_size"]:>5} {r["native_longedge"]:>6} {r["content_class"]:35s} {r["image_path"].split("/")[-1][:60]}')
print('\nclass coverage:', sorted(set(r['content_class'] for r in picks)))
print('n distinct classes among 16 picks:', len(set(r['content_class'] for r in picks)))
