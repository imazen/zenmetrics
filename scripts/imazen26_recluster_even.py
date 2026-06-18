#!/usr/bin/env python3
"""Even-only re-cluster of imazen-26 (image, crop) units for picker-training
source selection.

Fixes the holdout contamination in `imazen26_representatives_K500_2026-06-14.tsv`,
which clustered over ALL ids and so picked 202/414 odd (= holdout) representatives.
This replicates `zenanalyze/benchmarks/imazen26_cluster_ablation_2026-06-14.py`
exactly (same GEOM exclusion, z-score, KMeans n_init=10, centroid-nearest member)
but filters native rows to EVEN-id (train) images BEFORE clustering, KEEPING the
crop-level units (full + c50/c25 × {center,tl,tr,bl,br}). Odd ids stay a natural,
untouched holdout. The feature space (mean/std for z-scoring) is defined by the
even/train population only.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

# Geometry/size features excluded from selection (size is densified on the reps).
GEOM = ("pixel_count", "log_pixels", "bitmap_bytes", "min_dim", "max_dim",
        "aspect", "block_misalign", "log_padded", "channel_count")


def leading_id(path):
    tok = os.path.basename(path).split("_")[0]
    return int(tok) if tok.isdigit() else None


def load(parquet, parity):
    pf = pq.ParquetFile(parquet)
    feats = [n for n in pf.schema.names if n.startswith("feat_")]
    content = [n for n in feats if not any(k in n for k in GEOM)]
    t = pq.read_table(
        parquet,
        columns=["image_path", "crop_label", "content_class", "size_class"] + content,
    )
    t = t.filter(pc.equal(t["size_class"], "native"))
    paths = t["image_path"].to_pylist()
    crops = t["crop_label"].to_pylist()
    cc = t["content_class"].to_pylist()
    # EVEN-only (train) filter at crop granularity — keep every crop_label.
    keep = [i for i, p in enumerate(paths)
            if (leading_id(p) is not None and leading_id(p) % 2 == parity)]
    M = np.empty((len(keep), len(content)), np.float64)
    for j, name in enumerate(content):
        col = t[name].to_numpy(zero_copy_only=False).astype(np.float64)
        med = np.nanmedian(col)
        med = med if np.isfinite(med) else 0.0
        col = np.where(np.isfinite(col), col, med)
        M[:, j] = col[keep]
    paths = [paths[i] for i in keep]
    crops = [crops[i] for i in keep]
    cc = [cc[i] for i in keep]
    std = M.std(0)
    M = M[:, std > 1e-9]
    mu, sd = M.mean(0), M.std(0)
    sd[sd < 1e-9] = 1.0
    return (M - mu) / sd, paths, crops, cc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--select-k", type=int, required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity", type=int, default=0,
                    help="0 = even (train, default); 1 = odd (holdout)")
    a = ap.parse_args()

    Z, paths, crops, cc = load(a.parquet, a.parity)
    n = Z.shape[0]
    print(f"# parity={a.parity} units={n} content_feats={Z.shape[1]}", file=sys.stderr)

    km = KMeans(n_clusters=a.select_k, n_init=10, random_state=a.seed).fit(Z)
    sizes = np.bincount(km.labels_, minlength=a.select_k)
    rows = []
    for c in range(a.select_k):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1)
        r = int(idx[d.argmin()])
        rows.append((paths[r], crops[r], cc[r], c, int(sizes[c])))

    with open(a.out_manifest, "w") as f:
        f.write("image_path\tcrop_label\tcontent_class\tcluster_id\tcluster_size\n")
        for p, cr, klass, cid, sz in rows:
            f.write(f"{p}\t{cr}\t{klass}\t{cid}\t{sz}\n")

    odd = sum(1 for r in rows if (leading_id(r[0]) or 0) % 2 == 1)
    distinct = len({r[0] for r in rows})
    print(f"selected {len(rows)} reps ({distinct} distinct images) -> {a.out_manifest}",
          file=sys.stderr)
    print(f"ODD-id reps (MUST be 0 for parity=0): {odd}", file=sys.stderr)
    print(f"crop_label distribution: {dict(Counter(r[1] for r in rows))}", file=sys.stderr)
    print(f"singleton clusters (outliers kept): {int((sizes == 1).sum())}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
