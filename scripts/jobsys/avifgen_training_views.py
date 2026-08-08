#!/usr/bin/env python3
# avifgen_training_views.py — emit the wave-12 train/validate/test views for the
# avif944 corpus (zensim campaign appendix Z).
#
# Joins the writeback outputs (scores.parquet + features.parquet, keyed on the full
# cell identity) into one row per cell, attaches the origin split via the CANONICAL
# scripts/picker/origin_split.py rule (never re-derived), and writes
#   <outdir>/{train,validate,test}_944.parquet
#
# usage: avifgen_training_views.py <writeback_dir> <outdir>
import importlib.util
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

WB, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# import the split owner by path (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "origin_split",
    os.path.join(os.path.dirname(__file__), "..", "picker", "origin_split.py"),
)
origin_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(origin_split)

ID = ["image_path", "q", "knob_tuple_json", "encode_sha"]
sc = pq.read_table(os.path.join(WB, "scores.parquet")).to_pydict()
ft = pq.read_table(os.path.join(WB, "features.parquet"))
n_feat = sum(1 for c in ft.column_names if c.startswith("feat_"))
ft = ft.to_pydict()

fidx = {}
for i in range(len(ft["image_path"])):
    fidx[(ft["image_path"][i], ft["q"][i], ft["knob_tuple_json"][i], ft["encode_sha"][i])] = i

score_cols = [c for c in sc if c not in ID]
feat_cols = ["feat_%d" % i for i in range(n_feat)]
out_cols = ID + ["origin", "split"] + score_cols + (["zensim_score"] if "zensim_score" not in score_cols else []) + feat_cols
rows = {c: [] for c in out_cols}
miss_feat = 0
split_counts = {}
for i in range(len(sc["image_path"])):
    key = (sc["image_path"][i], sc["q"][i], sc["knob_tuple_json"][i], sc["encode_sha"][i])
    fi = fidx.get(key)
    if fi is None:
        miss_feat += 1
        continue
    name = sc["image_path"][i]
    sp = origin_split.split_of(name)
    if sp is None:
        continue
    rows["origin"].append(origin_split.origin_id(name))
    rows["split"].append(sp)
    for c in ID:
        rows[c].append(sc[c][i])
    for c in score_cols:
        rows[c].append(sc[c][i])
    if "zensim_score" not in score_cols:
        rows["zensim_score"].append(ft.get("zensim_score", [None] * (fi + 1))[fi])
    for j, c in enumerate(feat_cols):
        rows[c].append(ft[c][fi])
    split_counts[sp] = split_counts.get(sp, 0) + 1

t = pa.table(rows)
name_map = {"train": "train", "val": "validate", "test": "test"}
import pyarrow.compute as pc
for sp, out_name in name_map.items():
    sub = t.filter(pc.equal(t["split"], sp))
    p = os.path.join(OUT, f"{out_name}_944.parquet")
    pq.write_table(sub, p, compression="zstd")
    print(f"{out_name}: {sub.num_rows} rows -> {p}")
print(f"total joined {t.num_rows} (feat width {n_feat}; cells missing features: {miss_feat}); splits {split_counts}")
