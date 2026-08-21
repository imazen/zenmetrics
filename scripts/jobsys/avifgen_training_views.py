#!/usr/bin/env python3
# avifgen_training_views.py — emit the AC.R1-AMENDED wave-12 views for the avif944
# corpus (zensim campaign appendix Z / AC.R1, 2026-08-08; revival close-out 2026-08-21).
#
# The corpus is TRAIN-SPLIT-ONLY by construction (train_renditions_2026-06-14: every
# origin ends 0/2/4/6/8 — it IS the June even/odd train-side rendition set), so the
# original train/validate/test emit was structurally wrong (val + test were empty).
# AC.R1 amendment 2 registers instead:
#   train_944 = origins ending 0/2/4/6   (the wave-12 training view)
#   eval8_944 = origins ending 8         (leg-side eval holdout — never trained on)
# Any other terminal digit is a HARD ERROR (it would mean a non-train-side origin
# leaked into the corpus).
#
# Implementation is COLUMNAR over the row-aligned unified tables (identity-column
# equality asserted) — the previous per-row dict join OOM'd at 53.8 GB on the 564,300
# x 948 join (2026-08-08 incident); this build peaks ~8.5 GB.
#
# usage: avifgen_training_views.py <unified_dir> <outdir>
import importlib.util
import json
import os
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

U, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# import the split owner by path (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "origin_split",
    os.path.join(os.path.dirname(__file__), "..", "picker", "origin_split.py"),
)
origin_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(origin_split)

ID = ["image_path", "q", "knob_tuple_json", "encode_sha"]
sc = pq.read_table(os.path.join(U, "scores.parquet"))
ft = pq.read_table(os.path.join(U, "features.parquet"))
for c in ID:
    assert sc[c].equals(ft[c]), f"ID column {c} misaligned between scores and features"

names = sc["image_path"].to_pylist()
origins = [origin_split.origin_id(n) for n in names]
leg = []
for o in origins:
    if not o:
        raise SystemExit(f"unsplittable origin for a corpus row (origin_id returned {o!r})")
    d = o[-1]
    if d in "0246":
        leg.append("train")
    elif d == "8":
        leg.append("eval8")
    else:
        raise SystemExit(f"origin {o} ends in {d} — non-train-side origin in a train-only corpus")

t = sc
for c in ft.column_names:
    if c not in ID:
        t = t.append_column(c, ft[c])
t = t.append_column("origin", pa.array(origins)).append_column("leg", pa.array(leg))

counts, ocounts = {}, {}
for name in ("train", "eval8"):
    sub = t.filter(pc.equal(t["leg"], name))
    p = os.path.join(OUT, f"{name}_944.parquet")
    pq.write_table(sub, p, compression="zstd")
    counts[name] = sub.num_rows
    ocounts[name] = len({o for o, l in zip(origins, leg) if l == name})
    print(name, sub.num_rows, "rows ->", p, flush=True)
print("origins per leg:", ocounts)
json.dump({"rows": counts, "origins": ocounts}, open(os.path.join(OUT, "view_counts.json"), "w"))
print("VIEWS_DONE")
