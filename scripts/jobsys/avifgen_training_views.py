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
# Implementation is COLUMNAR: scores and features are INNER-JOINED on the identity
# key (image_path, encode_sha) via a take-index, then combined column-wise — never
# a per-row dict join (that OOM'd at 53.8 GB on the 564,300 x 948 join, 2026-08-08
# incident; this build peaks ~8.5 GB) and never a positional zip (the two tables are
# separate write-backs and can differ by a row, 2026-08-30 aom harvest). Rows present
# on only one side are DROPPED with a loud count + sample keys.
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
# The IDENTITY key. Scores and features are two independent write-backs of the
# same wave and are NOT guaranteed row-aligned: a cell can land a score row and
# no feature blob (or vice versa) whenever one of the two passes drained
# unevenly — the 2026-08-30 aom harvest had scores 125,688 vs features 125,687,
# which the old positional `sc[c].equals(ft[c])` assert reported as "ID column
# image_path misaligned" (a true statement about row ORDER that says nothing
# about which cell is missing). So: KEY-JOIN, never zip. The key is
# (image_path, encode_sha) — encode_sha alone is NOT unique, different sources
# can encode to byte-identical bytes (15 shas shared across 30 svt cells, 7/14
# on aom; zenmetrics `03fb7311`).
KEY = ["image_path", "encode_sha"]


def _find_features_table(d):
    """`features.parquet`, else the single regime-tagged `features_*.parquet`."""
    plain = os.path.join(d, "features.parquet")
    if os.path.exists(plain):
        return plain
    cands = sorted(
        f for f in os.listdir(d) if f.startswith("features_") and f.endswith(".parquet")
    )
    if len(cands) == 1:
        return os.path.join(d, cands[0])
    raise SystemExit(
        f"{d}: expected features.parquet or exactly one features_*.parquet, found {cands}"
    )


sc = pq.read_table(os.path.join(U, "scores.parquet"))
ft = pq.read_table(_find_features_table(U))


def _keys(t):
    return list(zip(*(t[c].to_pylist() for c in KEY)))


sc_keys, ft_keys = _keys(sc), _keys(ft)
for label, ks in (("scores", sc_keys), ("features", ft_keys)):
    if len(set(ks)) != len(ks):
        from collections import Counter

        dup = [k for k, n in Counter(ks).most_common(5) if n > 1]
        raise SystemExit(
            f"{label}: {KEY} is NOT unique ({len(ks) - len(set(ks))} duplicate rows); "
            f"first dups: {dup}"
        )
ft_pos = {k: i for i, k in enumerate(ft_keys)}
sc_take, ft_take = [], []
for i, k in enumerate(sc_keys):
    j = ft_pos.get(k)
    if j is not None:
        sc_take.append(i)
        ft_take.append(j)
matched = set(sc_keys) & set(ft_pos)
drop_sc = [k for k in sc_keys if k not in ft_pos]
drop_ft = [k for k in ft_keys if k not in matched]
print(
    f"[join] scores {sc.num_rows} rows x features {ft.num_rows} rows "
    f"-> {len(sc_take)} matched on {KEY}",
    flush=True,
)
for label, dropped in (("scores", drop_sc), ("features", drop_ft)):
    if dropped:
        print(
            f"[join] !! DROPPED {len(dropped)} {label}-only rows (no counterpart); "
            f"first {min(5, len(dropped))}: {dropped[:5]}",
            flush=True,
        )
if not sc_take:
    raise SystemExit(f"join on {KEY} matched ZERO rows — wrong harvest dir or key?")
idx_sc = pa.array(sc_take, type=pa.int64())
idx_ft = pa.array(ft_take, type=pa.int64())
sc = sc.take(idx_sc)
ft = ft.take(idx_ft)
# The remaining ID columns must AGREE on every joined row — they are redundant
# with the key, so a disagreement means the two write-backs disagree about the
# cell itself, not merely about row order.
for c in ID:
    if not sc[c].equals(ft[c]):
        a, b = sc[c].to_pylist(), ft[c].to_pylist()
        bad = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        raise SystemExit(
            f"ID column {c} DISAGREES on joined row {bad} "
            f"(key {sc_keys[sc_take[bad]]}): scores={a[bad]!r} features={b[bad]!r}"
        )

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
