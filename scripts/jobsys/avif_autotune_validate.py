#!/usr/bin/env python3
"""Validate a baked AVIF autotune picker on the eval8 leg, and emit the two
runtime LUTs `zenavif::auto_tune` consumes.

Computes NOTHING a canonical owner already owns: the forward pass is
`zenanalyze/zentrain/tools/_predict_lib.forward`, the rank statistic is
`zenstats` via `zensim/scripts/lib/zen_stats.py`, and the measured per-image
backend winner is read from the backend-race wave's own
`backend_per_image.parquet` (never re-derived).

WHAT IT REPORTS
  1. REGRET vs the per-row oracle — for each (image, size_class, target_zq) on
     the held-out `eval8` leg: the picker's cell vs the cheapest cell that
     actually reaches the target.  Broken out by coarse content class, by size
     class, and by target band.  `oracle` here is the best cell IN THE VIEW,
     which is an upper bound on achievable, not on possible.
  2. BACKEND-PICK ACCURACY vs the measured per-image winner
     (`backend_per_image.tsv::winner_full`, a BD-rate over the pooled
     best-over-speeds frontier).  Only defined where the picker can choose a
     backend at all — the cross-size-verified cell set cannot (zenrav1e has no
     native leg), and that is reported as NOT-APPLICABLE, never as a zero.
  3. SPEED-PREDICTION ERROR — the time head against the `encode_ms` column it
     was trained on.  That column is MODELLED (§encode_ms of the view manifest),
     so this is agreement with a model, and the instrument's own per-leg
     self-prediction error (-21.2% .. +13.9%) is the floor beneath it.

WHAT IT EMITS
  `<bake>_cellmap.json`      cell index -> the decoded knob tuple
  `<bake>_encode_ms_lut.json`   zenavif `EncodeMsLut` schema: median ms/MPx per
                                (cell, size_class)
  `<bake>_quality_lut.json`     zenavif `QualityLut` schema: per (cell,
                                target_zq) median q, -1 = unreachable
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/lilith/work/zen/zenanalyze/zentrain/tools")
from _predict_lib import forward  # noqa: E402

sys.path.insert(0, "/home/lilith/work/zen/zensim/scripts")
from lib.zen_stats import srocc  # noqa: E402

BACKEND_PER_IMAGE = "/mnt/v/output/avif-backend-2026-09-03/parquet/backend_per_image.parquet"
SIZE_CANON = ["tiny", "small", "medium", "large"]


def engineer(model, feat_rows, size_class, w, h, zq):
    """The trainer's `xe` layout, reconstructed FROM THE MODEL's own
    `extra_axes` so it cannot silently drift from what was trained."""
    fc = model["feat_cols"]
    axes = model["extra_axes"]
    size_axes = [a[len("size_"):] for a in axes if a.startswith("size_")]
    f = np.asarray([float(feat_rows[c]) for c in fc], dtype=np.float32)
    oh = np.zeros(len(size_axes), dtype=np.float32)
    # A size class outside the modelled grid maps to the NEAREST modelled class
    # (the trainer's `_scope_size_classes` docstring prescribes exactly this).
    if size_class in size_axes:
        oh[size_axes.index(size_class)] = 1.0
    else:
        want = SIZE_CANON.index(size_class)
        near = min(size_axes, key=lambda s: abs(SIZE_CANON.index(s) - want))
        oh[size_axes.index(near)] = 1.0
    log_px = math.log(max(1, w * h))
    zn = zq / 100.0
    xe = np.concatenate([
        f, oh,
        np.array([log_px, log_px * log_px, zn, zn * zn, zn * log_px], dtype=np.float32),
        zn * f, np.array([0.0], dtype=np.float32),
    ])
    if xe.shape[0] != model["n_inputs"]:
        raise SystemExit(
            f"engineered width {xe.shape[0]} != model n_inputs {model['n_inputs']} — "
            "the model's extra_axes and this layout disagree")
    return xe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the trainer's model JSON")
    ap.add_argument("--pareto", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = json.load(open(args.model))
    hm = model["hybrid_heads_manifest"]
    cells = [c["label"] for c in hm["cells"]]
    cell_axes = {c["label"]: c for c in hm["cells"]}
    n_cells = int(model["n_cells"])
    layout = hm["output_layout"]
    zq_targets = sorted(int(z) for z in model["reach_safety"]["by_zq"])

    # ---- data
    t = pq.read_table(args.pareto)
    cols = ["image_path", "size_class", "width", "height", "config_name", "q",
            "bytes", "ssim2", "encode_ms", "leg", "coarse_class", "backend",
            "speed", "corpus", "image"]
    d = {c: t[c].to_pylist() for c in cols}
    n = t.num_rows
    feats = {}
    with open(args.features) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            v = dict(zip(hdr, line.rstrip("\n").split("\t")))
            feats[v["image_path"]] = v

    # cell index by config_name (the trainer's cell label is a normalised form)
    cell_of = {lbl: i for i, lbl in enumerate(cells)}

    def cell_label(cfg):
        b, sp, *rest = cfg.split("-")
        bd = "bd10" if rest and rest[-1] == "bd10" else "bd8"
        if bd == "bd10":
            rest = rest[:-1]
        return f"{b}_{sp}_{'-'.join(rest) if rest else 'default'}_{bd}"

    # ---- ladders: (image_path, size_class) -> cell -> [(ssim2, bytes, q, ms)]
    lad = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for i in range(n):
        lbl = cell_label(d["config_name"][i])
        ci = cell_of.get(lbl)
        if ci is None:
            continue
        k = (d["image_path"][i], d["size_class"][i])
        lad[k][ci].append((d["ssim2"][i], d["bytes"][i], d["q"][i], d["encode_ms"][i]))
        meta[k] = (d["width"][i], d["height"][i], d["leg"][i], d["coarse_class"][i],
                   d["corpus"][i], d["image"][i])

    # ---- LUTs (from the whole view; a LUT is a lookup table, not a fit)
    def _sc(w, h):
        nn = w * h
        return "tiny" if nn < 64 * 64 else "small" if nn < 256 * 256 else \
            "medium" if nn < 1024 * 1024 else "large"

    ms_per_mpx = collections.defaultdict(list)
    q_at = collections.defaultdict(list)
    for k, per in lad.items():
        w, h, leg, cls, corpus, img = meta[k]
        sc = _sc(w, h)
        mpx = w * h / 1e6
        for ci, pts in per.items():
            for s, b, q, ms in pts:
                if ms is not None:
                    ms_per_mpx[(ci, sc)].append(ms / mpx)
            for tz in zq_targets:
                ok = [q for s, b, q, _m in pts if s >= tz]
                if ok:
                    q_at[(ci, tz)].append(min(ok))
    encode_ms_lut = {"median_ms_per_mpx": {}}
    for ci in range(n_cells):
        row = {}
        for sc in SIZE_CANON:
            v = ms_per_mpx.get((ci, sc))
            if v:
                row[sc] = round(statistics.median(v), 3)
        if row:
            # fill unmeasured size classes from the nearest measured one, and say so
            for sc in SIZE_CANON:
                if sc not in row:
                    near = min(row, key=lambda s: abs(SIZE_CANON.index(s) - SIZE_CANON.index(sc)))
                    row[sc] = row[near]
            encode_ms_lut["median_ms_per_mpx"][f"cell{ci}"] = row
    encode_ms_lut["_note"] = (
        "MODELLED, not measured: encode_ms in the training view is derived from the "
        "2026-09-03 speed instrument's alpha+beta fits (single-threaded wall time, "
        "q45-anchored, per-source fits on 5 of 32 sources). Size classes with no "
        "corpus coverage are filled from the nearest measured class.")
    encode_ms_lut["_cells"] = list(cells)
    quality_lut = {
        "schema_version": 1,
        "source": f"/mnt/v/zen/avif-autotune-2026-09-04 :: {args.name}",
        "cells": list(cells),
        "target_zqs": [int(z) for z in zq_targets],
        "median_q": [[int(statistics.median(q_at[(ci, tz)])) if q_at.get((ci, tz)) else -1
                      for tz in zq_targets] for ci in range(n_cells)],
        "_metric": "ssim2 (NOT zensim) — the only corpus-wide scalar response in the DOE",
    }
    cellmap = {"cells": [], "_grammar": "<backend>_s<speed>_<knobs|default>_bd<8|10>",
               "_output_layout": layout}
    for i, lbl in enumerate(cells):
        a = cell_axes[lbl]
        cellmap["cells"].append({
            "index": i, "label": lbl,
            "backend": {"svt": "zenav1-svt", "rav": "zenrav1e"}[a["backend"]],
            "speed": int(a["speed"][1:]),
            "knobs": [] if a["knobs"] == "default" else a["knobs"].split("-"),
            "bit_depth": 10 if a["bd"] == "bd10" else 8,
            "chroma": "4:2:0" if a["backend"] == "svt" else "4:4:4",
            "chroma_is_derived_from_backend": True,
        })

    # ---- inference on the eval8 leg
    rows = []
    for k, per in lad.items():
        w, h, leg, cls, corpus, img = meta[k]
        if leg != "eval8":
            continue
        fr = feats[k[0]]
        for tz in zq_targets:
            reach = {}
            for ci, pts in per.items():
                ok = [(b, ms) for s, b, _q, ms in pts if s >= tz]
                if ok:
                    reach[ci] = min(ok)
            if len(reach) < 2:
                continue
            xe = engineer(model, fr, k[1], w, h, tz)
            y = forward(model, xe.reshape(1, -1))[0]
            bl = y[:n_cells]
            order = np.argsort(bl)
            pick = int(order[0])
            if pick not in reach:
                pick = next((int(c) for c in order if int(c) in reach), None)
                if pick is None:
                    continue
            best = min(reach.values())[0]
            rows.append({
                "image": img, "corpus": corpus, "size_class": k[1], "class": cls,
                "zq": tz, "pick": pick, "pick_label": cells[pick],
                "pick_bytes": reach[pick][0], "oracle_bytes": best,
                "regret": reach[pick][0] / best - 1.0,
                "pick_backend": cellmap["cells"][pick]["backend"],
                "n_reachable": len(reach),
                "pred_ms": (float(np.exp(y[layout["time_log"][0] + pick]))
                            if "time_log" in layout else None),
                "label_ms": reach[pick][1],
            })

    def summarise(sub):
        if not sub:
            return None
        r = sorted(x["regret"] for x in sub)
        return {"n": len(r), "mean": round(float(np.mean(r)), 4),
                "p50": round(r[len(r) // 2], 4),
                "p90": round(r[int(0.9 * (len(r) - 1))], 4),
                "max": round(r[-1], 4),
                "frac_optimal": round(sum(1 for x in sub if x["regret"] < 1e-9) / len(sub), 4)}

    rep = {"bake": args.name, "n_cells": n_cells,
           "eval_leg": "eval8 (origins ending 8 — the registered even-only holdout)",
           "regret_overall": summarise(rows)}
    for key, fn in (("regret_by_class", lambda x: x["class"]),
                    ("regret_by_size", lambda x: x["size_class"]),
                    ("regret_by_corpus", lambda x: x["corpus"]),
                    ("regret_by_zq_band", lambda x: ("q<30" if x["zq"] < 30 else
                                                     "30-70" if x["zq"] < 70 else "70+"))):
        g = collections.defaultdict(list)
        for x in rows:
            g[fn(x)].append(x)
        rep[key] = {k: summarise(v) for k, v in sorted(g.items())}

    # ---- backend-pick accuracy
    backends_pickable = {c["backend"] for c in cellmap["cells"]}
    if len(backends_pickable) < 2:
        rep["backend_pick"] = {
            "status": "NOT-APPLICABLE",
            "reason": ("this cell set contains only "
                       f"{sorted(backends_pickable)} — a backend decision is not in its "
                       "output space, so accuracy is undefined (not zero)")}
    else:
        bt = pq.read_table(BACKEND_PER_IMAGE)
        win = dict(zip(bt["image"].to_pylist(), bt["winner_full"].to_pylist()))
        winb = dict(zip(bt["image"].to_pylist(), bt["winner_banded"].to_pylist()))
        # ⛔ THE REFERENCE IS A BUDGET-CORPUS VERDICT. Its `pixels` column reads
        # 1,048,576 for every cropped reference, i.e. the 1024^2 crop — the
        # `brsdr` (zenrav1e) arm only ever ran on the budget corpus, so there is
        # no native-size cross-backend measurement to compare a native pick
        # against. Scoring native rows against it would apply a crop verdict to
        # different pixels, which is precisely the corpus collision the DOE's
        # own gap-fill header documents. Native rows are counted as
        # NOT-COMPARABLE, never as misses.
        ref_px = dict(zip(bt["image"].to_pylist(), bt["pixels"].to_pylist()))
        NAME = {"zenrav1e": "zenrav1e", "svt": "zenav1-svt"}
        agree = tot = agree_b = 0
        n_native_skipped = 0
        per_img = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            if x["corpus"] != "budget":
                n_native_skipped += 1
                continue
            w = win.get(x["image"])
            if w is None:
                continue
            tot += 1
            if NAME[w] == x["pick_backend"]:
                agree += 1
                per_img[x["image"]][0] += 1
            per_img[x["image"]][1] += 1
            if NAME[winb[x["image"]]] == x["pick_backend"]:
                agree_b += 1
        # Baselines. A per-image binary decision must beat the trivial
        # always-majority rule or it is not a decision, it is a constant with
        # extra steps.
        n_by_winner = collections.Counter()
        for x in rows:
            if x["corpus"] != "budget":
                continue
            w = win.get(x["image"])
            if w is not None:
                n_by_winner[NAME[w]] += 1
        majority = max(n_by_winner.values()) / tot if tot else None
        rep["backend_pick"] = {
            "status": "MEASURED",
            "baseline_always_majority": round(majority, 4) if majority else None,
            "baseline_per_backend": {k: round(v / tot, 4) for k, v in n_by_winner.items()},
            "beats_majority_baseline": (agree / tot > majority) if tot and majority else None,
            "reference": (f"{BACKEND_PER_IMAGE}::winner_full — per-image BD-rate over the "
                          "pooled best-over-speeds frontier, measured on the 1024^2 "
                          "BUDGET corpus only"),
            "scope": "budget-corpus rows only",
            "n_decisions": tot,
            "n_native_rows_not_comparable": n_native_skipped,
            "why_native_excluded": ("the zenrav1e arm (brsdr) ran only on the budget "
                                    "corpus, so no native-size cross-backend "
                                    "measurement exists to compare against"),
            "agreement_full_ladder": round(agree / tot, 4) if tot else None,
            "agreement_banded_30_95": round(agree_b / tot, 4) if tot else None,
            "per_image": {k: {"agree": v[0], "n": v[1],
                              "measured_winner": win.get(k)} for k, v in sorted(per_img.items())},
            "CAVEAT": ("the reference is itself a (backend x chroma) verdict — svt is 4:2:0 "
                       "and zenrav1e 4:4:4 in every cell of this corpus, with no arm that "
                       "splits them"),
        }

    # ---- speed head
    pm = [(x["pred_ms"], x["label_ms"]) for x in rows
          if x["pred_ms"] is not None and x["label_ms"]]
    if pm:
        rel = [abs(p - l) / l for p, l in pm]
        rel.sort()
        rep["speed_head"] = {
            "n": len(pm),
            "rel_abs_err_p50": round(rel[len(rel) // 2], 4),
            "rel_abs_err_p90": round(rel[int(0.9 * (len(rel) - 1))], 4),
            "srocc_pred_vs_label": round(float(srocc([p for p, _ in pm], [l for _, l in pm])), 4),
            "LABEL_IS_MODELLED": ("the target is the view's modelled `encode_ms`, not a "
                                  "measurement; the speed instrument's own per-leg "
                                  "self-prediction error (-21.2%..+13.9%) is the floor"),
        }
    else:
        rep["speed_head"] = {"status": "ABSENT (no time head in this bake)"}

    (out / f"{args.name}_cellmap.json").write_text(json.dumps(cellmap, indent=2) + "\n")
    (out / f"{args.name}_encode_ms_lut.json").write_text(json.dumps(encode_ms_lut, indent=2) + "\n")
    (out / f"{args.name}_quality_lut.json").write_text(json.dumps(quality_lut, indent=2) + "\n")
    (out / f"{args.name}_validation.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "backend_pick"}, indent=2))
    if rep.get("backend_pick"):
        b = dict(rep["backend_pick"])
        b.pop("per_image", None)
        print(json.dumps({"backend_pick": b}, indent=2))


if __name__ == "__main__":
    main()
