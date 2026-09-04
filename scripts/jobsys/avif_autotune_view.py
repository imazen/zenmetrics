#!/usr/bin/env python3
"""Build THE canonical AVIF autotune training view from every scored AVIF DOE run.

One view, no new encodes. Reads the harvested per-cell parquets that
`avifdoe_harvest.py` produced for each wave, joins them into era-labelled thin
sidecars, and emits the `(PARETO, FEATURES)` pair that
`zenanalyze/zentrain/tools/train_hybrid.py` consumes.

WHY A NEW SCRIPT AND NOT AN EXTENSION
-------------------------------------
`avifdoe_harvest.py` owns blob->cell harvesting and `avifdoe_*_analyze.py` own
BD-rate/frontier statistics; neither owns "union every wave into one trainable
view". This is that step. It computes NO statistic that those files own — the
frontier / oracle helpers are IMPORTED from `avifdoe_brem_analyze.py`.

THE FIVE THINGS THAT BITE, ALL HANDLED HERE
-------------------------------------------
1. **Row identity is the CELL `(corpus, image, arm, q)`, never `encode_sha`.**
   Encoding is content-addressed: 111,400 harvested rows collapse to 63,148
   distinct bitstreams. Keying on the sha would drop one of every byte-identical
   pair and destroy the inertness signal. `encode_sha` is carried as a
   many-to-one ATTRIBUTE and used for (a) cross-era merging and (b) the measured
   inert-arm census.
2. **The two corpora share all 32 filenames and 13 are different pixels.**
   `image` alone is ambiguous. The run->corpus map below is transcribed from
   `avifdoe_declare.sh` (the declaring owner) and every row carries `corpus`.
   Features are per (corpus, image): the budget rows read the G-CROP
   re-extracted crop features, the native rows read the native table. Verified:
   the 19 pixel-shared images have byte-identical feature rows in both tables.
3. **Dead knobs are MEASURED here, not inherited from a doc.** Stage-A §3 and
   the era-delta record disagree about `scm3`'s liveness per speed, so this tool
   computes, per `(corpus, speed, devs)` arm, the fraction of its cells whose
   `encode_sha` equals the same-(corpus, speed) control's sha at the same
   `(image, q)`. `frac_inert == 1.0` => the arm is a relabelled control and is
   dropped. Pair arms that alias a single arm are collapsed to that single.
4. **Era labels per row; cross-era rows are merged only where the BYTES are
   identical.** Stage-A ran the old `zenav1-svt` pin, the era-delta wave the new
   one; 13,248 cell-pairs were proven byte-identical. This tool re-derives that
   itself: two rows of the same cell identity from different eras are one
   population iff their `encode_sha` matches, and a cell whose eras DISAGREE is
   dropped from the pooled view and reported.
5. **All 32 references are EVEN-origin by construction** (the corpus was
   k-means-selected under `--parity 0`), so the canonical `{1,3,5}` validate /
   `{7,9}` test buckets are STRUCTURALLY EMPTY. This tool applies the registered
   even-only sub-split (`DATA_SPLITS.md` line 158, the `avifgen-2026-08-06`
   precedent, owner `avifgen_training_views.py`): origins ending `{0,2,4,6}` =
   train, origins ending `{8}` = `eval8`, the leg-side holdout that is never
   trained on. It hard-errors on any odd-origin row.

USAGE
-----
    python3 avif_autotune_view.py --out /mnt/v/zen/avif-autotune-2026-09-04
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avifdoe_brem_analyze import interp_bytes, pooled_front  # noqa: E402
from avifdoe_stagea_analyze import COARSE, frontier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "picker"))
from origin_split import origin_id, split_of  # noqa: E402

# ---------------------------------------------------------------- inputs

BUDGET_SOURCES = "/mnt/v/output/avif-doe-1024-2026-09-01/sources"
NATIVE_SOURCES = "/mnt/v/output/avifsvt-subsample-2026-09-01/sources"

BUDGET_FEATURES = "/mnt/v/output/avif-doe-1024-2026-09-01/_gcrop/budget_features.tsv"
NATIVE_FEATURES = "/mnt/v/output/avifsvt-subsample-2026-09-01/_natfeat/native_features.tsv"
CROP_MANIFEST = "/mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv"
NATIVE_PICKS = "/mnt/v/output/avifsvt-subsample-2026-09-01/avif_subsample_picks_2026-09-01.tsv"

CHROMA_CENSUS = "/mnt/v/output/avif-backend-2026-09-03/chroma_census_2026-09-03.parquet"
SPEED_AB = "/mnt/v/output/avif-speed-instrument-2026-09-03/speed_alpha_beta.tsv"
SPEED_AB_SRC = "/mnt/v/output/avif-speed-instrument-2026-09-03/speed_alpha_beta_per_source.tsv"
SPEED_S1A = "/mnt/v/output/avif-speed-instrument-2026-09-03/s1a_pass3.tsv"
BACKEND_PER_IMAGE = "/mnt/v/output/avif-backend-2026-09-03/parquet/backend_per_image.parquet"

# source parquet -> era label.  Era = the (zenavif, zenav1-svt, zenrav1e) pin
# triple the bitstreams were produced under; transcribed from each wave's
# `_MANIFEST.json` / DATA_PROVENANCE entry.  NEVER pool across eras except
# through the byte-identity merge below.
SOURCES = [
    # (tag, path, era, has_butteraugli)
    ("stageA", "/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet", "2026-09-01", True),
    ("b6", "/mnt/v/output/zensim-avifdoe-b6/b6_scored_2026-09-02.parquet", "2026-09-01", False),
    ("naive", "/mnt/v/output/zensim-avifdoe-b6/naive_native_control_scored_2026-09-02.parquet", "2026-09-01", True),
    ("eradelta", "/mnt/v/output/zensim-avifdoe-eradelta/eradelta_scored_2026-09-03.parquet", "2026-09-03", False),
    ("c1ag", "/mnt/v/output/zensim-avifdoe-eradelta/c1_ag_scored_2026-09-03.parquet", "2026-09-03", True),
    ("br", "/mnt/v/output/avif-backend-2026-09-03/br_scored_2026-09-03.parquet", "2026-09-03", False),
]
HDR_SOURCE = ("t2", "/mnt/v/output/zensim-avifhbd-t2/t2_scored_2026-09-03.parquet", "2026-09-03")

# run -> corpus.  AUTHORITATIVE, transcribed from
# `zenmetrics/scripts/jobsys/avifdoe_declare.sh` (the declaring owner):
# a0r/a1/a2 use $SOURCES (budget), ag/b6/naive/c1/t1d/brnat use $NATIVE_SOURCES,
# brsdr uses $SOURCES.  Getting this wrong is the silent-garbage hazard the
# gapfill script's header documents.
RUN_CORPUS = {
    "a0r": "budget", "a1": "budget", "a2": "budget", "ag": "native",
    "b6": "native", "naive": "native", "b1": "budget", "c1": "native",
    "t1d": "native", "brnat": "native", "brsdr": "budget",
}
# c1_ag_scored carries the SUPERSEDED t1d run (24 of its 96 cells encoded
# 4.5-7 h before either zenav1-svt#18 fix landed; 8 of 32 images ran the broken
# multi-tile path).  Replacement = eradelta c1, same plan/corpus/ladder.
EXCLUDED_RUNS = {"t1d": "superseded by avifdoe-svt-eradelta-c1-20260903 (pre-#18 corruption, avif_eradelta_analysis_2026-09-03.md §4.1)"}

RE_SVT = re.compile(r"^s(\d+)-svt-420(?:-(.*))?$")
RE_RAV = re.compile(r"^s(\d+)-zenravif$")
RE_HDR_SVT = re.compile(r"^p(\d+)-zenav1svt-hdr10$")
RE_HDR_RAV = re.compile(r"^s(\d+)-zenavif-hdr10$")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_commit():
    """This checkout's commit. A `jj workspace` has no `.git`, so fall back to
    the colocated primary repo (`jj workspace list` names it `default: .`)."""
    here = Path(__file__).resolve().parents[2]
    for d in (here, here.parent / here.name.split("--")[0]):
        try:
            r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True)
            return r.stdout.strip()
        except Exception:
            continue
    return None


def parse_arm(arm):
    """`arm` -> (backend, speed, knobs, bit_depth).  None backend = unparseable."""
    m = RE_SVT.match(arm)
    if m:
        devs = m.group(2) or ""
        toks = [t for t in devs.split("-") if t] if devs else []
        bd = 10 if "bd10" in toks else 8
        knobs = "-".join(sorted(t for t in toks if t != "bd10"))
        return ("svt", int(m.group(1)), knobs, bd)
    m = RE_RAV.match(arm)
    if m:
        return ("rav", int(m.group(1)), "", 8)
    return (None, None, None, None)


def parse_hdr_arm(arm):
    m = RE_HDR_SVT.match(arm)
    if m:
        return ("svt", int(m.group(1)), "", 10)
    m = RE_HDR_RAV.match(arm)
    if m:
        return ("rav", int(m.group(1)), "", 10)
    return (None, None, None, None)


def size_class(w, h):
    """zenavif runtime's own buckets (`auto_tune.rs::size_class_idx`)."""
    n = int(w) * int(h)
    if n < 64 * 64:
        return "tiny"
    if n < 256 * 256:
        return "small"
    if n < 1024 * 1024:
        return "medium"
    return "large"


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_cells(tag, path, era, has_butter):
    t = pq.read_table(path)
    cn = t.column_names
    n = t.num_rows
    col = {c: t[c].to_pylist() for c in cn}
    out = []
    for i in range(n):
        run = col["run"][i]
        speed = col["speed"][i] if "speed" in col else None
        if speed is None and col.get("dial_kind", [None] * n)[i] == "speed":
            speed = col["dial"][i]
        r = {
            "source_table": tag, "era": era, "run": run,
            "corpus": RUN_CORPUS.get(run), "image": col["image"][i],
            "q": col["q"][i], "arm": col["arm"][i], "devs": col.get("devs", [None] * n)[i],
            "declared_speed": speed, "encode_sha": col["encode_sha"][i],
            "bytes": col["bytes"][i], "ssim2": col["m_ssim2"][i],
            "metrics_present": col.get("metrics_present", [None] * n)[i],
        }
        for b in ("m_butteraugli", "m_butteraugli_max", "m_butteraugli_pnorm3"):
            r[b] = col[b][i] if has_butter and b in col else None
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="ssim2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"inputs": {}, "exclusions": [], "gates": {}}

    # ---------------------------------------------------------- 1. cells
    rows = []
    for tag, path, era, hb in SOURCES:
        got = read_cells(tag, path, era, hb)
        report["inputs"][tag] = {
            "path": path, "era": era, "rows_read": len(got),
            "sha256": sha256_file(path), "bytes": os.path.getsize(path),
        }
        rows.extend(got)
    n_read = len(rows)

    dropped_runs = collections.Counter()
    keep = []
    for r in rows:
        if r["run"] in EXCLUDED_RUNS:
            dropped_runs[r["run"]] += 1
            continue
        if r["corpus"] is None:
            dropped_runs["UNMAPPED:" + str(r["run"])] += 1
            continue
        keep.append(r)
    for k, v in dropped_runs.items():
        report["exclusions"].append({
            "rule": "excluded_run", "run": k, "rows": v,
            "reason": EXCLUDED_RUNS.get(k, "run has no corpus in the declare-owner map"),
        })
    rows = keep

    # parse arms
    unparsed = collections.Counter()
    for r in rows:
        b, sp, kn, bd = parse_arm(r["arm"])
        if b is None:
            unparsed[r["arm"]] += 1
        r["backend"], r["speed"], r["knobs"], r["bit_depth"] = b, sp, kn, bd
    if unparsed:
        raise SystemExit(f"unparseable arms (refusing to guess): {dict(unparsed)}")
    # the declared `speed` column must agree with the label wherever it is present
    mism = [r for r in rows if r["declared_speed"] is not None and r["declared_speed"] != r["speed"]]
    report["gates"]["arm_label_speed_agrees_with_declared_column"] = {
        "checked": sum(1 for r in rows if r["declared_speed"] is not None),
        "mismatches": len(mism), "pass": not mism,
    }
    if mism:
        raise SystemExit(f"speed label/column disagreement on {len(mism)} rows")

    # ------------------------------------------------- 2. era merge on bytes
    bycell = collections.defaultdict(list)
    for r in rows:
        bycell[(r["corpus"], r["image"], r["arm"], r["q"])].append(r)
    merged, era_conflicts = [], []
    n_multi_era = n_agree = 0
    for key, group in bycell.items():
        eras = {g["era"] for g in group}
        shas = {g["encode_sha"] for g in group}
        if len(eras) > 1:
            n_multi_era += 1
            if len(shas) > 1:
                era_conflicts.append({
                    "cell": list(key), "eras": sorted(eras), "shas": sorted(shas)})
                continue
            n_agree += 1
        # one canonical row per cell: prefer a row that carries butteraugli,
        # then the newest era; every field but the metric extras is identical
        # by construction (same bytes => same bytes/score).
        group.sort(key=lambda g: (g["m_butteraugli"] is None, g["era"]), reverse=False)
        base = dict(group[0])
        base["eras"] = "|".join(sorted(eras))
        base["n_source_rows"] = len(group)
        base["source_tables"] = "|".join(sorted({g["source_table"] for g in group}))
        for b in ("m_butteraugli", "m_butteraugli_max", "m_butteraugli_pnorm3"):
            base[b] = next((g[b] for g in group if g[b] is not None), None)
        if base["ssim2"] is None:
            base["ssim2"] = next((g["ssim2"] for g in group if g["ssim2"] is not None), None)
        merged.append(base)
    report["gates"]["cross_era_cells"] = {
        "cells_seen_in_more_than_one_era": n_multi_era,
        "byte_identical_across_eras": n_agree,
        "conflicts_dropped": len(era_conflicts),
        "pass": len(era_conflicts) == 0,
    }
    if era_conflicts:
        report["exclusions"].append({
            "rule": "cross_era_byte_conflict", "rows": len(era_conflicts),
            "reason": "same cell identity, different bytes in two eras — not one population",
            "sample": era_conflicts[:20]})
    rows = merged

    # drop cells with no quality label (unscored stragglers)
    unscored = [r for r in rows if r["ssim2"] is None]
    rows = [r for r in rows if r["ssim2"] is not None]
    if unscored:
        report["exclusions"].append({
            "rule": "unscored_cell", "rows": len(unscored),
            "reason": "no m_ssim2 (metrics_present null) — never labelled, not a failure"})

    # ------------------------------------ 3. MEASURED inert-arm census
    # For each (corpus, speed, bd, knobs) arm, what fraction of its cells is
    # byte-identical to the same-(corpus, speed, bd) CONTROL at the same
    # (image, q)?  frac == 1.0 => the arm is the control relabelled.
    ctl = {}
    for r in rows:
        if r["knobs"] == "" and r["backend"] == "svt":
            ctl[(r["corpus"], r["speed"], r["bit_depth"], r["image"], r["q"])] = r["encode_sha"]
    census = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["backend"] != "svt" or r["knobs"] == "":
            continue
        c = ctl.get((r["corpus"], r["speed"], r["bit_depth"], r["image"], r["q"]))
        if c is None:
            continue
        key = (r["corpus"], r["speed"], r["bit_depth"], r["knobs"])
        census[key][1] += 1
        if c == r["encode_sha"]:
            census[key][0] += 1
    inert_rows = []
    inert_arms = set()
    for (corpus, sp, bd, kn), (ident, tot) in sorted(census.items()):
        frac = ident / tot if tot else float("nan")
        inert_rows.append({"corpus": corpus, "speed": sp, "bit_depth": bd, "knobs": kn,
                           "n_cells": tot, "n_identical_to_control": ident,
                           "frac_inert": frac})
        if tot and ident == tot:
            inert_arms.add((corpus, sp, bd, kn))
    # alias collapse: a multi-knob arm whose bytes equal a SINGLE-knob arm's on
    # every shared cell carries no information beyond that single.
    single_sha = collections.defaultdict(dict)
    for r in rows:
        if r["backend"] == "svt" and r["knobs"] and "-" not in r["knobs"]:
            single_sha[(r["corpus"], r["speed"], r["bit_depth"], r["knobs"])][(r["image"], r["q"])] = r["encode_sha"]
    pair_map = collections.defaultdict(list)
    for r in rows:
        if r["backend"] == "svt" and "-" in (r["knobs"] or ""):
            pair_map[(r["corpus"], r["speed"], r["bit_depth"], r["knobs"])].append(r)
    alias_rows, alias_arms = [], {}
    for key, group in sorted(pair_map.items()):
        corpus, sp, bd, kn = key
        for tok in kn.split("-"):
            tgt = single_sha.get((corpus, sp, bd, tok))
            if not tgt:
                continue
            shared = [g for g in group if (g["image"], g["q"]) in tgt]
            if not shared:
                continue
            ident = sum(1 for g in shared if tgt[(g["image"], g["q"])] == g["encode_sha"])
            if ident == len(shared):
                alias_rows.append({"corpus": corpus, "speed": sp, "bit_depth": bd,
                                   "knobs": kn, "aliases": tok, "n_cells": len(shared)})
                alias_arms[key] = tok
                break
    report["gates"]["inert_arm_census"] = {
        "arms_measured": len(census),
        "arms_fully_inert": len(inert_arms),
        "pair_arms_aliasing_a_single": len(alias_arms),
    }

    before = len(rows)
    rows = [r for r in rows
            if (r["corpus"], r["speed"], r["bit_depth"], r["knobs"]) not in inert_arms
            and (r["corpus"], r["speed"], r["bit_depth"], r["knobs"]) not in alias_arms]
    report["exclusions"].append({
        "rule": "measured_inert_or_aliased_arm", "rows": before - len(rows),
        "reason": ("arm's bitstream is byte-identical to its control (or to a single-knob "
                   "arm) on 100% of shared cells — a label with no causal content "
                   "(zenav1-svt#17). MEASURED here, not inherited."),
        "n_inert_arms": len(inert_arms), "n_alias_arms": len(alias_arms)})

    # speed-alias collapse: svt presets that are byte-identical to a lower speed
    sp_sha = collections.defaultdict(dict)
    for r in rows:
        if r["backend"] == "svt" and r["knobs"] == "":
            sp_sha[(r["corpus"], r["speed"], r["bit_depth"])][(r["image"], r["q"])] = r["encode_sha"]
    speed_alias, speed_alias_rows = {}, []
    for (corpus, sp, bd), m in sorted(sp_sha.items()):
        for lo in range(1, sp):
            other = sp_sha.get((corpus, lo, bd))
            if not other:
                continue
            shared = set(m) & set(other)
            if len(shared) < 32:
                continue
            if all(m[k] == other[k] for k in shared):
                speed_alias[(corpus, sp, bd)] = lo
                speed_alias_rows.append({"corpus": corpus, "speed": sp, "bit_depth": bd,
                                         "aliases_speed": lo, "n_shared_cells": len(shared)})
                break
    before = len(rows)
    rows = [r for r in rows if not (r["backend"] == "svt"
            and (r["corpus"], r["speed"], r["bit_depth"]) in speed_alias)]
    report["exclusions"].append({
        "rule": "measured_speed_alias", "rows": before - len(rows),
        "reason": "svt preset byte-identical to a lower preset on every shared cell — "
                  "the dial saturates; the higher label measures the lower encoder again",
        "aliases": speed_alias_rows})

    # ------------------------------------------------- 4. per-image metadata
    crop = {r["corpus_key"]: r for r in load_tsv(CROP_MANIFEST)}
    picks = {r["corpus_key"]: r for r in load_tsv(NATIVE_PICKS)}
    feats = {}
    featcols = None
    for corpus, path in (("budget", BUDGET_FEATURES), ("native", NATIVE_FEATURES)):
        recs = load_tsv(path)
        cols = [c for c in recs[0] if c.startswith("feat_")]
        if featcols is None:
            featcols = cols
        elif cols != featcols:
            raise SystemExit(f"feature column set differs between corpora: {path}")
        for rec in recs:
            feats[(corpus, os.path.basename(rec["image_path"]))] = rec
    # GATE: the 19 pixel-shared images must have byte-identical feature rows
    shared_ok = shared_n = 0
    for key in {k[1] for k in feats}:
        b, n = feats.get(("budget", key)), feats.get(("native", key))
        if not b or not n or b["image_sha"] != n["image_sha"]:
            continue
        shared_n += 1
        if all(b[c] == n[c] for c in featcols):
            shared_ok += 1
    report["gates"]["pixel_shared_images_have_identical_features"] = {
        "n_pixel_shared": shared_n, "n_identical": shared_ok, "pass": shared_ok == shared_n}
    if shared_ok != shared_n:
        raise SystemExit("feature drift between corpora on pixel-identical images")

    # ------------------------------------------------- 5. split (even-only)
    # DATA_SPLITS.md line 158 (avifgen-2026-08-06 precedent, owner
    # avifgen_training_views.py): the corpus is even-origin by construction, so
    # {1,3,5}/{7,9} are structurally empty.  {0,2,4,6} = train, {8} = eval8.
    def leg_of(image):
        s = split_of(image)
        if s != "train":
            raise SystemExit(
                f"{image}: canonical split_of == {s!r}, not 'train'. This corpus is "
                "even-origin BY CONSTRUCTION (--parity 0 k-means selection); an odd "
                "origin here means the corpus changed and the even-only sub-split no "
                "longer applies.")
        return "eval8" if origin_id(image)[-1] == "8" else "train"

    for r in rows:
        r["leg"] = leg_of(r["image"])
        fr = feats[(r["corpus"], r["image"])]
        r["width"], r["height"] = int(fr["width"]), int(fr["height"])
        r["size_class"] = size_class(r["width"], r["height"])
        r["content_class"] = fr["content_class"]
        r["coarse_class"] = COARSE.get(fr["content_class"], "other")
        r["origin_id"] = origin_id(r["image"])
        r["transform"] = crop[r["image"]]["transform"] if r["corpus"] == "budget" else "native-full"
        # image_path must distinguish the two corpora: same filename, 13 differ
        r["image_path"] = f"{r['corpus']}/{r['image']}"
        r["image_sha"] = fr["image_sha"]

    legs = collections.Counter(r["leg"] for r in rows)
    train_o = {r["origin_id"] for r in rows if r["leg"] == "train"}
    eval_o = {r["origin_id"] for r in rows if r["leg"] == "eval8"}
    report["gates"]["split"] = {
        "rule": "even-only sub-split: {0,2,4,6}=train, {8}=eval8 "
                "(DATA_SPLITS.md L158, avifgen-2026-08-06 precedent)",
        "rows_train": legs["train"], "rows_eval8": legs["eval8"],
        "origins_train": len(train_o), "origins_eval8": len(eval_o),
        "origin_overlap": sorted(train_o & eval_o),
        "n_rows_with_canonical_val_or_test_origin": 0,
        "pass": not (train_o & eval_o),
    }

    # ------------------------------------------------- 6. chroma (measured)
    cc = pq.read_table(CHROMA_CENSUS).to_pylist()
    by_sha = {c["encode_sha"]: c for c in cc}
    seen = collections.Counter()
    for r in rows:
        m = by_sha.get(r["encode_sha"])
        if m:
            r["chroma_measured"] = m["chroma"]
            r["seq_profile"] = m["seq_profile"]
            seen[(r["backend"], m["chroma"])] += 1
        else:
            r["chroma_measured"] = None
            r["seq_profile"] = None
    # backend -> chroma rule, derived from the census, applied where unmeasured
    rule = {}
    for (b, ch), n in seen.items():
        rule.setdefault(b, collections.Counter())[ch] = n
    chroma_rule = {b: c.most_common(1)[0][0] for b, c in rule.items()}
    inconsistent = {b: dict(c) for b, c in rule.items() if len(c) > 1}
    for r in rows:
        r["chroma"] = r["chroma_measured"] or chroma_rule.get(r["backend"])
        r["chroma_source"] = "av1C_census" if r["chroma_measured"] else "backend_rule"
    report["gates"]["chroma"] = {
        "census_rows": len(cc), "cells_matched_to_census": sum(seen.values()),
        "backend_to_chroma_rule": chroma_rule,
        "backends_with_mixed_chroma_in_census": inconsistent,
        "CONFOUND": ("backend and chroma are PERFECTLY COLLINEAR in this data "
                     "(svt=4:2:0 only, zenrav1e=4:4:4 only, 0 exceptions over 1,114 "
                     "av1C boxes). No chroma knob is wired in the AVIF sweep path. "
                     "A model fitted here cannot separate them; `chroma` is emitted as a "
                     "DERIVED attribute of backend, never as an independent axis."),
        "pass": not inconsistent,
    }

    # ------------------------------------------------- 7. cell (config) axis
    for r in rows:
        r["config_name"] = "-".join(
            [r["backend"], f"s{r['speed']}"]
            + ([r["knobs"]] if r["knobs"] else [])
            + ([f"bd{r['bit_depth']}"] if r["bit_depth"] != 8 else []))

    # ------------------------------------------------- 8. speed model
    ab = {(r["backend"], int(r["speed"])): r for r in load_tsv(SPEED_AB)}
    ab_src = {(r["backend"], int(r["speed"]), r["source"]): r for r in load_tsv(SPEED_AB_SRC)}
    BK = {"svt": "svt-rs", "rav": "zenravif"}
    n_src_fit = n_pooled = 0
    for r in rows:
        b = BK[r["backend"]]
        mp = r["width"] * r["height"] / 1e6
        s = ab_src.get((b, r["speed"], r["origin_id"]))
        if s:
            r["encode_ms"] = max(float(s["alpha_ms"]) + float(s["beta_ms_per_mp"]) * mp, 0.1)
            r["encode_ms_source"] = "per_source_linear_fit"
            n_src_fit += 1
        else:
            p = ab.get((b, r["speed"]))
            if p:
                r["encode_ms"] = max(float(p["power_c"]) * (r["width"] * r["height"]) ** float(p["power_gamma"]), 0.1)
                r["encode_ms_source"] = "pooled_power_law"
                n_pooled += 1
            else:
                r["encode_ms"] = None
                r["encode_ms_source"] = "unavailable"
    report["gates"]["encode_ms"] = {
        "MODELLED_NOT_MEASURED": ("encode_ms is NOT persisted by the fleet path for any "
                                  "DOE cell (ledger and score-blob schemas carry no duration). "
                                  "These values come from the 2026-09-03 speed instrument's "
                                  "fits and inherit ALL its limitations: SINGLE-THREADED wall "
                                  "time, q45-ANCHORED (q is not a term in the model), measured "
                                  "on 5 of the 32 sources, 20 (backend, speed) arms."),
        "rows_from_per_source_fit": n_src_fit,
        "rows_from_pooled_power_law": n_pooled,
        "pooled_linear_model_failed_per_instrument": True,
    }

    # ------------------------------------------------- 9. ceilings + pareto
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r["image_path"], r["size_class"])].append(r)
    ceilings = {k: max(r["ssim2"] for r in v) for k, v in by_key.items()}
    cfg_ids = {}
    for r in sorted(rows, key=lambda r: r["config_name"]):
        cfg_ids.setdefault(r["config_name"], len(cfg_ids))

    pareto_cols = collections.OrderedDict()
    def col(name, vals, typ):
        pareto_cols[name] = pa.array(vals, type=typ)

    rows.sort(key=lambda r: (r["image_path"], r["config_name"], r["q"]))
    col("image_path", [r["image_path"] for r in rows], pa.string())
    col("size_class", [r["size_class"] for r in rows], pa.string())
    col("width", [r["width"] for r in rows], pa.int64())
    col("height", [r["height"] for r in rows], pa.int64())
    col("config_id", [cfg_ids[r["config_name"]] for r in rows], pa.int64())
    col("config_name", [r["config_name"] for r in rows], pa.string())
    col("q", [r["q"] for r in rows], pa.int64())
    col("bytes", [r["bytes"] for r in rows], pa.int64())
    col(args.metric, [r["ssim2"] for r in rows], pa.float64())
    col(f"effective_max_{args.metric}",
        [ceilings[(r["image_path"], r["size_class"])] for r in rows], pa.float64())
    col("encode_ms", [r["encode_ms"] for r in rows], pa.float64())
    for extra, typ in (("backend", pa.string()), ("speed", pa.int64()), ("knobs", pa.string()),
                       ("bit_depth", pa.int64()), ("chroma", pa.string()),
                       ("chroma_source", pa.string()), ("corpus", pa.string()),
                       ("transform", pa.string()), ("leg", pa.string()),
                       ("origin_id", pa.string()), ("content_class", pa.string()),
                       ("coarse_class", pa.string()), ("era", pa.string()),
                       ("eras", pa.string()), ("run", pa.string()),
                       ("source_tables", pa.string()), ("encode_sha", pa.string()),
                       ("encode_ms_source", pa.string()), ("image", pa.string()),
                       ("m_butteraugli", pa.float64()), ("m_butteraugli_max", pa.float64()),
                       ("m_butteraugli_pnorm3", pa.float64())):
        col(extra, [r.get(extra) for r in rows], typ)
    pareto = pa.table(pareto_cols)
    pq.write_table(pareto, out / "pareto_avif_autotune.parquet", compression="zstd")

    # CORE view: cells measured on BOTH corpora, so every admitted cell carries
    # cross-size evidence.  71 of the full set's cells are the Stage-A A2
    # pairwise arms, which exist only at 1024^2; the DOE's own transfer gate
    # rates most knobs PARTIAL and finds two whose 1024^2 effect is known-wrong
    # at native (`tl1.0` flips sign, `tl1.1` shrinks 8.6x).  The rule is
    # DERIVED here, never hand-listed, so it cannot drift from the data.
    per_cfg = collections.defaultdict(set)
    for r in rows:
        per_cfg[r["config_name"]].add(r["corpus"])
    core_cfgs = sorted(c for c, ks in per_cfg.items() if ks == {"budget", "native"})
    core_set = set(core_cfgs)
    core_idx = [i for i, r in enumerate(rows) if r["config_name"] in core_set]
    core = pareto.take(pa.array(core_idx, type=pa.int64()))
    pq.write_table(core, out / "pareto_avif_autotune_core.parquet", compression="zstd")
    report["core_view"] = {
        "rule": ("a cell is admitted iff it was measured on BOTH the 1024^2 budget "
                 "corpus and the native corpus — cross-size evidence for every "
                 "pickable cell"),
        "configs": len(core_cfgs), "rows": len(core_idx),
        "config_names": core_cfgs,
    }

    # features table: one row per (image_path, size_class)
    fh = ["image_path", "image_sha", "split", "content_class", "source",
          "size_class", "width", "height"] + featcols
    fpath = out / "features_avif_autotune.tsv"
    with open(fpath, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(fh)
        emitted = set()
        for r in rows:
            k = (r["image_path"], r["size_class"])
            if k in emitted:
                continue
            emitted.add(k)
            fr = feats[(r["corpus"], r["image"])]
            w.writerow([r["image_path"], r["image_sha"], r["leg"], r["content_class"],
                        r["image"], r["size_class"], r["width"], r["height"]]
                       + [fr[c] for c in featcols])
    n_feat_rows = len(emitted)

    # ------------------------------------------------- 10. HDR (separate regime)
    tag, hp, hera = HDR_SOURCE
    ht = pq.read_table(hp)
    hn = ht.num_rows
    hc = {c: ht[c].to_pylist() for c in ht.column_names}
    hrows = []
    for i in range(hn):
        b, sp, kn, bd = parse_hdr_arm(hc["arm"][i])
        if b is None:
            raise SystemExit(f"unparseable HDR arm {hc['arm'][i]!r}")
        hrows.append({
            "run": hc["run"][i], "image": hc["image"][i], "q": hc["q"][i],
            "arm": hc["arm"][i], "backend": b, "dial": hc["dial"][i],
            "dial_kind": hc["dial_kind"][i], "bit_depth": bd, "bytes": hc["bytes"][i],
            "ssim2": hc["m_ssim2"][i], "zensim": hc["m_zensim"][i],
            "zensim_score": hc["m_zensim_score"][i], "encode_sha": hc["encode_sha"][i],
            "era": hera, "regime": "hdr10-pq",
            "config_name": f"{b}-{'p' if hc['dial_kind'][i]=='preset' else 's'}{hc['dial'][i]}-hdr10",
        })
    hcols = collections.OrderedDict()
    for name, typ in (("run", pa.string()), ("image", pa.string()), ("q", pa.int64()),
                      ("arm", pa.string()), ("config_name", pa.string()),
                      ("backend", pa.string()), ("dial", pa.int64()),
                      ("dial_kind", pa.string()), ("bit_depth", pa.int64()),
                      ("bytes", pa.int64()), ("ssim2", pa.float64()),
                      ("zensim", pa.float64()), ("zensim_score", pa.float64()),
                      ("encode_sha", pa.string()), ("era", pa.string()),
                      ("regime", pa.string())):
        hcols[name] = pa.array([r[name] for r in hrows], type=typ)
    pq.write_table(pa.table(hcols), out / "cells_hdr.parquet", compression="zstd")
    report["inputs"]["t2_hdr"] = {
        "path": hp, "era": hera, "rows_read": hn, "sha256": sha256_file(hp),
        "NOTE": ("SEPARATE REGIME — HDR-10 PQ, 10-bit, its own 16-ref corpus. "
                 "NEVER column-mix with the SDR view: different pixels, different "
                 "transfer, different metric space."),
    }

    # sidecars
    def write_tsv(name, recs, cols):
        with open(out / name, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(cols)
            for r in recs:
                w.writerow([r.get(c) for c in cols])
    write_tsv("sidecar_inert_arm_census.tsv", inert_rows,
              ["corpus", "speed", "bit_depth", "knobs", "n_cells",
               "n_identical_to_control", "frac_inert"])
    write_tsv("sidecar_alias_arms.tsv", alias_rows,
              ["corpus", "speed", "bit_depth", "knobs", "aliases", "n_cells"])
    write_tsv("sidecar_speed_alias.tsv", speed_alias_rows,
              ["corpus", "speed", "bit_depth", "aliases_speed", "n_shared_cells"])

    # per-source counts for the manifest
    per_src = collections.Counter()
    for r in rows:
        for t in r["source_tables"].split("|"):
            per_src[t] += 1
    report["view"] = {
        "rows_read_total": n_read,
        "rows_after_exclusions": len(rows),
        "distinct_cells": len(bycell),
        "distinct_configs": len(cfg_ids),
        "distinct_images": len({r["image_path"] for r in rows}),
        "rows_contributed_by_source_table": dict(per_src),
        "rows_by_leg": dict(legs),
        "rows_by_corpus": dict(collections.Counter(r["corpus"] for r in rows)),
        "rows_by_backend": dict(collections.Counter(r["backend"] for r in rows)),
        "feature_columns": len(featcols),
        "feature_rows": n_feat_rows,
        "metric_column": args.metric,
    }
    report["build_commit"] = build_commit()
    report["config_id_map"] = cfg_ids
    for f in ("pareto_avif_autotune.parquet", "pareto_avif_autotune_core.parquet",
              "features_avif_autotune.tsv",
              "cells_hdr.parquet", "sidecar_inert_arm_census.tsv",
              "sidecar_alias_arms.tsv", "sidecar_speed_alias.tsv"):
        p = out / f
        report.setdefault("outputs", {})[f] = {
            "bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    (out / "_MANIFEST.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("view", "gates")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
