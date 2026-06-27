#!/usr/bin/env python3
"""build_canonical.py — codec-agnostic canonical picker-dataset builder.

Builds the CANONICAL, documented, R2-stored, train/val/test-split training
dataset for ONE codec (split into lossy/lossless modes) from a mandfix-style
sweep run on the all-origin clean-picker corpus.

Per (codec, mode) it writes 3 zstd parquets — train.parquet / validate.parquet /
test.parquet — plus a _MANIFEST.json, locally and to R2.

One row == one (rendition, cell) encode+score. It is assembled by joining, per
box of the run:
  * omni TSV   (scores + encoded bytes/filename + timings), and
  * the 372-feature `box-N.feat.parquet` (per-cell zensim features),
joined 1:1 on the cell identity (image_path, codec, q, knob_tuple_json),
then left-joined to the per-rendition CONTENT features (clean_features_vn.tsv)
on `variant_name`, then stamped with run-level provenance.

The train/val/test split is the ONE canonical origin-level rule
(scripts/picker/origin_split.py): split is a pure function of the origin id, so
every sizing/crop/encode derivative of an origin lands in the same bucket and
nothing leaks. See DATA_PROVENANCE.md "Canonical picker datasets (2026-06-27)".

Usage:
  build_canonical.py --run <RUN> --codec <jpeg|webp|png|jxl|avif> \
      [--mode all|lossy|lossless] [--no-upload] [--max-boxes N] \
      [--local-root DIR] [--r2-root s3://...] [--build-ts ISO8601]

R2 access via aws-cli ONLY (never s5cmd here). Reads R2_ACCOUNT_ID,
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from the environment.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical split rule — the ONE source of truth. Imported, never re-implemented.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origin_split  # noqa: E402

# ---- constants -------------------------------------------------------------
CORPUS = "clean-picker-corpus-2026-06-26"
SOURCE_R2_PREFIX = f"s3://codec-corpus/{CORPUS}"
RUN_PREFIX_DEFAULT = "s3://zentrain/jxl-lossy/runs"
CONTENT_TSV_DEFAULT = f"/mnt/v/output/{CORPUS}/clean_features_vn.tsv"
LOCAL_ROOT_DEFAULT = "/mnt/v/output/canonical-picker-2026-06-27"
R2_ROOT_DEFAULT = "s3://zentrain/canonical/2026-06-27"
SWEEP_DATE = "2026-06-27"

# short codec label -> crate key inside the manifest's codec_commits dict
CODEC_CRATE = {"jpeg": "zenjpeg", "webp": "zenwebp", "png": "zenpng",
               "jxl": "zenjxl", "avif": "zenavif"}

# omni columns we keep (the join 4-key + scores/bytes/timings)
OMNI_KEEP = ["image_path", "codec", "q", "knob_tuple_json", "encoded_bytes",
             "encode_ms", "encoded_filename", "decode_ms", "score_ssim2",
             "score_zensim"]
KEY = ["image_path", "codec", "q", "knob_tuple_json"]
FEAT_N = [f"feat_{i}" for i in range(372)]

# content-TSV columns to merge (variant_name is the join key); the rest are
# renamed to avoid colliding with omni/feat-parquet columns.
CONTENT_RENAME = {"image_sha": "content_image_sha", "source": "content_source",
                  "split": "split_clean_features"}
CONTENT_DROP = ["image_path"]  # collides with omni image_path (host-specific path)
CONTENT_META = ["content_image_sha", "content_class", "content_source",
                "size_class", "width", "height", "split_clean_features"]


def mode_of(codec: str, cell: str) -> str:
    """lossy/lossless per the canonical rule (verified against actual cells)."""
    c = (codec or "").lower()
    cell = cell or ""
    if "png" in c:                 # zenpng — all lossless
        return "lossless"
    if cell.startswith("mod-"):    # jxl modular
        return "lossless"
    if cell.startswith("vp8l"):    # webp lossless (vp8l-*) vs lossy (vp8-*)
        return "lossless"
    return "lossy"                 # jpeg jp*/pw*/moz*, webp vp8-*, jxl vd-*, avif


def ep() -> str:
    acct = os.environ["R2_ACCOUNT_ID"]
    return f"https://{acct}.r2.cloudflarestorage.com"


def aws(*args, capture=False):
    cmd = ["aws", "s3", *args, "--endpoint-url", ep()]
    if capture:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    subprocess.run(cmd, check=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Backfill score-pairs file (PARQUET): (output column, source parquet column).
# `ref_path`+`dist_path` are the columns `zenmetrics score-pairs --pairs-tsv`
# requires; `dist_path` is the INDIVIDUAL encode object. The identity tuple
# (image_path/codec/q/knob_tuple_json) is carried so score-pairs passes it
# through to its sidecar, which then joins back onto the canonical parquet on
# that exact 4-key. dist_tar/dist_member let a worker fall back to tar
# extraction. Lets a later GPU pass score a new metric WITHOUT re-encoding
# (see DATA_PROVENANCE.md "metric-backfill recipe").
PAIRS_MAP = [("ref_path", "source_r2_url"), ("dist_path", "variant_r2_url"),
             ("image_path", "image_path"), ("codec", "codec"), ("q", "q"),
             ("knob_tuple_json", "knob_tuple_json"),
             ("dist_member", "encoded_filename"), ("dist_tar", "variant_tar_r2_url"),
             ("cell", "cell")]


def emit_pairs_parquet(src_parquet: str, pairs_parquet: str) -> int:
    cols = [s for _, s in PAIRS_MAP]
    t = pq.read_table(src_parquet, columns=cols)
    t = t.rename_columns([n for n, _ in PAIRS_MAP])
    pq.write_table(t, pairs_parquet, compression="zstd")
    return t.num_rows


def list_boxes(run_prefix: str, run: str) -> list:
    """Box ids that have BOTH an omni TSV and a feat parquet."""
    out = aws("ls", f"{run_prefix}/{run}/omni/", capture=True)
    omni = {int(p.split("box-")[1].split(".omni")[0])
            for p in out.split() if "box-" in p and ".omni.tsv" in p}
    out = aws("ls", f"{run_prefix}/{run}/features/", capture=True)
    feat = {int(p.split("box-")[1].split(".feat")[0])
            for p in out.split() if "box-" in p and ".feat.parquet" in p}
    return sorted(omni & feat)


def fetch(run_prefix, run, sub, fname, scratch):
    dst = os.path.join(scratch, f"{run}__{sub}__{fname}")
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        aws("cp", f"{run_prefix}/{run}/{sub}/{fname}", dst, "--no-progress")
    return dst


def load_content(content_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(content_tsv, sep="\t")
    df = df.drop(columns=[c for c in CONTENT_DROP if c in df.columns])
    df = df.rename(columns=CONTENT_RENAME)
    feat_named = [c for c in df.columns if c.startswith("feat_")]
    keep = ["variant_name"] + [c for c in CONTENT_META if c in df.columns] + feat_named
    return df[keep], feat_named


def build_box(omni_path, feat_path, run, box, codec_label, content_df,
              content_feats, prov):
    """Join one box's omni+feat+content, derive cols, return a tidy DataFrame."""
    omni = pd.read_csv(omni_path, sep="\t", dtype=str)[OMNI_KEEP]
    feat = pq.read_table(feat_path, columns=KEY + FEAT_N).to_pandas()
    # normalize the join key types: q float on both, knob/codec/path str
    omni["q"] = omni["q"].astype(float)
    feat["q"] = feat["q"].astype(float)
    for c in ("image_path", "codec", "knob_tuple_json"):
        feat[c] = feat[c].astype(str)
    n_omni = len(omni)
    m = omni.merge(feat, on=KEY, how="left", indicator=True, validate="1:1")
    dropped = int((m["_merge"] != "both").sum())
    m = m[m["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    # numeric coercion for omni score/byte/timing columns
    for c, t in (("encoded_bytes", "int64"), ("encode_ms", float),
                 ("decode_ms", float), ("score_ssim2", float),
                 ("score_zensim", float)):
        m[c] = pd.to_numeric(m[c], errors="coerce").astype(t if t != "int64" else "int64")
    # ---- derived columns ----
    m["ref_filename"] = m["image_path"].str.rsplit("/", n=1).str[-1]
    m["variant_name"] = m["ref_filename"].str.replace(r"\.png$", "", regex=True)
    knobs = m["knob_tuple_json"].map(json.loads)
    m["cell"] = knobs.map(lambda d: d.get("cell"))
    m["fp"] = knobs.map(lambda d: d.get("fp"))
    m["knob_plan"] = knobs.map(lambda d: d.get("plan"))
    m["mode"] = [mode_of(c, cell) for c, cell in zip(m["codec"], m["cell"])]
    m["split"] = m["ref_filename"].map(origin_split.split_of)
    m["origin_id"] = m["ref_filename"].map(origin_split.origin_id)
    m["box"] = int(box)
    m["source_r2_url"] = SOURCE_R2_PREFIX + "/" + m["ref_filename"]
    # PRIMARY encode reference: the INDIVIDUAL encode object, keyed on the omni
    # `codec` column + per-row `mode` + the per-cell `encoded_filename`. Uploaded
    # separately to exactly this scheme (one object per cell, no tar extraction).
    m["variant_r2_url"] = (prov["canonical_r2_root"] + "/" + m["codec"] + "_"
                           + m["mode"] + "/encodes/" + m["encoded_filename"])
    # SECONDARY: the by-box tar bundling every encode for this box (master record).
    m["variant_tar_r2_url"] = f"{prov['run_prefix']}/{run}/variants/box-{box}.tar"
    # ---- content features (left join on variant_name) ----
    m = m.merge(content_df, on="variant_name", how="left")
    m["_content_matched"] = m["content_class"].notna() if "content_class" in m else False
    # ---- provenance (constant per run) ----
    for k in ("codec_commit", "zenmetrics_commit", "run_plan", "corpus",
              "sweep_run", "sweep_date", "build_timestamp_utc",
              "codec_commits_json"):
        m[k] = prov[k]
    m["codec_label"] = codec_label
    # ---- enforce dtypes so arrow infers stable types (esp. all-null cols
    # like png's codec_commit -> typed string, not arrow `null`) ----
    str_cols = (["split", "mode", "codec_label", "origin_id", "variant_name",
                 "ref_filename", "codec", "cell", "fp", "knob_plan",
                 "knob_tuple_json", "encoded_filename", "source_r2_url",
                 "variant_r2_url", "variant_tar_r2_url", "image_path", "codec_commit",
                 "zenmetrics_commit", "run_plan", "corpus", "sweep_run",
                 "sweep_date", "build_timestamp_utc", "codec_commits_json"]
                + [c for c in CONTENT_META if c not in ("width", "height")])
    for c in str_cols:
        if c in m:
            m[c] = m[c].astype("string")
    for c in ("width", "height"):
        if c in m:
            m[c] = pd.to_numeric(m[c], errors="coerce").astype("float64")
    m["box"] = m["box"].astype("int32")
    return m, n_omni, dropped


def column_order(content_feats):
    return (
        ["split", "mode", "codec_label", "origin_id", "variant_name",
         "ref_filename", "box"]
        + ["codec", "q", "cell", "fp", "knob_plan", "knob_tuple_json",
           "encoded_bytes", "encode_ms", "decode_ms", "encoded_filename",
           "score_ssim2", "score_zensim"]
        + ["source_r2_url", "variant_r2_url", "variant_tar_r2_url", "image_path"]
        + ["codec_commit", "zenmetrics_commit", "run_plan", "corpus",
           "sweep_run", "sweep_date", "build_timestamp_utc", "codec_commits_json"]
        + CONTENT_META
        + content_feats
        + FEAT_N
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--codec", required=True, choices=sorted(CODEC_CRATE))
    ap.add_argument("--mode", default="all", choices=["all", "lossy", "lossless"])
    ap.add_argument("--run-prefix", default=RUN_PREFIX_DEFAULT)
    ap.add_argument("--content-tsv", default=CONTENT_TSV_DEFAULT)
    ap.add_argument("--local-root", default=LOCAL_ROOT_DEFAULT)
    ap.add_argument("--r2-root", default=R2_ROOT_DEFAULT)
    ap.add_argument("--build-ts", required=True, help="ISO-8601 UTC build timestamp")
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp/canonical-build"))
    ap.add_argument("--max-boxes", type=int, default=0, help="0 = all (smoke-test cap)")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.scratch, exist_ok=True)
    crate = CODEC_CRATE[args.codec]
    # Output dirs are keyed on the OMNI `codec` column (zenjpeg/zenwebp/zenpng/
    # zenjxl), NOT the short --codec label, so the {codec}_{mode}/encodes/ path in
    # variant_r2_url matches where the individual encode objects are stored. For
    # all current codecs the omni codec == the crate name (asserted per box).
    omni_codec = crate

    # provenance from box-0 manifest
    man_path = fetch(args.run_prefix, args.run, "manifests", "box-0.plan.json", args.scratch)
    man = json.load(open(man_path))
    codec_commits = man.get("codec_commits") or {}
    prov = {
        "run_prefix": args.run_prefix,
        "canonical_r2_root": args.r2_root,
        "codec_commit": codec_commits.get(crate),  # None if crate not tracked (e.g. zenpng)
        "zenmetrics_commit": man.get("zenmetrics_commit"),
        "run_plan": man.get("plan"),
        "corpus": CORPUS,
        "sweep_run": args.run,
        "sweep_date": SWEEP_DATE,
        "build_timestamp_utc": args.build_ts,
        "codec_commits_json": json.dumps(codec_commits, sort_keys=True),
    }
    print(f"[prov] codec={args.codec} crate={crate} codec_commit={prov['codec_commit']} "
          f"zenmetrics={prov['zenmetrics_commit']} plan={prov['run_plan']}", flush=True)

    content_df, content_feats = load_content(args.content_tsv)
    cols = column_order(content_feats)
    print(f"[content] {len(content_df)} renditions, {len(content_feats)} content feats; "
          f"output has {len(cols)} columns", flush=True)

    boxes = list_boxes(args.run_prefix, args.run)
    if args.max_boxes:
        boxes = boxes[:args.max_boxes]
    print(f"[boxes] {len(boxes)}: {boxes}", flush=True)

    wanted_modes = {"lossy", "lossless"} if args.mode == "all" else {args.mode}
    writers = {}           # (mode,split) -> ParquetWriter
    paths = {}             # (mode,split) -> local path
    frozen_schema = [None]
    origins = defaultdict(lambda: defaultdict(set))  # mode -> split -> {origin}
    rowcount = defaultdict(lambda: defaultdict(int))
    n_omni_total = n_dropped_total = n_content_matched = n_rows_total = 0

    def writer_for(mode, split):
        key = (mode, split)
        if key not in writers:
            d = os.path.join(args.local_root, f"{omni_codec}_{mode}")
            os.makedirs(d, exist_ok=True)
            fname = {"train": "train", "val": "validate", "test": "test"}[split]
            p = os.path.join(d, f"{fname}.parquet")
            writers[key] = pq.ParquetWriter(p, frozen_schema[0], compression="zstd")
            paths[key] = p
        return writers[key]

    for box in boxes:
        omni_p = fetch(args.run_prefix, args.run, "omni", f"box-{box}.omni.tsv", args.scratch)
        feat_p = fetch(args.run_prefix, args.run, "features", f"box-{box}.feat.parquet", args.scratch)
        df, n_omni, dropped = build_box(omni_p, feat_p, args.run, box,
                                        args.codec, content_df, content_feats, prov)
        assert (df["codec"] == omni_codec).all(), \
            f"box {box}: omni codec {set(df['codec'])} != expected {omni_codec}"
        n_omni_total += n_omni
        n_dropped_total += dropped
        n_content_matched += int(df["_content_matched"].sum())
        df = df.drop(columns="_content_matched")
        # split must never be None for a corpus rendition
        bad = df[df["split"].isna()]
        if len(bad):
            raise SystemExit(f"box {box}: {len(bad)} rows have no split (origin id); "
                             f"sample ref={bad['ref_filename'].iloc[0]}")
        df = df[cols]
        if frozen_schema[0] is None:
            frozen_schema[0] = pa.Table.from_pandas(df, preserve_index=False).schema
        for mode in sorted(set(df["mode"]) & wanted_modes):
            for split in ("train", "val", "test"):
                mask = (df["mode"] == mode) & (df["split"] == split)
                if not mask.any():
                    continue
                sp = pa.Table.from_pandas(df[mask], preserve_index=False).cast(frozen_schema[0])
                writer_for(mode, split).write_table(sp)
                rowcount[mode][split] += sp.num_rows
                origins[mode][split].update(df.loc[mask, "origin_id"].tolist())
                n_rows_total += sp.num_rows
        print(f"[box {box}] omni={n_omni} dropped={dropped} kept={len(df)} "
              f"modes={sorted(set(df['mode']))}", flush=True)

    for w in writers.values():
        w.close()

    # ---- per-(codec,mode) manifests + verification + upload ----
    results = {}
    for mode in sorted({m for (m, _s) in writers}):
        # leakage gate: origins disjoint across splits
        tr, va, te = (origins[mode]["train"], origins[mode]["val"], origins[mode]["test"])
        leaks = {"train&val": sorted(tr & va), "train&test": sorted(tr & te),
                 "val&test": sorted(va & te)}
        leaked = any(leaks.values())
        dataset = f"{omni_codec}_{mode}"
        d = os.path.join(args.local_root, dataset)
        split_info = {}
        for split, sp_name in (("train", "train"), ("val", "validate"), ("test", "test")):
            p = os.path.join(d, f"{sp_name}.parquet")
            if not os.path.exists(p):
                continue
            # convenience backfill pairs file (parquet), derived from the parquet
            pairs_p = os.path.join(d, f"pairs.{sp_name}.parquet")
            n_pairs = emit_pairs_parquet(p, pairs_p)
            split_info[sp_name] = {
                "path": p,
                "rows": int(rowcount[mode][split]),
                "distinct_origins": len(origins[mode][split]),
                "sha256": sha256_file(p),
                "bytes": os.path.getsize(p),
                "pairs_parquet": pairs_p,
                "pairs_rows": int(n_pairs),
                "pairs_sha256": sha256_file(pairs_p),
            }
        manifest = {
            "dataset": dataset,
            "codec": omni_codec,
            "codec_label": args.codec,
            "mode": mode,
            "corpus": CORPUS,
            "sweep_run": args.run,
            "sweep_date": SWEEP_DATE,
            "build_timestamp_utc": args.build_ts,
            "build_commit": prov["zenmetrics_commit"],
            "source_run_prefix": f"{args.run_prefix}/{args.run}",
            "split_rule": "scripts/picker/origin_split.py: last digit of origin id "
                          "{0,2,4,6,8}=train {1,3,5}=validate {7,9}=test (origin-level, deterministic)",
            "provenance": {k: prov[k] for k in
                           ("codec_commit", "zenmetrics_commit", "run_plan",
                            "corpus", "sweep_run", "sweep_date", "codec_commits_json")},
            "splits": split_info,
            "leakage_check": {"leaked": leaked, "overlaps": leaks},
            "columns": cols,
            "n_columns": len(cols),
            "local_root": d,
            "r2_root": f"{args.r2_root}/{dataset}",
        }
        man_out = os.path.join(d, "_MANIFEST.json")
        json.dump(manifest, open(man_out, "w"), indent=2)
        results[mode] = manifest
        print(f"\n=== {dataset} ===", flush=True)
        for sp_name, info in split_info.items():
            print(f"  {sp_name:9s} rows={info['rows']:>9d} origins={info['distinct_origins']:>4d} "
                  f"sha={info['sha256'][:12]} pairs={info['pairs_rows']}", flush=True)
        print(f"  leakage: {'LEAK!' if leaked else 'none (disjoint)'}", flush=True)
        if leaked:
            raise SystemExit(f"SPLIT LEAKAGE in {dataset}: {leaks}")
        # upload parquets + pairs + manifest
        if not args.no_upload:
            for info in split_info.values():
                aws("cp", info["path"], f"{manifest['r2_root']}/{os.path.basename(info['path'])}",
                    "--no-progress")
                aws("cp", info["pairs_parquet"],
                    f"{manifest['r2_root']}/{os.path.basename(info['pairs_parquet'])}", "--no-progress")
            aws("cp", man_out, f"{manifest['r2_root']}/_MANIFEST.json", "--no-progress")
            print(f"  uploaded -> {manifest['r2_root']}/", flush=True)

    print(f"\n[totals] omni_rows={n_omni_total} dropped_no_feat={n_dropped_total} "
          f"kept={n_omni_total - n_dropped_total} written={n_rows_total} "
          f"content_matched={n_content_matched}", flush=True)
    # emit a machine-readable summary for the driver
    summary = {"codec": args.codec, "n_omni": n_omni_total,
               "n_dropped_no_feat": n_dropped_total, "n_written": n_rows_total,
               "n_content_matched": n_content_matched,
               "modes": {m: {sp: {"rows": info["rows"], "origins": info["distinct_origins"]}
                             for sp, info in r["splits"].items()}
                         for m, r in results.items()}}
    print("SUMMARY_JSON " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
