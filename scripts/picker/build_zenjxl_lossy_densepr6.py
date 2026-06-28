#!/usr/bin/env python3
"""build_zenjxl_lossy_densepr6.py — PROVISIONAL zenjxl_lossy from dense-r6.

The mandfix sweep set has no JXL-lossy (VarDCT) run, so this fills the
`zenjxl_lossy` gap by REUSING the existing dense-r6 jxl-lossy VarDCT data
(`picker-pp/train/zenjxl_lossy_dense.{zensim,ssim2}.pareto.parquet` +
`.features.tsv`) — no new sweep. It is origin-split with the SAME canonical rule
(scripts/picker/origin_split.py).

**THIS IS NOT A SCHEMA-IDENTICAL PEER of the SDR-5 (zenjpeg/zenwebp/zenpng/zenjxl
lossless) canonical datasets. Verified divergences (all flagged in _MANIFEST):**

  * Corpus is `dense-corpus-r6-2026-06-26` (2000 renditions / 672 origins),
    NOT clean-picker-corpus-2026-06-26. Only 86 origins overlap; 0 rendition
    filenames match exactly. dense-r6 is train-biased (560 train / 64 val /
    48 test origins).
  * Features are the **117 NAMED content features** from the dense-r6
    features.tsv — NOT the 372-d `feat_0..feat_371` zensim feature vector the
    SDR-5 carry. (No 372-feat block exists for this data.)
  * **No encoded variant bytes** are persisted anywhere (R2 or local) → there is
    NO `variant_r2_url` / individual-encode path and NO metric-backfill pairs
    file. Adding any new metric (cvvdp/butteraugli/…) would require RE-ENCODING.
  * The rendition PNGs live LOCAL-ONLY at /mnt/v/output/dense-corpus-r6-2026-06-26/
    (not on R2) → `source_r2_url` is null; `source_local_path` is provided.
  * No provenance manifest exists for the dense-r6 jxl-lossy sweep →
    `codec_commit` / `zenmetrics_commit` are unknown.

Usage: build_zenjxl_lossy_densepr6.py --build-ts <ISO8601> [--no-upload]
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origin_split  # noqa: E402

PP = "/mnt/v/backups/home/picker-pp/train"
ZENSIM_PARQUET = f"{PP}/zenjxl_lossy_dense.zensim.pareto.parquet"
SSIM2_PARQUET = f"{PP}/zenjxl_lossy_dense.ssim2.pareto.parquet"
FEATURES_TSV = f"{PP}/zenjxl_lossy_dense.features.tsv"
CORPUS = "dense-corpus-r6-2026-06-26"
CORPUS_LOCAL = f"/mnt/v/output/{CORPUS}"
LOCAL_ROOT = "/mnt/v/output/canonical-picker-2026-06-27"
R2_ROOT = "s3://zentrain/canonical/2026-06-27"
DATASET = "zenjxl_lossy"
KEY = ["image_path", "config_name", "q"]


def ep():
    return f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"


def aws(*a):
    subprocess.run(["aws", "s3", *a, "--endpoint-url", ep()], check=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-ts", required=True)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    # scores: zensim file's `zensim` col = zensim; ssim2 file's `zensim` col = ssim2
    zf = pq.read_table(ZENSIM_PARQUET).to_pandas().rename(columns={"zensim": "score_zensim"})
    sf = pq.read_table(SSIM2_PARQUET, columns=KEY + ["zensim"]).to_pandas().rename(
        columns={"zensim": "score_ssim2"})
    df = zf.merge(sf, on=KEY, how="inner", validate="1:1")
    print(f"[scores] zensim={len(zf)} ssim2={len(sf)} joined={len(df)}", flush=True)

    # content features (117 named) per rendition, on image_path
    feats = pd.read_csv(FEATURES_TSV, sep="\t")
    feat_cols = [c for c in feats.columns if c.startswith("feat_")]
    df = df.merge(feats[["image_path"] + feat_cols], on="image_path", how="left")
    n_feat_matched = int(df[feat_cols[0]].notna().sum())
    print(f"[features] {len(feat_cols)} named feats; matched {n_feat_matched}/{len(df)} rows", flush=True)

    # derived
    df["codec"] = "zenjxl"
    df["codec_label"] = "jxl"
    df["mode"] = "lossy"
    df["cell"] = df["config_name"]
    df["encoded_bytes"] = df["bytes"].astype("int64")
    df["ref_filename"] = df["image_path"].astype(str) + ".png"
    df["variant_name"] = df["image_path"].astype(str)
    df["origin_id"] = df["ref_filename"].map(origin_split.origin_id)
    df["split"] = df["ref_filename"].map(origin_split.split_of)
    # references — renditions are LOCAL-ONLY; no R2 source, no variant bytes
    df["source_r2_url"] = pd.NA
    df["source_local_path"] = CORPUS_LOCAL + "/" + df["ref_filename"]
    df["variant_r2_url"] = pd.NA  # NO encoded bytes persisted (would need re-encode)
    # provenance (no manifest -> unknown commits)
    df["codec_commit"] = pd.NA
    df["zenmetrics_commit"] = pd.NA
    df["run_plan"] = "densepr6_vardct"
    df["corpus"] = CORPUS
    df["sweep_run"] = "picker-pp/train/zenjxl_lossy_dense (dense-r6)"
    df["sweep_date"] = "2026-06-26"
    df["build_timestamp_utc"] = args.build_ts
    df["schema_note"] = ("PROVISIONAL: dense-r6 corpus (NOT clean-picker), 117 named "
                         "feats (NOT 372 zensim feat_N), no variant bytes, provenance "
                         "unknown — NOT schema-compatible with the SDR-5 datasets")

    bad = df[df["split"].isna()]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows have no split; sample {bad['ref_filename'].iloc[0]}")

    cols = (["split", "mode", "codec_label", "origin_id", "variant_name", "ref_filename"]
            + ["codec", "q", "cell", "config_id", "encoded_bytes", "encode_ms",
               "total_ms", "effective_max_zensim", "score_ssim2", "score_zensim"]
            + ["source_r2_url", "source_local_path", "variant_r2_url"]
            + ["codec_commit", "zenmetrics_commit", "run_plan", "corpus", "sweep_run",
               "sweep_date", "build_timestamp_utc", "schema_note"]
            + ["size_class", "width", "height"]
            + feat_cols)
    # dtype hygiene: string cols typed string (so null cols aren't arrow `null`)
    for c in (["split", "mode", "codec_label", "origin_id", "variant_name", "ref_filename",
               "codec", "cell", "source_r2_url", "source_local_path", "variant_r2_url",
               "codec_commit", "zenmetrics_commit", "run_plan", "corpus", "sweep_run",
               "sweep_date", "build_timestamp_utc", "schema_note", "size_class"]):
        df[c] = df[c].astype("string")
    df = df[cols]

    d = os.path.join(LOCAL_ROOT, DATASET)
    os.makedirs(d, exist_ok=True)
    origins = {}
    split_info = {}
    for split, sp_name in (("train", "train"), ("val", "validate"), ("test", "test")):
        sub = df[df["split"] == split]
        if not len(sub):
            continue
        p = os.path.join(d, f"{sp_name}.parquet")
        pq.write_table(pa.Table.from_pandas(sub, preserve_index=False), p, compression="zstd")
        origins[split] = set(sub["origin_id"])
        split_info[sp_name] = {"path": p, "rows": int(len(sub)),
                               "distinct_origins": len(origins[split]),
                               "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
        print(f"  {sp_name:9s} rows={len(sub):>7d} origins={len(origins[split]):>4d}", flush=True)

    # leakage gate
    tr, va, te = origins.get("train", set()), origins.get("val", set()), origins.get("test", set())
    leaks = {"train&val": sorted(tr & va), "train&test": sorted(tr & te), "val&test": sorted(va & te)}
    if any(leaks.values()):
        raise SystemExit(f"SPLIT LEAKAGE: {leaks}")
    print(f"  leakage: none (disjoint); origins {len(tr)}/{len(va)}/{len(te)}", flush=True)

    manifest = {
        "dataset": DATASET, "codec": "zenjxl", "codec_label": "jxl", "mode": "lossy",
        "PROVISIONAL": True,
        "schema_compatible_with_sdr5": False,
        "divergences": [
            f"corpus={CORPUS} (local-only; NOT clean-picker; 86/672 origin overlap)",
            f"{len(feat_cols)} NAMED content features (feat_<name>) — NOT the 372 "
            "feat_0..feat_371 zensim vector the SDR-5 carry",
            "no encoded variant bytes persisted (R2 or local) -> no variant_r2_url, "
            "no encodes/, no pairs.* metric-backfill file; new metrics need re-encode",
            "source renditions LOCAL-ONLY at /mnt/v/output/dense-corpus-r6-2026-06-26/ "
            "(not on R2) -> source_r2_url is null, source_local_path provided",
            "no provenance manifest -> codec_commit / zenmetrics_commit unknown",
            "train-biased corpus (dense-r6 K500_even reps): thin val/test",
        ],
        "corpus": CORPUS, "corpus_local": CORPUS_LOCAL,
        "source": "picker-pp/train/zenjxl_lossy_dense.{zensim,ssim2}.pareto.parquet + .features.tsv",
        "sweep_date": "2026-06-26", "build_timestamp_utc": args.build_ts,
        "split_rule": "scripts/picker/origin_split.py (same as SDR-5): origin last digit "
                      "{0,2,4,6,8}=train {1,3,5}=validate {7,9}=test",
        "vardct_configs": sorted(df["cell"].dropna().unique().tolist()),
        "q_grid": sorted(int(q) for q in df["q"].dropna().unique()),
        "splits": split_info, "leakage_check": {"leaked": False, "overlaps": leaks},
        "columns": cols, "n_columns": len(cols),
        "local_root": d, "r2_root": f"{R2_ROOT}/{DATASET}",
    }
    man_out = os.path.join(d, "_MANIFEST.json")
    json.dump(manifest, open(man_out, "w"), indent=2)
    print(f"\n[{DATASET}] PROVISIONAL — schema_compatible_with_sdr5=False; "
          f"{len(df)} rows, {len(cols)} cols, {len(feat_cols)} named feats", flush=True)

    if not args.no_upload:
        for info in split_info.values():
            aws("cp", info["path"], f"{manifest['r2_root']}/{os.path.basename(info['path'])}",
                "--no-progress")
        aws("cp", man_out, f"{manifest['r2_root']}/_MANIFEST.json", "--no-progress")
        print(f"  uploaded -> {manifest['r2_root']}/", flush=True)


if __name__ == "__main__":
    main()
