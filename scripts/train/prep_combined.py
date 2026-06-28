#!/usr/bin/env python3
"""Build a combined (train+validate+test) parquet + a tiny image_path->split
map for one codec, projecting to only the columns picker_tree_ab needs.

This is the repo-tracked, source-parameterized version of the prep step the
dual-model A/B uses (the original lived in /tmp/dualmodel). The source dir of
the canonical {train,validate,test}.parquet files is `--src` (or env CANON_DIR);
on a fleet box the runner downloads them from R2 first and points --src at the
local download dir. Output dir is `--out` (default /work).

Streaming row-group copy keeps memory bounded. The `split` column is carried
through so picker_tree_ab can partition by the canonical origin split.
"""
import argparse
import os
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("codec", help="e.g. zenwebp_lossy")
    ap.add_argument(
        "--src",
        default=os.environ.get("CANON_DIR"),
        help="dir holding {train,validate,test}.parquet for the codec "
        "(default env CANON_DIR, else /mnt/v/output/canonical-picker-2026-06-27/<codec>)",
    )
    ap.add_argument("--out", default="/work", help="output dir")
    ap.add_argument(
        "--score-col",
        default=os.environ.get("SCORE_COL", "score_zensim"),
        help="metric column to use as the reach/oracle target (default score_zensim, "
        "env SCORE_COL). When != score_zensim (e.g. score_ssim2), it is RENAMED to "
        "score_zensim in the combined parquet so picker_tree_ab — which hardcodes the "
        "score_zensim column name in zenpicker-train::pareto_dataset — computes "
        "reach/oracle on that metric transparently (no zenanalyze-crate edit needed).",
    )
    args = ap.parse_args()

    codec = args.codec
    src = args.src or f"/mnt/v/output/canonical-picker-2026-06-27/{codec}"
    out = args.out
    score_col = args.score_col
    os.makedirs(out, exist_ok=True)

    # Discover columns from the validate file schema.
    sch = pq.ParquetFile(f"{src}/validate.parquet").schema_arrow
    allcols = list(sch.names)
    if score_col not in allcols:
        raise SystemExit(
            f"FATAL: requested --score-col {score_col!r} not in {src}/validate.parquet "
            f"(have score cols: {[c for c in allcols if c.startswith('score_')]})"
        )
    feat = [c for c in allcols if c.startswith("feat_")]
    need_meta = [
        "image_path", "codec", "q", "knob_tuple_json",
        score_col, "encoded_bytes", "split",
    ]
    need_meta = [c for c in need_meta if c in allcols]
    proj = feat + need_meta
    rename = score_col != "score_zensim"
    print(f"{codec}: {len(feat)} feat cols + {len(need_meta)} meta cols = {len(proj)} projected "
          f"(metric={score_col}{' -> renamed score_zensim' if rename else ''})", flush=True)

    combined = f"{out}/combined_{codec}.parquet"
    writer = None
    split_map = {}  # image_path -> split (origin-deterministic, first-seen wins)
    total = 0
    for split_file in ["train", "validate", "test"]:
        pf = pq.ParquetFile(f"{src}/{split_file}.parquet")
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=proj)
            if rename:
                # picker_tree_ab reads the literal column "score_zensim"; rename the
                # chosen metric into that slot so reach/oracle are computed on it.
                names = [("score_zensim" if n == score_col else n) for n in tbl.schema.names]
                tbl = tbl.rename_columns(names)
            if writer is None:
                writer = pq.ParquetWriter(combined, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            imgs = tbl.column("image_path").to_pylist()
            sps = tbl.column("split").to_pylist()
            for im, sp in zip(imgs, sps):
                if im not in split_map:
                    split_map[im] = sp
            total += tbl.num_rows
        print(f"  {split_file}: done (running total {total} rows)", flush=True)
    writer.close()
    print(f"wrote {combined} ({total} rows)", flush=True)

    items = sorted(split_map.items())
    smt = pa.table({"image_path": pa.array([k for k, _ in items]),
                    "split": pa.array([v for _, v in items])})
    smpath = f"{out}/splitmap_{codec}.parquet"
    pq.write_table(smt, smpath, compression="zstd")
    print(f"wrote {smpath}: {len(items)} distinct image_path; "
          f"split dist={dict(Counter(v for _, v in items))}", flush=True)


if __name__ == "__main__":
    main()
