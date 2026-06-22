#!/usr/bin/env python3
"""Adapt a zenmetrics sweep (omni TSV) + zenanalyze features TSV into the
zentrain picker trainer's input format (per-codec pareto parquet + features TSV).

The unified sweep emits:
  omni TSV: image_path, codec, q, knob_tuple_json={"cell","fp","plan"},
            encoded_bytes, encode_ms, encoded_filename, decode_ms, score_<metric>
The rendition features live in the render-time TSV keyed on `variant_name`
(= rendition basename without .png), with named feat_* columns.

The trainer (train_hybrid.py) wants:
  PARETO (parquet): image_path, size_class, width, height, config_id,
                    config_name, q, bytes, zensim (=target metric),
                    encode_ms, total_ms, effective_max_zensim
  FEATURES (tsv):   image_path, size_class, width, height, feat_*   (one row
                    per (image_path,size_class), joined to PARETO on those keys)

The config_name is the plan cell-id (opaque categorical for the picker — it
picks among the swept variants per (features, target)). The target metric
column is named `zensim` to satisfy the trainer regardless of which metric it
holds; pass --metric-col to choose the sweep score that fills it (ssim2 here).
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def size_class(px: int) -> str:
    if px <= 64 * 64:
        return "tiny"
    if px <= 256 * 256:
        return "small"
    if px <= 1024 * 1024:
        return "medium"
    return "large"


def variant_of(image_path: str) -> str:
    base = os.path.basename(image_path)
    for ext in (".png", ".PNG"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni", required=True, type=Path, help="sweep omni TSV")
    ap.add_argument("--features-tsv", required=True, type=Path, help="rendition features TSV (variant_name keyed)")
    ap.add_argument("--metric-col", default="score_ssim2_gpu", help="omni column to use as the target (-> 'zensim')")
    ap.add_argument("--out-pareto", required=True, type=Path)
    ap.add_argument("--out-features", required=True, type=Path)
    args = ap.parse_args()

    import pandas as pd

    omni = pd.read_csv(args.omni, sep="\t")
    if args.metric_col not in omni.columns:
        raise SystemExit(f"omni lacks metric col {args.metric_col!r}; have {list(omni.columns)}")
    omni["variant_name"] = omni["image_path"].map(variant_of)
    omni["config_name"] = omni["knob_tuple_json"].map(lambda s: json.loads(s)["cell"])
    omni["bytes"] = omni["encoded_bytes"].astype("int64")
    omni["zensim"] = omni[args.metric_col].astype("float64")
    if "encode_ms" not in omni.columns:
        omni["encode_ms"] = 0.0
    omni["total_ms"] = omni["encode_ms"].astype("float64")

    feats = pd.read_csv(args.features_tsv, sep="\t")
    feat_cols = [c for c in feats.columns if c.startswith("feat_")]
    if "variant_name" not in feats.columns:
        raise SystemExit(f"features TSV lacks 'variant_name'; have {list(feats.columns)[:8]}...")
    fkeep = feats[["variant_name", "width", "height", *feat_cols]].drop_duplicates("variant_name")

    merged = omni.merge(fkeep, on="variant_name", how="inner")
    dropped = omni["variant_name"].nunique() - merged["variant_name"].nunique()
    if dropped:
        print(f"WARNING: {dropped} sweep variants had no feature row (dropped)")
    merged["size_class"] = (merged["width"] * merged["height"]).map(size_class)
    # config_id: stable integer per distinct config_name (sorted for determinism)
    cfg_index = {c: i for i, c in enumerate(sorted(merged["config_name"].unique()))}
    merged["config_id"] = merged["config_name"].map(cfg_index).astype("int64")
    # effective_max_zensim: best achievable target per (image, size_class)
    merged["effective_max_zensim"] = merged.groupby(["variant_name", "size_class"])["zensim"].transform("max")

    pareto_cols = [
        "variant_name", "size_class", "width", "height", "config_id",
        "config_name", "q", "bytes", "zensim", "encode_ms", "total_ms",
        "effective_max_zensim",
    ]
    pareto = merged[pareto_cols].rename(columns={"variant_name": "image_path"})
    args.out_pareto.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pareto, preserve_index=False), args.out_pareto)

    feat_out = merged[["variant_name", "size_class", "width", "height", *feat_cols]].drop_duplicates(
        ["variant_name", "size_class"]
    ).rename(columns={"variant_name": "image_path"})
    feat_out.to_csv(args.out_features, sep="\t", index=False)

    n_cfg = len(cfg_index)
    print(
        f"pareto: {len(pareto)} rows, {n_cfg} configs, "
        f"sizes={sorted(pareto['size_class'].unique())}, "
        f"zensim(={args.metric_col}) range [{pareto['zensim'].min():.1f},{pareto['zensim'].max():.1f}]"
    )
    print(f"features: {len(feat_out)} rows ({len(feat_cols)} feat_ cols) -> {args.out_features}")
    # per (size_class) row counts — the trainer's DATA_STARVED gate needs >=50 per (size,zq)
    print(pareto.groupby("size_class").size().to_dict())


if __name__ == "__main__":
    main()
