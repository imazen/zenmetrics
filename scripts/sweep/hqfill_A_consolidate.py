#!/usr/bin/env python3
"""Consolidate the per-chunk hqfill-A sweep outputs into ONE canonical parquet.

Merges every chunk's TSV (identity + 6 metric scores + encoded pointer) with its
feature parquet (feat_0..feat_371) on (image_path, codec, q, knob_tuple_json).

Output canonical parquet columns:
  image_path, variant_name, codec, q, knob_tuple_json, distance, effort,
  encoded_bytes, encode_ms, decode_ms, encoded_filename,
  score_zensim (= profile A, from zensim-gpu==cpu),
  score_ssim2 (CPU, canonical-matching),
  score_butteraugli_max, score_butteraugli_pnorm3 (GPU),
  score_cvvdp (GPU, imazen v0.0.1),
  score_dssim (CPU),
  score_iwssim (GPU),
  feat_0 .. feat_371

The raw impl-tag column names are preserved in a `_impl_tags` metadata note.
"""
import sys, glob, json, os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv
import numpy as np

OUT = "/mnt/v/output/jxl-hqfill-A-2026-07-01"
FINAL = os.path.join(OUT, "zenjxl_lossy_hqfill_A_2026-07-01.parquet")

ID = ["image_path", "codec", "q", "knob_tuple_json"]

# raw sweep column -> canonical column
METRIC_RENAME = {
    "score_zensim_gpu": "score_zensim",
    "score_ssim2": "score_ssim2",
    "score_butteraugli_max_gpu": "score_butteraugli_max",
    "score_butteraugli_pnorm3_gpu": "score_butteraugli_pnorm3",
    "score_cvvdp_imazen_v0_0_1": "score_cvvdp",
    "score_dssim": "score_dssim",
    "score_iwssim_gpu": "score_iwssim",
}


def read_tsv(path):
    # all metric columns float64; identity/text as string; some may be empty -> null
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            column_types={
                "image_path": pa.string(),
                "codec": pa.string(),
                "q": pa.int64(),
                "knob_tuple_json": pa.string(),
                "encoded_bytes": pa.int64(),
                "encode_ms": pa.float64(),
                "encoded_filename": pa.string(),
                "decode_ms": pa.float64(),
                "score_zensim_gpu": pa.float64(),
                "score_ssim2": pa.float64(),
                "score_butteraugli_max_gpu": pa.float64(),
                "score_butteraugli_pnorm3_gpu": pa.float64(),
                "score_cvvdp_imazen_v0_0_1": pa.float64(),
                "score_dssim": pa.float64(),
                "score_iwssim_gpu": pa.float64(),
            },
            null_values=[""],
            strings_can_be_null=True,
        ),
    )


def main():
    tsvs = sorted(glob.glob(os.path.join(OUT, "tsv", "chunk_*.tsv")))
    feats = sorted(glob.glob(os.path.join(OUT, "features", "chunk_*.features.parquet")))
    print(f"chunks: {len(tsvs)} TSV, {len(feats)} feature parquets")
    assert len(tsvs) == len(feats), "chunk count mismatch"

    import pandas as pd

    tsv_frames, feat_frames = [], []
    for t in tsvs:
        tsv_frames.append(read_tsv(t).to_pandas())
    for f in feats:
        feat_frames.append(pq.read_table(f).to_pandas())

    tsv = pd.concat(tsv_frames, ignore_index=True)
    feat = pd.concat(feat_frames, ignore_index=True)
    print(f"TSV rows: {len(tsv)}, feature rows: {len(feat)}")

    # feature parquet: drop its duplicate zensim_score, keep identity + feat_*
    feat_cols = [c for c in feat.columns if c.startswith("feat_")]
    assert len(feat_cols) == 372, f"expected 372 feat cols, got {len(feat_cols)}"
    feat = feat[ID + feat_cols]

    # normalize q to int on both for a clean join
    tsv["q"] = tsv["q"].astype("int64")
    feat["q"] = feat["q"].astype("int64")

    merged = tsv.merge(feat, on=ID, how="inner", validate="one_to_one")
    print(f"merged rows: {len(merged)} (inner join on {ID})")
    if len(merged) != len(tsv):
        print(f"WARNING: merged {len(merged)} != TSV {len(tsv)} — some cells lack features!")

    # rename metric columns to canonical
    merged = merged.rename(columns=METRIC_RENAME)

    # derive distance + effort from knob_tuple_json (self-describing stratum)
    def parse_knob(k, key):
        try:
            return json.loads(k).get(key)
        except Exception:
            return None
    merged["distance"] = merged["knob_tuple_json"].map(lambda k: parse_knob(k, "distance"))
    merged["effort"] = merged["knob_tuple_json"].map(lambda k: parse_knob(k, "effort"))
    # variant_name = source rendition basename (matches canonical convention)
    merged["variant_name"] = merged["image_path"].map(lambda p: os.path.basename(p))

    # column order
    lead = ["image_path", "variant_name", "codec", "q", "knob_tuple_json",
            "distance", "effort", "encoded_bytes", "encode_ms", "decode_ms",
            "encoded_filename",
            "score_zensim", "score_ssim2", "score_butteraugli_max",
            "score_butteraugli_pnorm3", "score_cvvdp", "score_dssim", "score_iwssim"]
    ordered = lead + feat_cols
    merged = merged[ordered]

    # sanity report
    print("\n=== SANITY ===")
    for c in ["score_zensim", "score_ssim2", "score_butteraugli_max",
              "score_butteraugli_pnorm3", "score_cvvdp", "score_dssim", "score_iwssim"]:
        v = merged[c].to_numpy(dtype=float)
        nn = int(np.isfinite(v).sum())
        print(f"  {c:28s}: non-null {nn}/{len(v)}  min={np.nanmin(v):.4f} max={np.nanmax(v):.4f}")
    fm = merged[feat_cols].to_numpy(dtype=float)
    print(f"  feat_* : NaN={int(np.isnan(fm).sum())} Inf={int(np.isinf(fm).sum())} shape={fm.shape}")
    z = merged["score_zensim"].to_numpy(float)
    print(f"  zensim A-saturation check: max={np.nanmax(z):.4f} (A saturates ~97.69, never 100), n==100: {(z>=99.9999).sum()}")

    tbl = pa.Table.from_pandas(merged, preserve_index=False)
    meta = {
        b"sweep": b"jxl-lossy-hqfill-A-2026-07-01",
        b"zensim_profile": b"A (zensim-a; latest_preview==latest==A)",
        b"metric_impl_tags": json.dumps({
            "score_zensim": "zensim-gpu, ZensimProfile::A, WithIw(372)",
            "score_ssim2": "ssim2 CPU (fast-ssim2)",
            "score_butteraugli_max/pnorm3": "butteraugli-gpu",
            "score_cvvdp": "cvvdp-gpu imazen v0.0.1",
            "score_dssim": "dssim CPU",
            "score_iwssim": "iwssim-gpu (reflect-pad for <176px)",
        }).encode(),
    }
    tbl = tbl.replace_schema_metadata(meta)
    pq.write_table(tbl, FINAL, compression="zstd")
    print(f"\nWROTE {FINAL} ({os.path.getsize(FINAL)/1e6:.1f} MB, {tbl.num_rows} rows x {tbl.num_columns} cols)")


if __name__ == "__main__":
    main()
