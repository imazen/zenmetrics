#!/usr/bin/env python3
"""Merge the hqfill-A dataset: salvage chunks 0-10 (local) + remote chunks 11-29
(from the vast GPU fleet on R2) into ONE canonical parquet.

Both halves come from the SAME `zenmetrics sweep --feature-output` binary → identical
15-col TSV schema + 372-feat parquet → clean union. Output columns match the salvage:
identity + 7 metric score cols (salvage names) + feat_0..feat_371 + variant pointer.

Usage: hqfill_A_merge.py <remote_run_id>
"""
import sys, glob, os, json, subprocess
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.csv as pacsv
import numpy as np, pandas as pd

REMOTE_RUN = sys.argv[1] if len(sys.argv) > 1 else open("/tmp/hqfillA_smoke_run.txt").read().strip()
LOCAL = "/mnt/v/output/jxl-hqfill-A-2026-07-01"
FINAL = os.path.join(LOCAL, "zenjxl_lossy_hqfill_A_2026-07-01.parquet")
R2_PRE = f"jxl-lossy-hqfill-A/2026-07-01/remote/{REMOTE_RUN}"
ID = ["image_path", "codec", "q", "knob_tuple_json"]
SCORE_COLS = ["score_zensim_gpu", "score_ssim2", "score_butteraugli_max_gpu",
              "score_butteraugli_pnorm3_gpu", "score_cvvdp_imazen_v0_0_1",
              "score_dssim", "score_iwssim_gpu"]

acct = os.environ["R2_ACCOUNT_ID"]
EP = f"https://{acct}.r2.cloudflarestorage.com"
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def s5(*a): subprocess.run(["s5cmd", "--endpoint-url", EP, *a], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

TSV_TYPES = {"image_path": pa.string(), "codec": pa.string(), "q": pa.int64(),
             "knob_tuple_json": pa.string(), "encoded_bytes": pa.int64(),
             "encode_ms": pa.float64(), "encoded_filename": pa.string(),
             "decode_ms": pa.float64()}
for c in SCORE_COLS:
    TSV_TYPES[c] = pa.float64()

def read_tsv(p):
    return pacsv.read_csv(p, parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(column_types=TSV_TYPES, null_values=[""],
                                             strings_can_be_null=True)).to_pandas()

def collect_half(tsv_glob, feat_glob):
    tsvs = sorted(glob.glob(tsv_glob)); feats = sorted(glob.glob(feat_glob))
    tframes = [read_tsv(t) for t in tsvs]
    fframes = [pq.read_table(f).to_pandas() for f in feats]
    return pd.concat(tframes, ignore_index=True), pd.concat(fframes, ignore_index=True), len(tsvs)

# 1. salvage (local chunks 0-10)
sal_tsv, sal_feat, n_sal = collect_half(f"{LOCAL}/tsv/chunk_*.tsv", f"{LOCAL}/features/chunk_*.features.parquet")
print(f"salvage: {n_sal} chunks, {len(sal_tsv)} TSV rows, {len(sal_feat)} feature rows")

# 2. remote (download all chunk-*.tsv + features from R2)
rd = "/tmp/hqfillA_remote_dl"; os.makedirs(rd + "/tsv", exist_ok=True); os.makedirs(rd + "/features", exist_ok=True)
s5("cp", f"s3://zentrain/{R2_PRE}/tsv/*", rd + "/tsv/")
s5("cp", f"s3://zentrain/{R2_PRE}/features/*", rd + "/features/")
rem_tsv, rem_feat, n_rem = collect_half(f"{rd}/tsv/chunk-*.tsv", f"{rd}/features/chunk-*.features.parquet")
print(f"remote: {n_rem} chunks, {len(rem_tsv)} TSV rows, {len(rem_feat)} feature rows")

# 3. union both halves
tsv = pd.concat([sal_tsv, rem_tsv], ignore_index=True)
feat = pd.concat([sal_feat, rem_feat], ignore_index=True)
# The remote image_path is the box's /data/src/<name>; salvage is /mnt/v/.../<name>. Normalize to basename
# so identity joins across halves and matches canonical's variant_name convention.
for d in (tsv, feat):
    d["image_path"] = d["image_path"].map(lambda p: os.path.basename(str(p)))
    d["q"] = d["q"].astype("int64")
print(f"union: {len(tsv)} TSV rows, {len(feat)} feature rows")

# dedup on identity (a re-run/double-process could duplicate a cell; keep first)
tsv = tsv.drop_duplicates(subset=ID, keep="first")
feat = feat.drop_duplicates(subset=ID, keep="first")
print(f"after dedup: {len(tsv)} TSV, {len(feat)} feat")

fc = [c for c in feat.columns if c.startswith("feat_")]
assert len(fc) == 372, f"expected 372 feat cols, got {len(fc)}"
feat = feat[ID + fc]
merged = tsv.merge(feat, on=ID, how="inner", validate="one_to_one")
print(f"MERGED: {len(merged)} rows (inner join)")
if len(merged) != len(tsv):
    print(f"WARNING: merged {len(merged)} != TSV {len(tsv)} — some cells lack features")

# canonical names + derived cols
ren = {"score_zensim_gpu": "score_zensim", "score_butteraugli_max_gpu": "score_butteraugli_max",
       "score_butteraugli_pnorm3_gpu": "score_butteraugli_pnorm3", "score_cvvdp_imazen_v0_0_1": "score_cvvdp",
       "score_iwssim_gpu": "score_iwssim"}
merged = merged.rename(columns=ren)
def pk(k, key):
    try: return json.loads(k).get(key)
    except Exception: return None
merged["distance"] = merged["knob_tuple_json"].map(lambda k: pk(k, "distance"))
merged["effort"] = merged["knob_tuple_json"].map(lambda k: pk(k, "effort"))
merged["variant_name"] = merged["image_path"]

lead = ["image_path", "variant_name", "codec", "q", "knob_tuple_json", "distance", "effort",
        "encoded_bytes", "encode_ms", "decode_ms", "encoded_filename",
        "score_zensim", "score_ssim2", "score_butteraugli_max", "score_butteraugli_pnorm3",
        "score_cvvdp", "score_dssim", "score_iwssim"]
merged = merged[lead + fc]

print("\n=== FINAL SANITY ===")
for c in ["score_zensim", "score_ssim2", "score_butteraugli_max", "score_butteraugli_pnorm3",
          "score_cvvdp", "score_dssim", "score_iwssim"]:
    v = merged[c].to_numpy(float); print(f"  {c:26s} non-null {int(np.isfinite(v).sum())}/{len(v)} [{np.nanmin(v):.4f},{np.nanmax(v):.4f}]")
fm = merged[fc].to_numpy(float)
z = merged["score_zensim"].to_numpy(float)
print(f"  feat_* NaN={int(np.isnan(fm).sum())} Inf={int(np.isinf(fm).sum())}")
print(f"  zensim A: max={np.nanmax(z):.4f} (A~97.69 never 100), n==100={(z>=99.9999).sum()}")
print(f"  rows={len(merged)} (target 62958)  distances={sorted(merged['distance'].dropna().unique().tolist())}")
print(f"  renditions={merged['image_path'].nunique()} (target 4497)")

tbl = pa.Table.from_pandas(merged, preserve_index=False).replace_schema_metadata({
    b"sweep": b"jxl-lossy-hqfill-A-2026-07-01",
    b"zensim_profile": b"A (zensim-a)",
    b"halves": f"salvage local chunks 0-10 ({n_sal}) + remote vast-GPU chunks ({n_rem}, run {REMOTE_RUN})".encode(),
})
pq.write_table(tbl, FINAL, compression="zstd")
print(f"\nWROTE {FINAL} ({os.path.getsize(FINAL)/1e6:.1f} MB, {tbl.num_rows}x{tbl.num_columns})")
