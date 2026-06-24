#!/usr/bin/env python3
# Write-back: join a ScoreFile run's JSONL blobs (per-variant metric scores + 372-feature zensim sidecar,
# each keyed on encode_sha) back to the codec's (image_path, q, knob) CELL identity, producing two joinable
# parquet training sidecars per codec:
#   scores.parquet   : ID + every metric's flattened sub-scores (butteraugli_max_gpu, cvvdp_*, dssim_*,
#                      iwssim_*, ssim2_*, zensim_*) + the feature row's zensim_score
#   features.parquet : ID + feat_0..feat_371 (the with-iw 372-feature zensim sidecar)
# ID = (image_path, q, knob_tuple_json, encode_sha). The encode_sha -> cell map comes from the codec's
# pairs.tsv (basename(dist_path) -> variant name) joined to the variant CONTENT sha (sha256 of the bytes
# in variants.tar) — the same sha the executor scored under. Duplicate blobs (re-scores from claim races)
# dedup last-wins per (encode_sha, metric).
#   usage: writeback_scores.py <codec_dir> <ext> <run_id>
import json, csv, os, sys, tarfile, subprocess, hashlib, glob
import pyarrow as pa, pyarrow.parquet as pq
codec, ext = sys.argv[1], sys.argv[2]
RUNS = sys.argv[3].split(",")  # comma-sep: merge blobs from multiple runs (e.g. main + gap-fill)
DGP = os.environ.get("ZEN_DATAGEN_PREFIX", "picker-sweep-2026-06-22/datagen-2026-06-23")
OUTDIR = os.environ.get("ZEN_WRITEBACK_DIR", "/mnt/v/zen/zensim-training/2026-06-24/unified/%s" % codec)
METRICS = ["butteraugli-gpu", "cvvdp", "dssim-gpu", "iwssim-gpu", "ssim2-gpu", "zensim-gpu"]
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def s5(*a): subprocess.run(["s5cmd", "--endpoint-url", ep, *a], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def s5cat(key): return subprocess.run(["s5cmd", "--endpoint-url", ep, "cat", "s3://codec-corpus/%s" % key],
                                      env=env, check=True, capture_output=True).stdout.decode()
csv.field_size_limit(1 << 24)
work = "/mnt/v/zen/writeback-%s" % codec; os.makedirs(work, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)

# 1) download all blobs
bdir = "%s/blobs" % work; os.makedirs(bdir, exist_ok=True)
for RUN in RUNS:
    print("downloading blobs from %s..." % RUN, flush=True)
    s5("cp", "s3://codec-corpus/jobs/%s/blobs/*" % RUN, bdir + "/")
blobs = glob.glob(bdir + "/*")
print("  %d blobs" % len(blobs), flush=True)

# 2) parse blobs -> metric_data[(sha,metric)] = scores{}, feat_data[sha] = (zensim_score, [372])
metric_data = {}; feat_data = {}
for bp in blobs:
    with open(bp) as fh:
        for line in fh:
            if not line.strip(): continue
            r = json.loads(line)
            k = r.get("kind")
            if k == "metric":
                metric_data[(r["encode_sha"], r["metric"])] = r.get("scores") or {r["metric"]: r.get("score")}
            elif k == "feature":
                feat_data[r["encode_sha"]] = (r.get("zensim_score"), r.get("features"))
print("  metric entries=%d, feature entries=%d" % (len(metric_data), len(feat_data)), flush=True)

# 3) name -> content sha from variants.tar bytes (the sha the executor scored under)
tar_local = "%s/variants.tar" % work
print("downloading variants.tar...", flush=True)
s5("cp", "s3://codec-corpus/%s/%s/variants.tar" % (DGP, codec), tar_local)
name2sha = {}
with tarfile.open(tar_local, "r") as tf:
    for m in tf:
        if m.isfile():
            name2sha[os.path.basename(m.name)] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
print("  %d variant shas" % len(name2sha), flush=True)

# 4) iterate pairs.tsv CELLS -> join encode_sha -> blob data
all_score_cols = set()
score_rows = []; feat_rows = []
miss_sha = miss_score = 0
for c in csv.DictReader(s5cat("%s/%s/pairs.tsv" % (DGP, codec)).splitlines(), delimiter="\t"):
    name = os.path.basename(c.get("dist_path", "")); sha = name2sha.get(name)
    if not sha: miss_sha += 1; continue
    try: q = int(c["q"])
    except (ValueError, KeyError): q = -1
    base = {"image_path": os.path.basename(c["image_path"]), "q": q,
            "knob_tuple_json": c.get("knob_tuple_json", ""), "encode_sha": sha}
    srow = dict(base); got = False
    for m in METRICS:
        sc = metric_data.get((sha, m))
        if sc:
            srow.update(sc); all_score_cols.update(sc.keys()); got = True
    ft = feat_data.get(sha)
    if ft and ft[1]:
        srow["zensim_score"] = ft[0]; all_score_cols.add("zensim_score")
        frow = dict(base); frow["zensim_score"] = ft[0]
        for i, v in enumerate(ft[1]): frow["feat_%d" % i] = v
        feat_rows.append(frow); got = True
    if got: score_rows.append(srow)
    else: miss_score += 1
print("  score_rows=%d feat_rows=%d (miss_sha=%d miss_score=%d)"
      % (len(score_rows), len(feat_rows), miss_sha, miss_score), flush=True)

# 5) write parquet (ragged-safe: r.get(col) -> None fill)
ID = ["image_path", "q", "knob_tuple_json", "encode_sha"]
scols = ID + sorted(all_score_cols)
pq.write_table(pa.table({c: [r.get(c) for r in score_rows] for c in scols}),
               "%s/scores.parquet" % OUTDIR, compression="zstd")
fcols = ID + ["zensim_score"] + ["feat_%d" % i for i in range(372)]
pq.write_table(pa.table({c: [r.get(c) for r in feat_rows] for c in fcols}),
               "%s/features.parquet" % OUTDIR, compression="zstd")
print("WROTE %s/{scores,features}.parquet — scores %d rows x %d cols, features %d rows x 372 feat"
      % (OUTDIR, len(score_rows), len(scols), len(feat_rows)), flush=True)
