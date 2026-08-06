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
# Two-stage (Encode->ScoreFile) runs: jobs live in the RUN bucket (zentrain), not codec-corpus,
# and the cell->encode_sha map comes from pairs_from_encode_ledger.py's parquet instead of
# hashing variants.tar members. Both default OFF = byte-identical June behaviour.
JOBS_BUCKET = os.environ.get("ZEN_JOBS_BUCKET", "codec-corpus")
PAIRS_PARQUET = os.environ.get("ZEN_PAIRS_PARQUET")  # bridge parquet: image_path/q/knob_tuple_json/encode_sha
METRICS = [m for m in os.environ.get(
    "ZEN_WRITEBACK_METRICS",
    "butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu,ssim2-gpu,zensim-gpu,zensim-foldapp2,zensim-foldapp").split(",") if m]
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def s5(*a): subprocess.run(["s5cmd", "--endpoint-url", ep, *a], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def s5cat(key): return subprocess.run(["s5cmd", "--endpoint-url", ep, "cat", "s3://codec-corpus/%s" % key],
                                      env=env, check=True, capture_output=True).stdout.decode()
csv.field_size_limit(1 << 24)
# Work dir is (codec, first-run)-scoped: a codec-only dir accumulates blobs across UNRELATED runs
# and step 2 would parse every stale cached blob (harmless for content-sha joins, expensive and
# confusing). Old codec-only dirs remain valid caches for reruns of their own runs.
work = "/mnt/v/zen/writeback-%s-%s" % (codec, RUNS[0]); os.makedirs(work, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)

# 1) download all blobs
bdir = "%s/blobs" % work; os.makedirs(bdir, exist_ok=True)
for RUN in RUNS:
    print("downloading blobs from %s..." % RUN, flush=True)
    s5("cp", "s3://%s/jobs/%s/blobs/*" % (JOBS_BUCKET, RUN), bdir + "/")
blobs = glob.glob(bdir + "/*")
print("  %d blobs" % len(blobs), flush=True)

# 2) parse blobs -> metric_data[(sha,metric)] = scores{}, feat_data[sha] = (zensim_score, [feat...])
# Full-URI declares (declare_direct_objects.py ZEN_FULL_URI=1) put the whole s3://.../blobs/<sha>
# in encode_sha — normalize to the bare content sha so the pairs join keys match.
def norm_sha(s): return os.path.basename(s or "")
metric_data = {}; feat_data = {}
for bp in blobs:
    with open(bp) as fh:
        for line in fh:
            if not line.strip(): continue
            r = json.loads(line)
            k = r.get("kind")
            if k == "metric":
                metric_data[(norm_sha(r["encode_sha"]), r["metric"])] = r.get("scores") or {r["metric"]: r.get("score")}
            elif k == "feature":
                feat_data[norm_sha(r["encode_sha"])] = (r.get("zensim_score"), r.get("features"))
print("  metric entries=%d, feature entries=%d" % (len(metric_data), len(feat_data)), flush=True)

# 3+4) CELL rows -> encode_sha. Two sources:
#   - ZEN_PAIRS_PARQUET (two-stage runs): the bridge parquet ALREADY carries the content sha per cell.
#   - classic June layout: hash variants.tar members, join via pairs.tsv basename(dist_path).
cells = []
if PAIRS_PARQUET:
    t = pq.read_table(PAIRS_PARQUET,
                      columns=["image_path", "q", "knob_tuple_json", "encode_sha"]).to_pydict()
    for i in range(len(t["image_path"])):
        cells.append({"image_path": t["image_path"][i], "q": t["q"][i],
                      "knob_tuple_json": t["knob_tuple_json"][i], "dist_sha": norm_sha(t["encode_sha"][i])})
    print("  %d cells from %s" % (len(cells), PAIRS_PARQUET), flush=True)
else:
    tar_local = "%s/variants.tar" % work
    print("downloading variants.tar...", flush=True)
    s5("cp", "s3://codec-corpus/%s/%s/variants.tar" % (DGP, codec), tar_local)
    name2sha = {}
    with tarfile.open(tar_local, "r") as tf:
        for m in tf:
            if m.isfile():
                name2sha[os.path.basename(m.name)] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    print("  %d variant shas" % len(name2sha), flush=True)
    for c in csv.DictReader(s5cat("%s/%s/pairs.tsv" % (DGP, codec)).splitlines(), delimiter="\t"):
        cells.append({"image_path": c.get("image_path", ""), "q": c.get("q"),
                      "knob_tuple_json": c.get("knob_tuple_json", ""),
                      "dist_sha": name2sha.get(os.path.basename(c.get("dist_path", "")))})

all_score_cols = set()
score_rows = []; feat_rows = []
miss_sha = miss_score = 0
for c in cells:
    sha = c["dist_sha"]
    if not sha: miss_sha += 1; continue
    try: q = int(c["q"])
    except (ValueError, TypeError): q = -1
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

# 5) write parquet (ragged-safe: r.get(col) -> None fill). Feature width is taken from the
# data (372 = with-iw v1, 720 = V2Ab, 924 = foldapp, 944 = foldapp2) — never assumed.
ID = ["image_path", "q", "knob_tuple_json", "encode_sha"]
scols = ID + sorted(all_score_cols)
pq.write_table(pa.table({c: [r.get(c) for r in score_rows] for c in scols}),
               "%s/scores.parquet" % OUTDIR, compression="zstd")
n_feat = max((len(f[1]) for f in feat_data.values() if f and f[1]), default=372)
fcols = ID + ["zensim_score"] + ["feat_%d" % i for i in range(n_feat)]
pq.write_table(pa.table({c: [r.get(c) for r in feat_rows] for c in fcols}),
               "%s/features.parquet" % OUTDIR, compression="zstd")
print("WROTE %s/{scores,features}.parquet — scores %d rows x %d cols, features %d rows x %d feat"
      % (OUTDIR, len(score_rows), len(scols), len(feat_rows), n_feat), flush=True)
