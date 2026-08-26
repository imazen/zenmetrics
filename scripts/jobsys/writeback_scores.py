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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from zen_s3env import resolve  # noqa: E402  (ZEN_S3_ENDPOINT overrides; default = R2, unchanged)
ep, _ak, _sk = resolve()
env = dict(os.environ, AWS_ACCESS_KEY_ID=_ak, AWS_SECRET_ACCESS_KEY=_sk, AWS_REGION="auto")
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
if os.environ.get("ZEN_SKIP_DOWNLOAD") == "1":
    print("ZEN_SKIP_DOWNLOAD=1 — using existing local blobs", flush=True)
else:
    for RUN in RUNS:
        print("downloading blobs from %s..." % RUN, flush=True)
        s5("cp", "s3://%s/jobs/%s/blobs/*" % (JOBS_BUCKET, RUN), bdir + "/")
blobs = glob.glob(bdir + "/*")
print("  %d blobs" % len(blobs), flush=True)

# 2) parse blobs -> metric_data[(sha,metric)] = scores{}, feat_data[sha] = (zensim_score, [feat...])
# Full-URI declares (declare_direct_objects.py ZEN_FULL_URI=1) put the whole s3://.../blobs/<sha>
# in encode_sha — normalize to the bare content sha so the pairs join keys match.
def norm_sha(s): return os.path.basename(s or "")
metric_data = {}; feat_data = {}; err_rows = 0
for bp in blobs:
    with open(bp) as fh:
        for line in fh:
            if not line.strip(): continue
            r = json.loads(line)
            k = r.get("kind")
            if k == "metric":
                # Per-cell ERROR records ride the blob stream (kind:"metric" + `error`, no
                # metric key — e.g. the hdrgrid R2-era failures). Skip + count; a retry that
                # later succeeded emits a separate scored row under the same encode_sha.
                if "metric" not in r or r.get("error"):
                    err_rows += 1
                    continue
                metric_data[(norm_sha(r["encode_sha"]), r["metric"])] = r.get("scores") or {r["metric"]: r.get("score")}
            elif k == "feature":
                feat_data[norm_sha(r["encode_sha"])] = (r.get("zensim_score"), r.get("features"), r.get("regime"))
print("  metric entries=%d, feature entries=%d, error rows skipped=%d" % (len(metric_data), len(feat_data), err_rows), flush=True)

# 3+4) CELL rows -> encode_sha. Two sources:
#   - ZEN_PAIRS_PARQUET (two-stage runs): the bridge parquet ALREADY carries the content sha per cell.
#   - classic June layout: hash variants.tar members, join via pairs.tsv basename(dist_path).
cells = []
if PAIRS_PARQUET:
    # codec rides through in two-stage mode: multi-arm manifests (hdrgrid: zenjxl /
    # zenav1-svt / jpeg-gainmap in ONE run) need the arm identity in the output tables.
    _pf = pq.read_table(PAIRS_PARQUET)
    _cols = ["image_path", "q", "knob_tuple_json", "encode_sha"] + (["codec"] if "codec" in _pf.column_names else [])
    t = _pf.select(_cols).to_pydict()
    for i in range(len(t["image_path"])):
        cells.append({"image_path": t["image_path"][i], "q": t["q"][i],
                      "knob_tuple_json": t["knob_tuple_json"][i], "dist_sha": norm_sha(t["encode_sha"][i]),
                      "codec": (t.get("codec") or [None]*len(t["image_path"]))[i]})
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

ID_COLS = ["image_path", "q", "knob_tuple_json", "encode_sha"]
if cells and cells[0].get("codec"):
    ID_COLS = ["image_path", "codec", "q", "knob_tuple_json", "encode_sha"]
all_score_cols = set()
score_rows = []

# Per-regime batched feature writers (regime purity; bounded memory).
_FEAT_BATCH = int(os.environ.get("ZEN_FEAT_BATCH", "20000"))
_fw = {}  # regime -> {"writer": ParquetWriter|None, "rows": [], "n": int, "width": int}
def _feat_batch_add(base, ft):
    zs, feats, regime = ft[0], ft[1], (ft[2] or "unknown")
    st = _fw.setdefault(regime, {"writer": None, "rows": [], "n": 0, "width": len(feats)})
    if len(feats) != st["width"]:
        raise SystemExit("feature width %d != %d within regime %r — refusing to mix"
                         % (len(feats), st["width"], regime))
    row = dict(base); row["zensim_score"] = zs
    for i, v in enumerate(feats): row["feat_%d" % i] = v
    st["rows"].append(row); st["n"] += 1
    if len(st["rows"]) >= _FEAT_BATCH: _feat_flush(regime)
def _feat_flush(regime):
    st = _fw[regime]
    if not st["rows"]: return
    cols = ID_COLS + ["zensim_score"] + ["feat_%d" % i for i in range(st["width"])]
    tbl = pa.table({c: [r.get(c) for r in st["rows"]] for c in cols})
    if st["writer"] is None:
        os.makedirs(OUTDIR, exist_ok=True)
        st["writer"] = pq.ParquetWriter("%s/features_%s.parquet" % (OUTDIR, regime),
                                        tbl.schema, compression="zstd")
    st["writer"].write_table(tbl); st["rows"] = []
miss_sha = miss_score = 0
for c in cells:
    sha = c["dist_sha"]
    if not sha: miss_sha += 1; continue
    try: q = int(c["q"])
    except (ValueError, TypeError): q = -1
    base = {"image_path": os.path.basename(c["image_path"]), "q": q,
            "knob_tuple_json": c.get("knob_tuple_json", ""), "encode_sha": sha}
    if c.get("codec"): base["codec"] = c["codec"]
    srow = dict(base); got = False
    for m in METRICS:
        sc = metric_data.get((sha, m))
        if sc:
            srow.update(sc); all_score_cols.update(sc.keys()); got = True
    ft = feat_data.get(sha)
    if ft and ft[1]:
        srow["zensim_score"] = ft[0]; all_score_cols.add("zensim_score")
        got = True
        # Features are written INCREMENTALLY, one parquet PER REGIME (regime purity:
        # never column-mix 372/720/924/944 rows), via batched ParquetWriter — the
        # old whole-table feat_rows list OOM-killed a 60G box (57.9G rss, 2026-08-26).
        if os.environ.get("ZEN_SKIP_FEATURES") != "1":
            _feat_batch_add(base, ft)
    if got: score_rows.append(srow)
    else: miss_score += 1
print("  score_rows=%d feat_streamed=%d (miss_sha=%d miss_score=%d)"
      % (len(score_rows), sum(st["n"] for st in _fw.values()), miss_sha, miss_score), flush=True)

# 5) write parquet (ragged-safe: r.get(col) -> None fill). Feature width is taken from the
# data (372 = with-iw v1, 720 = V2Ab, 924 = foldapp, 944 = foldapp2) — never assumed.
ID = ["image_path", "q", "knob_tuple_json", "encode_sha"]
if any("codec" in r for r in score_rows[:1]):
    ID = ["image_path", "codec", "q", "knob_tuple_json", "encode_sha"]
scols = ID + sorted(all_score_cols)
pq.write_table(pa.table({c: [r.get(c) for r in score_rows] for c in scols}),
               "%s/scores.parquet" % OUTDIR, compression="zstd")
if os.environ.get("ZEN_SKIP_FEATURES") == "1":
    print("ZEN_SKIP_FEATURES=1 — features NOT written (scores-only pass)", flush=True)
else:
    for _rg in list(_fw):
        _feat_flush(_rg)
        if _fw[_rg]["writer"] is not None: _fw[_rg]["writer"].close()
        print("  features_%s.parquet: %d rows x %d feat" % (_rg, _fw[_rg]["n"], _fw[_rg]["width"]), flush=True)
print("WROTE %s — scores %d rows x %d cols (+ per-regime feature files above)"
      % (OUTDIR, len(score_rows), len(scols)), flush=True)
