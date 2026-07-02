#!/usr/bin/env python3
# Build ScoreFile job-system inputs from a CANONICAL pairs.parquet + one generation box-tar —
# the fill path for corpora whose encodes live in per-box generation tars (dist_tar/dist_member
# columns) rather than a single datagen variants.tar. Same outputs as build_scorefile_manifest.py:
#   jobs/<run>/variant_index.tsv : sha \t offset \t size \t name   (4-col: enables tar-shard mode)
#   jobs/<run>/manifest.json[.gz]: one DesiredJob per source-file CHUNK (kind=score_file)
# The tar is STREAMED from R2 (s5cmd cat | tarfile r|) — no local disk, one pass computes
# offset+size+sha256 per member. Cells come from the pairs parquet rows whose dist_tar basename
# matches this tar; refs resolve via ZEN_CORPUS_PREFIX at launch (pairs ref_path bucket/prefix).
#   usage: build_scorefile_from_pairs.py <pairs.parquet[,pairs2,...]> <tar_uri> <run_id>
#   env:   ZEN_SCOREFILE_CHUNK (default 12), ZEN_SKIP_SHAS_FILE (gap-fill)
import json, os, sys, tarfile, subprocess, gzip, hashlib
import pyarrow.parquet as pq

pairs_arg, TAR_URI, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
METRICS = ["butteraugli-gpu", "cvvdp", "dssim-gpu", "iwssim-gpu", "ssim2-gpu", "zensim-gpu"]
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")

def r2cp(src, dst):
    subprocess.run(["s5cmd", "--endpoint-url", ep, "cp", src, dst], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

tar_base = os.path.basename(TAR_URI)
work = "/mnt/v/zen/scorefile-frompairs-%s" % RUN
os.makedirs(work, exist_ok=True)

# 1. cells for THIS tar, from the pairs parquet(s), matched on dist_tar basename
files = {}   # source basename -> {codec, members:[dist_member,...]}
want = {}    # dist_member -> source basename
for pp in pairs_arg.split(","):
    t = pq.read_table(pp, columns=["image_path", "codec", "dist_tar", "dist_member"])
    for ip, codec, dt, dm in zip(t["image_path"].to_pylist(), t["codec"].to_pylist(),
                                 t["dist_tar"].to_pylist(), t["dist_member"].to_pylist()):
        if os.path.basename(dt) != tar_base:
            continue
        bn = os.path.basename(ip)
        want[dm] = bn
        files.setdefault(bn, {"codec": codec, "shas": []})
print("pairs: %d cells across %d source files reference %s" % (len(want), len(files), tar_base), flush=True)
if not want:
    print("FATAL: no pairs rows reference this tar", flush=True); sys.exit(1)

# 2. stream the tar once: offset + size + sha256 per wanted member (no local disk)
SKIP = set()
skf = os.environ.get("ZEN_SKIP_SHAS_FILE")
if skf and os.path.exists(skf):
    SKIP = {l.strip() for l in open(skf) if l.strip()}
    print("skip-shas: %d loaded (gap-fill mode)" % len(SKIP), flush=True)
proc = subprocess.Popen(["s5cmd", "--endpoint-url", ep, "cat", TAR_URI], env=env,
                        stdout=subprocess.PIPE, bufsize=1 << 22)
idx_path = "%s/variant_index.tsv" % work
n_idx = n_skip = 0
with tarfile.open(fileobj=proc.stdout, mode="r|") as tf, open(idx_path, "w") as fidx:
    for m in tf:
        if not m.isfile():
            continue
        name = os.path.basename(m.name)
        src = want.get(name)
        if src is None:
            continue  # member not referenced by canonical pairs (e.g. non-canonical extras)
        b = tf.extractfile(m).read()
        sha = hashlib.sha256(b).hexdigest()
        fidx.write("%s\t%d\t%d\t%s\n" % (sha, m.offset_data, m.size, m.name))
        n_idx += 1
        if sha in SKIP:
            n_skip += 1
            continue
        files[src]["shas"].append(sha)
        if n_idx % 20000 == 0:
            print("  indexed %d members..." % n_idx, flush=True)
rc = proc.wait()
if rc != 0:
    print("FATAL: s5cmd cat rc=%d" % rc, flush=True); sys.exit(1)
print("indexed %d/%d wanted members (skipped %d already-scored)" % (n_idx, len(want), n_skip), flush=True)
if n_idx < len(want) * 0.98:
    print("FATAL: >2%% of wanted members missing from tar — wrong tar or truncated", flush=True); sys.exit(1)

# 3. chunked manifest (identical shape to build_scorefile_manifest.py)
manifest = []
for bn, info in files.items():
    shas = info["shas"]
    for i in range(0, len(shas), CHUNK):
        manifest.append({"kind": {"kind": "score_file", "metrics": METRICS}, "inputs": shas[i:i + CHUNK],
                         "cell": {"image_path": bn, "codec": info["codec"], "q": -1,
                                  "knob_tuple_json": "scorefile"}, "hint": None})
mpath = "%s/manifest.json" % work
json.dump(manifest, open(mpath, "w"))
with open(mpath, "rb") as fi, gzip.open(mpath + ".gz", "wb") as g:
    g.write(fi.read())
r2cp(idx_path, "s3://codec-corpus/jobs/%s/variant_index.tsv" % RUN)
r2cp(mpath, "s3://codec-corpus/jobs/%s/manifest.json" % RUN)
r2cp(mpath + ".gz", "s3://codec-corpus/jobs/%s/manifest.json.gz" % RUN)
tot = sum(len(i["shas"]) for i in files.values())
print("uploaded run %s: %d chunk jobs, %d variants (chunk=%d) for %d files"
      % (RUN, len(manifest), tot, CHUNK, len(files)), flush=True)
