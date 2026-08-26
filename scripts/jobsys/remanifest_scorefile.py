#!/usr/bin/env python3
# Re-emit a ScoreFile manifest for an EXISTING run's variant_index.tsv with a
# different metric set / chunk size — no tar streaming (the index already
# carries sha/offset/size/name). Use case: moving a fill from GPU boxes to
# Hetzner CPU boxes (CPU metric names) without recomputing 189k member shas.
#   usage: remanifest_scorefile.py <src_run> <dst_run> <pairs.parquet[,...]> \
#              --metrics butteraugli,cvvdp,dssim,iwssim [--chunk 12]
import argparse, gzip, json, os, subprocess, sys, tempfile
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from zen_s3env import resolve_full  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src_run"); ap.add_argument("dst_run"); ap.add_argument("pairs")
ap.add_argument("--metrics", required=True)
ap.add_argument("--chunk", type=int, default=int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12")))
a = ap.parse_args()
METRICS = a.metrics.split(",")
# Store resolution via scripts/lib/zen_s3env.py -- defaults to the LAN store,
# ZEN_STORE=r2 opts out. Fixed 2026-08-26 (was: unconditional R2_* env vars).
ep, _ak, _sk, _store_kind, _reachable = resolve_full()
print(f"remanifest_scorefile: store={_store_kind} endpoint={ep}", file=sys.stderr)
env = dict(os.environ, AWS_ACCESS_KEY_ID=_ak, AWS_SECRET_ACCESS_KEY=_sk, AWS_REGION="auto")
def r2(*args):
    subprocess.run(["s5cmd", "--endpoint-url", ep, *args], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

work = tempfile.mkdtemp(prefix="remanifest-")
idx_local = os.path.join(work, "variant_index.tsv")
r2("cp", "s3://codec-corpus/jobs/%s/variant_index.tsv" % a.src_run, idx_local)
name2sha = {}
for line in open(idx_local):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 4 and p[3]:
        name2sha[os.path.basename(p[3])] = p[0]
print("index: %d named members" % len(name2sha), flush=True)
if not name2sha:
    print("FATAL: index has no 4th name column — cannot remanifest without it"); sys.exit(1)

files = {}
matched = 0
for pp in a.pairs.split(","):
    t = pq.read_table(pp, columns=["image_path", "codec", "dist_member"])
    for ip, codec, dm in zip(t["image_path"].to_pylist(), t["codec"].to_pylist(),
                             t["dist_member"].to_pylist()):
        sha = name2sha.get(dm)
        if sha is None:
            continue
        matched += 1
        files.setdefault(os.path.basename(ip), {"codec": codec, "shas": []})["shas"].append(sha)
print("pairs matched to index: %d cells across %d files" % (matched, len(files)), flush=True)

manifest = []
for bn, info in files.items():
    shas = info["shas"]
    for i in range(0, len(shas), a.chunk):
        manifest.append({"kind": {"kind": "score_file", "metrics": METRICS}, "inputs": shas[i:i + a.chunk],
                         "cell": {"image_path": bn, "codec": info["codec"], "q": -1,
                                  "knob_tuple_json": "scorefile"}, "hint": None})
mpath = os.path.join(work, "manifest.json")
json.dump(manifest, open(mpath, "w"))
with open(mpath, "rb") as fi, gzip.open(mpath + ".gz", "wb") as g:
    g.write(fi.read())
r2("cp", idx_local, "s3://codec-corpus/jobs/%s/variant_index.tsv" % a.dst_run)
r2("cp", mpath, "s3://codec-corpus/jobs/%s/manifest.json" % a.dst_run)
r2("cp", mpath + ".gz", "s3://codec-corpus/jobs/%s/manifest.json.gz" % a.dst_run)
print("uploaded run %s: %d chunk jobs (chunk=%d, metrics=%s)" % (a.dst_run, len(manifest), a.chunk, METRICS), flush=True)
