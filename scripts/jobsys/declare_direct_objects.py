#!/usr/bin/env python3
# Declare ScoreFile jobs for a corpus whose variants are individually addressable R2
# objects (encodes/<name>). NO tar streaming, NO byte-range index — the job input IS
# the encode filename; the worker GETs <ZEN_ENCODES_PREFIX>/<name> in-process.
#   usage: declare_direct_objects.py <pairs.parquet[,...]> <run_id> <jobs_bucket>
import json, os, sys, gzip, subprocess, pyarrow.parquet as pq
pairs_arg, RUN, BUCKET = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
METRICS = [m for m in os.environ.get("ZEN_SCOREFILE_METRICS", "zensim-gpu").split(",") if m]
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def r2cp(local, key): subprocess.run(["s5cmd","--endpoint-url",ep,"cp",local,"s3://%s/%s"%(BUCKET,key)], env=env, check=True, stdout=subprocess.DEVNULL)
files = {}  # ref basename -> [dist_member,...]
for pp in pairs_arg.split(","):
    have = set(pq.read_schema(pp).names)
    full = os.environ.get("ZEN_FULL_URI") == "1"
    ipc = "ref_path" if full else "image_path"
    memc = "dist_path" if full else "dist_member"
    t = pq.read_table(pp, columns=[ipc, memc])
    for ip, dm in zip(t[ipc].to_pylist(), t[memc].to_pylist()):
        if not ip or not dm: continue
        key = ip if full else os.path.basename(ip)   # full s3 uri, or ref basename
        val = dm if full else os.path.basename(dm)
        files.setdefault(key, []).append(val)
manifest = []
for bn, members in files.items():
    for i in range(0, len(members), CHUNK):
        manifest.append({"kind": {"kind": "score_file", "metrics": METRICS},
                         "inputs": members[i:i+CHUNK],
                         "cell": {"image_path": bn, "codec": "zenjpeg", "q": -1, "knob_tuple_json": "scorefile"},
                         "hint": None})
work = "/home/lilith/tmp/hz720"; os.makedirs(work, exist_ok=True)
mp = "%s/manifest_direct.json" % work
json.dump(manifest, open(mp, "w"))
with open(mp,"rb") as fi, gzip.open(mp+".gz","wb") as g: g.write(fi.read())
r2cp(mp, "jobs/%s/manifest.json" % RUN); r2cp(mp+".gz", "jobs/%s/manifest.json.gz" % RUN)
open(work+"/ctl.json","w").write('{"paused":false}'); r2cp(work+"/ctl.json","jobs/%s/control.json"%RUN)
print("declared %d chunk jobs for %d sources -> s3://%s/jobs/%s/ (direct-object, no index)" % (len(manifest), len(files), BUCKET, RUN))
