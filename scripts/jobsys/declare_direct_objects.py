#!/usr/bin/env python3
# Declare ScoreFile jobs for a corpus whose variants are individually addressable R2
# objects (encodes/<name>). NO tar streaming, NO byte-range index — the job input IS
# the encode filename; the worker GETs <ZEN_ENCODES_PREFIX>/<name> in-process.
#   usage: declare_direct_objects.py <pairs.parquet[,...]> <run_id> <jobs_bucket>
import json, os, sys, gzip, subprocess, pyarrow.parquet as pq
pairs_arg, RUN, BUCKET = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
METRICS = [m for m in os.environ.get("ZEN_SCOREFILE_METRICS", "zensim-gpu").split(",") if m]
# Cell codec label (metadata only — the join key is the variant filename in `inputs`, which already
# carries the codec). Default zenjpeg for back-compat; set ZEN_CELL_CODEC per codec for correct sidecars.
CELL_CODEC = os.environ.get("ZEN_CELL_CODEC", "zenjpeg")
# Optional scheduling hint attached to every declared job (does NOT enter the JobId).
# ZEN_SCOREFILE_HINT_MEM_GB sizes BoxBudget admission (host-RAM proxy; use it to bound
# concurrent GPU processes per box — the 2026-08-07 avifgen sf-gpu OOM storm was N unhinted
# GPU jobs admitted by host RAM on 8 GB cards). ZEN_SCOREFILE_HINT_THREADS bounds by cores.
_hm = os.environ.get("ZEN_SCOREFILE_HINT_MEM_GB")
_ht = os.environ.get("ZEN_SCOREFILE_HINT_THREADS")
HINT = None
if _hm or _ht:
    HINT = {"peak_mem_bytes": int(float(_hm or 0.5) * (1 << 30)), "threads": int(_ht or 1)}
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def r2cp(local, key): subprocess.run(["s5cmd","--endpoint-url",ep,"cp",local,"s3://%s/%s"%(BUCKET,key)], env=env, check=True, stdout=subprocess.DEVNULL)
files = {}  # ref basename -> [dist_member,...]
cellinfo = {}  # dist member -> (codec, q, knob_tuple_json) when the pairs carry identity
for pp in pairs_arg.split(","):
    have = set(pq.read_schema(pp).names)
    full = os.environ.get("ZEN_FULL_URI") == "1"
    ipc = "ref_path" if full else "image_path"
    memc = "dist_path" if full else "dist_member"
    cols = [ipc, memc]
    id_cols = [c for c in ("codec", "q", "knob_tuple_json") if c in have]
    t = pq.read_table(pp, columns=cols + id_cols)
    idlists = {c: t[c].to_pylist() for c in id_cols}
    for i, (ip, dm) in enumerate(zip(t[ipc].to_pylist(), t[memc].to_pylist())):
        if not ip or not dm: continue
        key = ip if full else os.path.basename(ip)   # full s3 uri, or ref basename
        val = dm if full else os.path.basename(dm)
        files.setdefault(key, []).append(val)
        if len(id_cols) == 3:
            cellinfo[val] = (idlists["codec"][i], idlists["q"][i], idlists["knob_tuple_json"][i])
# HDR ScoreFile (HDR-corpus): ZEN_SCOREFILE_HDR=1 adds "hdr":true (+ optional
# ZEN_SCOREFILE_HDR_TRANSFER=pu-rescale|pq) to the job kind — the executor then
# decodes ref+variants to absolute nits and applies the per-metric HDR feeding.
# The flag changes every content-addressed job id (correct: different work).
# Absent => byte-identical SDR manifests (append-only schema, like the Rust side).
HDR = os.environ.get("ZEN_SCOREFILE_HDR") == "1"
HDR_TRANSFER = os.environ.get("ZEN_SCOREFILE_HDR_TRANSFER", "")
# ZEN_DECLARE_KIND=diffmap (HDR-corpus B2): emit `diffmap` jobs instead of
# `score_file` — ONE job per (variant x metric), inputs = [the one variant URI]
# (the executor's one-blob-per-job contract; output = the gzip'd PFM map).
# CHUNK is ignored in this mode. Same full-URI resolution path as score_file.
DECLARE_KIND = os.environ.get("ZEN_DECLARE_KIND", "score_file")
manifest = []
for bn, members in files.items():
    if DECLARE_KIND == "diffmap":
        # Cell identity: the variant's TRUE (codec, q, knobs) when the pairs
        # parquet carries them (pairs_from_encode_ledger does) — the diffmap
        # ledger rows then join to encode cells directly. Multi-codec safe:
        # the executor derives the variant decode ext from cell.codec.
        for m in members:
            ci = cellinfo.get(m)
            cell = ({"image_path": bn, "codec": ci[0], "q": int(ci[1]), "knob_tuple_json": ci[2]}
                    if ci else
                    {"image_path": bn, "codec": CELL_CODEC, "q": -1, "knob_tuple_json": "diffmap"})
            for metric in METRICS:
                kind = {"kind": "diffmap", "metric": metric}
                if HDR:
                    kind["hdr"] = True
                manifest.append({"kind": kind, "inputs": [m], "cell": cell, "hint": None})
        continue
    for i in range(0, len(members), CHUNK):
        kind = {"kind": "score_file", "metrics": METRICS}
        if HDR:
            kind["hdr"] = True
            if HDR_TRANSFER:
                kind["hdr_transfer"] = HDR_TRANSFER
        manifest.append({"kind": kind,
                         "inputs": members[i:i+CHUNK],
                         "cell": {"image_path": bn, "codec": CELL_CODEC, "q": -1, "knob_tuple_json": "scorefile"},
                         "hint": HINT})
# ZEN_MANIFEST_OUT: write the manifest JSON locally and SKIP the upload — for
# multi-invocation runs (e.g. per-codec passes over one run id) whose parts are
# merged and uploaded once by the caller. Uploading per invocation would
# overwrite jobs/<run>/manifest.json with only the last part.
mo = os.environ.get("ZEN_MANIFEST_OUT", "")
if mo:
    json.dump(manifest, open(mo, "w"))
    print("wrote %d jobs for %d sources -> %s (no upload)" % (len(manifest), len(files), mo))
    sys.exit(0)
work = "/home/lilith/tmp/hz720"; os.makedirs(work, exist_ok=True)
mp = "%s/manifest_direct.json" % work
json.dump(manifest, open(mp, "w"))
with open(mp,"rb") as fi, gzip.open(mp+".gz","wb") as g: g.write(fi.read())
r2cp(mp, "jobs/%s/manifest.json" % RUN); r2cp(mp+".gz", "jobs/%s/manifest.json.gz" % RUN)
open(work+"/ctl.json","w").write('{"paused":false}'); r2cp(work+"/ctl.json","jobs/%s/control.json"%RUN)
print("declared %d chunk jobs for %d sources -> s3://%s/jobs/%s/ (direct-object, no index)" % (len(manifest), len(files), BUCKET, RUN))
