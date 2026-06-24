#!/usr/bin/env python3
# Build the ScoreFile job-system inputs for a codec, grouping by SOURCE FILE (no re-encode, no
# variants re-upload — the executor byte-range-fetches each variant out of the existing variants.tar):
#   - jobs/<run>/variant_index.tsv : sha \t offset \t size   (byte ranges from variants.tar headers)
#   - jobs/<run>/manifest.json[.gz]: ONE DesiredJob per source file (kind=score_file, inputs=that
#     file's variant shas, cell=the file). jobexec then decodes the ref ONCE and scores every metric
#     for every variant — replacing the per-(cell,metric) re-encode path.
#   usage: build_scorefile_manifest.py <codec_dir> <ext> <run_id>
import json, csv, os, sys, tarfile, subprocess, gzip
codec, ext, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
DGP = os.environ.get("ZEN_DATAGEN_PREFIX", "picker-sweep-2026-06-22/datagen-2026-06-23")
METRICS = ["butteraugli-gpu", "cvvdp", "dssim-gpu", "iwssim-gpu", "ssim2-gpu", "zensim-gpu"]
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def r2(*a): subprocess.run(["s5cmd", "--endpoint-url", ep, *a], env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def r2cat(key): return subprocess.run(["s5cmd", "--endpoint-url", ep, "cat", "s3://codec-corpus/%s" % key],
                                      env=env, check=True, capture_output=True).stdout.decode()
csv.field_size_limit(1 << 24)
work = "/mnt/v/zen/scorefile-%s" % codec; os.makedirs(work, exist_ok=True)
tar_local = "%s/variants.tar" % work
print("downloading variants.tar...", flush=True)
r2("cp", "s3://codec-corpus/%s/%s/variants.tar" % (DGP, codec), tar_local)
print("reading tar member offsets...", flush=True)
name2off = {}
with tarfile.open(tar_local, "r") as tf:
    for m in tf:
        if m.isfile():
            # tar members carry a './' prefix (hetzner_cpu_sweep tarred with `tar -C dir .`); the
            # encode_index keys on the bare filename — join on basename.
            name2off[os.path.basename(m.name)] = (m.offset_data, m.size)
print("  %d tar members" % len(name2off), flush=True)
name2sha = {}
for r in csv.DictReader(r2cat("%s/%s/encode_index.tsv" % (DGP, codec)).splitlines(), delimiter="\t"):
    if r.get("name") and r.get("sha256"):
        name2sha[r["name"]] = r["sha256"]
idx_path = "%s/variant_index.tsv" % work
n_idx = 0
with open(idx_path, "w") as f:
    for name, (off, sz) in name2off.items():
        sha = name2sha.get(name)
        if sha:
            f.write("%s\t%d\t%d\n" % (sha, off, sz)); n_idx += 1
r2("cp", idx_path, "s3://codec-corpus/jobs/%s/variant_index.tsv" % RUN)
print("uploaded variant_index.tsv (%d shas)" % n_idx, flush=True)
# group omni rows by source file basename -> per-file ScoreFile jobs
files = {}
for r in csv.DictReader(r2cat("%s/%s/omni.tsv" % (DGP, codec)).splitlines(), delimiter="\t"):
    ef = r.get("encoded_filename")
    sha = name2sha.get(ef) if ef else None
    if not sha:
        continue
    bn = os.path.basename(r["image_path"])
    files.setdefault(bn, {"codec": r["codec"], "shas": []})["shas"].append(sha)
# Chunk each file's variants so resumability + the OOM blast radius are FINER than per-file: a file
# could have thousands of variants, and a per-file job that OOMs or dies partway would lose all of
# them. Each chunk is an independent DesiredJob (its own content-addressed job_id), so the job system
# retries/poisons per chunk and converges across passes (declare -> gap -> reconcile) — an OOM that
# fails pass 1 is just retried. The ref (a cheap PNG decode) is re-decoded per chunk (negligible vs
# the eliminated re-encode); the variant decode-once-for-all-6-metrics win is fully kept.
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
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
r2("cp", mpath, "s3://codec-corpus/jobs/%s/manifest.json" % RUN)
r2("cp", mpath + ".gz", "s3://codec-corpus/jobs/%s/manifest.json.gz" % RUN)
tot = sum(len(i["shas"]) for i in files.values())
print("uploaded manifest: %d per-file jobs, %d variants total (vs %d per-(cell,metric) jobs)"
      % (len(manifest), tot, tot * len(METRICS)), flush=True)
