#!/usr/bin/env python3
# Build ScoreFile job-system inputs for a codec, grouped by SOURCE FILE (no re-encode, no variants
# re-upload — the executor byte-range-fetches each variant out of the EXISTING variants.tar). Computes
# the variant content sha from the bytes (no encode_index dependency — some codecs lack one). Outputs:
#   jobs/<run>/variant_index.tsv : sha \t offset \t size   (byte ranges into variants.tar)
#   jobs/<run>/manifest.json[.gz]: ONE DesiredJob per source-file CHUNK (kind=score_file, inputs = the
#     chunk's variant shas). jobexec decodes the ref ONCE, fetches+decodes each variant once, scores all
#     6 metrics + 372 features. Chunked (ZEN_SCOREFILE_CHUNK, default 12) so resumability + the OOM blast
#     radius are finer than per-file.
#   usage: build_scorefile_manifest.py <codec_dir> <ext> <run_id>
import json, csv, os, sys, tarfile, subprocess, gzip, hashlib
codec, ext, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
DGP = os.environ.get("ZEN_DATAGEN_PREFIX", "picker-sweep-2026-06-22/datagen-2026-06-23")
METRICS = ["butteraugli-gpu", "cvvdp", "dssim-gpu", "iwssim-gpu", "ssim2-gpu", "zensim-gpu"]
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
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
print("reading tar members (offset + size + sha256 from bytes)...", flush=True)
name2info = {}  # basename -> (offset_data, size, sha256-of-bytes)
with tarfile.open(tar_local, "r") as tf:
    for m in tf:
        if m.isfile():
            b = tf.extractfile(m).read()
            name2info[os.path.basename(m.name)] = (m.offset_data, m.size, hashlib.sha256(b).hexdigest())
print("  %d tar members" % len(name2info), flush=True)
idx_path = "%s/variant_index.tsv" % work
with open(idx_path, "w") as f:
    for name, (off, sz, sha) in name2info.items():
        f.write("%s\t%d\t%d\n" % (sha, off, sz))
r2("cp", idx_path, "s3://codec-corpus/jobs/%s/variant_index.tsv" % RUN)
print("uploaded variant_index.tsv (%d shas)" % len(name2info), flush=True)
# Group by source file via pairs.tsv (ref_path, dist_path) — the canonical (ref, variant) map. The
# omni's encoded_filename is unreliable across codecs (webp's is truncated, missing _q<Q>_<hash>.<ext>);
# the pairs.tsv dist_path is the actual persisted variant filename (matches the tar member basename).
files = {}
for r in csv.DictReader(r2cat("%s/%s/pairs.tsv" % (DGP, codec)).splitlines(), delimiter="\t"):
    dp = r.get("dist_path")
    name = os.path.basename(dp) if dp else None
    info = name2info.get(name) if name else None
    if not info:
        continue
    bn = os.path.basename(r["image_path"])
    files.setdefault(bn, {"codec": r["codec"], "shas": []})["shas"].append(info[2])
# Chunk each file's variants so resumability + the OOM blast radius are FINER than per-file: a file
# could have thousands of variants, and a per-file job that OOMs/dies partway would lose all of them.
# Each chunk is an independent DesiredJob (own content-addressed job_id) so the job system retries/
# poisons per chunk. The ref (a cheap PNG decode) is re-decoded per chunk; the variant decode-once-
# for-all-6-metrics win is fully kept.
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
print("uploaded manifest: %d chunk jobs, %d variants (chunk=%d) for %d files"
      % (len(manifest), tot, CHUNK, len(files)), flush=True)
