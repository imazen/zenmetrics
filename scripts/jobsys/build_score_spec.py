#!/usr/bin/env python3
# Build a zenfleet-ctl declare spec.json from a codec's omni + encode_index on R2.
# Joins omni.encoded_filename <-> encode_index.name -> encode_sha; image_path = basename
# (matches the ref renditions under <PREFIX>/ref/ that jobexec resolves via ZEN_CORPUS_PREFIX).
#   usage: build_score_spec.py <codec> [out.json]   (env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY)
import csv, json, os, sys, subprocess
codec = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/%s_spec.json" % codec
P = os.environ.get("ZEN_DATAGEN_PREFIX", "picker-sweep-2026-06-22/datagen-2026-06-23")
METRICS = ["butteraugli-gpu", "cvvdp", "dssim-gpu", "iwssim-gpu", "ssim2-gpu", "zensim-gpu"]
ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto")
def dl(key, dst):
    subprocess.run(["s5cmd", "--endpoint-url", ep, "cp",
                    "s3://codec-corpus/%s/%s/%s" % (P, codec, key), dst],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
csv.field_size_limit(1 << 24)
dl("omni.tsv", "/tmp/%s_omni.tsv" % codec)
dl("encode_index.tsv", "/tmp/%s_idx.tsv" % codec)
sha = {}
with open("/tmp/%s_idx.tsv" % codec) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r.get("name") and r.get("sha256"):
            sha[r["name"]] = r["sha256"]
items, skipped = [], 0
with open("/tmp/%s_omni.tsv" % codec) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ef = r.get("encoded_filename") or ""
        if not ef or ef not in sha:
            skipped += 1; continue
        try:
            q = int(r["q"])
        except (KeyError, ValueError):
            skipped += 1; continue
        items.append({"image_path": os.path.basename(r["image_path"]),
                      "codec": r["codec"], "q": q,
                      "knob_tuple_json": r.get("knob_tuple_json", ""),
                      "encode_sha": sha[ef]})
json.dump({"items": items, "metrics": METRICS}, open(out, "w"))
print("spec: %d items x %d metrics = %d metric jobs -> %s (skipped %d)"
      % (len(items), len(METRICS), len(items) * len(METRICS), out, skipped))
