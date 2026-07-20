#!/usr/bin/env python3
# Runs ON the Hetzner index box. Streams ONE per-box variants tar ONCE (r| mode, no whole-file
# download-to-disk), and from its member headers builds BOTH:
#   * a byte-range index  (dist_member \t offset \t size \t dist_member)  -> jobs/<run>/variant_index.tsv
#   * a ScoreFile manifest (score_file jobs, inputs = dist_members, cell.image_path = derived ref)
# The ref is derived from the variant filename: <refbase>_<16hex refhash>_<codec>_q<q>_<hash>.<ext>
# -> <refbase>.png. No pairs parquet needed; the tar's members ARE the variant set.
#   usage: index_and_declare.py <tar_s3_uri> <cell_codec> <run> <bucket>
import sys, os, re, json, gzip, subprocess, tarfile, time
tar_uri, CODEC, RUN, BUCKET = sys.argv[1:5]
EP = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
ENV = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["AWS_ACCESS_KEY_ID"],
           AWS_SECRET_ACCESS_KEY=os.environ["AWS_SECRET_ACCESS_KEY"], AWS_REGION="auto")
if os.environ.get("AWS_SESSION_TOKEN"):
    ENV["AWS_SESSION_TOKEN"] = os.environ["AWS_SESSION_TOKEN"]
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
METRICS = [m for m in os.environ.get("ZEN_SCOREFILE_METRICS", "zensim-gpu").split(",") if m]
REFHASH = re.compile(r'(.+?)_[0-9a-f]{16}_')

def ref_of(dm):
    m = REFHASH.match(dm)
    return (m.group(1) + '.png') if m else None

t0 = time.time()
p = subprocess.Popen(["s5cmd", "--endpoint-url", EP, "cat", tar_uri], stdout=subprocess.PIPE, env=ENV)
tf = tarfile.open(fileobj=p.stdout, mode="r|")
idx = []
by_ref = {}
noref = 0
for m in tf:
    if not m.isfile():
        continue
    dm = os.path.basename(m.name)
    idx.append("%s\t%d\t%d\t%s" % (dm, m.offset_data, m.size, dm))
    r = ref_of(dm)
    if r is None:
        noref += 1
        continue
    by_ref.setdefault(r, []).append(dm)
rc = p.wait()
if rc != 0:
    sys.exit("s5cmd cat rc=%d" % rc)
manifest = []
for ref, members in by_ref.items():
    for i in range(0, len(members), CHUNK):
        manifest.append({"kind": {"kind": "score_file", "metrics": METRICS},
                         "inputs": members[i:i + CHUNK],
                         "cell": {"image_path": ref, "codec": CODEC, "q": -1, "knob_tuple_json": "scorefile"},
                         "hint": None})
work = "/root/idxwork/%s" % RUN
os.makedirs(work, exist_ok=True)
open(work + "/variant_index.tsv", "w").write("\n".join(idx) + "\n")
json.dump(manifest, open(work + "/manifest.json", "w"))
with open(work + "/manifest.json", "rb") as fi, gzip.open(work + "/manifest.json.gz", "wb") as g:
    g.write(fi.read())
open(work + "/control.json", "w").write('{"paused":false}')

def up(f, k):
    subprocess.run(["s5cmd", "--endpoint-url", EP, "cp", work + "/" + f,
                    "s3://%s/jobs/%s/%s" % (BUCKET, RUN, k)], env=ENV, check=True,
                   stdout=subprocess.DEVNULL)

for f in ("variant_index.tsv", "manifest.json", "manifest.json.gz", "control.json"):
    up(f, f)
print("run=%s variants=%d jobs=%d refs=%d noref=%d elapsed=%.0fs -> s3://%s/jobs/%s/"
      % (RUN, len(idx), len(manifest), len(by_ref), noref, time.time() - t0, BUCKET, RUN))
# emit the derived-ref sample so the caller can sanity-check ref existence
print("SAMPLE_REF " + (next(iter(by_ref)) if by_ref else "<none>"))
