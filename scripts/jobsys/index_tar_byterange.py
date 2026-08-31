#!/usr/bin/env python3
# Runs ON the Hetzner index box. Streams ONE per-box variants tar ONCE (r| mode, no whole-file
# download-to-disk), and from its member headers builds BOTH:
#   * a byte-range index  (dist_member \t offset \t size \t dist_member)  -> jobs/<run>/variant_index.tsv
#   * a ScoreFile manifest (score_file jobs, inputs = dist_members, cell.image_path = derived ref)
# The ref is derived from the variant filename: <refbase>_<16hex refhash>_<codec>_q<q>_<hash>.<ext>
# -> <refbase>.png. No pairs parquet needed; the tar's members ARE the variant set.
#   usage: index_tar_byterange.py <tar_s3_uri> <cell_codec> <run> <bucket>
#
# ZEN_INDEX_ONLY=1 builds and uploads ONLY variant_index.tsv — no manifest.json,
# no manifest.json.gz, no control.json. Use it when you want the byte-range index
# for a tar WITHOUT declaring a job: writing control.json + manifest.json into
# jobs/<run>/ is exactly what makes a run claimable, so an index-only need must not
# go through the declaring path. (Added 2026-08-30 to index the mandfix2-zenjpeg
# tars on the LAN store, which had no index and therefore no tar-range fetch path.)
import sys, os, re, json, gzip, subprocess, tarfile, time
tar_uri, CODEC, RUN, BUCKET = sys.argv[1:5]
INDEX_ONLY = os.environ.get("ZEN_INDEX_ONLY") == "1"
EP = os.environ.get("ZEN_S3_ENDPOINT") or "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
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
for ref, members in ({} if INDEX_ONLY else by_ref).items():
    for i in range(0, len(members), CHUNK):
        manifest.append({"kind": {"kind": "score_file", "metrics": METRICS},
                         "inputs": members[i:i + CHUNK],
                         "cell": {"image_path": ref, "codec": CODEC, "q": -1, "knob_tuple_json": "scorefile"},
                         "hint": None})
work = os.environ.get("ZEN_IDX_WORK", "/root/idxwork") + "/%s" % RUN
os.makedirs(work, exist_ok=True)
open(work + "/variant_index.tsv", "w").write("\n".join(idx) + "\n")
if not INDEX_ONLY:
    json.dump(manifest, open(work + "/manifest.json", "w"))
    with open(work + "/manifest.json", "rb") as fi, gzip.open(work + "/manifest.json.gz", "wb") as g:
        g.write(fi.read())
    open(work + "/control.json", "w").write('{"paused":false}')

def up(f, k):
    subprocess.run(["s5cmd", "--endpoint-url", EP, "cp", work + "/" + f,
                    "s3://%s/jobs/%s/%s" % (BUCKET, RUN, k)], env=ENV, check=True,
                   stdout=subprocess.DEVNULL)

for f in (("variant_index.tsv",) if INDEX_ONLY
          else ("variant_index.tsv", "manifest.json", "manifest.json.gz", "control.json")):
    up(f, f)
print("run=%s variants=%d jobs=%d refs=%d noref=%d elapsed=%.0fs -> s3://%s/jobs/%s/"
      % (RUN, len(idx), len(manifest), len(by_ref), noref, time.time() - t0, BUCKET, RUN))
# emit the derived-ref sample so the caller can sanity-check ref existence
print("SAMPLE_REF " + (next(iter(by_ref)) if by_ref else "<none>"))
