#!/usr/bin/env python3
# Build ScoreFile job-system inputs for a PERSISTED-PAIRS HDR corpus (kadis-hdr layout:
# refs/ + dist/ + pairs.tsv already on R2) — the job-system replacement for the ad-hoc
# `hdr_pairs_chunk_worker.sh` bash fleet. Same outputs as build_scorefile_from_pairs.py,
# same resolution mechanism (variants.tar + 4-col index; NO new resolution path):
#   jobs/<run>/variants.tar      : the corpus dist objects, tarred (executor tar-shard / range-GET)
#   jobs/<run>/variant_index.tsv : sha \t offset \t size \t name  (4-col — name carries the REAL
#                                  extension, which the HDR decode dispatches on)
#   jobs/<run>/manifest.json[.gz]: one DesiredJob per source-file CHUNK with
#                                  kind = {"kind":"score_file","metrics":[...],"hdr":true}
# The refs stay in place: cell.image_path = the ref's corpus-relative path, resolved at run
# time via ZEN_CORPUS_BUCKET/ZEN_CORPUS_PREFIX exactly like every other ScoreFile run.
#
#   usage: build_scorefile_hdr_pairs.py <corpus_prefix_s3> <run_id>
#          e.g. build_scorefile_hdr_pairs.py s3://codec-corpus/kadis-hdr-2026-07-13 kadis-hdr-jobs-1
#   env:   R2_ACCOUNT_ID + R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY   (creds)
#          ZEN_SCOREFILE_CHUNK   dists per job (default 12)
#          ZEN_HDR_TRANSFER      omit (default; executor defaults pu-rescale) or "pq"
#          ZEN_METRICS           comma-list; default
#                                zensim-gpu,ssim2-gpu,cvvdp,iwssim-gpu,butteraugli-gpu
#                                (dssim is HDR-Unsupported BY DESIGN — never default it here)
#          ZEN_LIMIT_REFS        smoke mode: only the first N refs (default 0 = all)
#          ZEN_WORK_DIR          scratch root (default /mnt/v/zen — NOT /tmp, box reboots wipe it)
#
# pairs.tsv contract (frozen by the corpus consolidation, header + 1 row per cell):
# column 5 = ref relative path (refs/<name>.png), column 6 = dist relative path
# (dist/<name>.png) — the same columns hdr_pairs_chunk_worker.sh consumed.
#
# ⚠ Executor version gate: HDR manifests need an executor image built WITH the `hdr`
# feature and ≥ the version shipping the ScoreFile HDR arm (docs/RUNNING_JOBS.md,
# "HDR ScoreFile"). An older executor would not know the flag.
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from zen_s3env import resolve_full  # noqa: E402

CORPUS, RUN = sys.argv[1].rstrip("/"), sys.argv[2]
METRICS = os.environ.get(
    "ZEN_METRICS", "zensim-gpu,ssim2-gpu,cvvdp,iwssim-gpu,butteraugli-gpu"
).split(",")
CHUNK = int(os.environ.get("ZEN_SCOREFILE_CHUNK", "12"))
TRANSFER = os.environ.get("ZEN_HDR_TRANSFER", "")
LIMIT = int(os.environ.get("ZEN_LIMIT_REFS", "0"))
if "dssim" in ",".join(METRICS):
    print("FATAL: dssim has no HDR path by design — remove it from ZEN_METRICS", flush=True)
    sys.exit(1)
if TRANSFER not in ("", "pu-rescale", "pq"):
    print("FATAL: ZEN_HDR_TRANSFER must be empty, pu-rescale, or pq", flush=True)
    sys.exit(1)

# Store resolution via scripts/lib/zen_s3env.py -- defaults to the LAN store,
# ZEN_STORE=r2 opts out. Fixed 2026-08-26 (was: unconditional R2_* env vars).
ep, _ak, _sk, _store_kind, _reachable = resolve_full()
print(f"build_scorefile_hdr_pairs: store={_store_kind} endpoint={ep}", file=sys.stderr)
env = dict(os.environ, AWS_ACCESS_KEY_ID=_ak, AWS_SECRET_ACCESS_KEY=_sk, AWS_REGION="auto")


def s5(args, **kw):
    return subprocess.run(
        ["s5cmd", "--endpoint-url", ep] + args, env=env, check=True, **kw
    )


work = os.path.join(os.environ.get("ZEN_WORK_DIR", "/mnt/v/zen"), "scorefile-hdr-%s" % RUN)
os.makedirs(os.path.join(work, "dist"), exist_ok=True)

# 1. pairs.tsv → (ref relpath → [dist relpath…]), preserving corpus order.
s5(["cp", CORPUS + "/pairs.tsv", os.path.join(work, "pairs.tsv")],
   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
refs = {}  # ref relpath -> list of dist relpaths
with open(os.path.join(work, "pairs.tsv")) as f:
    header = f.readline()
    for line in f:
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 6:
            continue
        ref_rel, dist_rel = cols[4], cols[5]
        refs.setdefault(ref_rel, []).append(dist_rel)
if LIMIT:
    refs = dict(list(refs.items())[:LIMIT])
n_dists = sum(len(v) for v in refs.values())
print("pairs: %d refs × %d dists (limit=%s)" % (len(refs), n_dists, LIMIT or "off"), flush=True)
if not refs:
    print("FATAL: pairs.tsv yielded no rows — wrong corpus prefix?", flush=True)
    sys.exit(1)

# 2. sync the dist objects (one parallel s5cmd run, resumable re-run) …
runfile = os.path.join(work, "_dl.run")
with open(runfile, "w") as f:
    for dists in refs.values():
        for d in dists:
            local = os.path.join(work, "dist", os.path.basename(d))
            if not os.path.exists(local):
                f.write("cp %s/%s %s\n" % (CORPUS, d, local))
if os.path.getsize(runfile):
    print("syncing dists from %s …" % CORPUS, flush=True)
    s5(["run", runfile], stdout=subprocess.DEVNULL)

# 3. … tar them with a 4-column index (offset/size/sha256/NAME — the name keeps the
#    real extension the executor's HDR decode dispatches on).
tar_path = os.path.join(work, "variants.tar")
idx_path = os.path.join(work, "variant_index.tsv")
sha_by_rel = {}
with tarfile.open(tar_path, "w") as tf:
    for dists in refs.values():
        for d in dists:
            if d in sha_by_rel:
                continue
            local = os.path.join(work, "dist", os.path.basename(d))
            with open(local, "rb") as fh:
                b = fh.read()
            if not b:
                print("FATAL: empty dist object %s" % d, flush=True)
                sys.exit(1)
            sha_by_rel[d] = hashlib.sha256(b).hexdigest()
            info = tarfile.TarInfo(name=os.path.basename(d))
            info.size = len(b)
            import io

            tf.addfile(info, io.BytesIO(b))
# offsets from a second header pass (offset_data is only stable on read).
with tarfile.open(tar_path, "r") as tf, open(idx_path, "w") as fidx:
    by_name = {}
    for m in tf:
        if m.isfile():
            by_name[m.name] = (m.offset_data, m.size)
    for rel, sha in sha_by_rel.items():
        off, size = by_name[os.path.basename(rel)]
        fidx.write("%s\t%d\t%d\t%s\n" % (sha, off, size, os.path.basename(rel)))
print("tar: %d members, %.1f MiB" % (len(sha_by_rel), os.path.getsize(tar_path) / 2**20), flush=True)

# 4. chunked manifest — kind carries hdr:true (+ hdr_transfer only when non-default,
#    keeping the content-addressed id minimal; absent = executor default pu-rescale).
# 4. DesiredJob emission via THE owner (`zenfleet-ctl declare-scorefiles`,
#    2026-08-27): bridge parquet with VERBATIM keys (--full-uri: ref_path =
#    the corpus-relative ref path, dist_path = the content sha), cell knobs
#    "scorefile-hdr", --hdr; --hdr-transfer only when non-default (the id-
#    minimality rule the inline kind build kept). Chunking, requires,
#    manifest uploads + control.json (create-if-absent) are owner-side.
import pyarrow as pa
import pyarrow.parquet as pq
bridge = os.path.join(work, "bridge_pairs.parquet")
rp_col, dp_col = [], []
for ref_rel, dists in refs.items():
    for d in dists:
        rp_col.append(ref_rel); dp_col.append(sha_by_rel[d])
pq.write_table(pa.table({"ref_path": rp_col, "dist_path": dp_col}), bridge, compression="zstd")
JOBCTL = os.environ.get("ZEN_JOBCTL", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "..", "target", "release", "zenfleet-ctl"))
_cmd = [JOBCTL, "declare-scorefiles", "--pairs", bridge, "--run", RUN,
        "--bucket", "codec-corpus", "--metrics", ",".join(METRICS),
        "--chunk", str(CHUNK), "--full-uri", "--hdr",
        "--cell-knobs", "scorefile-hdr", "--cell-codec", os.path.basename(CORPUS)]
if TRANSFER and TRANSFER != "pu-rescale":
    _cmd += ["--hdr-transfer", TRANSFER]
subprocess.run(_cmd, env=env, check=True)
manifest = []  # kept for the summary line below (job count now printed by the CLI)

# 5. dist_sha_map.tsv: dist relpath -> sha256, so the write-back can rejoin each
#    JSONL row (keyed on encode_sha) to the corpus pairs.tsv row (keyed on
#    dist_path) and recover the per-cell identity tuple (q / knob_tuple_json —
#    the ScoreFile cell carries the SOURCE identity only, by convention).
map_path = os.path.join(work, "dist_sha_map.tsv")
with open(map_path, "w") as f:
    f.write("dist_path\tencode_sha\n")
    for rel, sha in sha_by_rel.items():
        f.write("%s\t%s\n" % (rel, sha))

# 6. upload run inputs. Launch with ZEN_CORPUS_PREFIX=<corpus prefix path> so
#    cell.image_path (refs/…) resolves against the corpus bucket, and the usual
#    ZEN_VARIANTS_TAR_URI / ZEN_VARIANT_INDEX_URI pointing at jobs/<run>/.
bucket_prefix = "s3://codec-corpus/jobs/%s" % RUN
for name in (
    "variants.tar",
    "variant_index.tsv",
    # manifest.json(.gz) + control.json are uploaded by zenfleet-ctl declare-scorefiles
    "dist_sha_map.tsv",
):
    s5(["cp", os.path.join(work, name), "%s/%s" % (bucket_prefix, name)],
       stdout=subprocess.DEVNULL)
print(
    "uploaded run %s inputs: %d chunk jobs declared by zenfleet-ctl, %d variants (chunk=%d) for %d refs — hdr:true%s\n"
    "local staging kept at %s (delete after the run verifies)"
    % (
        RUN,
        len(manifest),
        len(sha_by_rel),
        CHUNK,
        len(refs),
        ", transfer=%s" % TRANSFER if TRANSFER else "",
        work,
    ),
    flush=True,
)
