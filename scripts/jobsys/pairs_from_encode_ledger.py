#!/usr/bin/env python3
# pairs_from_encode_ledger.py — the two-stage Encode→ScoreFile bridge.
#
# Reads an encode run's ledger (local dir of sidecar parquets, or an s3:// prefix),
# reduces latest-wins per job_id, keeps DONE rows, and emits:
#   pairs.parquet  — ref_path (full URI) + dist_path (full URI of the content-addressed
#                    blob) + the full cell identity (image_path/codec/q/knob_tuple_json)
#                    + encode_sha + worker/provider provenance. This is the input
#                    declare_direct_objects.py consumes with ZEN_FULL_URI=1, and the
#                    join table writeback_scores.py uses to rejoin ScoreFile JSONL rows
#                    (keyed encode_sha) back to per-cell identity.
#   pairs.tsv      — same rows, tab-separated, for eyeballing / shell tooling.
#
# usage:
#   pairs_from_encode_ledger.py <ledger_dir_or_s3_prefix> <refs_uri_prefix> \
#       <blobs_uri_prefix> <out_basename>
# example:
#   pairs_from_encode_ledger.py s3://zentrain/jobs/avifgen-enc-20260806/ledger/ \
#       s3://zentrain/refs/train-renditions-2026-06-14 \
#       s3://zentrain/jobs/avifgen-enc-20260806/blobs \
#       /mnt/v/output/avifgen-2026-08-06/pairs
#
# R2 access: sources root creds from ~/.config/cloudflare/r2-credentials when the
# ledger argument is s3:// (workstation-side tool; never runs on workers).
import os
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

LEDGER, REFS, BLOBS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
REFS = REFS.rstrip("/")
BLOBS = BLOBS.rstrip("/")

if LEDGER.startswith("s3://"):
    tmp = tempfile.mkdtemp(prefix="pairs_ledger_")
    creds = {}
    with open(os.path.expanduser("~/.config/cloudflare/r2-credentials")) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip().strip('"').strip("'")
    env = dict(
        os.environ,
        AWS_ACCESS_KEY_ID=creds["R2_ACCESS_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=creds["R2_SECRET_ACCESS_KEY"],
        AWS_REGION="auto",
    )
    ep = os.environ.get("ZEN_S3_ENDPOINT") or "https://%s.r2.cloudflarestorage.com" % creds["R2_ACCOUNT_ID"]
    subprocess.run(
        ["aws", "s3", "cp", "--endpoint-url", ep, LEDGER, tmp, "--recursive", "--quiet"],
        env=env,
        check=True,
    )
    ledger_dir = tmp
else:
    ledger_dir = LEDGER

best = {}  # job_id -> row dict (latest ts wins)
files = [f for f in sorted(os.listdir(ledger_dir)) if f.endswith(".parquet")]
for f in files:
    t = pq.read_table(os.path.join(ledger_dir, f)).to_pydict()
    n = len(t.get("job_id", []))
    for i in range(n):
        jid = t["job_id"][i]
        ts = t["ts"][i]
        prev = best.get(jid)
        if prev is None or ts >= prev["ts"]:
            best[jid] = {k: t[k][i] for k in t}

done = [r for r in best.values() if str(r.get("status", "")).lower() == "done"]
skipped = len(best) - len(done)
cols = {
    "ref_path": [f"{REFS}/{r['image_path']}" for r in done],
    "dist_path": [f"{BLOBS}/{r['output_sha']}" for r in done],
    "image_path": [r["image_path"] for r in done],
    "codec": [r["codec"] for r in done],
    "q": [int(r["q"]) for r in done],
    "knob_tuple_json": [r["knob_tuple_json"] for r in done],
    "encode_sha": [r["output_sha"] for r in done],
    "worker": [r.get("worker", "") for r in done],
    "provider": [r.get("provider", "") for r in done],
}
table = pa.table(cols)
pq.write_table(table, OUT + ".parquet", compression="zstd")
with open(OUT + ".tsv", "w") as f:
    keys = list(cols.keys())
    f.write("\t".join(keys) + "\n")
    for i in range(len(done)):
        f.write("\t".join(str(cols[k][i]) for k in keys) + "\n")
print(
    f"pairs: {len(done)} DONE cells ({skipped} non-done job_ids skipped) "
    f"from {len(files)} ledger files -> {OUT}.parquet / .tsv"
)
