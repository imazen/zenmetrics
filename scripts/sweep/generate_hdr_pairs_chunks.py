#!/usr/bin/env python3
"""Generate chunks.jsonl for an HDR persisted-pairs metric fleet.

The persisted-pairs shape of the metric-backfill flow: variants already
exist (e.g. the kadis-hdr distortion corpus at
s3://codec-corpus/kadis-hdr-2026-07-13/), so workers score `pairs.tsv`
row-ranges with `score-pairs --hdr` — no re-encode, no regeneration.
Driven by `onstart_hdr_pairs.sh` + `hdr_pairs_chunk_worker.sh` on the
stock sweep image with an SWEEP_BIN_OVERRIDE hdr binary (the
`hetzner_cpu_sweep.sh` precedent).

  usage: generate_hdr_pairs_chunks.py <run_id> <pairs_r2> <data_prefix> \
             <out_prefix> [chunk_rows=600] > chunks.jsonl

Row ranges are half-open [start, end) over the pairs.tsv DATA rows
(header excluded). The worker slices with awk, so chunk boundaries are
plain line arithmetic — the pairs file must not change after chunking
(it is content-frozen in R2 for the run).
"""
import json
import os
import subprocess
import sys

run_id, pairs_r2, data_prefix, out_prefix = sys.argv[1:5]
chunk_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 600

ep = "https://%s.r2.cloudflarestorage.com" % os.environ["R2_ACCOUNT_ID"]
env = dict(
    os.environ,
    AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
    AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"],
    AWS_REGION="auto",
)
pairs = subprocess.run(
    ["s5cmd", "--endpoint-url", ep, "cat", pairs_r2],
    env=env, check=True, capture_output=True,
).stdout.decode()
n_rows = len(pairs.splitlines()) - 1  # header
assert n_rows > 0, "empty pairs.tsv"

for i, start in enumerate(range(0, n_rows, chunk_rows)):
    end = min(start + chunk_rows, n_rows)
    print(json.dumps({
        "chunk_id": f"{run_id}-{i:04d}",
        "run_id": run_id,
        "pairs_r2": pairs_r2,
        "row_range": [start, end],
        "data_prefix": data_prefix,
        "out_prefix": out_prefix,
    }, separators=(",", ":")))
print(f"# {n_rows} rows -> {(n_rows + chunk_rows - 1)//chunk_rows} chunks", file=sys.stderr)
