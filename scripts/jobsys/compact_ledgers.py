#!/usr/bin/env python3
"""Compact one run's ledger dir -> a done-set snapshot parquet (schema-preserved).

The pool worker (fleet-entrypoint.sh, POOL mode) passes NO --ledger-in, so its reconcile
view is empty and its gap is *all* cells every pass; dedup falls to a 10-min claim lease.
On a resumed run whose earlier claims have expired (e.g. a fleet's prior work), that means
re-scoring already-done cells — a measured 2.0-3.4x tax on the big runs. Feeding the worker
a done-set snapshot as --ledger-in makes its gap = only-undone, killing the tax.

This writes snapshot = all status=='done' rows, deduped by job_id (keep first), TAKEn from the
worker's own ledger parquets so the arrow schema matches read_ledger exactly. Read-only against
the live ledger/ dir; the caller uploads the result to jobs/<run>/ledger_snapshot.parquet.

  python3 compact_ledgers.py <run>        # writes $ZEN_SNAP_DIR/snap_<run>.parquet (default ~/tmp/zen-snaps)
"""
import os, sys, time
import pyarrow as pa, pyarrow.dataset as ds, pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc

E = dict(os.environ)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from zen_s3env import resolve  # noqa: E402  (ZEN_S3_ENDPOINT overrides; default = R2, unchanged)

_ep, _ak, _sk = resolve()
S3 = fs.S3FileSystem(
    access_key=_ak, secret_key=_sk,
    endpoint_override=_ep, region="auto")

run = sys.argv[1]
bucket = E.get("ZEN_BUCKET", "zentrain")
snap_dir = os.path.expanduser(E.get("ZEN_SNAP_DIR", "~/tmp/zen-snaps"))
os.makedirs(snap_dir, exist_ok=True)

t0 = time.time()
t = ds.dataset("%s/jobs/%s/ledger/" % (bucket, run), filesystem=S3, format="parquet").to_table()
rows = t.num_rows
done = t.filter(pc.equal(t.column("status"), "done"))
seen, keep = set(), []
for i, j in enumerate(done.column("job_id").to_pylist()):
    if j not in seen:
        seen.add(j); keep.append(i)
snap = done.take(keep)
out = os.path.join(snap_dir, "snap_%s.parquet" % run)
# zstd, NOT pyarrow's default snappy: zenfleet-ledger's parquet reader is built with
# features=["arrow","zstd"] (no "snap"), so the worker reads this snapshot as --ledger-in.
# A snappy snapshot errors "Disabled feature at compile time: snap" on a correctly-built worker.
pq.write_table(snap, out, compression="zstd")
print("run=%s read_rows=%d distinct_done=%d wrote=%s (%.1fs)" % (run, rows, snap.num_rows, out, time.time() - t0))
