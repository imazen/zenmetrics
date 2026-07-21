#!/usr/bin/env python3
"""Fast pool-progress readout for the zensim-720 backfill (or any pool run set).

Sums distinct-done by reading the per-run ledger_snapshot.parquet FOOTER (num_rows only — one
metadata read per run, no full ledger scan), so it's seconds, not the ~40min a live-ledger count
takes. Snapshots are maintained by refresh_snapshots.sh (~30min cron); this reflects the last
refresh. Runs with no snapshot yet (never-worked) count as 0.

  python3 pool_progress.py [total_jobs]     # default total 490173 (zensim-720 corpus)
"""
import os, sys, time, concurrent.futures as cf
import pyarrow.fs as fs, pyarrow.parquet as pq

E = dict(os.environ)
for line in open(os.path.expanduser("~/.config/cloudflare/r2-credentials")):
    line = line.strip()
    if line.startswith("R2_") and "=" in line:
        k, v = line.split("=", 1)
        E[k] = v.strip().strip('"').strip("'")
S3 = fs.S3FileSystem(access_key=E["R2_ACCESS_KEY_ID"], secret_key=E["R2_SECRET_ACCESS_KEY"],
    endpoint_override="https://%s.r2.cloudflarestorage.com" % E["R2_ACCOUNT_ID"], region="auto")
BUCKET = E.get("ZEN_BUCKET", "zentrain")
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 490173

with S3.open_input_file("%s/jobs/_pool/runlist.tsv" % BUCKET) as f:
    runs = [ln.split("\t")[0] for ln in f.read().decode().splitlines() if ln.startswith("bf-")]

def rows(run):
    try:
        with S3.open_input_file("%s/jobs/%s/ledger_snapshot.parquet" % (BUCKET, run)) as f:
            return (run, pq.read_metadata(f).num_rows)
    except Exception:
        return (run, 0)

T = int(time.time())
with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(rows, runs))
total = sum(r for _, r in res)
missing = [run for run, r in res if r == 0]
print("distinct_done=%d / %d = %.2f%%  (from %d/%d snapshots)  T=%d"
      % (total, TOTAL, 100 * total / TOTAL, len(runs) - len(missing), len(runs), T))
if missing:
    print("no-snapshot runs (~0 done):", " ".join(missing))
