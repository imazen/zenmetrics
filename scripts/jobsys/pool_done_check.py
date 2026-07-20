#!/usr/bin/env python3
# Prints "DONE <pct>" if the byte-range backfill is complete (>= 99.5% of all pool runs' jobs done),
# else "NOTDONE <pct>". Concurrent ledger read. Used by pool_cron to stop self-renewing when finished.
import os, re, subprocess
import pyarrow.dataset as ds, pyarrow.fs as fs
from concurrent.futures import ThreadPoolExecutor

E = dict(os.environ)
for line in open(os.path.expanduser("~/.config/cloudflare/r2-credentials")):
    line = line.strip()
    if line.startswith("R2_") and "=" in line:
        k, v = line.split("=", 1); E[k] = v.strip().strip('"').strip("'")
EP = "https://%s.r2.cloudflarestorage.com" % E["R2_ACCOUNT_ID"]
S3 = fs.S3FileSystem(access_key=E["R2_ACCESS_KEY_ID"], secret_key=E["R2_SECRET_ACCESS_KEY"], endpoint_override=EP, region="auto")
TOTAL_JOBS = 490173  # 5,882,076 variants / 12 (approx; used only for the completion ratio)

def runs():
    r = subprocess.run(["s5cmd", "--endpoint-url", EP, "ls", "s3://zentrain/jobs/"],
                       env=dict(E, AWS_ACCESS_KEY_ID=E["R2_ACCESS_KEY_ID"], AWS_SECRET_ACCESS_KEY=E["R2_SECRET_ACCESS_KEY"], AWS_REGION="auto"),
                       capture_output=True, text=True)
    return sorted(set(m.group(0).rstrip("/") for l in r.stdout.splitlines()
                      for m in [re.search(r"bf-(zavif|zjxll|zwebp|zjxlm|zpng)-t\d+|bf-zjl2", l)] if m))

def done(run):
    try:
        t = ds.dataset("zentrain/jobs/%s/ledger/" % run, filesystem=S3, format="parquet").to_table(columns=["job_id", "status"])
        return len(set(j for j, s in zip(t.column("job_id").to_pylist(), t.column("status").to_pylist()) if s == "done"))
    except Exception:
        return 0

def main():
    rs = runs()
    with ThreadPoolExecutor(max_workers=16) as ex:
        tot = sum(ex.map(done, rs))
    pct = 100 * tot / TOTAL_JOBS
    print("%s %.1f" % ("DONE" if pct >= 99.5 else "NOTDONE", pct))

if __name__ == "__main__":
    main()
