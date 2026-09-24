#!/usr/bin/env python3
"""Independent manifest/ledger/blob completeness check for the frozen run."""
import argparse
import glob
import hashlib
import json
import os

import pyarrow.parquet as pq


def job_id(job):
    # zenfleet-core/src/ids.rs: JobIdCanonForm field order is inputs, kind.
    canon = {"inputs": sorted(set(job["inputs"])), "kind": job["kind"]}
    wire = json.dumps(canon, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(wire).hexdigest()


def tier(worker):
    for name in ("r5600g", "r3500", "tower"):
        if name in worker.lower():
            return name
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/var/tmp/cvvdp-safesyn")
    a = ap.parse_args()
    d = a.dir
    manifest = json.load(open(os.path.join(d, "store_manifest.json")))
    expected = {job_id(j) for j in manifest}
    assert len(manifest) == len(expected), "duplicate manifest job_id"
    blobs = {os.path.basename(line.split()[-1]) for line in open(
        os.path.join(d, "store_blobs.ls")) if line.strip()}
    latest = {}
    ledger_rows = 0
    by_status = {}
    by_tier = {}
    for path in sorted(glob.glob(os.path.join(d, "store_ledger", "*.parquet"))):
        tab = pq.read_table(path, columns=["job_id", "output_sha", "status", "ts", "worker"])
        cols = tab.to_pydict()
        ledger_rows += tab.num_rows
        for row in zip(*(cols[c] for c in cols)):
            r = dict(zip(cols, row))
            jid = r["job_id"]
            if jid not in latest or r["ts"] >= latest[jid]["ts"]:
                latest[jid] = r
    missing_ledger = sorted(expected - set(latest))
    unexpected_ledger = sorted(set(latest) - expected)
    missing_blob = []
    for jid, r in latest.items():
        status = str(r["status"])
        by_status[status] = by_status.get(status, 0) + 1
        t = tier(str(r["worker"]))
        by_tier[t] = by_tier.get(t, 0) + 1
        if jid in expected and (not r["output_sha"] or r["output_sha"] not in blobs):
            missing_blob.append(jid)
    failed_or_not_done = sorted(jid for jid in expected & set(latest)
                                if str(latest[jid]["status"]).lower() != "done")
    report = {
        "manifest_jobs": len(manifest), "ledger_rows": ledger_rows,
        "ledger_jobs": len(latest), "blob_objects": len(blobs),
        "missing_ledger": missing_ledger, "unexpected_ledger": unexpected_ledger,
        "missing_blob": sorted(missing_blob),
        "failed_or_not_done": failed_or_not_done,
        "status": by_status, "tier": by_tier,
    }
    out = os.path.join(d, "store_verify.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"manifest_jobs={len(manifest)} ledger_rows={ledger_rows} "
          f"ledger_jobs={len(latest)} blob_objects={len(blobs)}")
    print(f"missing_ledger={len(missing_ledger)} unexpected_ledger={len(unexpected_ledger)} "
          f"missing_blob={len(missing_blob)} failed_or_not_done={len(failed_or_not_done)}")
    print("status=" + json.dumps(by_status, sort_keys=True))
    print("tier=" + json.dumps(by_tier, sort_keys=True))
    print("report=" + out)
    if missing_ledger or unexpected_ledger or missing_blob or failed_or_not_done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
