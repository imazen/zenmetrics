#!/usr/bin/env python3
"""Independent keyed SafeSyn SSIM2 check, including worker tier attribution."""
import csv
import glob
import json
import os
import struct
from collections import defaultdict

import pyarrow.parquet as pq

ROOT = "/var/tmp/cvvdp-safesyn"
AUDIT = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-audit.jsonl"
PAIRS = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-pairs.tsv"
CACHE = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train944.parquet"
S3 = "s3://codec-corpus/safesyn-rev2-2026-09-06/images/"
IMG = "/mnt/v/input/zensim/images/"


def tier(worker):
    for name in ("r5600g", "r3500", "tower"):
        if name in worker.lower():
            return name
    return "unknown"


def main():
    pair_id = {}
    with open(PAIRS) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            uri = S3 + "/".join(r["dist_path"].split("/")[-3:])
            assert uri not in pair_id
            pair_id[uri] = int(r["row_id"])
    audit = {}
    sept14_4k_rows = 0
    with open(AUDIT) as f:
        for line in f:
            a = json.loads(line)
            uri = S3 + a["distorted"].replace(IMG, "")
            assert uri not in audit
            audit[uri] = float(a["peer_ssim2"]["score"])
            if any("cvvdp" in k.lower() for k in a):
                sept14_4k_rows += 1
            if any("cvvdp" in str(v).lower() for v in a.get("extra_targets", [])):
                sept14_4k_rows += 1
    cache_4k_cols = [c for c in pq.ParquetFile(CACHE).schema.names
                     if "cvvdp" in c.lower()]
    sc = pq.read_table(os.path.join(ROOT, "safesyn_cvvdp_sidecar.parquet"),
                       columns=["row_id", "ssim2_fresh", "cvvdp_jod_standard_4k"])
    sidecar = {rid: (s, c) for rid, s, c in zip(
        sc["row_id"].to_pylist(), sc["ssim2_fresh"].to_pylist(),
        sc["cvvdp_jod_standard_4k"].to_pylist())}
    assert len(sidecar) == sc.num_rows == len(pair_id) == len(audit) == 196086

    latest = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "store_ledger", "*.parquet"))):
        cols = pq.read_table(path, columns=["job_id", "output_sha", "ts", "worker"]).to_pydict()
        for jid, sha, ts, worker in zip(*(cols[c] for c in cols)):
            if jid not in latest or ts >= latest[jid][0]:
                latest[jid] = (ts, sha, tier(worker))
    blob_tier = {sha: host for _ts, sha, host in latest.values() if sha}

    groups = defaultdict(lambda: [0, 0, 0.0])  # rows, nonexact, max absolute delta
    seen = set()
    unmatched = raw_sidecar_mismatch = unknown_tier = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "harvest_blobs", "*"))):
        if not os.path.isfile(path):
            continue
        host = blob_tier.get(os.path.basename(path), "unknown")
        if host == "unknown":
            unknown_tier += 1
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("metric") != "ssim2" or "error" in r:
                    continue
                uri = r["encode_sha"]
                rid = pair_id.get(uri)
                if rid is None or uri not in audit:
                    unmatched += 1
                    continue
                score = float(r["score"])
                if struct.pack("!d", score) != struct.pack("!d", sidecar[rid][0]):
                    raw_sidecar_mismatch += 1
                delta = abs(score - audit[uri])
                for key in ((uri.split("/")[-2], host), ("ALL", "ALL")):
                    cell = groups[key]
                    cell[0] += 1
                    cell[1] += delta != 0.0
                    cell[2] = max(cell[2], delta)
                seen.add(rid)
    rep = {
        "unique_row_ids_checked": len(seen), "unmatched_raw_rows": unmatched,
        "raw_sidecar_mismatch": raw_sidecar_mismatch,
        "unknown_tier_blobs": unknown_tier,
        "sept14_4k_audit_rows": sept14_4k_rows,
        "sept14_4k_cache_columns": cache_4k_cols,
        "groups": {"/".join(k): {"rows": v[0], "nonexact": v[1], "max_abs_delta": v[2]}
                   for k, v in sorted(groups.items())},
    }
    out = os.path.join(ROOT, "independent_scores.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    print(f"ssim2_unique={len(seen)} unmatched={unmatched} "
          f"raw_sidecar_mismatch={raw_sidecar_mismatch} unknown_tier_blobs={unknown_tier}")
    print("ssim2_ALL=" + json.dumps(rep["groups"].get("ALL/ALL")))
    print(f"sept14_4k_audit_rows={sept14_4k_rows} "
          f"sept14_4k_cache_columns={len(cache_4k_cols)}")
    print("report=" + out)
    assert len(seen) == 196086 and not (unmatched or raw_sidecar_mismatch or unknown_tier)


if __name__ == "__main__":
    main()
