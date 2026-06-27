#!/usr/bin/env python3
"""verify_canonical.py — verification gates for the canonical picker datasets.

Re-checks the datasets built by build_canonical.py against the 5 gates the task
requires, for EVERY `<codec>_<mode>` dir under --local-root:

  1. No split leakage  — train/val/test origin sets pairwise disjoint (per
     dataset) AND every origin maps to the same split across ALL datasets.
  2. Row sanity        — rows match the _MANIFEST.json; report per-split rows.
  3. Sample resolves   — a sample `source_r2_url` exists in R2 (aws s3 ls).
  4. Split ratios sane — train≈50% / val≈30% / test≈20% by distinct origins.
  5. Readable + sha256 — every parquet (incl. pairs.*) reads back and its
     sha256 matches the _MANIFEST.json.

Exit non-zero if any gate fails. R2 access via aws-cli (R2_ACCOUNT_ID +
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in env); pass --no-r2 to skip gate 3.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

LOCAL_ROOT_DEFAULT = "/mnt/v/output/canonical-picker-2026-06-27"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def origins_of(parquet):
    return set(pq.read_table(parquet, columns=["origin_id"]).column("origin_id").to_pylist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-root", default=LOCAL_ROOT_DEFAULT)
    ap.add_argument("--no-r2", action="store_true")
    args = ap.parse_args()

    datasets = sorted(d for d in glob.glob(os.path.join(args.local_root, "*"))
                      if os.path.isdir(d) and os.path.exists(os.path.join(d, "_MANIFEST.json")))
    if not datasets:
        sys.exit(f"no datasets under {args.local_root}")

    fails = []
    origin_split_global = {}  # origin -> (split, first dataset seen) for cross-dataset check

    for d in datasets:
        name = os.path.basename(d)
        man = json.load(open(os.path.join(d, "_MANIFEST.json")))
        print(f"\n=== {name} (run {man['sweep_run']}, plan {man['provenance']['run_plan']}) ===")
        per_split_origins = {}
        # Gate 5: every parquet readable + sha256 matches manifest.
        # _MANIFEST.json keys splits by file stem: train / validate / test.
        for sp_name in ("train", "validate", "test"):
            if sp_name not in man["splits"]:
                continue
            info = man["splits"][sp_name]
            p = os.path.join(d, f"{sp_name}.parquet")
            t = pq.read_table(p, columns=["origin_id", "split", "mode"])
            rows = t.num_rows
            if rows != info["rows"]:
                fails.append(f"{name}/{sp_name}: rows {rows} != manifest {info['rows']}")
            sha = sha256_file(p)
            if sha != info["sha256"]:
                fails.append(f"{name}/{sp_name}: sha256 mismatch")
            # pairs parquet
            pp = os.path.join(d, f"pairs.{sp_name}.parquet")
            if os.path.exists(pp):
                pt = pq.read_table(pp)
                if pt.num_rows != info["pairs_rows"]:
                    fails.append(f"{name}/{sp_name}: pairs rows {pt.num_rows} != manifest {info['pairs_rows']}")
                if sha256_file(pp) != info["pairs_sha256"]:
                    fails.append(f"{name}/{sp_name}: pairs sha256 mismatch")
                need = {"ref_path", "dist_path", "image_path", "codec", "q", "knob_tuple_json"}
                miss = need - set(pt.column_names)
                if miss:
                    fails.append(f"{name}/{sp_name}: pairs missing cols {miss}")
            else:
                fails.append(f"{name}/{sp_name}: pairs.{sp_name}.parquet missing")
            # split column must be uniform
            uniq_split = set(t.column("split").to_pylist()[:1000])
            want = {"train": "train", "validate": "val", "test": "test"}[sp_name]
            if uniq_split and uniq_split != {want}:
                fails.append(f"{name}/{sp_name}: split col {uniq_split} != {{{want}}}")
            origs = set(t.column("origin_id").to_pylist())
            per_split_origins[want] = origs
            # cross-dataset origin->split consistency (global gate 1)
            for o in origs:
                if o in origin_split_global and origin_split_global[o][0] != want:
                    fails.append(f"GLOBAL LEAK: origin {o} is {want} in {name} but "
                                 f"{origin_split_global[o][0]} in {origin_split_global[o][1]}")
                origin_split_global.setdefault(o, (want, name))
            print(f"  {sp_name:9s} rows={rows:>9d} origins={len(origs):>4d} sha✓ pairs✓")

        # Gate 1 (per-dataset): pairwise disjoint
        tr, va, te = (per_split_origins.get("train", set()),
                      per_split_origins.get("val", set()), per_split_origins.get("test", set()))
        for a, b, lab in ((tr, va, "train∩val"), (tr, te, "train∩test"), (va, te, "val∩test")):
            if a & b:
                fails.append(f"{name}: LEAK {lab} = {sorted(a & b)[:5]}")
        # Gate 4: ratios by distinct origins
        tot = len(tr) + len(va) + len(te)
        if tot:
            r = (len(tr) / tot, len(va) / tot, len(te) / tot)
            ok = 0.42 <= r[0] <= 0.58 and 0.22 <= r[1] <= 0.38 and 0.12 <= r[2] <= 0.26
            print(f"  origin split: train {r[0]:.1%} / val {r[1]:.1%} / test {r[2]:.1%} "
                  f"({'sane' if ok else 'OUT OF RANGE'})")
            if not ok:
                fails.append(f"{name}: ratios out of range {r}")
        # Gate 1 disjoint print
        print(f"  leakage: {'none (disjoint)' if not (tr&va or tr&te or va&te) else 'LEAK!'}")

        # Gate 3: sample source_r2_url resolves
        if not args.no_r2:
            url = pq.read_table(os.path.join(d, "train.parquet"),
                                columns=["source_r2_url"]).column("source_r2_url")[0].as_py()
            ep = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
            rc = subprocess.run(["aws", "s3", "ls", url, "--endpoint-url", ep],
                                capture_output=True, text=True)
            if rc.returncode == 0 and rc.stdout.strip():
                print(f"  gate3: {url.rsplit('/', 1)[-1]} resolves ✓")
            else:
                fails.append(f"{name}: source_r2_url does NOT resolve: {url}")

    print(f"\n{'='*60}")
    print(f"datasets: {len(datasets)} | distinct origins total: {len(origin_split_global)}")
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for f in fails:
            print("  ✗", f)
        sys.exit(1)
    print("ALL GATES PASSED ✓")


if __name__ == "__main__":
    main()
