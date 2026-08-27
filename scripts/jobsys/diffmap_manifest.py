#!/usr/bin/env python3
"""Deterministic _MANIFEST for a diffmap run's corpus slice.

Regenerable at any time from the live ledger + the emitted join table — run it
again at gap=0 and the counts/shas update in place. Fields follow the ML data
pipeline discipline: build_commit, per-artifact sha256, counts by metric/codec,
executor provenance, and pointers to the incident record (the failure history
is part of the corpus's provenance, not something to hide).

  usage: diffmap_manifest.py <run_id> <pairs_parquet> <out_json>
"""
import hashlib
import json
import os
import subprocess
import sys

import pyarrow.parquet as pq

run, pairs_p, out = sys.argv[1], sys.argv[2], sys.argv[3]

def sha256(p, cap=1 << 30):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

t = pq.read_table(pairs_p, columns=["metric", "codec", "encode_sha"])
by_metric, by_codec = {}, {}
for m in t.column("metric").to_pylist():
    by_metric[m] = by_metric.get(m, 0) + 1
for c in t.column("codec").to_pylist():
    by_codec[c] = by_codec.get(c, 0) + 1

def git_head(repo):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"

manifest = {
    "corpus_slice": "hdrgrid per-pixel diffmaps (gzip'd PFM, content-addressed)",
    "run": run,
    "declared": 193574,
    "join_table": {
        "path": pairs_p,
        "sha256": sha256(pairs_p),
        "rows": t.num_rows,
        "by_metric": by_metric,
        "by_codec": by_codec,
    },
    "blob_prefix": f"s3://zentrain/jobs/{run}/blobs/",
    "refs_prefix": "s3://zentrain/refs/imazen-26-hdr-grid-2026-06-14/",
    "build_commit": {
        "manifest_generator": git_head(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "diffmap_executor": "9093cc23 (B2) + images exec-zensim944hdr-526c84b8 / exec-gpu-399abe82",
    },
    "provenance_notes": [
        "zero data-genuine failures: every failure in the run's history was environmental "
        "(missing-hdr-gainmap image era, store outages) — see the incident arc + amnesty "
        "records in zensim docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md (2026-08-27 entries)",
        "holdout constraint: refs are imazen-26 eval-family sources — id+dHash audit required "
        "before ANY training use (registered; id half PASS in zensim "
        "benchmarks/imazen26_id_audit_2026-08-27.md; dHash+eye half pending)",
        "maps parameterization matches the recorded scalars (HDR: measured display peak, "
        "appendix AA; butteraugli CPU reference crate; cvvdp in-tree port)",
    ],
}
json.dump(manifest, open(out, "w"), indent=1)
print(f"manifest -> {out} (rows={t.num_rows}, sha={manifest['join_table']['sha256'][:12]}…)")
