#!/usr/bin/env python3
"""Add the remaining input hashes to the gated harvest manifest, atomically."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/var/tmp/cvvdp-safesyn")
MANIFEST = ROOT / "safesyn_cvvdp_sidecar_MANIFEST.json"
AUDIT = Path("/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-audit.jsonl")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = json.load(open(MANIFEST))
    assert man["build_commit"] == "9f36f88b8a23e645bd791c1193842d950eccf4be"
    assert man["rows"] == 196086 and man["blob_count"] == 3218
    assert man["sidecar_sha256"] == sha(ROOT / "safesyn_cvvdp_sidecar.parquet")
    old = man["inputs"]
    assert old["pairs_tsv_sha256"] == sha(old["pairs_tsv"])
    man["inputs"] = {
        "pairs_tsv": old["pairs_tsv"],
        "pairs_tsv_sha256": old["pairs_tsv_sha256"],
        "pairs_parquet": old["pairs_parquet"],
        "pairs_parquet_sha256": sha(old["pairs_parquet"]),
        "blob_prefix": old["blob_prefix"],
        "audit_jsonl": old["audit_jsonl"],
        "audit_jsonl_sha256": sha(AUDIT),
        "declared_manifest": str(ROOT / "safesyn_manifest4.json"),
        "declared_manifest_sha256": sha(ROOT / "safesyn_manifest4.json"),
    }
    tmp = MANIFEST.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, MANIFEST)
    print("manifest_sha256=" + sha(MANIFEST))
    for key in ("pairs_tsv_sha256", "pairs_parquet_sha256", "audit_jsonl_sha256",
                "declared_manifest_sha256"):
        print(key + "=" + man["inputs"][key])


if __name__ == "__main__":
    main()
