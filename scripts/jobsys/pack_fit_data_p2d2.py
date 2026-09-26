#!/usr/bin/env python3
"""Pack the P2 (GMSD-peer) and D2 (source-held-out) MLP inputs into one content-addressed archive.

Superset of `pack_fit_data.py`'s P0 archive: the eight P0 sets plus `konjnd_bpg_val` (the D2 eval
substitute for konjnd_bpg_train), the reviewed peer GMSD tables for exactly those nine sets, and the
D2 fold tables with their receipts. Bytes are copied and hash-checked against the registered
receipts; no label column is decoded. CID22-B, AIC-4, CSIQ, KonJND JPEG and every other peer/bank
set are never included. The caller must have logged the transport in the owning exposure ledger.
"""

import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from pack_fit_data import ADMIT_RECEIPT_SHA, PINNED, PINNED_REGISTRATION, ROOT, add_file, digest

# The promoted bank moved on after the registered era; the pinned snapshot holds the registered bytes
# (its receipt is pinned by admit_bank.py). Archive member names stay under rev4-featbank/bank/.
BANK = ROOT / "bank_snapshot"
SNAPSHOT_RECEIPT_SHA = "1c61f154e3f8f263e2c7e2f3ac125f98a9d94653cf616518087bd00048e899fc"

PEER = Path("/var/tmp/gmsbank/peer_gmsd")
PEER_MANIFEST_SHA = "8d2799c8a9ebaa3f25fa7518b3d7591ee7742b696d7f5592b9ace9974d0d27fc"
PEER_VERIFY_SHA = "5429a455120bd41a0c729514e61670c124b85be576dc2fc49a8a9cf5bf695f5f"
SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train", "konjnd_bpg_val",
        "cid22_a25", "aic3", "kadid_select", "konfig_val")
D2_ARMS = ("r0", "p2", "p2_perm")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if digest(BANK / "SNAPSHOT_RECEIPT.json") != SNAPSHOT_RECEIPT_SHA:
        raise ValueError("bank snapshot receipt changed")
    receipt_path = ROOT / "admit_bank.jsonl"
    if digest(receipt_path) != ADMIT_RECEIPT_SHA:
        raise ValueError("admitted-table receipt changed from the registered target-free era")
    receipts = {json.loads(line)["set"]: json.loads(line) for line in receipt_path.read_text().splitlines()}
    files = {"rev4-featpot/admit_bank.jsonl": receipt_path}
    for name, expected in PINNED.items():
        manifest_path = BANK / name / "_MANIFEST.json"
        if digest(manifest_path) != expected:
            raise ValueError(f"{name}: bank manifest is not the pinned era")
        files[f"rev4-featbank/bank/{name}/_MANIFEST.json"] = manifest_path
        if name not in SETS:
            continue
        manifest = json.loads(manifest_path.read_text())
        for filename, record in manifest["files"].items():
            if filename != "keys.parquet" and not filename.startswith(("features__", "labels__")):
                continue
            path = BANK / name / filename
            if digest(path) != record["sha256"]:
                raise ValueError(f"{name}/{filename}: component hash mismatch")
            files[f"rev4-featbank/bank/{name}/{filename}"] = path
        admitted = ROOT / "admitted" / f"POT_{name}_rev3_944.parquet"
        if digest(admitted) != receipts[name]["output_sha256"]:
            raise ValueError(f"{name}: admitted table differs from registered receipt")
        files[f"rev4-featpot/admitted/{admitted.name}"] = admitted
    for name in ("folds.json", "sampler_preflight.json"):
        if digest(ROOT / name) != PINNED_REGISTRATION[name]:
            raise ValueError(f"{name}: differs from the preregistered hash")
        files[f"rev4-featpot/{name}"] = ROOT / name
    # Peer GMSD tables (label-free), exactly the nine sets.
    if digest(PEER / "_MANIFEST.json") != PEER_MANIFEST_SHA or digest(PEER / "verification.json") != PEER_VERIFY_SHA:
        raise ValueError("peer manifest or verification receipt changed")
    peer = json.loads((PEER / "_MANIFEST.json").read_text())
    files["gmsbank/peer_gmsd/_MANIFEST.json"] = PEER / "_MANIFEST.json"
    files["gmsbank/peer_gmsd/verification.json"] = PEER / "verification.json"
    for name in SETS:
        path = PEER / f"{name}.parquet"
        if digest(path) != peer["sets"][name]["parquet_sha256"]:
            raise ValueError(f"{name}: peer parquet hash mismatch")
        files[f"gmsbank/peer_gmsd/{name}.parquet"] = path
    # D2 fold tables per arm, verified against each arm's receipt.
    for arm in D2_ARMS:
        receipt = ROOT / "p2/d2_mlp/tables" / arm / "receipt.json"
        value = json.loads(receipt.read_text())
        if value["schema"] != "rev4-featpot-d2-mlp-tables-v1" or value["arm"] != arm:
            raise ValueError(f"{arm}: D2 table receipt identity mismatch")
        files[f"rev4-featpot/p2/d2_mlp/tables/{arm}/receipt.json"] = receipt
        for name, record in value["tables"].items():
            if name not in SETS:
                raise ValueError(f"{arm}/{name}: table for a set outside the registered nine")
            for kind, item in record["tables"].items():
                path = Path(item["path"])
                if digest(path) != item["sha256"] or digest(Path(f"{path}.manifest.json")) != item["manifest_sha256"]:
                    raise ValueError(f"{path}: table hash mismatch")
                for source in (path, Path(f"{path}.manifest.json")):
                    if not str(source).startswith(str(ROOT) + "/"):
                        raise ValueError(f"{source}: table outside {ROOT}")
                    files[f"rev4-featpot/{source.relative_to(ROOT)}"] = source
    forbidden = ("cid22_b", "aic4", "csiq", "konjnd_jpeg", "mcljci", "safesyn", "kadid_terminal", "cid22_train")
    bad = [n for n in files if any(f in n for f in forbidden)]
    if bad:
        raise ValueError(f"forbidden populations in archive: {bad[:3]}")
    hashes = {name: digest(path) for name, path in sorted(files.items())}
    inventory = json.dumps({"schema": "zenfleet-fit-data-v1", "profile": "p2d2", "sets": sorted(SETS),
                            "files": hashes}, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            info = tarfile.TarInfo("input_inventory.json")
            info.size = len(inventory)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(inventory))
            for name, path in sorted(files.items()):
                add_file(tar, path, name)
    print(json.dumps({"sets": sorted(SETS), "files": len(files), "archive_bytes": args.out.stat().st_size,
                      "sha256": digest(args.out), "out": str(args.out)}))


if __name__ == "__main__":
    main()
