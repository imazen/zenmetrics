#!/usr/bin/env python3
"""Pack explicitly admitted P0 fit inputs into one content-addressed archive.

This copies bytes without decoding target columns. The caller must have logged
any held-out population exposure in zensim/docs/DATA_SPLITS.md first.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


BANK = Path("/var/tmp/rev4-featbank/bank")
ROOT = Path("/var/tmp/rev4-featpot")
ADMIT_RECEIPT_SHA = "aaf927dc2a8c8dd6066ce740bef53f137a61cee8e26bb3210d440d38465cdbcb"
PINNED = {
    "kadid_train": "6976a3ce05fa52552883ac380c85bdfd4cbeb800be23077299069d2b78769b30",
    "tid2013": "ec305b5e610dc36815a91c7faf533dd1000cdba5f4faa59a14f4ab924bb69c1d",
    "konfig_train": "2a48a0c3f3686dce24ef9e1288cec0907df17531df32f345d91eda0b1ad8e405",
    "konjnd_bpg_train": "f0badfe8d26e69fb79839458749c385b32d04734e36116bd5c40f25749046292",
    "konjnd_bpg_val": "59f7099930cde8b2899ad3f76f826a7e12f3615938356fa370ec0ad79d9c2cf9",
    "cid22_a25": "38474bc10a3ad6d89965dcf3cabdc10e7fb677b0ce76606ccd9e73c4ef9009f4",
    "aic3": "e73441296ec64433198a537b5fe0dd340003a3bced446848a8b38c4ba22dc389",
    "kadid_select": "c8807a25089e0d354c8cff068477edca410b33cc82085ddc2ec86930af24183b",
    "konfig_val": "b2fc90e103312ec86968ede3fc0d820b4dc45a48df7e37c3bebd4efff945605e",
}
TRAIN = {"kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train"}
P0_SETS = {"kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
           "cid22_a25", "aic3", "kadid_select", "konfig_val"}


# Preregistered hashes of the fold assignment and the sampler preflight (potential lane's prereg).
PINNED_REGISTRATION = {
    "folds.json": "8b5b922f3d57ee5149249e597fd717a566f457957692959af8286394c81e20dc",
    "sampler_preflight.json": "724a49f8aee1dba6a1881122e4d1608619f22f9233ac1194d9804c213030bebf",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_file(tar: tarfile.TarFile, path: Path, name: str) -> None:
    member = tar.gettarinfo(str(path), arcname=name)
    member.mtime = 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    with path.open("rb") as stream:
        tar.addfile(member, stream)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", required=True, help="comma-separated admitted set names")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sets = args.sets.split(",")
    if len(sets) != len(set(sets)) or not sets or any(s not in P0_SETS for s in sets):
        p.error("sets must be unique P0 runner names")
    receipt_path = ROOT / "admit_bank.jsonl"
    if digest(receipt_path) != ADMIT_RECEIPT_SHA:
        raise ValueError("admitted-table receipt changed from the registered target-free era")
    receipt_rows = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    receipts = {row["set"]: row for row in receipt_rows}
    if len(receipt_rows) != len(PINNED) or set(receipts) != set(PINNED):
        raise ValueError("admitted-table receipt has missing or duplicate sets")
    files = {"rev4-featpot/admit_bank.jsonl": receipt_path}
    for name, expected in PINNED.items():
        manifest_path = BANK / name / "_MANIFEST.json"
        if digest(manifest_path) != expected:
            raise ValueError(f"{name}: bank manifest is not the pinned era")
        files[f"rev4-featbank/bank/{name}/_MANIFEST.json"] = manifest_path
        if name not in sets:
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
    hashes = {name: digest(path) for name, path in sorted(files.items())}
    inventory = json.dumps({"schema": "zenfleet-fit-data-v1", "sets": sorted(sets),
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
    print(json.dumps({"sets": sorted(sets), "train_only": set(sets) <= TRAIN,
                      "files": len(files), "archive_bytes": args.out.stat().st_size,
                      "sha256": digest(args.out), "out": str(args.out)}))


if __name__ == "__main__":
    main()
