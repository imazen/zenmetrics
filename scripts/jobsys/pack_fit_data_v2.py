#!/usr/bin/env python3
"""Pack the Rev4 potential Instrument v2 wide tables into one content-addressed fit-data archive.

Governing record: zensim benchmarks/rev4_featpot_v2_amendment_2026-09-30.md (revision R1, layout R1.1). Members
are rev4-featpot/v2/wide/{main,aux}/{real,p1,p2,p3}/* (tables, provenance sidecars, held-out keys, receipts) and
rev4-featpot/v2/wide/keep_lists.json. Every table is hash-checked against its variant receipt before it is
copied; nothing is decoded. One archive serves the whole grid because the fit-cell executor binds
/var/tmp/rev4-featpot to a single extracted archive per container. `--select family/variant` (repeatable) packs
only those variant directories, for a jobset that reads no others; the inventory records the selection. The
caller must have logged the transport in the owning exposure ledger.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

WIDE = Path("/var/tmp/rev4-featpot/v2/wide")
FAMILIES = ("main", "aux")
VARIANTS = ("real", "p1", "p2", "p3")


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


def variant_files(family: str, variant: str) -> dict[str, Path]:
    vdir = WIDE / family / variant
    receipt = json.loads((vdir / "receipt.json").read_text())
    if (receipt.get("schema") != "rev4-featpot-v2-wide-v2" or receipt.get("family") != family
            or receipt.get("variant") != variant):
        raise ValueError(f"{family}/{variant}: wide receipt identity mismatch")
    files = {"receipt.json": vdir / "receipt.json"}
    for leg, rec in receipt["legs"].items():
        for part in ("full", "fit", "dev"):
            if part not in rec:
                continue
            path = Path(rec[part]["path"])
            if path.parent != vdir or digest(path) != rec[part]["sha256"] or \
                    digest(Path(f"{path}.manifest.json")) != rec[part]["manifest_sha256"]:
                raise ValueError(f"{family}/{variant}/{leg}/{part}: table changed after its receipt")
            files[path.name] = path
            files[f"{path.name}.manifest.json"] = Path(f"{path}.manifest.json")
        if "keys_sha256" in rec:
            keys = vdir / f"{leg}.keys.parquet"
            if digest(keys) != rec["keys_sha256"]:
                raise ValueError(f"{family}/{variant}/{leg}: held-out keys changed after the receipt")
            files[keys.name] = keys
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--select", action="append", default=None, metavar="FAMILY/VARIANT",
                   help="pack only these variant directories (repeatable); default: all")
    args = p.parse_args()
    every = [(f, v) for f in FAMILIES for v in VARIANTS]
    chosen = every if args.select is None else [tuple(s.split("/", 1)) for s in args.select]
    unknown = [c for c in chosen if c not in every]
    if unknown or len(set(chosen)) != len(chosen):
        raise SystemExit(f"--select: unknown or repeated variant directories {unknown or chosen}")
    members = {"rev4-featpot/v2/wide/keep_lists.json": WIDE / "keep_lists.json"}
    for family, variant in [c for c in every if c in chosen]:
        for name, path in variant_files(family, variant).items():
            members[f"rev4-featpot/v2/wide/{family}/{variant}/{name}"] = path
    inventory = {"schema": "zenfleet-fit-data-v1", "label": "POTENTIAL — ceiling, not a model score",
                 "program": "Rev4 potential Instrument v2 (R915 sampling)",
                 "files": {name: digest(path) for name, path in sorted(members.items())}}
    if args.select is not None:  # absent for the full archive, so its bytes are unchanged
        inventory["variant_dirs"] = [f"{f}/{v}" for f, v in every if (f, v) in chosen]
    inv_bytes = json.dumps(inventory, sort_keys=True, indent=2).encode() + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0,
                                                   compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            info = tarfile.TarInfo("input_inventory.json")
            info.size = len(inv_bytes)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(inv_bytes))
            for name, path in sorted(members.items()):
                add_file(tar, path, name)
    print(json.dumps({"sha256": digest(args.out), "bytes": args.out.stat().st_size,
                      "members": len(members), "out": str(args.out)}))


if __name__ == "__main__":
    main()
