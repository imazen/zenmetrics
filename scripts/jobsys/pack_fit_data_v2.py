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
    p.add_argument("--byc-root", type=Path, help="frozen by-construction TRAIN root")
    p.add_argument("--byc-score-root", type=Path, help="frozen TRAIN new-group score root")
    p.add_argument("--byc-label")
    args = p.parse_args()
    if args.byc_root is not None:
        if args.select is not None or args.byc_score_root is None or not args.byc_label:
            p.error("byc needs both frozen roots and its arm label, without --select")
        pack_byc(args)
        return
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


def pack_byc(args):
    """Preserve frozen root bytes and original aliases; no labels are interpreted."""
    import re
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", args.byc_label):
        raise ValueError("unsafe byc label")
    members, bindings = {}, {}
    for role, root in (("fit", args.byc_root), ("score", args.byc_score_root)):
        if not root.is_absolute() or not root.is_relative_to("/var/tmp") or ".." in root.parts:
            raise ValueError("byc requires an absolute staged TRAIN root")
        frozen = json.loads((root / "wide/frozen.json").read_text())
        if (frozen["schema"] != "rev5-recipe-admission-freeze-v1"
                or digest(root / "datasets/receipt.json") != frozen["dataset_legs_sha256"]):
            raise ValueError("byc dataset root is not frozen")
        prefix = f"rev4-featpot/byc/inputs/{role}"
        bindings[str(root)] = prefix
        packet = json.loads((root / "datasets/receipt.json").read_text())
        # A staged root can retain its canonical alias in frozen group paths.
        # Bind that alias to the same bytes, after checking each referenced file.
        references = [(Path(g["path"]), g["table_sha256"]) for g in packet["groups"]]
        references.append((Path(packet["family_guard_path"]), packet["family_guard_sha256"]))
        for path, pin in references:
            alias = path.parent.parent
            if (path.parent.name != "datasets" or not alias.is_absolute() or ".." in alias.parts
                    or not (alias.is_relative_to("/var/tmp") or alias.is_relative_to("/mnt/v/output"))
                    or digest(root / "datasets" / path.name) != pin):
                raise ValueError("byc frozen dataset alias differs from staged bytes")
            if str(alias) in bindings and bindings[str(alias)] != prefix:
                raise ValueError("byc roots require conflicting aliases")
            bindings[str(alias)] = prefix
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("byc frozen roots must contain regular files")
            if path.is_file():
                members[f"{prefix}/{path.relative_to(root)}"] = path
    inventory = dict(schema="zenfleet-fit-data-v1", byc=dict(schema="byc-fit-input-v1",
        label=args.byc_label, root=str(args.byc_root), score_root=str(args.byc_score_root), bindings=bindings),
        files={name: digest(path) for name, path in sorted(members.items())})
    inv = json.dumps(inventory, sort_keys=True, indent=2).encode() + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0,
                                                  compresslevel=1) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            info = tarfile.TarInfo("input_inventory.json")
            info.size = len(inv)
            tar.addfile(info, io.BytesIO(inv))
            for name, path in sorted(members.items()):
                add_file(tar, path, name)
    print(json.dumps(dict(sha256=digest(args.out), bytes=args.out.stat().st_size,
                          members=len(members), out=str(args.out))))


if __name__ == "__main__":
    main()
