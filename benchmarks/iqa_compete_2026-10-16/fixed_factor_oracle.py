#!/usr/bin/env python3
"""Fixed-factor oracle source patcher — scratch-copy contract.

Patches the automatic downscale-factor assignment in a *verified* oracle
source to a literal factor K in {1,2,3,4}, preserving every other byte
(including newline style and the line terminator). Emits a JSON manifest
per patch for the integrity checker.

This module is benchmark-harness glue for the AIC2026 fixed-factor
verification (``benchmarks/iqa_compete_2026-10-16``). It performs no
metric or oracle evaluation, no imports beyond the stdlib, and no
network access. Originals are never modified.

Source paths are supplied explicitly: ``--source-path <name>=<path>``
(repeatable) and/or ``--source-root <dir>`` (repeatable; each registered
source is resolved as ``<root>/<relpath>`` and the first existing
candidate wins — the registered sha256 still binds either way). There
are no built-in host paths.

Roles: ``primary`` = the oracle that binds the ported metric's parity
contract (the authors' MATLAB implementation for FSIM/VSI; pyiqa for the
``*-pyiqa`` SSIM-family metric names, whose oracle is pyiqa itself).
``diagnostic`` = a separately labeled variant comparison — pyiqa's FSIM/
VSI patch does not displace MATLAB parity.

Runtime contract notes (recorded, not enforced here):
- pyiqa SSIM/MS-SSIM additionally require ``downsample=True`` at the
  call site for the patched ``f`` to take effect; the flag is a
  constructor/function argument, not a source mutation.
- For pyiqa ``ms_ssim``, ``downsample=True`` applies the patched ``f``
  at *every* pyramid level (per-level consequence); this patcher fixes
  ``f`` once in the shared ``ssim`` function — there is no separate
  pre-pool mutation.

Write discipline: every produced artifact (patched copy, manifest) is
created with ``O_EXCL`` — an existing *different* artifact is preserved
and fails loudly; an existing *identical* output is reported
``exists-identical`` and not rewritten. Source, output and manifest
paths are resolved and required distinct before anything is read or
written.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "fixed-factor-oracle-patch/1"
VALID_FACTORS = (1, 2, 3, 4)


class PatchError(Exception):
    """Loud failure: hash drift, duplicate/missing target, bad factor,
    path aliasing, or a conflicting pre-existing artifact."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    role: str  # "primary" | "diagnostic"
    relpath: str  # resolved under each --source-root
    sha256: str
    target: bytes
    origin: str
    runtime_contract: str = ""


SOURCES: dict[str, SourceSpec] = {
    "fsim-matlab": SourceSpec(
        name="fsim-matlab",
        role="primary",
        relpath="fsim/FeatureSIM.m",
        sha256="4b23dc2881c71dfa996e40fd84b9b946f3934b36ecfb3ee4dfbafa5505bcbba9",
        target=b"max(1,round(minDimension / 256))",
        origin=(
            "Author release FeatureSIM.m, source URL "
            "https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m "
            "(edu/research-only terms in file header). "
            "Assignment at line 72."
        ),
    ),
    "vsi-matlab": SourceSpec(
        name="vsi-matlab",
        role="primary",
        relpath="vsi/VSI.m",
        sha256="5939b9053966f58377e08052b5e2d0e0e3f954040fe732625591363285433954",
        target=b"max(1,round(minDimension / 256))",
        origin=(
            "Author release VSI.m, source URL "
            "https://cslinzhang.github.io/home/VSI/Files/VSI.m "
            "(edu/research-only terms in file header). "
            "Assignment at line 55."
        ),
    ),
    "fsim-pyiqa": SourceSpec(
        name="fsim-pyiqa",
        role="diagnostic",
        relpath="pyiqa/archs/fsim_arch.py",
        sha256="c8b2320fea6fa662a1346d705932f280357c3e93f6be8800a6910cb24458ef05",
        target=b"max(1, round(min(x.shape[-2:]) / 256))",
        origin=(
            "pyiqa 0.1.16 installed package file pyiqa/archs/fsim_arch.py, "
            "upstream https://github.com/chaofengc/IQA-PyTorch. "
            "Assignment at line 77."
        ),
    ),
    "vsi-pyiqa": SourceSpec(
        name="vsi-pyiqa",
        role="diagnostic",
        relpath="pyiqa/archs/vsi_arch.py",
        sha256="a519f83346101e552955f874bb188fdf98fa42ca225e12ab4868c61a8cc4eb13",
        target=b"max(1, round(min(vs_x.size()[-2:]) / 256))",
        origin=(
            "pyiqa 0.1.16 installed package file pyiqa/archs/vsi_arch.py, "
            "upstream https://github.com/chaofengc/IQA-PyTorch. "
            "Assignment at line 118."
        ),
    ),
    "ssim-pyiqa": SourceSpec(
        name="ssim-pyiqa",
        role="primary",
        relpath="pyiqa/archs/ssim_arch.py",
        sha256="a0eb8e0bedfae99d9d014371a77b7d9afb110c71a22b6cda06a4992da0d951e8",
        target=b"max(1, round(min(X.size()[-2:]) / 256))",
        origin=(
            "pyiqa 0.1.16 installed package file pyiqa/archs/ssim_arch.py, "
            "upstream https://github.com/chaofengc/IQA-PyTorch. "
            "Assignment at line 47 inside ssim(); shared by ms_ssim()."
        ),
        runtime_contract=(
            "requires downsample=True at the call site for the patched "
            "factor to take effect (constructor/function argument, not a "
            "source mutation); under ms_ssim the flag applies the patched "
            "f at every pyramid level (per-level consequence)."
        ),
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_path(spec: SourceSpec, source_path: str | None,
                 roots: list[str]) -> Path:
    if source_path:
        return Path(source_path).expanduser()
    for root in roots:
        cand = Path(root).expanduser() / spec.relpath
        if cand.is_file():
            return cand
    raise PatchError(
        f"{spec.name}: unresolved — pass --source-path {spec.name}=<path> "
        f"or a --source-root containing {spec.relpath}"
    )


def preflight_distinct(**paths: Path) -> None:
    """All produced/consumed paths must resolve to distinct files —
    checked before anything is read or written."""
    items = [
        (label, p.expanduser().resolve()) for label, p in paths.items()
    ]
    for (la, a), (lb, b) in itertools.combinations(items, 2):
        if a == b:
            raise PatchError(f"{la} and {lb} resolve to the same path: {a}")


def load_and_verify(path: Path, spec: SourceSpec) -> bytes:
    if not path.is_file():
        raise PatchError(f"{spec.name}: source missing: {path}")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != spec.sha256:
        raise PatchError(
            f"{spec.name}: hash drift — expected {spec.sha256}, got {actual} "
            f"at {path} (source changed; re-audit required)"
        )
    count = data.count(spec.target)
    if count == 0:
        raise PatchError(f"{spec.name}: targeted factor assignment not found in {path}")
    if count > 1:
        raise PatchError(
            f"{spec.name}: targeted factor assignment occurs {count} times in {path}"
        )
    return data


def patch_bytes(data: bytes, spec: SourceSpec, factor: int) -> tuple[bytes, int]:
    """Replace the target expression with a literal factor; returns
    (new_bytes, byte-offset of the single replacement)."""
    # strict type: bool True and float 1.0 compare equal to 1 but would
    # inject an invalid literal token into the patched source.
    if type(factor) is not int or factor not in VALID_FACTORS:
        raise PatchError(f"illegal factor {factor!r}: expected an int in {VALID_FACTORS}")
    idx = data.find(spec.target)
    if idx < 0:
        raise PatchError(f"{spec.name}: target absent at patch time")
    out = data[:idx] + str(factor).encode("ascii") + data[idx + len(spec.target):]
    if out == data or data.count(spec.target) != 1:
        raise PatchError(f"{spec.name}: replacement produced no change or non-unique target")
    return out, idx


def write_artifact(path: Path, content: bytes) -> str:
    """Create a NEW file atomically (O_EXCL — no exists()+write race).
    An existing *different* file is preserved and fails loudly; an
    existing *identical* file is reported, not rewritten."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() == content:
            return "exists-identical"
        raise PatchError(
            f"artifact already exists and differs: {path} "
            "(preserved — choose a fresh path)"
        )
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return "written"


def make_manifest(
    spec: SourceSpec,
    src_path: Path,
    out_path: Path,
    factor: int,
    offset: int,
    before_sha: str,
    after_sha: str,
    status: str,
) -> dict:
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "name": spec.name,
            "role": spec.role,
            "path": str(src_path),
            "origin": spec.origin,
            "runtime_contract": spec.runtime_contract,
        },
        "patch": {
            "factor": factor,
            "target_bytes": spec.target.decode("utf-8"),
            "replacement_bytes": str(factor),
            "byte_offset": offset,
            "before_sha256": before_sha,
            "after_sha256": after_sha,
        },
        "output": {"path": str(out_path), "status": status},
    }


def _parse_source_paths(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise PatchError(f"--source-path expects <name>=<path>, got {p!r}")
        name, path = p.split("=", 1)
        if name not in SOURCES:
            raise PatchError(f"--source-path: unknown source {name!r}")
        out[name] = path
    return out


def cmd_list(args: argparse.Namespace) -> int:
    overrides = _parse_source_paths(args.source_path or [])
    for spec in SOURCES.values():
        try:
            p = resolve_path(spec, overrides.get(spec.name), args.source_root or [])
        except PatchError:
            print(f"{spec.name}\t{spec.role}\tunresolved\t{spec.sha256[:16]}…")
            continue
        state = (
            "ok"
            if sha256_bytes(p.read_bytes()) == spec.sha256
            else "DRIFT"
        )
        print(f"{spec.name}\t{spec.role}\t{state}\t{spec.sha256[:16]}…\t{p}")
    return 0


def cmd_patch(args: argparse.Namespace) -> int:
    spec = SOURCES[args.name]
    src = resolve_path(spec, args.source_path, args.source_root or [])
    out_path = Path(args.out)
    mpath = Path(args.manifest) if args.manifest else Path(str(out_path) + ".manifest.json")
    # Aliasing must be ruled out before any read or mutation.
    preflight_distinct(source=src, output=out_path, manifest=mpath)
    data = load_and_verify(src, spec)
    out, offset = patch_bytes(data, spec, args.factor)
    status = write_artifact(out_path, out)
    manifest = make_manifest(
        spec, src, out_path, args.factor, offset,
        sha256_bytes(data), sha256_bytes(out), status,
    )
    write_artifact(mpath, (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("list", help="list known oracle sources + drift state")
    lp.add_argument("--source-root", action="append", help="root dir to resolve sources under (repeatable)")
    lp.add_argument("--source-path", action="append", help="explicit <name>=<path> (repeatable)")
    pp = sub.add_parser("patch", help="write a factor-patched scratch copy + manifest")
    pp.add_argument("--name", required=True, choices=sorted(SOURCES))
    pp.add_argument("--source-path", default=None, help="explicit input path for --name")
    pp.add_argument("--source-root", action="append", help="root dir to resolve the source under (repeatable)")
    pp.add_argument("--factor", required=True, type=int, choices=VALID_FACTORS)
    pp.add_argument("--out", required=True, help="NEW output path (exclusive create)")
    pp.add_argument("--manifest", default=None, help="manifest path (default: <out>.manifest.json)")
    args = ap.parse_args(argv)
    try:
        return {"list": cmd_list, "patch": cmd_patch}[args.cmd](args)
    except PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
