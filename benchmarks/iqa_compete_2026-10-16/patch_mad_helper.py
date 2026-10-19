#!/usr/bin/env python3
"""MAD helper source overlay — hash-guarded scratch-copy contract.

Produces a *compatibility-patched* scratch copy of the original upstream
``ical_std.c`` (Netflix/vmaf ``STMAD_2011_MatlabCode``). The patch is
ownership-only and fixed: an ``mxArray*`` handle is held separately from
the ``double*`` data pointer so the cleanup path calls
``mxDestroyArray`` on a real ``mxArray*`` — not on a ``double*``, which
is what made the original MEX fault inside
``mxArray::~mxArray``/``mxDestroyArray`` on the x86 Octave 6.4 build
tested (SIGSEGV; ``ical_std.c:159`` tail call, per the contained gdb
backtrace and the module disassembly).

Guarantees enforced here:
- the input file must sha256-match the registered original source
  (``4a3370ec…``) — hash drift fails loudly;
- each of the exactly three regions must occur exactly once;
- every other byte is preserved (the patch is a pure splice);
- the produced bytes must sha256-match the already-validated
  compatibility copy (``79fb753e…``) — if they differ, the run fails
  rather than emitting an unvalidated artifact;
- outputs are created ``O_EXCL``: an existing *different* artifact is
  preserved and fails loudly; an identical one is reported
  ``exists-identical`` and not rewritten. Source, output and manifest
  paths must be distinct before anything is read or written.

This module is benchmark-harness glue (``benchmarks/iqa_compete_2026-
10-16``). It performs **no** compile, metric import, oracle/metric
execution or network access — it writes a source copy plus a JSON
provenance manifest. Originals are never modified. The produced copy is
a compatibility-patched reference only and must never be relabeled the
original canonical artifact (the original remains a failing artifact:
on the x86 Octave 6.4 build tested, its compiled MEX ``24d5beb5…``
SIGSEGVs on the first valid call).

Source paths are supplied explicitly: ``--source-root <dir>``
(repeatable; each registered source is resolved as
``<root>/<relpath>`` and the first existing candidate wins — the
registered sha256 still binds either way) and/or ``--source-path`` —
whose form differs per command: ``patch`` takes a plain
``--source-path FILE`` for its ``--name``; ``list``/``selfcheck`` take
``--source-path <name>=<path>`` (repeatable). There are no built-in
host paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_factor_oracle import (  # noqa: E402
    PatchError,
    preflight_distinct,
    sha256_bytes,
    write_artifact,
)

SCHEMA = "mad-helper-overlay/1"

# sha256 of the byte stream this patcher is required to produce — the
# exact scratch copy that passed the six contained helper cases
# (16x16, 17x19, 64x64; textured+constant) cleanly.
EXPECTED_AFTER_SHA256 = (
    "79fb753e584603cdd8af8b5e7bd6933eaf6a5ef78dc778ff306a1842c592b5cd")


@dataclass(frozen=True)
class Region:
    name: str
    old: bytes
    new: bytes


# The three ownership-only edits, exactly as applied to the validated
# compatibility copy. Loops, float constants, indexing, types and the
# plhs output order are untouched.
REGIONS: tuple[Region, ...] = (
    Region(
        name="declare-handle",
        old=b"    double *TMP;\n    int i,j,iB,jB;",
        new=b"    double *TMP;\n    mxArray *tmpArr;\n    int i,j,iB,jB;",
    ),
    Region(
        name="split-alloc-getpr",
        old=b"    TMP = mxGetPr( mxCreateDoubleMatrix(colLen, rowLen, mxREAL) );",
        new=(
            b"    tmpArr = mxCreateDoubleMatrix(colLen, rowLen, mxREAL);\n"
            b"    TMP = mxGetPr( tmpArr );"
        ),
    ),
    Region(
        name="destroy-handle",
        old=b"    mxDestroyArray(TMP);",
        new=b"    mxDestroyArray(tmpArr);",
    ),
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    role: str
    relpath: str  # resolved under each --source-root
    sha256: str
    origin: str
    source_url: str
    commit: str
    failing_mex_sha256: str


SOURCES: dict[str, SourceSpec] = {
    "mad-ical_std-c": SourceSpec(
        name="mad-ical_std-c",
        role="compatibility-reference",
        relpath="mad/STMAD-ical_std.c",
        sha256="4a3370ece03c7090f682b7f1c6db04e27e535f7933b0b37fcdddc4e6eb56c2f2",
        origin=(
            "Original upstream ical_std.c from Netflix/vmaf "
            "matlab/STMAD_2011_MatlabCode @ "
            "229f2478ab53ae0fd6b07328802294ba15fd0cdd (no license header "
            "in file; terms unclear — private oracle/test material only, "
            "not a shipped dependency)."
        ),
        source_url=(
            "https://raw.githubusercontent.com/Netflix/vmaf/"
            "229f2478ab53ae0fd6b07328802294ba15fd0cdd/"
            "matlab/STMAD_2011_MatlabCode/ical_std.c"
        ),
        commit="229f2478ab53ae0fd6b07328802294ba15fd0cdd",
        failing_mex_sha256=(
            "24d5beb5e36b20ade2cd1bcf686b265d4101189b9f0f0cb84db84ed8f56f0198"),
    ),
}


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
    for r in REGIONS:
        count = data.count(r.old)
        if count == 0:
            raise PatchError(
                f"{spec.name}: region {r.name} pattern absent in {path}")
        if count > 1:
            raise PatchError(
                f"{spec.name}: region {r.name} pattern occurs {count} "
                f"times in {path}")
    return data


def patch_bytes(data: bytes, spec: SourceSpec) -> tuple[bytes, list[dict]]:
    """Splice the three fixed regions; returns (new_bytes, edits) where
    each edit records the byte offset in the *original* stream plus the
    old/new byte strings for the manifest."""
    edits: list[dict] = []
    out = data
    for r in REGIONS:
        idx = out.find(r.old)
        if idx < 0:
            raise PatchError(f"{spec.name}: region {r.name} absent at patch time")
        # Splice position comes from the evolving stream; the manifest
        # records the region's position in the ORIGINAL source so the
        # checker can re-derive it without replaying earlier splices.
        orig_idx = data.find(r.old)
        out = out[:idx] + r.new + out[idx + len(r.old):]
        edits.append({
            "region": r.name,
            "byte_offset": orig_idx,
            "old_bytes": r.old.decode("utf-8"),
            "new_bytes": r.new.decode("utf-8"),
            "old_len": len(r.old),
            "new_len": len(r.new),
        })
    if out == data:
        raise PatchError(f"{spec.name}: replacement produced no change")
    after = sha256_bytes(out)
    if after != EXPECTED_AFTER_SHA256:
        raise PatchError(
            f"{spec.name}: result drift — produced {after}, expected the "
            f"validated compatibility bytes {EXPECTED_AFTER_SHA256}; "
            "refusing to emit an unvalidated artifact"
        )
    return out, edits


def make_manifest(
    spec: SourceSpec,
    src_path: Path,
    out_path: Path,
    edits: list[dict],
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
            "source_url": spec.source_url,
            "commit": spec.commit,
            "failing_mex_sha256": spec.failing_mex_sha256,
        },
        "patch": {
            "kind": "ownership-cleanup-compat",
            "regions": edits,
            "region_count": len(edits),
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "expected_after_sha256": EXPECTED_AFTER_SHA256,
        },
        "output": {"path": str(out_path), "status": status},
        "notes": (
            "Compatibility-patched reference copy only. The original "
            "ical_std.c and its compiled MEX (failing_mex_sha256, "
            "SIGSEGV at the :159 mxDestroyArray(TMP) tail call on the "
            "x86 Octave 6.4 build tested — Octave "
            "typedefs mxArray to void in C mode so the double*->void* "
            "conversion compiles silently) remain the canonical "
            "artifacts. Never relabel this copy canonical. Numeric body "
            "(loops, /255.0 and /63.0 divisors, indexing, plhs output "
            "order [std,modified_std,mean]) is byte-identical outside "
            "the three regions by construction; validated on six "
            "contained helper cases, which is bounded coverage, not a "
            "metric or parity qualification."
        ),
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
    out, edits = patch_bytes(data, spec)
    status = write_artifact(out_path, out)
    manifest = make_manifest(
        spec, src, out_path, edits,
        sha256_bytes(data), sha256_bytes(out), status,
    )
    write_artifact(mpath, (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("list", help="list known overlay sources + drift state")
    lp.add_argument("--source-root", action="append", help="root dir to resolve sources under (repeatable)")
    lp.add_argument("--source-path", action="append", help="explicit <name>=<path> (repeatable)")
    pp = sub.add_parser("patch", help="write the compat-patched scratch copy + manifest")
    pp.add_argument("--name", required=True, choices=sorted(SOURCES))
    pp.add_argument("--source-path", default=None, help="explicit input path for --name")
    pp.add_argument("--source-root", action="append", help="root dir to resolve the source under (repeatable)")
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
