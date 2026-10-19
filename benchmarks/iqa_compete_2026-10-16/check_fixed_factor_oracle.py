#!/usr/bin/env python3
"""Independent integrity checks for fixed_factor_oracle.py outputs.

Verifies that a produced scratch copy differs from its verified source at
*exactly one* byte region — the recorded expression replaced by the
literal factor — with no other mutation (newline style, ordering, every
other byte preserved). The source must hash-match the *registry* spec
(not merely the manifest's claim), and the manifest's schema, role,
origin and runtime_contract must bind to the registry — forged metadata
or a co-rewritten source+manifest is rejected.

Also runs the built-in negative-control suite.

Pure stdlib; performs no metric/oracle execution.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fixed_factor_oracle as ffo  # noqa: E402


class CheckError(Exception):
    pass


def verify_patch(src_data: bytes, spec: ffo.SourceSpec, factor: int,
                 out_data: bytes) -> None:
    """Independent one-replacement diff: out must equal src with exactly
    the target expression replaced by str(factor) at its single offset."""
    # strict type: bool/float compare equal to a valid factor but are not
    # valid literals — a manifest could carry JSON true or 1.0.
    if type(factor) is not int or factor not in ffo.VALID_FACTORS:
        raise CheckError(f"illegal factor {factor!r}")
    n = src_data.count(spec.target)
    if n != 1:
        raise CheckError(f"source target count {n}, expected 1")
    i = src_data.find(spec.target)
    expect = src_data[:i] + str(factor).encode("ascii") + src_data[i + len(spec.target):]
    if out_data != expect:
        raise CheckError("output is not the expected single-replacement result")
    if src_data.count(b"\n") != out_data.count(b"\n") or src_data.count(b"\r\n") != out_data.count(b"\r\n"):
        raise CheckError("newline count/style changed by the patch")
    if out_data.count(spec.target):
        raise CheckError("residual unpatched target expression in output")
    # `expect` is constructed as a splice at the single target offset, so
    # equality already proves a one-region diff.


def check_manifest(mpath: Path) -> dict:
    m = json.loads(mpath.read_text())
    if m.get("schema") != ffo.SCHEMA:
        raise CheckError(f"manifest schema {m.get('schema')!r} != {ffo.SCHEMA!r}")
    name = m.get("source", {}).get("name")
    if name not in ffo.SOURCES:
        raise CheckError(f"manifest source name {name!r} not in registry")
    spec = ffo.SOURCES[name]
    s = m["source"]
    # Metadata must bind to the registry — a forged manifest cannot
    # re-label a diagnostic source primary or rewrite its provenance.
    if s.get("role") != spec.role:
        raise CheckError(f"role {s.get('role')!r} != registry {spec.role!r}")
    if s.get("origin") != spec.origin:
        raise CheckError("origin does not match registry")
    if s.get("runtime_contract", "") != spec.runtime_contract:
        raise CheckError("runtime_contract does not match registry")
    src = Path(s["path"]).expanduser()
    out = Path(m["output"]["path"]).expanduser()
    src_data = src.read_bytes()
    out_data = out.read_bytes()
    src_sha = ffo.sha256_bytes(src_data)
    if src_sha != spec.sha256:
        raise CheckError(
            "source sha256 does not match registry spec.sha256 "
            "(tampered source + co-rewritten manifest is still rejected)"
        )
    if src_sha != m["patch"]["before_sha256"]:
        raise CheckError("before_sha256 mismatch")
    if ffo.sha256_bytes(out_data) != m["patch"]["after_sha256"]:
        raise CheckError("after_sha256 mismatch")
    try:
        rep = int(m["patch"]["replacement_bytes"])
    except (TypeError, ValueError):
        raise CheckError("replacement_bytes is not an integer literal")
    if type(m["patch"]["factor"]) is not int or m["patch"]["factor"] != rep:
        raise CheckError("factor/replacement mismatch or non-int factor")
    if m["patch"]["factor"] not in ffo.VALID_FACTORS:
        raise CheckError("manifest factor outside allowed set")
    if spec.target.decode() != m["patch"]["target_bytes"]:
        raise CheckError("manifest target mismatch registry")
    if src_data.find(spec.target) != m["patch"]["byte_offset"]:
        raise CheckError("recorded byte_offset does not match target position")
    if m["output"]["status"] not in ("written", "exists-identical"):
        raise CheckError(f"output status {m['output']['status']!r} invalid")
    verify_patch(src_data, spec, m["patch"]["factor"], out_data)
    return m


def _expect_fail(desc: str, fn) -> str:
    try:
        fn()
    except (ffo.PatchError, CheckError) as e:
        return f"  PASS  {desc} -> rejected ({e})"
    return f"  FAIL  {desc} -> unexpectedly accepted"


def _forge(m: dict, **kv) -> dict:
    m = json.loads(json.dumps(m))  # deep copy
    for k, v in kv.items():
        node = m
        *pre, last = k.split(".")
        for p in pre:
            node = node[p]
        node[last] = v
    return m


def selfcheck(evidence: Path, source_paths: dict[str, str],
              roots: list[str]) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    results = []
    log = evidence / "selfcheck.log"

    def record(line: str) -> None:
        print(line)
        results.append(line)

    # --- positive sweep: every known source x every factor -------------
    for name, spec in ffo.SOURCES.items():
        try:
            src = ffo.resolve_path(spec, source_paths.get(name), roots)
        except ffo.PatchError as e:
            record(f"  FAIL  {name}: {e}")
            continue
        for k in ffo.VALID_FACTORS:
            out = evidence / f"{name}-k{k}{src.suffix}"
            mpath = evidence / f"{name}-k{k}.manifest.json"
            try:
                ffo.preflight_distinct(source=src, output=out, manifest=mpath)
                data = ffo.load_and_verify(src, spec)
                patched, _off = ffo.patch_bytes(data, spec, k)
                status = ffo.write_artifact(out, patched)
                man = ffo.make_manifest(spec, src, out, k, _off,
                                        ffo.sha256_bytes(data),
                                        ffo.sha256_bytes(patched), status)
                ffo.write_artifact(
                    mpath, (json.dumps(man, indent=2) + "\n").encode())
                check_manifest(mpath)
                record(f"  PASS  {name} k={k} ({status})")
            except (ffo.PatchError, CheckError) as e:
                record(f"  FAIL  {name} k={k}: {e}")

    # --- negative controls --------------------------------------------
    ref = ffo.SOURCES["fsim-matlab"]
    src = ffo.resolve_path(ref, source_paths.get(ref.name), roots)
    good = src.read_bytes()

    with tempfile.TemporaryDirectory(dir=evidence) as td:
        td = Path(td)
        corrupt = td / "corrupt.m"
        bad = bytearray(good)
        bad[10] ^= 0x01
        corrupt.write_bytes(bytes(bad))
        record(_expect_fail(
            "corrupt source (hash drift)",
            lambda: ffo.load_and_verify(corrupt, ref)))

        # Crafted specs bind sha to the synthetic file so the target-count
        # branches (not the hash branch) are exercised for real.
        dup = td / "dup.m"
        dup.write_bytes(good + b"\nF2 = " + ref.target + b";\n")
        dup_spec = dataclasses.replace(
            ref, sha256=ffo.sha256_bytes(dup.read_bytes()))
        record(_expect_fail(
            "duplicate target assignment",
            lambda: ffo.load_and_verify(dup, dup_spec)))

        missing = td / "missing.m"
        missing.write_bytes(good.replace(ref.target, b"1", 1))
        missing_spec = dataclasses.replace(
            ref, sha256=ffo.sha256_bytes(missing.read_bytes()))
        record(_expect_fail(
            "missing target assignment",
            lambda: ffo.load_and_verify(missing, missing_spec)))

        record(_expect_fail(
            "illegal factor 5",
            lambda: ffo.patch_bytes(good, ref, 5)))
        record(_expect_fail(
            "illegal factor 0",
            lambda: ffo.patch_bytes(good, ref, 0)))
        # bool/float compare equal to 1 but inject invalid literal tokens
        record(_expect_fail(
            "illegal factor True (bool)",
            lambda: ffo.patch_bytes(good, ref, True)))
        record(_expect_fail(
            "illegal factor 1.0 (float)",
            lambda: ffo.patch_bytes(good, ref, 1.0)))

        existing = td / "exists.m"
        existing.write_bytes(b"different content entirely")
        record(_expect_fail(
            "existing different output",
            lambda: ffo.write_artifact(existing, b"proposed content")))

        # tampered output caught by checker
        patched, _ = ffo.patch_bytes(good, ref, 3)
        tampered = bytearray(patched)
        tampered[100] ^= 0x01
        record(_expect_fail(
            "tampered output (checker)",
            lambda: verify_patch(good, ref, 3, bytes(tampered))))

        # idempotent identical output allowed
        same = td / "same.m"
        same.write_bytes(patched)
        try:
            st = ffo.write_artifact(same, patched)
            record(f"  PASS  existing identical output -> {st}")
        except ffo.PatchError as e:
            record(f"  FAIL  existing identical output: {e}")

        # --- path aliasing: source/output/manifest must be distinct ---
        outp = td / "out.m"
        for desc, kwargs in (
            ("alias: manifest == source",
             dict(source=src, output=outp, manifest=src)),
            ("alias: manifest == output",
             dict(source=src, output=outp, manifest=outp)),
            ("alias: output == source",
             dict(source=src, output=src, manifest=td / "m.json")),
        ):
            before = src.read_bytes()
            record(_expect_fail(
                desc, lambda kw=kwargs: ffo.preflight_distinct(**kw)))
            if src.read_bytes() != before:
                record(f"  FAIL  {desc}: source mutated despite rejection")

        # --- overwrite refusal: existing different manifest preserved ---
        mp = td / "existing.manifest.json"
        mp.write_bytes(b'{"forged": true}')
        record(_expect_fail(
            "existing different manifest preserved",
            lambda: ffo.write_artifact(mp, b'{"new": 1}')))
        if mp.read_bytes() != b'{"forged": true}':
            record("  FAIL  existing manifest content was overwritten")

        # --- forged manifest / co-rewritten source ---------------------
        real_out = evidence / "fsim-matlab-k3.m"
        real_m = json.loads(
            (evidence / "fsim-matlab-k3.manifest.json").read_text())

        # tamper a copy of the source and co-rewrite before_sha to match
        tsrc = td / "tampered-src.m"
        tdata = bytearray(good)
        tdata[200] ^= 0x01
        tsrc.write_bytes(bytes(tdata))
        forged = _forge(
            real_m,
            **{"source.path": str(tsrc),
               "patch.before_sha256": ffo.sha256_bytes(bytes(tdata))})
        fm = td / "forged-hash.manifest.json"
        fm.write_text(json.dumps(forged))
        record(_expect_fail(
            "forged before_sha256 + tampered source",
            lambda: check_manifest(fm)))

        for desc, key, val in (
            ("forged role", "source.role", "diagnostic"),
            ("forged schema", "schema", "other/9"),
            ("forged origin", "source.origin", "invented origin"),
            ("forged runtime_contract", "source.runtime_contract", "none"),
            ("forged output status", "output.status", "bogus"),
            ("forged factor", "patch.factor", 9),
            # JSON true/1.0 parse to bool/float and compare equal to 1 —
            # strict-int binding must reject both at manifest level.
            ("forged factor true", "patch.factor", True),
            ("forged factor 1.0", "patch.factor", 1.0),
            ("malformed replacement_bytes", "patch.replacement_bytes", "x"),
        ):
            val = val
            fm2 = td / f"forged-{len(results)}.manifest.json"
            fm2.write_text(json.dumps(_forge(real_m, **{key: val})))
            record(_expect_fail(desc, lambda f=fm2: check_manifest(f)))

    n_fail = sum(1 for r in results if r.startswith("  FAIL"))
    record(f"selfcheck: {len(results) - n_fail}/{len(results)} pass, {n_fail} fail")
    ffo.write_artifact(log, ("\n".join(results) + "\n").encode())
    return 1 if n_fail else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("check", help="verify one produced patch via its manifest")
    cp.add_argument("--manifest", required=True)
    sp = sub.add_parser("selfcheck", help="full positive sweep + negative controls")
    sp.add_argument("--evidence", required=True, help="NEW output dir for scratch copies + log")
    sp.add_argument("--source-root", action="append",
                    help="root dir to resolve sources under (repeatable)")
    sp.add_argument("--source-path", action="append",
                    help="explicit <name>=<path> (repeatable)")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        try:
            m = check_manifest(Path(args.manifest).expanduser())
            print(f"OK {m['source']['name']} k={m['patch']['factor']} -> {m['output']['path']}")
            return 0
        except CheckError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
    try:
        spaths = ffo._parse_source_paths(args.source_path or [])
    except ffo.PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return selfcheck(Path(args.evidence).expanduser(), spaths,
                     args.source_root or [])


if __name__ == "__main__":
    raise SystemExit(main())
