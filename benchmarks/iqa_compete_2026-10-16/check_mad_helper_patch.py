#!/usr/bin/env python3
"""Independent integrity checks for patch_mad_helper.py outputs.

Verifies that a produced scratch copy differs from its verified source
at *exactly* the three registered ownership regions — nothing else
changed (the checker recomputes the splice independently rather than
trusting the manifest). The source must hash-match the *registry* spec,
and the manifest's schema, role, origin, URL, commit and failing-MEX
identity must bind to the registry — forged metadata or a co-rewritten
source+manifest is rejected. The produced bytes must also hash-match
the validated compatibility copy (`EXPECTED_AFTER_SHA256`), so a
numeric-body mutation anywhere outside the three regions fails the
result-drift guard even if it slips past region accounting.

Also runs the built-in negative-control suite.

Pure stdlib + the shared fixed_factor_oracle helpers; performs no
metric imports, compile, or oracle execution.
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

import patch_mad_helper as pmh  # noqa: E402


class CheckError(Exception):
    pass


def _expected_splice(src_data: bytes) -> bytes:
    """Independent recomputation of the three-region splice, in registry
    order. Used to prove the output differs at exactly those regions."""
    out = src_data
    for r in pmh.REGIONS:
        i = out.find(r.old)
        if i < 0:
            raise CheckError(f"region {r.name} pattern absent in source")
        out = out[:i] + r.new + out[i + len(r.old):]
    return out


def verify_patch(src_data: bytes, spec: pmh.SourceSpec,
                 out_data: bytes) -> None:
    for r in pmh.REGIONS:
        n = src_data.count(r.old)
        if n != 1:
            raise CheckError(f"region {r.name} count {n}, expected 1")
    expect = _expected_splice(src_data)
    if out_data != expect:
        raise CheckError("output is not the expected three-region splice")
    if pmh.sha256_bytes(out_data) != pmh.EXPECTED_AFTER_SHA256:
        raise CheckError("output sha256 != validated compatibility bytes")
    for r in pmh.REGIONS:
        if out_data.count(r.old):
            raise CheckError(f"residual unpatched region {r.name} in output")
    # Numeric-body spot guard: the upstream divisors must survive
    # untouched (the splice equality above already proves it, but the
    # explicit check keeps a cheap readable assertion at review level).
    for tok in (b"255.0", b"63.0", b"64.0"):
        if out_data.count(tok) != src_data.count(tok):
            raise CheckError(f"numeric token {tok!r} count changed")
    if src_data.count(b"\r\n") != out_data.count(b"\r\n"):
        raise CheckError("CRLF style changed by the patch")


def check_manifest(mpath: Path) -> dict:
    m = json.loads(mpath.read_text())
    if m.get("schema") != pmh.SCHEMA:
        raise CheckError(f"manifest schema {m.get('schema')!r} != {pmh.SCHEMA!r}")
    name = m.get("source", {}).get("name")
    if name not in pmh.SOURCES:
        raise CheckError(f"manifest source name {name!r} not in registry")
    spec = pmh.SOURCES[name]
    s = m["source"]
    # Metadata must bind to the registry — a forged manifest cannot
    # relabel the compat copy canonical or rewrite provenance.
    if s.get("role") != spec.role:
        raise CheckError(f"role {s.get('role')!r} != registry {spec.role!r}")
    if s.get("origin") != spec.origin:
        raise CheckError("origin does not match registry")
    if s.get("source_url") != spec.source_url:
        raise CheckError("source_url does not match registry")
    if s.get("commit") != spec.commit:
        raise CheckError("commit does not match registry")
    if s.get("failing_mex_sha256") != spec.failing_mex_sha256:
        raise CheckError("failing_mex_sha256 does not match registry")
    src = Path(s["path"]).expanduser()
    out = Path(m["output"]["path"]).expanduser()
    src_data = src.read_bytes()
    out_data = out.read_bytes()
    src_sha = pmh.sha256_bytes(src_data)
    if src_sha != spec.sha256:
        raise CheckError(
            "source sha256 does not match registry spec.sha256 "
            "(tampered source + co-rewritten manifest is still rejected)"
        )
    p = m["patch"]
    if src_sha != p["before_sha256"]:
        raise CheckError("before_sha256 mismatch")
    if pmh.sha256_bytes(out_data) != p["after_sha256"]:
        raise CheckError("after_sha256 mismatch")
    if p["after_sha256"] != pmh.EXPECTED_AFTER_SHA256:
        raise CheckError("after_sha256 != validated compatibility bytes")
    if p.get("expected_after_sha256") != pmh.EXPECTED_AFTER_SHA256:
        raise CheckError("expected_after_sha256 does not match registry")
    if p.get("kind") != "ownership-cleanup-compat":
        raise CheckError(f"patch kind {p.get('kind')!r} invalid")
    regions = p.get("regions")
    if not isinstance(regions, list) or len(regions) != len(pmh.REGIONS):
        raise CheckError("region records missing or wrong count")
    if p.get("region_count") != len(pmh.REGIONS):
        raise CheckError("region_count does not match registry")
    for rec, r in zip(regions, pmh.REGIONS):
        if rec.get("region") != r.name:
            raise CheckError(f"region order/name {rec.get('region')!r} != {r.name!r}")
        if rec.get("old_bytes") != r.old.decode("utf-8") or \
           rec.get("new_bytes") != r.new.decode("utf-8"):
            raise CheckError(f"region {r.name} byte strings do not match registry")
        if rec.get("byte_offset") != src_data.find(r.old):
            raise CheckError(f"region {r.name} byte_offset does not match source")
        if rec.get("old_len") != len(r.old) or rec.get("new_len") != len(r.new):
            raise CheckError(f"region {r.name} lengths do not match registry")
    if m["output"]["status"] not in ("written", "exists-identical"):
        raise CheckError(f"output status {m['output']['status']!r} invalid")
    verify_patch(src_data, spec, out_data)
    return m


def _expect_fail(desc: str, fn) -> str:
    try:
        fn()
    except (pmh.PatchError, CheckError) as e:
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

    spec = pmh.SOURCES["mad-ical_std-c"]

    # --- positive: real verified source -> byte-identical to validated
    # compatibility copy --------------------------------------------
    try:
        src = pmh.resolve_path(spec, source_paths.get(spec.name), roots)
        out = evidence / "ical_std_compat.c"
        mpath = evidence / "ical_std_compat.manifest.json"
        pmh.preflight_distinct(source=src, output=out, manifest=mpath)
        data = pmh.load_and_verify(src, spec)
        patched, edits = pmh.patch_bytes(data, spec)
        status = pmh.write_artifact(out, patched)
        man = pmh.make_manifest(
            spec, src, out, edits,
            pmh.sha256_bytes(data), pmh.sha256_bytes(patched), status)
        pmh.write_artifact(
            mpath, (json.dumps(man, indent=2) + "\n").encode())
        check_manifest(mpath)
        if pmh.sha256_bytes(out.read_bytes()) == pmh.EXPECTED_AFTER_SHA256:
            record(f"  PASS  patch -> byte-identical to validated compat copy ({status})")
        else:
            record("  FAIL  patch output != validated compat bytes")
    except (pmh.PatchError, CheckError) as e:
        record(f"  FAIL  positive patch: {e}")
        src = None

    with tempfile.TemporaryDirectory(dir=evidence) as td:
        td = Path(td)
        if src is not None:
            good = src.read_bytes()

            corrupt = td / "corrupt.c"
            bad = bytearray(good)
            bad[10] ^= 0x01
            corrupt.write_bytes(bytes(bad))
            record(_expect_fail(
                "corrupt source (hash drift)",
                lambda: pmh.load_and_verify(corrupt, spec)))

            # Crafted specs bind sha to the synthetic file so the
            # region-count branches (not the hash branch) are exercised.
            dup = td / "dup.c"
            dup.write_bytes(good + b"\n" + pmh.REGIONS[2].old + b"\n")
            dup_spec = dataclasses.replace(
                spec, sha256=pmh.sha256_bytes(dup.read_bytes()))
            record(_expect_fail(
                "duplicate destroy region",
                lambda: pmh.load_and_verify(dup, dup_spec)))

            missing = td / "missing.c"
            missing.write_bytes(good.replace(pmh.REGIONS[0].old,
                                             pmh.REGIONS[0].new, 1))
            missing_spec = dataclasses.replace(
                spec, sha256=pmh.sha256_bytes(missing.read_bytes()))
            record(_expect_fail(
                "missing declare-handle region",
                lambda: pmh.load_and_verify(missing, missing_spec)))

            # Numeric-body mutation must trip the result-drift guard:
            # patch the real source then tamper a numeric constant.
            patched_ok, _ = pmh.patch_bytes(good, spec)
            tampered = bytearray(patched_ok)
            pos = patched_ok.find(b"255.0")
            tampered[pos + 2] = ord("4")  # 255.0 -> 254.0
            record(_expect_fail(
                "changed numeric body (/255.0 -> /254.0)",
                lambda: verify_patch(good, spec, bytes(tampered))))

            # verify_patch must catch any non-splice byte change too.
            tampered2 = bytearray(patched_ok)
            tampered2[0] ^= 0x01
            record(_expect_fail(
                "tampered output outside regions (checker)",
                lambda: verify_patch(good, spec, bytes(tampered2))))

            existing = td / "exists.c"
            existing.write_bytes(b"different content entirely")
            record(_expect_fail(
                "existing different output",
                lambda: pmh.write_artifact(existing, b"proposed content")))

            same = td / "same.c"
            same.write_bytes(patched_ok)
            try:
                st = pmh.write_artifact(same, patched_ok)
                record(f"  PASS  existing identical output -> {st}")
            except pmh.PatchError as e:
                record(f"  FAIL  existing identical output: {e}")

            # --- path aliasing -------------------------------------
            outp = td / "out.c"
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
                    desc, lambda kw=kwargs: pmh.preflight_distinct(**kw)))
                if src.read_bytes() != before:
                    record(f"  FAIL  {desc}: source mutated despite rejection")

            # --- overwrite refusal ----------------------------------
            mp = td / "existing.manifest.json"
            mp.write_bytes(b'{"forged": true}')
            record(_expect_fail(
                "existing different manifest preserved",
                lambda: pmh.write_artifact(mp, b'{"new": 1}')))
            if mp.read_bytes() != b'{"forged": true}':
                record("  FAIL  existing manifest content was overwritten")

            # --- forged manifest / co-rewritten source --------------
            real_m = json.loads(
                (evidence / "ical_std_compat.manifest.json").read_text())

            tsrc = td / "tampered-src.c"
            tdata = bytearray(good)
            tdata[200] ^= 0x01
            tsrc.write_bytes(bytes(tdata))
            forged = _forge(
                real_m,
                **{"source.path": str(tsrc),
                   "patch.before_sha256": pmh.sha256_bytes(bytes(tdata))})
            fm = td / "forged-hash.manifest.json"
            fm.write_text(json.dumps(forged))
            record(_expect_fail(
                "forged before_sha256 + tampered source",
                lambda: check_manifest(fm)))

            for desc, key, val in (
                ("forged role canonical", "source.role", "canonical"),
                ("forged schema", "schema", "other/9"),
                ("forged origin", "source.origin", "invented origin"),
                ("forged source_url", "source.source_url", "https://example.invalid/x"),
                ("forged commit", "source.commit", "deadbeef"),
                ("forged failing_mex_sha256", "source.failing_mex_sha256", "0" * 64),
                ("forged expected_after", "patch.expected_after_sha256", "0" * 64),
                ("forged output status", "output.status", "bogus"),
                ("forged region_count", "patch.region_count", 4),
                ("forged patch kind", "patch.kind", "canonical-fix"),
            ):
                fm2 = td / f"forged-{len(results)}.manifest.json"
                fm2.write_text(json.dumps(_forge(real_m, **{key: val})))
                record(_expect_fail(desc, lambda f=fm2: check_manifest(f)))
        else:
            record("  FAIL  negatives skipped: source unresolved")

    n_fail = sum(1 for r in results if r.startswith("  FAIL"))
    record(f"selfcheck: {len(results) - n_fail}/{len(results)} pass, {n_fail} fail")
    pmh.write_artifact(log, ("\n".join(results) + "\n").encode())
    return 1 if n_fail else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("check", help="verify one produced copy via its manifest")
    cp.add_argument("--manifest", required=True)
    sp = sub.add_parser("selfcheck", help="positive patch + negative controls")
    sp.add_argument("--evidence", required=True, help="NEW output dir for scratch copies + log")
    sp.add_argument("--source-root", action="append",
                    help="root dir to resolve sources under (repeatable)")
    sp.add_argument("--source-path", action="append",
                    help="explicit <name>=<path> (repeatable)")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        try:
            m = check_manifest(Path(args.manifest).expanduser())
            print(f"OK {m['source']['name']} -> {m['output']['path']}")
            return 0
        except CheckError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
    try:
        spaths = pmh._parse_source_paths(args.source_path or [])
    except pmh.PatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return selfcheck(Path(args.evidence).expanduser(), spaths,
                     args.source_root or [])


if __name__ == "__main__":
    raise SystemExit(main())
