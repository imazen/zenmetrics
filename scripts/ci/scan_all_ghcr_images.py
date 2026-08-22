#!/usr/bin/env python3
"""Scan EVERY published ghcr.io/imazen container image for leaked credentials.

These images are PUBLIC (world-pullable), so any baked credential is a public leak.
This enumerates every container package under the imazen org, lists its tagged
versions, DEDUPS by image digest (many tags -> one digest; scan each digest once),
and runs scripts/ci/scan_image_secrets.sh against one tag per digest.

Companion to audit_ghcr_org.py (which guards package *names*). This guards package
*contents*. Never pushes, retags, or deletes anything — read-only on the registry.

Selection (env SCAN_SCOPE):
  recent   (default) — newest N tags per package + any tag named latest/kadis/exec*/base*
  all                — every tagged digest under every package (slow; weekly sweep)

Env:
  ORG=imazen  SCAN_SCOPE=recent|all  RECENT_N=12  ONLY_PACKAGES=a,b  SCAN_NO_GREP=0
Exit: 0 all clean · 1 leak found · 2 could not enumerate/scan (broken, not "clean").
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ORG = os.environ.get("ORG", "imazen")
SCOPE = os.environ.get("SCAN_SCOPE", "recent")
RECENT_N = int(os.environ.get("RECENT_N", "12"))
ONLY = {p for p in os.environ.get("ONLY_PACKAGES", "").split(",") if p}
HERE = Path(__file__).resolve().parent
SCANNER = HERE / "scan_image_secrets.sh"
ALWAYS_TAGS = ("latest", "kadis", "exec", "exec-gpu", "persistent",
               "base-x86", "base-x86-cuda", "base-arm")


def gh_json(path: str):
    out = subprocess.run(
        ["gh", "api", path, "--paginate"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    # --paginate concatenates JSON arrays as separate documents; join them.
    docs, dec = [], json.JSONDecoder()
    i, s = 0, out.strip()
    while i < len(s):
        obj, j = dec.raw_decode(s, i)
        docs.append(obj)
        i = j
        while i < len(s) and s[i] in " \t\r\n":
            i += 1
    merged = []
    for d in docs:
        merged.extend(d if isinstance(d, list) else [d])
    return merged


def list_packages() -> list[str]:
    pkgs = [p["name"] for p in gh_json(f"/orgs/{ORG}/packages?package_type=container&per_page=100")]
    return sorted(p for p in pkgs if not ONLY or p in ONLY)


def versions(pkg: str) -> list[dict]:
    return gh_json(f"/orgs/{ORG}/packages/container/{pkg}/versions?per_page=100")


def pick_tags(pkg: str, vers: list[dict]) -> list[tuple[str, str]]:
    """Return [(tag, digest)] to scan: one tag per unique digest, honoring scope."""
    chosen: dict[str, str] = {}      # digest -> tag
    # newest first (the API returns newest-first by created_at)
    for idx, v in enumerate(vers):
        digest = v.get("name", "")   # the sha256 digest
        tags = (v.get("metadata", {}).get("container", {}) or {}).get("tags", []) or []
        if not tags:
            continue                 # untagged digest (by-digest push / buildcache) — skip
        # skip buildcache pseudo-tags (not runnable images)
        tags = [t for t in tags if not t.startswith("buildcache")]
        if not tags:
            continue
        keep = (SCOPE == "all"
                or idx < RECENT_N
                or any(t in ALWAYS_TAGS for t in tags))
        if keep and digest not in chosen:
            chosen[digest] = sorted(tags, key=len)[0]   # shortest tag is the friendly one
    return [(t, d) for d, t in chosen.items()]


def main() -> int:
    if shutil.which("gh") is None:
        print("FATAL: gh not on PATH", file=sys.stderr); return 2
    if not SCANNER.exists():
        print(f"FATAL: scanner missing: {SCANNER}", file=sys.stderr); return 2
    try:
        pkgs = list_packages()
    except subprocess.CalledProcessError as e:
        print(f"FATAL: cannot list {ORG} packages (need read:packages + org access):\n{e.stderr}",
              file=sys.stderr); return 2

    refs: list[str] = []
    print(f"== ghcr.io/{ORG}: {len(pkgs)} container packages (scope={SCOPE}) ==")
    for pkg in pkgs:
        try:
            vs = versions(pkg)
        except subprocess.CalledProcessError as e:
            print(f"  {pkg}: WARN cannot list versions: {e.stderr.strip()}", file=sys.stderr)
            continue
        sel = pick_tags(pkg, vs)
        print(f"  {pkg}: {len(vs)} versions -> {len(sel)} unique-digest tags to scan")
        for tag, _digest in sel:
            refs.append(f"ghcr.io/{ORG}/{pkg}:{tag}")

    if not refs:
        print("FATAL: nothing to scan (enumeration empty)", file=sys.stderr); return 2

    print(f"\n== scanning {len(refs)} unique images ==", flush=True)
    rc = subprocess.run(["bash", str(SCANNER), *refs]).returncode
    print(f"\n== scan_all done: scanner rc={rc} over {len(refs)} images ==")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
