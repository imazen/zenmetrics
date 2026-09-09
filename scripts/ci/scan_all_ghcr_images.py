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

Enumeration has two sources, in order:
  1. the GitHub packages API (`/orgs/<org>/packages`), which also finds packages
     nobody remembered to register; needs a token with `read:packages` AND org
     access, and
  2. `ghcr-packages.json` + `crane ls`, which needs neither.

Actions' default `GITHUB_TOKEN` is a REPOSITORY-scoped installation token and
cannot enumerate ORG-level packages at all -- it returns `HTTP 400 Invalid
argument`, not a 403, and no `permissions:` block changes that (`packages: read`
grants access to this repo's own packages, not org enumeration). So on a stock
Actions run source 1 always fails and source 2 is what runs. Supply a PAT with
`read:packages` as `GH_TOKEN` to get the wider enumeration.

Source 2 is not merely a degraded fallback: `ghcr-packages.json` is the ENFORCED
source of truth for package names (`just ghcr-check` fails any infra file
referencing a package not listed there), so it is the same set the naming guard
already polices, and it covers the grandfathered `deprecated` names too because
those images are still public and still pullable.

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


def manifest_packages() -> list[str]:
    """Package names from `ghcr-packages.json` — canonical plus the grandfathered
    `deprecated` splinters, which are still public and still pullable."""
    mf = HERE.parent.parent / "ghcr-packages.json"
    d = json.loads(mf.read_text())
    names = [p["name"] for p in d.get("packages", [])]
    names += [k for k in d.get("deprecated", {}) if not k.startswith("_")]
    return sorted(set(names))


def list_packages() -> tuple[list[str], str]:
    """(packages, source). Prefers the API; falls back to the manifest."""
    try:
        pkgs = [
            p["name"]
            for p in gh_json(f"/orgs/{ORG}/packages?package_type=container&per_page=100")
        ]
        source = "github-api"
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip().replace("\n", " ")[:200]
        print(
            f"NOTE: org package enumeration unavailable ({err}); "
            f"falling back to ghcr-packages.json + crane.\n"
            f"      (Actions' GITHUB_TOKEN is repo-scoped and cannot list ORG packages. "
            f"Supply a PAT with read:packages as GH_TOKEN for the wider sweep.)",
            file=sys.stderr,
        )
        pkgs, source = manifest_packages(), "ghcr-packages.json"
    return sorted(p for p in pkgs if not ONLY or p in ONLY), source


def crane_versions(pkg: str) -> list[dict]:
    """Version records shaped like the API's, built from `crane ls` + `crane digest`.

    `crane` is already installed and authenticated by the workflow, and these
    packages are public, so this needs no GitHub packages scope. Tags come back
    newest-last from the registry, so reverse to match the API's newest-first
    ordering that `pick_tags` relies on.
    """
    ref = f"ghcr.io/{ORG}/{pkg}"
    out = subprocess.run(
        ["crane", "ls", ref], capture_output=True, text=True, timeout=120, check=True
    ).stdout
    tags = [t for t in (line.strip() for line in out.splitlines()) if t]
    tags.reverse()
    # Resolve only the tags that could be selected, so a package with hundreds of
    # tags does not cost hundreds of HEAD requests.
    candidate = [
        t
        for i, t in enumerate(tags)
        if SCOPE == "all" or i < RECENT_N or t in ALWAYS_TAGS
    ]
    vers: list[dict] = []
    for tag in candidate:
        if tag.startswith("buildcache"):
            continue
        try:
            digest = subprocess.run(
                ["crane", "digest", f"{ref}:{tag}"],
                capture_output=True, text=True, timeout=120, check=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            continue  # tag vanished mid-run, or is a manifest crane won't resolve
        vers.append({"name": digest, "metadata": {"container": {"tags": [tag]}}})
    return vers


def versions(pkg: str, source: str) -> list[dict]:
    if source == "github-api":
        return gh_json(f"/orgs/{ORG}/packages/container/{pkg}/versions?per_page=100")
    return crane_versions(pkg)


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
    if shutil.which("crane") is None:
        print("FATAL: crane not on PATH", file=sys.stderr); return 2
    if not SCANNER.exists():
        print(f"FATAL: scanner missing: {SCANNER}", file=sys.stderr); return 2
    try:
        pkgs, source = list_packages()
    except Exception as e:  # manifest unreadable / malformed JSON
        print(f"FATAL: cannot enumerate {ORG} packages by any source: {e}",
              file=sys.stderr); return 2

    refs: list[str] = []
    print(f"== ghcr.io/{ORG}: {len(pkgs)} container packages "
          f"(scope={SCOPE}, enumerated via {source}) ==")
    for pkg in pkgs:
        try:
            vs = versions(pkg, source)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            # A name in the manifest's `deprecated` map may never have been
            # created, or may not be public. Either way it is out of scope for a
            # PUBLIC-image leak scan — say so quietly instead of emitting a
            # permanent CI warning that trains people to ignore warnings.
            if "DENIED" in err or "UNAUTHORIZED" in err or "NAME_UNKNOWN" in err:
                print(f"  {pkg}: absent or not public — skipped")
            else:
                print(f"  {pkg}: WARN cannot list versions: {err}", file=sys.stderr)
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
