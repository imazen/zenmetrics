#!/usr/bin/env python3
"""Guard against ghcr.io/imazen container-package-name splintering.

THE PROBLEM this exists to prevent: every iteration the worker/sweep image got
RENAMED instead of re-tagged, accreting near-duplicate ghcr packages
(zen-metrics-sweep -> zenmetrics-sweep; zen-jobworker -> zenfleet-worker ->
zenfleet-worker-exec -> zenfleet-worker-exec-gpu -> zen-jobworker-exec — five
names for one worker). Tags are free; package names are forever and the user
has to look at all of them.

THE RULE (single source of truth = ghcr-packages.json at the repo root):
  ONE ghcr package per artifact; variants are TAGS, never new package names.

WHAT THIS SCRIPT DOES: scan the repo for `ghcr.io/imazen/<name>` references and
classify each `<name>`:
  * canonical (in packages[])                 -> OK
  * deprecated (grandfathered splinter)       -> WARN in infra (migrate to a tag),
                                                 tolerated in docs (historical record)
  * unknown (a NEW name in neither list)      -> ERROR in infra, WARN in docs

So a brand-new invented name fails CI the moment it lands in a workflow,
Dockerfile, launch script, or Rust default-image constant. The ONLY way to add
a legitimately-new package is to add it to packages[] in ghcr-packages.json in
the same reviewed change — which is exactly the human chokepoint we want.

Exit code: 0 if no errors, 1 otherwise. `--strict` promotes deprecated-in-infra
and unknown-in-docs to errors (the post-migration gate; flip it on once the
`deprecated` map in the manifest is empty). Stdlib only (Python 3.8+).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Repo root is two levels up from scripts/ci/.
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ghcr-packages.json"

# Match `ghcr.io/imazen/<name>`; <name> stops before a tag (:), digest (@), or
# any non-name character. ghcr package names are lowercase [a-z0-9._-].
REF_RE = re.compile(r"ghcr\.io/imazen/([a-z0-9][a-z0-9._-]*)", re.IGNORECASE)

# Files whose ghcr references actually drive image creation / pulls.
INFRA_SUFFIXES = {
    ".yml", ".yaml", ".sh", ".bash", ".rs", ".py", ".toml", ".json", ".env",
}
INFRA_NAMES = {"justfile", "Justfile", "Dockerfile"}
DOC_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}

# Directories never scanned (build output, vcs internals, agent scratch, venvs).
SKIP_DIRS = {
    ".git", ".jj", ".claude", "target", "node_modules", ".venv", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build",
}

# Files excluded because they LEGITIMATELY enumerate deprecated names (self-
# reference): the manifest, this guard's own directory, and the policy doc.
SKIP_FILES = {
    MANIFEST.resolve(),
    (ROOT / "docs" / "GHCR_PACKAGES.md").resolve(),
}
SKIP_DIR_PATHS = {(ROOT / "scripts" / "ci").resolve()}

MAX_BYTES = 5 * 1024 * 1024  # skip files larger than 5 MB (data, not infra)


def classify_file(path: Path) -> str | None:
    """Return 'infra', 'doc', or None (skip)."""
    name = path.name
    if name in INFRA_NAMES or name.startswith("Dockerfile") or name.endswith(".dockerfile"):
        return "infra"
    suffix = path.suffix.lower()
    if suffix in INFRA_SUFFIXES:
        return "infra"
    if suffix in DOC_SUFFIXES:
        return "doc"
    return None


def iter_files(root: Path):
    for path in root.rglob("*"):
        # Prune skip dirs cheaply by checking path parts.
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.resolve() in SKIP_FILES:
            continue
        if any(str(path.resolve()).startswith(str(d)) for d in SKIP_DIR_PATHS):
            continue
        kind = classify_file(path)
        if kind is None:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        yield path, kind


def load_manifest():
    if not MANIFEST.exists():
        sys.exit(f"FATAL: {MANIFEST} not found — the package allowlist is the source of truth.")
    data = json.loads(MANIFEST.read_text())
    canonical = {p["name"] for p in data.get("packages", [])}
    deprecated = {k: v for k, v in data.get("deprecated", {}).items() if not k.startswith("_")}
    if not canonical:
        sys.exit("FATAL: manifest lists no canonical packages.")
    return canonical, deprecated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="promote deprecated-in-infra and unknown-in-docs to errors (post-migration gate)")
    ap.add_argument("--list", action="store_true", help="print canonical + deprecated names and exit")
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root to scan (default: auto)")
    args = ap.parse_args()

    canonical, deprecated = load_manifest()

    if args.list:
        print("canonical packages (use these; variants are TAGS):")
        for n in sorted(canonical):
            print(f"  {n}")
        print("\ndeprecated splinters (migrate -> canonical[:tag], then delete):")
        for k in sorted(deprecated):
            print(f"  {k}  ->  {deprecated[k]}")
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    ok_count = 0

    for path, kind in iter_files(args.root):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "ghcr.io/imazen/" not in text:
            continue
        rel = path.relative_to(args.root)
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in REF_RE.finditer(line):
                name = m.group(1).lower().rstrip(".")
                loc = f"{rel}:{lineno}"
                if name in canonical:
                    ok_count += 1
                elif name in deprecated:
                    target = deprecated[name]
                    msg = f"{loc}  {name}  ->  use {target}"
                    if kind == "infra" and args.strict:
                        errors.append(f"[deprecated-in-infra] {msg}")
                    elif kind == "infra":
                        warnings.append(f"[grandfathered splinter] {msg}")
                    # deprecated-in-doc: tolerated (historical record), silent.
                else:  # unknown — a NEW name in neither list
                    msg = (f"{loc}  {name}  is NOT in ghcr-packages.json. "
                           f"Add it to packages[] (if a real new artifact, in THIS change) "
                           f"or use a canonical package + tag.")
                    if kind == "infra":
                        errors.append(f"[NEW/unapproved package name] {msg}")
                    elif args.strict:
                        errors.append(f"[unknown-in-doc] {msg}")
                    else:
                        warnings.append(f"[unknown name in doc] {msg}")

    print(f"ghcr package-name guard — scanned repo for ghcr.io/imazen/<name> references")
    print(f"  canonical refs OK: {ok_count}   warnings: {len(warnings)}   errors: {len(errors)}")
    if warnings:
        print("\nWARNINGS (not failing; migrate these to canonical tags — see docs/GHCR_PACKAGES.md):")
        for w in warnings:
            print(f"  WARN  {w}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ERR   {e}")
        print("\nFAIL: a ghcr.io/imazen image name is not a canonical package.")
        print("      One package per artifact; variants are TAGS. To add a genuinely new")
        print("      artifact, add its name to ghcr-packages.json in the same change.")
        return 1
    print("\nPASS: every active-infra ghcr.io/imazen reference uses a canonical package name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
