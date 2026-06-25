#!/usr/bin/env python3
"""Guard against fleet-tooling sprawl — ONE correct way per concern.

The fleet accreted 6 launchers, 11 onstart scripts, 4 bash chunk-workers and 6
monitoring scripts for what should be a handful of canonical entry points. This
guard makes that impossible to repeat: it scans `scripts/` for fleet tooling
(launch_*, onstart_*, *_chunk_worker, fleet*, *_watch, *cost_watch, teardown_fleet)
and FAILS if any such script is neither `canonical` nor `deprecated` in
`fleet-tools.json`.

So a new monitoring/launch/onstart script can't land — you add a SUBCOMMAND to
`scripts/jobsys/fleet` (the one tool) instead. The `deprecated` forks are
grandfathered (warn, not fail) until Phase E deletes them; `--strict` fails on
them too (the post-Phase-E gate).

Exit 0 if no errors, 1 otherwise. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fleet-tools.json"
SCRIPTS = ROOT / "scripts"

# A file is "fleet tooling" (and thus must be in the manifest) if its basename matches.
TOOL_RE = re.compile(
    r"^(launch_.*\.sh|onstart_.*\.sh|.*_chunk_worker\.sh|fleet|fleet_.*|.*_watch\.sh|.*cost_watch.*\.sh|teardown_fleet\.sh)$"
)
SKIP_DIRS = {".git", ".jj", "__pycache__", ".venv", "venv", "node_modules"}


def load_manifest():
    if not MANIFEST.exists():
        sys.exit(f"FATAL: {MANIFEST} not found — it is the fleet-tooling source of truth.")
    d = json.loads(MANIFEST.read_text())
    canonical = set()
    for cat, lst in d.get("canonical", {}).items():
        if cat.startswith("_"):
            continue
        canonical.update(lst)
    deprecated = {k: v for k, v in d.get("deprecated", {}).items() if not k.startswith("_")}
    return canonical, deprecated


def find_tool_scripts():
    out = []
    for p in SCRIPTS.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and TOOL_RE.match(p.name):
            out.append(p.relative_to(ROOT).as_posix())
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="also fail on grandfathered (deprecated) forks")
    ap.add_argument("--list", action="store_true", help="print canonical + deprecated and exit")
    args = ap.parse_args()

    canonical, deprecated = load_manifest()

    if args.list:
        print("canonical fleet tooling (the one true way):")
        for c in sorted(canonical):
            print(f"  {c}")
        print("\ndeprecated forks (Phase E will delete; use the canonical instead):")
        for k in sorted(deprecated):
            print(f"  {k}  ->  {deprecated[k]}")
        return 0

    errors, warnings, ok = [], [], 0
    found = find_tool_scripts()
    for rel in found:
        if rel in canonical:
            ok += 1
        elif rel in deprecated:
            msg = f"{rel}  (use {deprecated[rel]})"
            (errors if args.strict else warnings).append(f"[deprecated fork] {msg}")
        else:
            errors.append(
                f"[NEW fleet script — sprawl] {rel} is not in fleet-tools.json. "
                f"Add a subcommand to scripts/jobsys/fleet instead of a new script "
                f"(or, if truly a new canonical artifact, add it to fleet-tools.json in this change)."
            )

    # Canonical entries that no longer exist (stale manifest).
    for c in sorted(canonical):
        if not (ROOT / c).exists():
            errors.append(f"[canonical missing] {c} is in fleet-tools.json but does not exist.")

    print(f"fleet-tools guard — {len(found)} fleet scripts scanned   "
          f"canonical={ok}  warnings={len(warnings)}  errors={len(errors)}")
    if warnings:
        print("\nWARNINGS (grandfathered forks — Phase E deletes them):")
        for w in warnings:
            print(f"  WARN  {w}")
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ERR   {e}")
        print("\nFAIL: fleet tooling outside the canonical set. There is ONE way per concern;")
        print("      add a subcommand to `fleet`, don't fork a new script.")
        return 1
    print("\nPASS: every fleet script is canonical (or a grandfathered fork).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
