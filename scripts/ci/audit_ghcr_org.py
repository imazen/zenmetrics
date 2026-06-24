#!/usr/bin/env python3
"""Audit the LIVE ghcr.io/imazen container packages against ghcr-packages.json.

The guard (check_ghcr_packages.py) stops NEW splinters landing in the repo.
This audit reports the splinters that ALREADY EXIST on ghcr.io so they can be
cleaned up: it lists every container package under the imazen org and diffs it
against the manifest.

  * canonical present  -> ✓
  * canonical missing  -> ✗ (declared but not on ghcr — maybe never pushed)
  * orphan, known      -> ⚠ a deprecated splinter; prints the migrate+delete recipe
  * orphan, unknown    -> ‼ on ghcr but in NEITHER list — investigate before touching

It NEVER deletes anything. Deleting a ghcr package is destructive and externally
visible — the migrate/delete commands are printed COMMENTED OUT for a human to
run after re-tagging. Needs `gh` authenticated with read:packages + org access.

Exit: 0 = audited (even if orphans exist — reporting, not gating);
      2 = could not list packages (gh missing/unauth) — soft skip, not a failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ghcr-packages.json"
ORG = "imazen"


def load_manifest():
    data = json.loads(MANIFEST.read_text())
    canonical = {p["name"] for p in data.get("packages", [])}
    deprecated = {k: v for k, v in data.get("deprecated", {}).items() if not k.startswith("_")}
    return canonical, deprecated


def list_org_packages() -> list[str] | None:
    if shutil.which("gh") is None:
        print("SKIP: `gh` not found on PATH — cannot list org packages.", file=sys.stderr)
        return None
    try:
        out = subprocess.run(
            ["gh", "api", f"/orgs/{ORG}/packages?package_type=container&per_page=100",
             "--paginate", "-q", ".[].name"],
            capture_output=True, text=True, timeout=60, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"SKIP: gh api failed (need read:packages + {ORG} access?):\n{e.stderr.strip()}",
              file=sys.stderr)
        return None
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"SKIP: gh api error: {e}", file=sys.stderr)
        return None
    return sorted({n.strip() for n in out.stdout.splitlines() if n.strip()})


def main() -> int:
    canonical, deprecated = load_manifest()
    live = list_org_packages()
    if live is None:
        return 2

    live_set = set(live)
    print(f"ghcr.io/{ORG} container packages: {len(live)} live, "
          f"{len(canonical)} canonical, {len(deprecated)} deprecated\n")

    print("CANONICAL:")
    for n in sorted(canonical):
        print(f"  {'✓ present' if n in live_set else '✗ MISSING (declared, not on ghcr)'}   {n}")

    known_orphans: list[str] = []
    unknown_orphans: list[str] = []
    for n in live:
        if n in canonical:
            continue
        (known_orphans if n in deprecated else unknown_orphans).append(n)

    if known_orphans:
        print("\nKNOWN ORPHANS (deprecated splinters still on ghcr — migrate then delete):")
        for n in known_orphans:
            print(f"  ⚠ {n}   ->   {deprecated[n]}")
    if unknown_orphans:
        print("\n‼ UNKNOWN ORPHANS (on ghcr but in NEITHER canonical NOR deprecated — investigate):")
        for n in unknown_orphans:
            print(f"  ‼ {n}   (add to packages[] if real, or to deprecated[] with a target, then delete)")

    if known_orphans or unknown_orphans:
        print("\n--- cleanup recipe (REVIEW; destructive + externally visible — run by hand) ---")
        print("# 1) Re-tag the live image under its canonical name so nothing breaks first, e.g.:")
        print("#    docker buildx imagetools create -t ghcr.io/imazen/zenmetrics-sweep:salad \\")
        print("#        ghcr.io/imazen/zenmetrics-sweep-salad:v6")
        print("# 2) Flip code defaults + scripts to the canonical name:tag (see `just ghcr-check`).")
        print("# 3) Only AFTER nothing references the old name, delete the orphan package:")
        for n in known_orphans + unknown_orphans:
            print(f"#    gh api -X DELETE /orgs/{ORG}/packages/container/{n}")
    else:
        print("\nPASS: no orphan packages — every live package is canonical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
