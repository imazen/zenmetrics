#!/usr/bin/env bash
# GUARD: no launcher script may hard-code a non-canonical ghcr.io/imazen image package name.
#
# This is the MACHINE-ENFORCED version of the CLAUDE.md "ONE ghcr package per artifact — variants are
# TAGS, never new package names" rule. The rule-as-documentation failed to stop FIVE worker renames and
# FOUR sweep renames (see scripts/jobsys/fleet.env), because the rename happens at `docker build -t`,
# nowhere near the doc. So this fails CI instead: a new image package name cannot be merged silently.
#
# Allowed = the canonical four packages (any TAG on them is fine). New launchers must
# `source scripts/jobsys/fleet.env` and use $ZEN_FLEET_IMAGE / $ZEN_SWEEP_IMAGE / etc.
#
# exit 0 = clean, exit 1 = a non-canonical package name is hard-coded in scripts/*.sh, exit 2 = bad cwd.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

ANY='ghcr\.io/imazen/[a-zA-Z0-9._-]+'
# Canonical package names, plus the two legacy executor images (CPU + GPU) explicitly MIGRATING onto
# zenfleet-worker:exec / zenfleet-worker:gpu tags (then deleted). The trailing (:|"|'|space|EOL) boundary
# stops `zenfleet-worker` from matching inside `zenfleet-worker-exec`. Removing the two -exec entries below
# (after the re-tag + package delete) makes the guard STRICTLY the 4 canonical packages.
OK='ghcr\.io/imazen/(zenfleet-worker|zenmetrics-sweep|pycvvdp-scorer|zen-train|zenfleet-worker-exec-gpu|zenfleet-worker-exec)([^a-zA-Z0-9_-]|$)'

# fleet.env (the source of truth) and this guard are allowed to spell the names out.
viol=$(grep -rnE "$ANY" scripts/ --include="*.sh" 2>/dev/null \
  | grep -vE 'scripts/jobsys/(fleet\.env|check_canonical_images\.sh)' \
  | grep -vE "$OK" || true)

if [ -n "$viol" ]; then
  echo "FAIL — non-canonical ghcr.io/imazen image package names hard-coded in scripts/*.sh:"
  echo "$viol"
  echo
  echo "Variants are TAGS on the canonical packages, NEVER new package names:"
  echo "  zenfleet-worker   zenmetrics-sweep   pycvvdp-scorer   zen-train"
  echo "New launchers must:  source scripts/jobsys/fleet.env  &&  use \$ZEN_FLEET_IMAGE / \$ZEN_SWEEP_IMAGE"
  exit 1
fi
echo "OK: scripts/*.sh reference only canonical image packages (or source fleet.env)."
