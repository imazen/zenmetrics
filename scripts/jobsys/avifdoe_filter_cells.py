#!/usr/bin/env python3
"""Select a stratum subset of the canonical builder's emitted declare items.

The builder mints the cells; this only SELECTS, so every declared CellId is
bit-identical to the one the full plan would mint and the block joins the
existing identity space (and its budget-size twin) for free.

A requested stratum that does not resolve is a HARD ERROR, never a silent
shrink: picking a stratum the plan stopped spelling is exactly how a block
quietly measures 11 strata instead of 27 (plan doc section 15.2's
--max-deviations trap, in a different disguise).
"""
import json
import sys

src, allowf, dst = sys.argv[1], sys.argv[2], sys.argv[3]
want = {l.strip() for l in open(allowf) if l.strip() and not l.startswith("#")}
seen, kept, total = set(), [], 0
for line in open(src):
    total += 1
    rec = json.loads(line)
    cell = json.loads(rec["knob_tuple_json"])["cell"]
    seen.add(cell)
    if cell in want:
        kept.append(line)
missing = sorted(want - seen)
if missing:
    sys.exit(
        "FATAL: %d requested strata do not exist in this plan: %s\n"
        "(a stratum the plan no longer spells would silently shrink the block "
        "-- fix the allowlist or the plan, do not proceed)"
        % (len(missing), ", ".join(missing))
    )
open(dst, "w").writelines(kept)
print(
    "    strata requested: %d   all present   cells kept: %d of %d"
    % (len(want), len(kept), total)
)
