#!/usr/bin/env python3
"""Select the smallest matched-quality payload under each time budget.

Consumes analyze.py's matched.tsv from ONE hardware/timing cohort. Does not
interpolate presets or extrapolate quality, and preserves quantizer brackets.
The input's time estimates are not runtime deadlines or confidence bounds.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def select(rows, budgets):
    groups = defaultdict(list)
    identity = (
        "source", "width", "height", "backend", "depth", "chroma", "threads",
        "tune", "scm", "sb128", "target_ssim2",
    )
    provenance = (
        "svt_reference", "zen_intra_edge_filter", "binary_sha256", "source_sha256", "timing_scope",
    )
    for row in rows:
        if int(row["rounds"]) < 3:
            raise ValueError("time-budget comparisons require three completed rounds")
        groups[tuple(row[k] for k in identity) + tuple(row.get(k, "") for k in provenance)].append(row)
    for _, choices in sorted(groups.items()):
        for budget in budgets:
            eligible = [r for r in choices if float(r["estimated_ms"]) <= budget]
            if not eligible:
                continue
            best = min(eligible, key=lambda r: (float(r["estimated_bytes"]), float(r["estimated_ms"])))
            yield dict(best, budget_ms=budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched")
    parser.add_argument("out")
    parser.add_argument("--cohort", required=True, help="machine/run identity; never pool different CPUs")
    parser.add_argument("--budgets", default="5,10,20,30,50,100,200,500,1000,2000,5000")
    args = parser.parse_args()
    budgets = [float(b) for b in args.budgets.split(",")]
    if not budgets or any(b <= 0 for b in budgets):
        parser.error("budgets must be positive milliseconds")
    with Path(args.matched).open() as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    selected = [dict(r, timing_cohort=args.cohort) for r in select(rows, budgets)]
    if not selected:
        raise SystemExit("no bracketed quality point fits the requested budgets")
    with Path(args.out).open("w") as target:
        writer = csv.DictWriter(target, fieldnames=list(selected[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print(f"{len(selected)} budget-constrained points; cohort={args.cohort}")


if __name__ == "__main__":
    main()
