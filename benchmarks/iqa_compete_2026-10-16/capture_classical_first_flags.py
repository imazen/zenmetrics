#!/usr/bin/env python3
"""Capture completed classical cold-call flags without qualifying regressions."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics


PHASE = "scorer_construct_first_eval"


def capture(roots, regressions):
    completed, flags = [], []
    recorded = regressions.read_text() if regressions.exists() else ""
    for root in roots:
        for path in sorted(root.rglob("summary.json")):
            # A controller may still be writing its create-once summary.
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(row, list):
                aggregate_plan = path.parent / "plan.json"
                if not aggregate_plan.is_file():
                    raise ValueError("aggregate summary has no sibling plan: " + str(path))
                declared = json.loads(aggregate_plan.read_text())
                identities = [c["id"] for c in declared["cases"]]
                acquired = [r.get("case") for r in row if isinstance(r, dict)]
                if (declared.get("phase") != PHASE or len(acquired) != len(row)
                        or len(identities) != len(set(identities))
                        or len(acquired) != len(set(acquired))
                        or set(acquired) != set(identities)):
                    raise ValueError("aggregate summary differs from declared cases: " + str(path))
                for record in row:
                    retained = path.parent / record["case"] / "summary.json"
                    if (record.get("phase") != PHASE
                            or json.loads(retained.read_text()) != record):
                        raise ValueError("aggregate differs from retained case summary: " + str(path))
                # Count and flag the separately retained case files exactly once.
                continue
            if not isinstance(row, dict):
                raise ValueError("unexpected summary type: " + str(path))
            if row.get("phase") != PHASE:
                raise ValueError("unexpected summary phase: " + str(path))
            plan_path = path.parent.parent / "plan.json"
            plan = json.loads(plan_path.read_text())
            case = next(c for c in plan["cases"] if c["id"] == row["case"])
            completed.append(str(path))
            ratio = row.get("ratio")
            if ratio is None:
                continue
            if (not math.isfinite(ratio) or ratio <= 0
                    or row.get("scope_reasons")):
                raise ValueError("invalid matched ratio: " + str(path))
            if ratio <= 1.08:
                continue
            blocks = row["blocks"]
            if any(b.get("errors") or b.get("passed") is False for b in blocks):
                raise ValueError("ratio contains failed block: " + str(path))
            native = statistics.median(b["arms"]["native"]["ns"] for b in blocks) / 1e6
            canonical = statistics.median(b["arms"]["canonical"]["ns"] for b in blocks) / 1e6
            flag = dict(case=row["case"], metric=case["metric"],
                        width=case["width"], height=case["height"],
                        threads=case["threads"], blocks=len(blocks),
                        native_ms=native, canonical_ms=canonical, ratio=ratio,
                        summary=str(path), absolute_bound=row.get("absolute_bound"),
                        relative_bound=row.get("relative_bound"),
                        regression_qualified=False)
            flags.append(flag)
            if str(path) not in recorded:
                stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                line = (f"{stamp} | {case['metric']} first-call | "
                        f"{case['width']}x{case['height']} | {case['threads']} | "
                        f"ours {native} ms | canonical {canonical} ms | {ratio} | "
                        f"measured-here {path}; descriptive matched constructor/first-evaluation "
                        "scope; runtime cleanliness unqualified; fixed-stage confirmation and profile required\n")
                with regressions.open("a") as stream:
                    stream.write(line)
                recorded += line
    return dict(phase=PHASE, completed_summaries=completed, flags=flags,
                qualification="Initial acquired comparisons; no confidence interval or runtime-cleanliness claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--regressions", type=Path, required=True)
    args = parser.parse_args()
    if any(not root.is_dir() for root in args.root):
        parser.error("every explicitly selected results root must exist")
    if args.output.exists():
        parser.error("new snapshot output required")
    result = capture(args.root, args.regressions)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(completed=len(result["completed_summaries"]), flags=len(result["flags"]))))


if __name__ == "__main__":
    main()
