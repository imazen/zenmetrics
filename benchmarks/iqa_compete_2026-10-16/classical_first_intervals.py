#!/usr/bin/env python3
"""Describe fixed classical first-call stages without accepting a regression."""
import argparse
import json
import math
from pathlib import Path

from canonical_intervals import interval
from compare_canonical import compare, sha


PHASE = "scorer_construct_first_eval"


def analyze(root, additional, case_ids=None):
    roots = [root, *additional]
    plans = [json.loads((p / "plan.json").read_text()) for p in roots]
    base = plans[0]
    if base.get("phase") != PHASE or len(base["cases"]) != len({c["id"] for c in base["cases"]}):
        raise ValueError("distinct classical first-call cases required")
    initial_cases = {c["id"]: c for c in base["cases"]}
    selected = set(initial_cases) if case_ids is None else set(case_ids)
    if not selected or not selected <= initial_cases.keys():
        raise ValueError("selected case missing from initial confirmation")
    for other in plans[1:]:
        successor_cases = {c["id"]: c for c in other["cases"]}
        if (other.get("phase") != PHASE
                or len(successor_cases) != len(other["cases"])
                or not selected <= successor_cases.keys()
                or any(initial_cases.get(key) != value for key, value in successor_cases.items())
                or other.get("preceding_plan_sha256") != sha(root / "plan.json")):
            raise ValueError("successor changes the case contracts or preceding plan")
        for path, digest in base["bindings"].items():
            if other["bindings"].get(path) != digest:
                raise ValueError("successor changes an original artifact binding")
    rows = []
    for case in base["cases"]:
        if case["id"] not in selected:
            continue
        ratios, blocks, paths, gates, summaries = [], [], [], [], []
        absolute, relative = case["absolute_bound"], case.get("relative_bound", 0)
        for bound in [absolute, relative]:
            if bound is not None and (type(bound) not in (int, float) or not math.isfinite(bound) or bound < 0):
                raise ValueError("invalid numerical bound")
        for stage, plan in zip(roots, plans):
            summary_path = stage / case["id"] / "summary.json"
            summary = json.loads(summary_path.read_text())
            if (summary.get("phase") != PHASE or summary["case"] != case["id"]
                    or summary["plan_sha256"] != sha(stage / "plan.json")
                    or summary.get("scope_reasons") or summary.get("ratio") is None
                    or summary["first_call_scope"] != case["first_call_scope"]
                    or summary["absolute_bound"] != absolute
                    or summary.get("relative_bound", 0) != relative):
                raise ValueError("unmatched or inconsistent first-call summary")
            acquired = sorted((stage / case["id"]).glob("block-*/result.json"))
            values = [json.loads(p.read_text()) for p in acquired]
            if values != summary["blocks"]:
                raise ValueError("summary does not retain every ordered block")
            expected_count = 20 if stage == root else 40
            if len(values) != expected_count:
                raise ValueError("complete fixed twenty-block initial or forty-block successor stage required")
            for number, block in enumerate(values):
                expected = ["canonical", "native"] if number % 2 == 0 else ["native", "canonical"]
                if (block.get("phase") != PHASE or block.get("errors")
                        or block.get("scope_reasons") or block["order"] != expected
                        or set(block["arms"]) != {"native", "canonical"}):
                    raise ValueError("failed, unmatched or reordered block")
                native, canonical = [block["arms"][role] for role in ["native", "canonical"]]
                if any(not math.isfinite(arm["ns"]) or arm["ns"] <= 0 for arm in [native, canonical]):
                    raise ValueError("invalid first-call duration")
                delta, passed = compare(native["score"], canonical["score"], absolute, relative)
                if passed is False or block.get("passed") is not passed or block.get("delta") != delta:
                    raise ValueError("numerical gate failed or recorded gate differs")
                ratios.append(native["ns"] / canonical["ns"])
                gates.append(passed)
            paths.extend(acquired)
            blocks.extend(values)
            summaries.append(summary_path)
        if len(paths) != len({p.resolve() for p in paths}):
            raise ValueError("duplicate acquired block")
        result = interval(ratios)
        rows.append(dict(case=case["id"], metric=case["metric"],
                         width=case["width"], height=case["height"], threads=case["threads"],
                         phase=PHASE, blocks=len(blocks), **result,
                         borderline=result["lower_95"] <= 1.08 <= result["upper_95"],
                         absolute_bound=absolute, relative_bound=relative,
                         numerical_qualified=all(g is True for g in gates),
                         max_abs_delta=max(b["delta"] for b in blocks),
                         rust_source=case["rust_source"],
                         plan_sha256=[sha(p / "plan.json") for p in roots],
                         stage_summary_sha256=[sha(p) for p in summaries],
                         block_sha256=[sha(p) for p in paths],
                         regression_qualified=False))
    return dict(method="20,000 bootstrap resamples of complete alternating-block ratios; median estimator; seed 20261020",
                qualification="Descriptive first-call variability; runtime cleanliness not independently qualified. Null numerical bounds remain unqualified.",
                analysis_source_sha256={str(Path(__file__).resolve()): sha(__file__),
                                        str(Path(__file__).with_name("canonical_intervals.py")): sha(Path(__file__).with_name("canonical_intervals.py")),
                                        str(Path(__file__).with_name("compare_canonical.py")): sha(Path(__file__).with_name("compare_canonical.py"))},
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--additional", type=Path, action="append", default=[])
    parser.add_argument("--case", action="append", help="analyze a selected original case; repeatable")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    result = analyze(args.results, args.additional, args.case)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["rows"]))


if __name__ == "__main__":
    main()
