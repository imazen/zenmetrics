#!/usr/bin/env python3
"""Gate on a cvvdp-backfill manifest's parity stats.

Consumes the `manifest.json` produced by `finalize.sh`, applies
configurable thresholds (mean, median, max abs |diff|; minimum joined
row count per source), and exits non-zero on the first failure.

Pipeline placement:

    finalize.sh   ->  manifest.json       (consolidation, ALWAYS writes)
    assert_parity ->  exit 0/1            (CI / automation gate, OPTIONAL)

`finalize.sh` is descriptive — it always produces a manifest no matter
how blown out parity is, so a human can inspect the numbers.
`assert_parity.py` is prescriptive — wrap it around finalize when an
automation step (a nightly fleet run, a release gate, a PR job) needs a
machine-readable pass/fail on those same stats.

Defaults match the smoke-tested tolerances from the n=4 sentinel +
the dual-impl-chunk acceptance line in `crates/cvvdp-gpu/docs/
CVVDP_SIDECAR_SCHEMA.md`:

    --max-mean-abs-diff    0.10   (mean of |cvvdp_imazen - cvvdp_pycvvdp_v054|)
    --max-median-abs-diff  0.10   (50th percentile)
    --max-max-abs-diff     0.50   (worst single row)
    --min-joined-per-src   1      (each source must have >=1 joinable row)

Skipped (parity == null) sources are tolerated by default — many
sources will have only one impl present during partial runs — but
`--require-parity-on-all` flips that to a hard failure.

Exit codes:
    0  all sources pass
    1  manifest parse / file-missing error
    2  one or more sources fail a threshold
    3  one or more sources missing parity when --require-parity-on-all set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path, help="Path to manifest.json from finalize.sh")
    p.add_argument("--max-mean-abs-diff", type=float, default=0.10,
                   help="Fail if any source's mean abs diff exceeds this (default 0.10)")
    p.add_argument("--max-median-abs-diff", type=float, default=0.10,
                   help="Fail if any source's median abs diff exceeds this (default 0.10)")
    p.add_argument("--max-max-abs-diff", type=float, default=0.50,
                   help="Fail if any source's max abs diff exceeds this (default 0.50)")
    p.add_argument("--min-joined-per-src", type=int, default=1,
                   help="Fail if any source has fewer joined rows than this (default 1)")
    p.add_argument("--require-parity-on-all", action="store_true",
                   help="Fail when any source has parity=null (only one impl present). "
                        "By default null parity is tolerated.")
    p.add_argument("--only-sources", default="",
                   help="Comma-separated list of source stems to gate. "
                        "Other sources are reported but don't affect exit code.")
    p.add_argument("--json-summary", type=Path, default=None,
                   help="If set, write a machine-readable pass/fail summary to this path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.manifest.is_file():
        print(f"[assert_parity] manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(args.manifest.read_text())
    except json.JSONDecodeError as exc:
        print(f"[assert_parity] manifest is not valid JSON: {exc}", file=sys.stderr)
        return 1

    sources = manifest.get("sources") or {}
    if not isinstance(sources, dict) or not sources:
        print("[assert_parity] manifest has no 'sources' entries", file=sys.stderr)
        return 1

    only = {s.strip() for s in args.only_sources.split(",") if s.strip()}

    failures: list[str] = []
    null_parity: list[str] = []
    pass_count = 0
    skip_count = 0
    summary: dict[str, dict] = {}

    for stem, info in sorted(sources.items()):
        gated = (not only) or (stem in only)
        parity = info.get("parity") if isinstance(info, dict) else None

        record = {"gated": gated, "parity": parity}
        summary[stem] = record

        if parity is None:
            if gated and args.require_parity_on_all:
                null_parity.append(stem)
                print(f"  FAIL  {stem}: parity is null (one impl missing); "
                      f"--require-parity-on-all set", file=sys.stderr)
            else:
                skip_count += 1
                print(f"  skip  {stem}: parity is null (one impl missing)")
            continue

        if not isinstance(parity, dict):
            failures.append(f"{stem}: parity is not a dict ({type(parity).__name__})")
            print(f"  FAIL  {stem}: parity is not a dict", file=sys.stderr)
            continue

        joined = parity.get("joined", 0)
        mean = parity.get("mean_abs_diff")
        median = parity.get("median_abs_diff")
        worst = parity.get("max_abs_diff")

        src_fails: list[str] = []
        if gated:
            if joined < args.min_joined_per_src:
                src_fails.append(f"joined={joined} < --min-joined-per-src={args.min_joined_per_src}")
            if mean is None or mean > args.max_mean_abs_diff:
                src_fails.append(f"mean={mean} > --max-mean-abs-diff={args.max_mean_abs_diff}")
            if median is None or median > args.max_median_abs_diff:
                src_fails.append(f"median={median} > --max-median-abs-diff={args.max_median_abs_diff}")
            if worst is None or worst > args.max_max_abs_diff:
                src_fails.append(f"max={worst} > --max-max-abs-diff={args.max_max_abs_diff}")

        if src_fails:
            for reason in src_fails:
                failures.append(f"{stem}: {reason}")
            mean_s = "?" if mean is None else f"{mean:.4f}"
            median_s = "?" if median is None else f"{median:.4f}"
            worst_s = "?" if worst is None else f"{worst:.4f}"
            print(
                f"  FAIL  {stem}: n={joined} mean={mean_s} median={median_s} "
                f"max={worst_s}  ({'; '.join(src_fails)})",
                file=sys.stderr,
            )
            record["pass"] = False
            record["reasons"] = src_fails
        else:
            pass_count += 1
            mean_s = "?" if mean is None else f"{mean:.4f}"
            median_s = "?" if median is None else f"{median:.4f}"
            worst_s = "?" if worst is None else f"{worst:.4f}"
            tag = "PASS" if gated else "info"
            print(f"  {tag}  {stem}: n={joined} mean={mean_s} median={median_s} max={worst_s}")
            record["pass"] = True if gated else None

    if args.json_summary is not None:
        args.json_summary.write_text(json.dumps({
            "run_id": manifest.get("run_id"),
            "thresholds": {
                "max_mean_abs_diff": args.max_mean_abs_diff,
                "max_median_abs_diff": args.max_median_abs_diff,
                "max_max_abs_diff": args.max_max_abs_diff,
                "min_joined_per_src": args.min_joined_per_src,
                "require_parity_on_all": args.require_parity_on_all,
            },
            "totals": {
                "pass": pass_count,
                "fail": len(failures),
                "null_parity": len(null_parity),
                "skipped_ungated": skip_count if only else 0,
            },
            "sources": summary,
        }, indent=2))

    print(
        f"[assert_parity] pass={pass_count} fail={len(failures)} "
        f"null_parity={len(null_parity)}",
        file=sys.stderr,
    )

    if failures:
        return 2
    if null_parity:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
