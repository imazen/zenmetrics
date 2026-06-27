#!/usr/bin/env python3
"""Mandatory-sweep-axis coverage gate for picker training data.

The "never silently ship a crippled picker" guardrail (docs/MANDATORY_SWEEP_AXES.md).
Before a picker trains, assert the swept pareto actually COVERS every first-class
mode the codec must be able to choose — color mode, subsampling, and sub-30s
effort tiers. A budget tail-shed, an rd_core-instead-of-modes_full sweep, or a
plan omission all manifest the same way here: a mandatory token that matches ZERO
config_names. We FAIL LOUD with a re-sweep instruction rather than train a picker
that can never select that mode (the zenjpeg-XYB disaster, 2026-06-27).

Usage:
    check_mandatory_coverage.py --pareto <parquet> --codec zenjpeg
    check_mandatory_coverage.py --pareto <parquet> --codec zenjxl --picker modular

Exit 0 = all mandatory modes present. Exit 1 = a mandatory mode is ABSENT.
"""
import argparse
import re
import sys

import pyarrow.parquet as pq

# Each check: (human label, predicate(names:set[str]) -> bool). The predicate is
# True when the mode is COVERED. Most are "some config_name matches regex R"; a
# few encode a default mode as "some config_name LACKS token(s)".
def has(rx):
    p = re.compile(rx)
    return lambda names: any(p.search(n) for n in names)

def lacks_all(*toks):  # a default mode present = some name carries none of these
    return lambda names: any(all(t not in n for t in toks) for n in names)

MANDATORY = {
    # zenjpeg grammar: {fam}_{trellis}_{scan}_{color}[-flags]; color is a token.
    "zenjpeg": [
        ("subsampling 4:2:0", has(r"_420(?:[-_]|$)")),
        ("subsampling 4:4:4", has(r"_444(?:[-_]|$)")),
        ("subsampling 4:2:2", has(r"_422(?:[-_]|$)")),
        ("XYB B-subsampled (xybBq) — REQUIRED", has(r"xybBq")),
    ],
    # zenavif grammar: sN[-noqm][-420][-bd*][-rgb][-vaq*][-trel][-probe].
    # 4:4:4 is the DEFAULT (no -420 token); 4:2:0 is the explicit -420 token.
    "zenavif": [
        ("subsampling 4:2:0", has(r"-420(?:[-_]|$)")),
        ("subsampling 4:4:4 (default)", lacks_all("-420")),
        ("RGB color model", has(r"-rgb(?:[-_]|$)")),
    ],
    # zenwebp grammar: vp8-mN_def (lossy) / vp8l-mN[-qlN] (lossless); -syuv flag.
    "zenwebp": [
        ("lossy mode (vp8)", has(r"vp8-")),
        ("lossless mode (vp8l)", has(r"vp8l")),
        ("sharp_yuv", has(r"-syuv(?:[-_]|$)")),
    ],
    # zenpng grammar: png-<preset>[-iq<N>|-zq<N>]. The mandatory quantize
    # axis must sweep BOTH palette backends: imagequant (-iq<N>) AND
    # zenquant (-zq<N>), across the color ladder {256,128,64,32}.
    "zenpng": [
        ("palette imagequant", has(r"-iq\d")),
        ("palette zenquant", has(r"-zq\d")),
    ],
    # zenjxl lossy: full effort ladder e1..e9 (the ablation program mandates it).
    "zenjxl_lossy": [(f"effort e{n}", has(rf"(?<![0-9a-z])e{n}(?![0-9])")) for n in range(1, 10)],
    # zenjxl modular: full ladder e1..e10 + at least one explicit predictor probe.
    "zenjxl_modular": [(f"effort e{n}", has(rf"(?<![0-9a-z])e{n}(?![0-9])")) for n in range(1, 11)]
    + [("modular predictor probe", has(r"(?:wp|pred|rct|gss)"))],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", required=True, help="pareto parquet with a config_name column")
    ap.add_argument("--codec", required=True)
    ap.add_argument("--picker", default="", help="for zenjxl: lossy|modular")
    ap.add_argument("--warn-only", action="store_true", help="report but exit 0 (for diagnostics)")
    a = ap.parse_args()

    key = a.codec
    if a.codec == "zenjxl":
        picker = a.picker or ("lossy" if "lossy" in a.pareto else "modular")
        if picker not in ("lossy", "modular"):
            sys.exit("zenjxl requires --picker lossy|modular")
        key = f"zenjxl_{picker}"
    checks = MANDATORY.get(key)
    if checks is None:
        sys.exit(f"no mandatory-coverage spec for {key!r}; add one to {sys.argv[0]} (see docs/MANDATORY_SWEEP_AXES.md)")

    names = set(
        pq.read_table(a.pareto, columns=["config_name"]).column("config_name").to_pylist()
    )
    print(f"[coverage] {key}: {len(names)} distinct configs in {a.pareto}")
    missing = []
    for label, pred in checks:
        ok = pred(names)
        print(f"  [{'OK ' if ok else 'MISS'}] {label}")
        if not ok:
            missing.append(label)

    if missing:
        print(
            f"\n  ✗ MANDATORY COVERAGE FAILED for {key}: {len(missing)} first-class "
            f"mode(s) ABSENT from the training data:\n    - " + "\n    - ".join(missing),
            file=sys.stderr,
        )
        print(
            "  This picker would be CRIPPLED (cannot choose these modes). Re-sweep with the\n"
            "  mandatory axes pinned — see docs/MANDATORY_SWEEP_AXES.md. Do NOT train/ship on this data.",
            file=sys.stderr,
        )
        sys.exit(0 if a.warn_only else 1)
    print(f"  ✓ all mandatory modes present for {key}")


if __name__ == "__main__":
    main()
