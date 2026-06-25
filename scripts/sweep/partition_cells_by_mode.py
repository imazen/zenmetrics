#!/usr/bin/env python3
"""Partition a plan's emit-cells JSONL into lossy vs lossless by cell-id prefix.

`zenmetrics sweep --plan modes_full ... --dry-run --emit-cells cells.jsonl` writes one
JSON line per (image x plan cell), each carrying a `knob_tuple_json` of
`{"cell":"<stratum-id>","fp":"<hex>","plan":"<name>"}`. The stratum id encodes the
mode: jxl `vd-*` = lossy VarDCT, `mod-*` = lossless modular; webp `vp8-*` lossy /
`vp8l-*` lossless. This splits one cells JSONL into `<stem>.lossy.jsonl` and
`<stem>.lossless.jsonl` so the two halves can be declared + run as SEPARATE sweeps
(declare each with `zenfleet-ctl declare-encodes`), e.g. run the cheap lossy half
FIRST (web priority) and defer the heavy modular half.

WHY split: lossy and lossless have very different cost (measured 2026-06-25, 3.15 MP:
lossy 0.20 GB/encode, lossless modular 1.50 GB/encode), and modes_full is ~96%
modular cells. Partitioning lets the web-relevant lossy data land first + cheap, and
quarantines the heavy archival modular work. The two halves also feed different
pickers (lossy-only / lossless-only / cross). Lossless rows have no quality axis
(they ride the q=0 sentinel) — that's expected.

Usage:
  partition_cells_by_mode.py cells.jsonl [--out-stem PATH]
                             [--lossy-prefix vd,vp8] [--lossless-prefix mod,vp8l]
Default prefixes cover jxl (vd/mod) + webp (vp8/vp8l). An "other" bucket
(unmatched ids) is written + warned only if non-empty — never silently dropped.
"""
import argparse
import json
import sys


def cell_id(item):
    kt = item.get("knob_tuple_json")
    if isinstance(kt, str):
        kt = json.loads(kt)
    return (kt or {}).get("cell", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", help="emit-cells JSONL from `zenmetrics sweep --emit-cells`")
    ap.add_argument("--out-stem", help="output stem (default: input path without .jsonl)")
    ap.add_argument("--lossy-prefix", default="vd,vp8",
                    help="comma-sep cell-id prefixes that are LOSSY (default jxl vd, webp vp8)")
    ap.add_argument("--lossless-prefix", default="mod,vp8l",
                    help="comma-sep cell-id prefixes that are LOSSLESS (default jxl mod, webp vp8l)")
    a = ap.parse_args()

    # Longest-prefix-first so vp8l (lossless) is tested before vp8 (lossy).
    lossless = sorted((p for p in a.lossless_prefix.split(",") if p), key=len, reverse=True)
    lossy = sorted((p for p in a.lossy_prefix.split(",") if p), key=len, reverse=True)
    stem = a.out_stem or (a.cells[:-6] if a.cells.endswith(".jsonl") else a.cells)

    buckets = {"lossy": [], "lossless": [], "other": []}
    for line in open(a.cells):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        cid = cell_id(item)
        if any(cid.startswith(p) for p in lossless):
            buckets["lossless"].append(line)
        elif any(cid.startswith(p) for p in lossy):
            buckets["lossy"].append(line)
        else:
            buckets["other"].append(line)

    for name in ("lossy", "lossless"):
        path = f"{stem}.{name}.jsonl"
        with open(path, "w") as f:
            f.write("\n".join(buckets[name]) + ("\n" if buckets[name] else ""))
        print(f"  {name:9s}: {len(buckets[name]):>7d} cells -> {path}")
    if buckets["other"]:
        path = f"{stem}.other.jsonl"
        with open(path, "w") as f:
            f.write("\n".join(buckets["other"]) + "\n")
        print(f"  WARNING: {len(buckets['other'])} cells matched neither prefix set "
              f"(see {path}); add their prefix to --lossy-prefix/--lossless-prefix",
              file=sys.stderr)


if __name__ == "__main__":
    main()
