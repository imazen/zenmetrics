#!/usr/bin/env python3
# jxlav1_cells.py — emit the encode-declare cells JSONL for the imazen-26-JXL
# zenav1-svt / zenav1-aom config-trial sweep (2026-11-05 pareto-refresh run).
#
#   usage: jxlav1_cells.py <subset.tsv> <out_cells.jsonl>
#                          [--speeds 4,6,8] [--backend svt-rs|aom-rs]
#                          [--exclude-min-dim LO:HI]
#
# Same cell shape as avifsvt_cells.py (codec zenavif, knob_tuple_json
# {"backend":B,"speed":S}); sources are .jxl instead of .png — decode.rs sniffs
# JXL magic, so the executor's native decoder handles them. source_sha comes
# straight out of subset.tsv (manifest-verified at selection time).
#
# Grid (uniform, additive-later):
#   - q: 1 + step 5 across 5..70 + step 2 across 72..100 = 30 points (the avif944
#     grid, dense at BOTH ends per the sweep discipline).
#   - speeds: zenavif speed 1..=10 -> SVT preset 0..=13 linear; default {4,6,8}.
#   - --exclude-min-dim 720:1080 skips sources whose min(w,h) sits in the
#     zenav1-aom divergence band (imazen/zenav1-aom#14); names land in
#     <out>.excluded.txt for later additive re-declare.
import csv, json, sys

def q_grid():
    qs = [1] + list(range(5, 71, 5)) + list(range(72, 101, 2))
    assert len(qs) == 30, len(qs)
    return qs

def main():
    subset_tsv, out_path = sys.argv[1], sys.argv[2]
    speeds = [int(x) for x in _opt("--speeds", "4,6,8").split(",")]
    backend = _opt("--backend", "svt-rs")
    if backend not in ("svt-rs", "aom-rs"):
        sys.exit("--backend must be svt-rs or aom-rs")
    band = None
    if "--exclude-min-dim" in sys.argv:
        lo, hi = _opt("--exclude-min-dim").split(":")
        band = (int(lo), int(hi))
    rows = list(csv.DictReader(open(subset_tsv), delimiter="\t"))
    names, band_excluded = [], []
    for r in rows:
        w, h = int(r["width"]), int(r["height"])
        if band is not None and band[0] <= min(w, h) < band[1]:
            band_excluded.append(r["name"])
        else:
            names.append((r["name"], r["sha256"]))
    if not names:
        sys.exit("no eligible sources in %s" % subset_tsv)
    qs = q_grid()
    cells = 0
    with open(out_path, "w") as out:
        for n, sha in names:
            for sp in speeds:
                knobs = json.dumps({"backend": backend, "speed": sp}, separators=(",", ":"))
                for q in qs:
                    out.write(json.dumps({
                        "image_path": n, "codec": "zenavif", "q": q,
                        "knob_tuple_json": knobs, "source_sha": sha,
                    }, separators=(",", ":")) + "\n")
                    cells += 1
    expect = len(names) * len(speeds) * len(qs)
    assert cells == expect, (cells, expect)
    if band_excluded:
        with open(out_path + ".excluded.txt", "w") as f:
            f.write("".join(n + "\n" for n in band_excluded))
    print("wrote %d cells (%d sources x %d speeds x %d q; backend %s; %d excluded in min-dim band %s) -> %s"
          % (cells, len(names), len(speeds), len(qs), backend, len(band_excluded), band, out_path))

def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

if __name__ == "__main__":
    main()
