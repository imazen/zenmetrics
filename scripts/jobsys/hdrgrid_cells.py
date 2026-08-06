#!/usr/bin/env python3
# HDR phase-2 corpus (zensim campaign appendix S): emit the encode-declare cells
# JSONL for the registered grid — 1,140 PQ-PNG sources (76 origins x 15 scales)
# x 3 codec arms x 30 q points, all cells hdr:true.
#
#   usage: hdrgrid_cells.py <source_dir> <out_cells.jsonl> [--sha-cache <tsv>]
#
# Grid, verbatim from appendix S (S.4):
#   - q: step 5 across 0..70 (0,5,...,70) + step 2 across 70..100 (72,...,100)
#     = 30 points, dense at BOTH ends per the sweep discipline.
#   - sizes: ALL 15 ladder steps (every file in the source dir).
#   - preset: the QUALITY tier — svt preset 6 / jxl effort 7 / uhdr crate
#     defaults (gainmap q85, scale 4; recorded as defaults, not knobs).
#   - arms: zenjxl / zenav1-svt / jpeg-gainmap. The AVIF arm is ABSENT per the
#     standing B5 record (additive-later; rows are independent so adding it
#     later touches nothing).
#
# `image_path` is the source BASENAME (workers resolve it against the run's
# corpus prefix); `source_sha` is the sha256 of the source bytes (the encode
# job's content-addressed input). The sha pass reads ~7.8 GB once; --sha-cache
# persists it so re-declares are instant.
import hashlib
import json
import os
import sys

def q_grid():
    qs = list(range(0, 71, 5)) + list(range(72, 101, 2))
    assert len(qs) == 30, len(qs)
    return qs

ARMS = [
    ("zenjxl", json.dumps({"effort": 7}, separators=(",", ":"))),
    ("zenav1-svt", json.dumps({"preset": 6}, separators=(",", ":"))),
    ("jpeg-gainmap", "{}"),
]

def main():
    src_dir, out_path = sys.argv[1], sys.argv[2]
    sha_cache = None
    if "--sha-cache" in sys.argv:
        sha_cache = sys.argv[sys.argv.index("--sha-cache") + 1]
    names = sorted(n for n in os.listdir(src_dir) if n.endswith(".png"))
    if not names:
        sys.exit("no .png sources in %s" % src_dir)
    cache = {}
    if sha_cache and os.path.exists(sha_cache):
        for line in open(sha_cache):
            n, s = line.rstrip("\n").split("\t")
            cache[n] = s
    shas = {}
    for i, n in enumerate(names):
        if n in cache:
            shas[n] = cache[n]
            continue
        h = hashlib.sha256()
        with open(os.path.join(src_dir, n), "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        shas[n] = h.hexdigest()
        if (i + 1) % 100 == 0:
            print("sha %d/%d" % (i + 1, len(names)), file=sys.stderr)
    if sha_cache:
        with open(sha_cache, "w") as f:
            for n in names:
                f.write("%s\t%s\n" % (n, shas[n]))
    qs = q_grid()
    cells = 0
    with open(out_path, "w") as out:
        for n in names:
            for codec, knobs in ARMS:
                for q in qs:
                    out.write(json.dumps({
                        "image_path": n,
                        "codec": codec,
                        "q": q,
                        "knob_tuple_json": knobs,
                        "source_sha": shas[n],
                        "hdr": True,
                    }, separators=(",", ":")) + "\n")
                    cells += 1
    expect = len(names) * len(ARMS) * len(qs)
    assert cells == expect, (cells, expect)
    print("wrote %d cells (%d sources x %d arms x %d q) -> %s"
          % (cells, len(names), len(ARMS), len(qs), out_path))

if __name__ == "__main__":
    main()
