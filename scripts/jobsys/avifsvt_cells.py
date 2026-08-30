#!/usr/bin/env python3
# avifsvt_cells.py — emit the encode-declare cells JSONL for the fresh SDR AVIF wave
# on zenavif's svt-rs backend (Av1Backend::SvtRs = the pure-Rust zenav1-svt port,
# 4:2:0 still encoder, muxed by zenavif-serialize). Registered: zensim
# benchmarks/balance_campaign_2026-08-28.md "FRESH AVIF WAVE — SVT BACKEND".
#
#   usage: avifsvt_cells.py <source_dir> <out_cells.jsonl> [--sha-cache <tsv>]
#                          [--speeds 4,6,8] [--max-mp 16]
#
# Grid (uniform, additive-later):
#   - q: 1 + step 5 across 5..70 + step 2 across 72..100 = 30 points (the avif944
#     grid, dense at BOTH ends per the sweep discipline).
#   - speeds: zenavif speed 1..=10 -> SVT preset 0..=13 linear; default {4,6,8}.
#   - sources: every .png in <source_dir> whose `.scale<W>x<H>.` area is <= --max-mp
#     megapixels (the avifgen monster-tier exclusion, 27 renditions > 16 MP).
#   - knob_tuple_json = {"backend":"svt-rs","speed":S} — a knob-grid cell (no plan
#     identity), executed by `encode_cell` via the executor's `backend` knob
#     (zenmetrics-cli `avif-svt` feature). 4:2:0 is implied by the backend.
#
# `image_path` is the source BASENAME (workers resolve it against ZEN_CORPUS_PREFIX);
# `source_sha` is the sha256 of the source bytes. Same shape as hdrgrid_cells.py /
# the avifgen cells_*_declare.jsonl consumed by `zenfleet-ctl declare-encodes`.
import hashlib, json, os, re, sys

def q_grid():
    qs = [1] + list(range(5, 71, 5)) + list(range(72, 101, 2))
    assert len(qs) == 30, len(qs)
    return qs

def main():
    src_dir, out_path = sys.argv[1], sys.argv[2]
    sha_cache = sys.argv[sys.argv.index("--sha-cache") + 1] if "--sha-cache" in sys.argv else None
    speeds = [int(x) for x in (sys.argv[sys.argv.index("--speeds") + 1] if "--speeds" in sys.argv else "4,6,8").split(",")]
    max_mp = float(sys.argv[sys.argv.index("--max-mp") + 1]) if "--max-mp" in sys.argv else 16.0
    dim_re = re.compile(r"\.scale(\d+)x(\d+)\.")
    names, excluded = [], []
    for n in sorted(os.listdir(src_dir)):
        if not n.endswith(".png"):
            continue
        m = dim_re.search(n)
        if not m:
            sys.exit(f"cannot parse scaleWxH from {n}")
        if int(m.group(1)) * int(m.group(2)) > max_mp * 1e6:
            excluded.append(n)
        else:
            names.append(n)
    if not names:
        sys.exit("no eligible .png sources in %s" % src_dir)
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
            for sp in speeds:
                knobs = json.dumps({"backend": "svt-rs", "speed": sp}, separators=(",", ":"))
                for q in qs:
                    out.write(json.dumps({
                        "image_path": n, "codec": "zenavif", "q": q,
                        "knob_tuple_json": knobs, "source_sha": shas[n],
                    }, separators=(",", ":")) + "\n")
                    cells += 1
    expect = len(names) * len(speeds) * len(qs)
    assert cells == expect, (cells, expect)
    print("wrote %d cells (%d sources x %d speeds x %d q; %d excluded > %.0f MP) -> %s"
          % (cells, len(names), len(speeds), len(qs), len(excluded), max_mp, out_path))

if __name__ == "__main__":
    main()
