#!/usr/bin/env python3
# jxlav1_subset.py — select a representative train-bucket subset of the
# imazen-26 JXL render-v4 corpus for the zenav1-svt / zenav1-aom config-trial
# sweep, and stage it under <stage_dir> with manifest-verified sha256.
#
#   usage: jxlav1_subset.py <manifest.tsv> <corpus_root> <stage_dir> [--total 84]
#                          [--max-mp 16]
#
#   Selection is deterministic (folder-stratified, pixel-count order) — no RNG.
#
# Design (2026-11-05 pareto-refresh run):
#   - bucket == "train" only: sweep rows feed codec-choice heuristics, so the
#     validate/test buckets stay untouched.
#   - <= --max-mp megapixels, the avifgen monster-tier exclusion (encode cost
#     is per-pixel; the corpus tail reaches 124 MP).
#   - Stratified by source folder: k = max(2, round(total * folder_share)),
#     size-spread inside each folder (evenly spaced picks over the pixel-count
#     ordering) so each folder contributes its small AND its large renditions
#     instead of a random draw.
#   - Output: <stage_dir>/<basename>.jxl copies + <stage_dir>/subset.tsv
#     (name, folder, width, height, mp, form, colour, sha256, manifest_path).
#     Every copied file is hash-verified against the manifest row.
import csv, hashlib, os, shutil, sys
from collections import defaultdict

def main():
    manifest_path, corpus_root, stage_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    total = int(_opt("--total", 84))
    max_mp = float(_opt("--max-mp", 16.0))
    rows = [r for r in csv.DictReader(open(manifest_path), delimiter="\t")
            if r["bucket"] == "train" and int(r["width"]) * int(r["height"]) <= max_mp * 1e6]
    if not rows:
        sys.exit("no train-bucket rows under the MP cap")
    by_folder = defaultdict(list)
    for r in rows:
        by_folder[r["path"].split("/")[1]].append(r)
    picks = []
    for folder, rs in sorted(by_folder.items()):
        k = min(len(rs), max(2, round(total * len(rs) / len(rows))))
        rs.sort(key=lambda r: int(r["width"]) * int(r["height"]))
        if k < len(rs):
            rs = [rs[round(i * (len(rs) - 1) / (k - 1))] for i in range(k)]
        picks.extend(rs)
    os.makedirs(stage_dir, exist_ok=True)
    out_tsv = os.path.join(stage_dir, "subset.tsv")
    with open(out_tsv, "w") as f:
        f.write("name\tfolder\twidth\theight\tmp\tform\tcolour\tsha256\tmanifest_path\n")
        for r in picks:
            src = os.path.join(corpus_root, r["path"])
            name = os.path.basename(src)
            dst = os.path.join(stage_dir, name)
            h = hashlib.sha256()
            with open(src, "rb") as fd, open(dst, "wb") as fo:
                for blk in iter(lambda: fd.read(1 << 20), b""):
                    h.update(blk)
                    fo.write(blk)
            if h.hexdigest() != r["sha256"]:
                os.remove(dst)
                sys.exit(f"sha256 mismatch on {name}: manifest {r['sha256']} got {h.hexdigest()}")
            mp = int(r["width"]) * int(r["height"]) / 1e6
            f.write("\t".join([name, r["path"].split("/")[1], r["width"], r["height"],
                               f"{mp:.3f}", r["form"], r["colour"], r["sha256"], r["path"]]) + "\n")
    print(f"wrote {len(picks)} sources -> {stage_dir} (max {max_mp} MP)")

def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

if __name__ == "__main__":
    main()
