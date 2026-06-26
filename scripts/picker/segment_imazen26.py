#!/usr/bin/env python3
"""Materialize the imazen-26 even/odd train/val/test segmentation from the
manifest + the canonical `origin_split` rule. Deterministic — re-run any time.

Outputs (under /mnt/v/output/imazen-26-features/ by default):
  - imazen26_split_evenodd.tsv  : stem split manifest_split content_class source original_path
  - optionally, train/ validate/ test/ subfolders of SYMLINKS to the originals
    (--subfolders <dir>) so the corpus is browseable per the user's request,
    without duplicating GBs.

Usage:
  python3 scripts/picker/segment_imazen26.py
  python3 scripts/picker/segment_imazen26.py --subfolders /mnt/v/output/imazen-26-split
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from origin_split import split_of  # noqa: E402

MANIFEST = "/mnt/v/output/imazen-26-features/imazen26_manifest.tsv"
OUT_TSV = "/mnt/v/output/imazen-26-features/imazen26_split_evenodd.tsv"
SUBDIR = {"train": "train", "val": "validate", "test": "test"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default=OUT_TSV)
    ap.add_argument("--subfolders", default=None, help="root to populate train/validate/test symlinks")
    args = ap.parse_args()

    rows = []
    counts = {"train": 0, "val": 0, "test": 0, None: 0}
    with open(args.manifest) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            c = line.rstrip("\n").split("\t")
            stem = c[0]
            man = c[1] if len(c) > 1 else ""
            cc = c[2] if len(c) > 2 else ""
            src = c[3] if len(c) > 3 else ""
            path = c[4] if len(c) > 4 else ""
            sp = split_of(stem)
            counts[sp] = counts.get(sp, 0) + 1
            rows.append((stem, sp, man, cc, src, path))

    with open(args.out, "w") as o:
        o.write("stem\tsplit\tmanifest_split\tcontent_class\tsource\toriginal_path\n")
        for r in rows:
            o.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {len(rows)} origins -> {args.out}  splits={counts}")

    if args.subfolders:
        made = {"train": 0, "val": 0, "test": 0}
        for sp in SUBDIR.values():
            os.makedirs(os.path.join(args.subfolders, sp), exist_ok=True)
        for stem, sp, _man, _cc, _src, path in rows:
            if sp is None or not path or not os.path.exists(path):
                continue
            dst = os.path.join(args.subfolders, SUBDIR[sp], os.path.basename(path))
            if not os.path.lexists(dst):
                os.symlink(path, dst)
                made[sp] += 1
        print(f"symlinked originals into {args.subfolders}/{{train,validate,test}}: {made}")


if __name__ == "__main__":
    main()
