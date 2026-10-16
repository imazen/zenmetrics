#!/usr/bin/env python3
"""PIQE goldens via the pypiqe oracle.

Regenerate (from the repo root):

    ~/tmp/nr-classic-scratch/venv/bin/python crates/piqe/tests/golden_piqe.py \
        > crates/piqe/tests/goldens_piqe.tsv

Rows are keyed by image path relative to `--root` (default `~/tmp`);
the parity test resolves them against `$ZENMETRICS_NR_GOLDEN_ROOT`.

pypiqe is a pure function of the decoded pixels; cv2.imread loads PNG as
BGR and piqe() converts BGR->GRAY internally, which is exactly what the
Rust port does from RGB8 (same luma formula).
"""
import sys
from pathlib import Path

import cv2
from pypiqe import piqe

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "tmp"
LADDER = ROOT / "nn-phase0/ladder"
PAIRS = ROOT / "konjnd-pairs/pairs_abs.tsv"


def rel(path: str) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def score(path: str) -> float:
    img = cv2.imread(path)  # BGR u8
    if img is None:
        raise ValueError(f"cannot decode {path}")
    s, *_ = piqe(img)
    return s


def main() -> None:
    rows = []
    for p in sorted(LADDER.iterdir()):
        if p.suffix == ".png":
            rows.append((str(p), score(str(p))))

    with open(PAIRS) as f:
        header = f.readline()
        lines = [ln.rstrip("\n").split("\t") for ln in f]
    ref_i, dist_i, codec_i = (
        header.rstrip("\n").split("\t").index(k) for k in ("ref_path", "dist_path", "codec")
    )
    picked = [ln for ln in lines if ln[codec_i] == "JPEG"][:32]
    picked += [ln for ln in lines if ln[codec_i] == "BPG"][:32]
    for ln in picked:
        rows.append((ln[ref_i], score(ln[ref_i])))
        rows.append((ln[dist_i], score(ln[dist_i])))

    import numpy
    print("# piqe goldens — regenerate (from the repo root):")
    print("#   ~/tmp/nr-classic-scratch/venv/bin/python \\")
    print("#       crates/piqe/tests/golden_piqe.py > crates/piqe/tests/goldens_piqe.tsv")
    print(f"# oracle: pypiqe==1.2, opencv-python=={cv2.__version__}, "
          f"numpy=={numpy.__version__}, python=={sys.version.split()[0]}")
    print("# paths are relative to $ZENMETRICS_NR_GOLDEN_ROOT (default ~/tmp)")
    print("# path\tpiqe")
    for p, s in rows:
        print(f"{rel(p)}\t{s:.17g}")


if __name__ == "__main__":
    sys.exit(main())
