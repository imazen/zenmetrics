#!/usr/bin/env python3
"""NIQE goldens via the BasicSR oracle (LIVE pristine params).

Regenerate (from the repo root):

    ~/tmp/nr-classic-scratch/venv/bin/python crates/niqe/tests/golden_niqe.py \
        > crates/niqe/tests/goldens_niqe.tsv

The oracle environment versions that produced this file are pinned in
the generated TSV's `# oracle:` header. basicsr 1.4.2's package __init__
imports a torchvision symbol that no longer exists, so this script shims
`torchvision.transforms.functional_tensor` before importing — the NIQE
path itself does not use torchvision. `torch.set_num_threads(1)` pins the
deterministic single-thread reduction order the Rust port reproduces.

Rows are keyed by image path relative to `--root` (default `~/tmp`);
the parity test resolves them against `$ZENMETRICS_NR_GOLDEN_ROOT`.

`calculate_niqe(img, crop_border=0)` on a cv2 BGR u8 decode: BGR→Y is
the `bgr2ycbcr` Y channel (studio-swing BT.601, not MATLAB `rgb2gray`)
evaluated in the oracle's f32/f64 cast chain, then two-scale AGGD
features vs the LIVE pristine MVG — identical to what the Rust port
computes from RGB8.

Inputs with fewer than two complete 96×96 blocks (the 64px and 128px
ladder rows) make the oracle crash (`np.cov` of <2 rows is NaN →
`np.linalg.pinv`: "SVD did not converge"); the generator emits NaN for
those rows and the parity test asserts the port reports the matching
error kind (`TooSmall` <2 blocks, `Degenerate` on flat ≥2-block input).
"""
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def _shim_basicsr_import():
    """Satisfy basicsr 1.4.2's `__init__`, which imports
    `torchvision.transforms.functional_tensor` (removed in current
    torchvision) for unrelated dataset utilities. NIQE never touches it."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except Exception:
        pass
    import torchvision
    mod = types.ModuleType("torchvision.transforms.functional_tensor")
    mod.rgb_to_grayscale = lambda *a, **k: None
    torchvision.transforms.functional_tensor = mod
    sys.modules["torchvision.transforms.functional_tensor"] = mod


_shim_basicsr_import()
import torch  # noqa: E402

torch.set_num_threads(1)
from basicsr.metrics.niqe import calculate_niqe  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "tmp"
LADDER = ROOT / "nn-phase0/ladder"
PAIRS = ROOT / "konjnd-pairs/pairs_abs.tsv"


def rel(path: str) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def score(path: str) -> float:
    img = cv2.imread(path)  # BGR u8
    if img is None:
        raise ValueError(f"cannot decode {path}")
    try:
        s = calculate_niqe(img, crop_border=0)
    except Exception:
        # The oracle crashes on images with no complete 96×96 block
        # (np.linalg.pinv on a NaN covariance → "SVD did not converge");
        # the Rust port reports ErrorKind::TooSmall there. Emit NaN so the
        # parity test can check that mapping.
        return float("nan")
    return float(s)


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

    import scipy, torchvision
    print("# niqe goldens — regenerate (from the repo root):")
    print("#   ~/tmp/nr-classic-scratch/venv/bin/python \\")
    print("#       crates/niqe/tests/golden_niqe.py > crates/niqe/tests/goldens_niqe.tsv")
    print(f"# oracle: basicsr==1.4.2, torch=={torch.__version__}, "
          f"torchvision=={torchvision.__version__}, numpy=={np.__version__}, "
          f"scipy=={scipy.__version__}, opencv-python=={cv2.__version__}, "
          f"python=={sys.version.split()[0]}, BLAS=MKL, torch threads=1")
    print("# paths are relative to $ZENMETRICS_NR_GOLDEN_ROOT (default ~/tmp)")
    print("# path\tniqe")
    for p, s in rows:
        print(f"{rel(p)}\t{s:.17g}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
