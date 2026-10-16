#!/usr/bin/env python3
"""BRISQUE goldens via the piq oracle (LIVE allmodel/allrange SVR weights).

Regenerate (from the repo root):

    ~/tmp/nr-classic-scratch/venv/bin/python crates/brisque/tests/golden_brisque.py \
        > crates/brisque/tests/goldens_brisque.tsv

The oracle environment versions that produced this file are pinned in
the generated TSV's `# oracle:` header. `torch.set_num_threads(1)` pins
the deterministic single-thread reduction order the Rust port
reproduces.

Rows are keyed by image path relative to the first positional root argument
(default `$ZENMETRICS_NR_GOLDEN_ROOT`, then `~/tmp`);
the parity test resolves them against `$ZENMETRICS_NR_GOLDEN_ROOT`.

`piq.brisque(x, data_range=255.0)` on an RGB u8 decode cast to f32:
RGB→Y is piq's `rgb2yiq` Y channel rounded to u8 (`x/255*255` is the
identity on u8), then two scales of zero-pad-conv MSCN + GGD/AGGD
features, SVM-scale to [-1,1], epsilon-SVR predict against piq's
`brisque_svm_weights.pt` — bitwise identical to LIVE's released
`allmodel` (774 SVs, gamma 0.05, rho -153.591).

Images that make the oracle raise (flat inputs → 'non zero variance'
assert, single-sign MSCN products → AGGD asserts, degenerate geometry)
emit NaN; the parity test asserts the port reports the matching error
kind (`TooSmall` for degenerate geometry, `Degenerate` otherwise).
"""
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)
import piq  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ZENMETRICS_NR_GOLDEN_ROOT", Path.home() / "tmp"))
LADDER = ROOT / "nn-phase0/ladder"
PAIRS = ROOT / "konjnd-pairs/pairs_abs.tsv"


def rel(path: str) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def score(path: str) -> float:
    img = cv2.imread(path)  # BGR u8
    if img is None:
        raise ValueError(f"cannot decode {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    try:
        s = piq.brisque(x, kernel_size=7, kernel_sigma=7.0 / 6.0,
                        data_range=255.0, reduction='none')
    except Exception:
        # Oracle raises on degenerate inputs (zero-variance MSCN,
        # single-sign AGGD products); the Rust port reports an error
        # kind there. Emit NaN so the parity test can check the mapping.
        return float("nan")
    return float(s.item())


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

    print("# brisque goldens — regenerate (from the repo root):")
    print("#   ~/tmp/nr-classic-scratch/venv/bin/python \\")
    print("#       crates/brisque/tests/golden_brisque.py > crates/brisque/tests/goldens_brisque.tsv")
    print(f"# oracle: piq=={piq.__version__}, torch=={torch.__version__}, "
          f"numpy=={np.__version__}, opencv-python=={cv2.__version__}, "
          f"python=={sys.version.split()[0]}, torch threads=1, "
          "SVR model=piq brisque_svm_weights.pt (bitwise LIVE allmodel)")
    print(f"# CPU context: libc={platform.libc_ver()}, capability={torch.backends.cpu.get_cpu_capability()}, threads=1; MKL available={torch.backends.mkl.is_available()}")
    for line in torch.__config__.show().splitlines():
        if "Math Kernel Library Version" in line:
            print(f"# MKL build: {line.strip()}")
    print("# CPU environment: " + ", ".join(f"{key}={os.environ.get(key, 'unset')}" for key in ("MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS", "ATEN_CPU_CAPABILITY")))
    print("# paths are relative to $ZENMETRICS_NR_GOLDEN_ROOT (default ~/tmp)")
    print("# path\tbrisque")
    for p, s in rows:
        print(f"{rel(p)}\t{s:.17g}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
