#!/usr/bin/env python3
"""Capture IW-SSIM goldens from the canonical Python reference.

Generates ~10 reference scores from Python-IW-SSIM
(https://github.com/Jack-guo-xy/Python-IW-SSIM) on deterministic
fixtures, plus the actual Python reference's bundled Ref/Dist images,
and writes the scores + fixture metadata to
``python_iwssim_2026-05-27.json``.

Usage:

    PYTHON_IWSSIM_REF=/path/to/Python-IW-SSIM \\
        VENV=/path/to/venv \\
        $VENV/bin/python3 capture_python_goldens.py [out_json]

Re-runnable: the only randomness is `np.random.seed(0xC0FFEE)` for the
synthetic fixtures, which is deterministic across numpy versions for
the integer types we use.

The script reads ``$PYTHON_IWSSIM_REF`` (defaults to
``~/work/python-iwssim-ref``). Imports the upstream's `IW_SSIM` class
directly from that path. The CPU port verifies its scores against this
JSON via ``tests/parity_python.rs``.
"""

import os
import sys
import json
import hashlib
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PYIWSSIM_REF = Path(os.environ.get(
    "PYTHON_IWSSIM_REF", os.path.expanduser("~/work/python-iwssim-ref")))
sys.path.insert(0, str(PYIWSSIM_REF))


# torch.eig was removed in PyTorch 1.9+; the reference still calls
# `torch.eig(C_u, eigenvectors=True)` which expects a (N, 2) tensor of
# eigenvalues (real, imag) and a (N, N) eigenvector matrix. Force-replace
# the stub so the unmodified reference code runs against modern torch.
def _torch_eig_shim(a, eigenvectors=False):
    # The IW-SSIM reference always sets eigenvectors=True.
    # C_u is real symmetric (= Yᵀ Y), so use the SYMMETRIC eigh which
    # returns real eigenvalues + an orthogonal eigenvector matrix —
    # numerically equivalent to torch.eig (which used LAPACK *geev*)
    # but always real for symmetric input.
    L_real, V_real = torch.linalg.eigh(a)
    L_imag = torch.zeros_like(L_real)
    # The reference reads `eig_values[:, 0]` and treats it as the real
    # part. We mirror that contract: column 0 = real, column 1 = imag.
    L_pair = torch.stack([L_real, L_imag], dim=-1)
    return L_pair, V_real
torch.eig = _torch_eig_shim
import torch._linalg_utils  # noqa: E402
torch._linalg_utils.eig = _torch_eig_shim


from IW_SSIM_PyTorch import IW_SSIM  # noqa: E402
from utils import rgb2gray  # noqa: E402


def hash_array(arr: np.ndarray) -> str:
    """Stable SHA256 of an ndarray's contents (dtype-aware)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def make_synthetic_pair(name: str, w: int, h: int, distort: str,
                        seed: int) -> dict:
    """Make a (ref_rgb, dist_rgb) pair deterministically from a 32-bit seed.

    The Rust harness recreates the same RGB bytes via the documented
    XorShift64 sequence below — no PNG fixtures need to be committed.
    """
    # XorShift64 PRNG, identical sequence in Python + Rust.
    state = [seed & 0xFFFFFFFFFFFFFFFF]
    def xorshift64() -> int:
        s = state[0]
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        state[0] = s
        return s

    ref = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = xorshift64()
            ref[y, x, 0] = (v >> 0) & 0xFF
            ref[y, x, 1] = (v >> 8) & 0xFF
            ref[y, x, 2] = (v >> 16) & 0xFF

    if distort == "identical":
        dist = ref.copy()
    elif distort == "offset":
        # Shift by 5 brightness units, clamped.
        dist = np.clip(ref.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    elif distort == "shift1px":
        # Shift by 1 pixel horizontally (zero-fill the new column).
        # Deterministic across implementations — no convolution drift.
        dist = np.zeros_like(ref)
        dist[:, 1:, :] = ref[:, :-1, :]
    elif distort == "swap":
        # Swap RG channels — strong distortion.
        dist = ref[..., [1, 0, 2]].copy()
    else:
        raise ValueError(f"unknown distortion {distort!r}")

    return {
        "name": name,
        "kind": "synthetic",
        "distortion": distort,
        "width": w,
        "height": h,
        "seed": seed,
        "ref_sha": hash_array(ref),
        "dist_sha": hash_array(dist),
        "_ref_arr": ref,
        "_dist_arr": dist,
    }


def score_pair(scorer: IW_SSIM, pair: dict) -> float:
    ref = pair["_ref_arr"]
    dist = pair["_dist_arr"]
    g1 = rgb2gray(ref)
    g2 = rgb2gray(dist)
    out = scorer.test(g1, g2)
    # `out` is a torch scalar tensor.
    return float(out.item())


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "python_iwssim_2026-05-27.json")

    # The reference's class defaults match config.py exactly (see
    # demo.py for the CLI mapping).
    scorer = IW_SSIM(
        iw_flag=True,
        Nsc=5,
        blSzX=3,
        blSzY=3,
        parent=True,
        sigma_nsq=0.4,
        use_cuda=False,
        use_double=False,
    )

    # Build the fixture list.
    pairs = []

    # NOTE: We previously included the upstream's bundled Ref.bmp /
    # Dist.jpg pairs but ended up with 200 KB PNG fixtures per side
    # (random noise doesn't compress). The fixtures are reconstructible
    # from a seed via the XorShift64 sequence so we only commit the
    # manifest + seed; the Rust harness regenerates the bytes locally.
    #
    # The upstream's Ref/Dist BMP/JPEG pair is non-deterministic content
    # we can't reconstruct without committing 2 MB of pixels. We leave
    # those out of the goldens manifest by design — the parity tests
    # against synthetic seeds are sufficient to lock the algorithm.

    # Synthetic random fixtures at 256x256 (just barely above 176).
    # Fixed seeds (low-entropy values for human readability).
    synth_seeds = {
        "identical": 1,
        "offset": 2,
        "shift1px": 3,
        "swap": 4,
    }
    for distort, seed in synth_seeds.items():
        pairs.append(make_synthetic_pair(
            f"synth_256_{distort}", 256, 256, distort, seed))

    # Bigger fixtures at 320x240 (above 176 on both axes).
    big_seeds = {
        "identical": 10,
        "shift1px": 11,
    }
    for distort, seed in big_seeds.items():
        pairs.append(make_synthetic_pair(
            f"synth_320x240_{distort}", 320, 240, distort, seed))

    # And one minimally-just-above-176 fixture to stress the smallest
    # supported input.
    pairs.append(make_synthetic_pair(
        "synth_176_shift1px", 176, 176, "shift1px", 100))

    # Score each pair. No PNG fixtures committed — all fixtures are
    # reconstructible from `seed` + XorShift64 (synthetic) or from the
    # upstream's bundled BMPs (kind == "upstream", excluded here).
    json_pairs = []
    for p in pairs:
        score = score_pair(scorer, p)
        print(f"  {p['name']:36s}  score = {score:.10f}")
        rec = {k: v for k, v in p.items() if not k.startswith("_")}
        rec["score"] = score
        json_pairs.append(rec)

    out = {
        "schema_version": 1,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_reference": {
            "repo": "https://github.com/Jack-guo-xy/Python-IW-SSIM",
            "commit": "f9de37cdb4c459fd188c611e377a87a54516d3cd",
            "use_double": False,
            "iw_flag": True,
            "Nsc": 5,
            "blSzX": 3,
            "blSzY": 3,
            "parent": True,
            "sigma_nsq": 0.4,
        },
        "pairs": json_pairs,
    }
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(json_pairs)} goldens -> {out_path}")


if __name__ == "__main__":
    main()
