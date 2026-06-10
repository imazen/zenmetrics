#!/usr/bin/env python3
"""Capture plain-MS-SSIM goldens (``iw_flag=False``) from the same Python
reference + identical deterministic fixtures as ``capture_python_goldens.py``.

With ``iw_flag=False`` the reference (Wang & Li 2011 framework) pools each
scale's contrast-structure map by plain mean under the standard MS-SSIM
exponents — i.e. canonical MS-SSIM (Wang et al. 2003). These goldens lock the
crate's ``IwssimParams { iw_flag: false }`` mode to the reference.

Usage (same env as the base script):
    PYTHON_IWSSIM_REF=~/work/python-iwssim-ref python3 capture_python_msssim_goldens.py
"""
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import capture_python_goldens as base  # noqa: E402  (reuses shim + fixtures)
from IW_SSIM_PyTorch import IW_SSIM  # noqa: E402


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "python_msssim_2026-06-10.json")
    scorer = IW_SSIM(iw_flag=False, Nsc=5, blSzX=3, blSzY=3, parent=True,
                     sigma_nsq=0.4, use_cuda=False, use_double=False)

    pairs = []
    for distort, seed in {"identical": 1, "offset": 2, "shift1px": 3, "swap": 4}.items():
        pairs.append(base.make_synthetic_pair(f"synth_256_{distort}", 256, 256, distort, seed))
    for distort, seed in {"identical": 10, "shift1px": 11}.items():
        pairs.append(base.make_synthetic_pair(f"synth_320x240_{distort}", 320, 240, distort, seed))
    pairs.append(base.make_synthetic_pair("synth_176_shift1px", 176, 176, "shift1px", 100))

    json_pairs = []
    for p in pairs:
        score = base.score_pair(scorer, p)
        print(f"  {p['name']:36s}  msssim = {score:.10f}")
        rec = {k: v for k, v in p.items() if not k.startswith("_")}
        rec["score"] = score
        json_pairs.append(rec)

    out = {
        "schema_version": 1,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_reference": {
            "repo": "https://github.com/Jack-guo-xy/Python-IW-SSIM",
            "commit": "f9de37cdb4c459fd188c611e377a87a54516d3cd",
            "use_double": False, "iw_flag": False, "Nsc": 5,
            "blSzX": 3, "blSzY": 3, "parent": True, "sigma_nsq": 0.4,
        },
        "pairs": json_pairs,
    }
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
