#!/usr/bin/env python3
"""
Build per-stage golden tensors and final JOD scores from the pinned
ColorVideoVDP Python reference (v0.5.4), for cvvdp-gpu's parity tests.

Run once locally, upload outputs to R2 via `upload_to_r2.sh`. The
Rust tests fetch the artifacts on first run and cache them locally.

Inputs:
  --pairs <file>     JSON: list of {"name", "ref", "dist"} entries
                     pointing at PNG files relative to --image-root.
  --image-root <dir> Root directory for the PNG paths.
  --out <dir>        Output directory. Writes:
                       manifest.json
                       <name>.stage_color.bin       (3 f32 planes)
                       <name>.stage_pyramid_l<k>_c<ch>.bin
                       <name>.final.json            (JOD + intermediate scalars)

The script does NOT touch R2 itself — it just produces files. Use
`upload_to_r2.sh` (sibling script) to publish to the public bucket.

Why a Python helper instead of a Rust port of pycvvdp first:
The whole point of the goldens is to lock parity against the published
reference. The script must call the reference verbatim — any "helpful"
preprocessing we add here would defeat the purpose.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_f32(path: Path, arr: np.ndarray) -> dict:
    """Write a contiguous f32 array, return manifest entry."""
    arr = np.ascontiguousarray(arr.astype(np.float32))
    path.write_bytes(arr.tobytes())
    return {
        "path": path.name,
        "dtype": "f32",
        "shape": list(arr.shape),
        "sha256": sha256_file(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--display",
        default="standard_4k",
        help="pycvvdp display model name (default: standard_4k)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(args.image_root)
    with open(args.pairs) as f:
        pairs = json.load(f)

    # Imported lazily so `--help` works without the dep installed.
    import pycvvdp  # noqa: F401  (used reflectively below)
    from pycvvdp import cvvdp

    metric = cvvdp(display_name=args.display, heatmap=None)
    manifest: dict = {
        "reference": "gfxdisp/ColorVideoVDP",
        "reference_version": "v0.5.4",
        "display_model": args.display,
        "pairs": {},
    }

    def to_json_friendly(value):
        """Convert a stat value into something json.dump can serialize."""
        if hasattr(value, "detach"):  # torch tensors
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):  # numpy arrays
            return value.tolist()
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return repr(value)

    for entry in pairs:
        name = entry["name"]
        ref_path = image_root / entry["ref"]
        dist_path = image_root / entry["dist"]
        print(f"[{name}] {ref_path.name} vs {dist_path.name}", file=sys.stderr)

        ref = np.asarray(Image.open(ref_path).convert("RGB"))
        dist = np.asarray(Image.open(dist_path).convert("RGB"))

        # pycvvdp.predict signature: (test, reference, dim_order, fps).
        # Note the test-first argument order — flipping it gives a
        # near-identical JOD on symmetric distortions but is technically
        # wrong on directional ones, so don't simplify.
        jod, stats = metric.predict(dist, ref, dim_order="HWC")

        pair_entry = {
            "ref_path": entry["ref"],
            "dist_path": entry["dist"],
            "jod": float(jod),
            "stats": {
                k: to_json_friendly(v)
                for k, v in (
                    stats.items() if hasattr(stats, "items") else stats.__dict__.items()
                )
            },
        }

        manifest["pairs"][name] = pair_entry

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
