#!/usr/bin/env python3
"""
Build pycvvdp v0.5.4 conformance goldens for the cvvdp-conformance
matrix.

Consumes the manifest emitted by:

    cargo run -p cvvdp-conformance --bin emit_situations -- <situations_dir>

For every (situation, display) cell it loads the situation's
ref.png + dist.png (the EXACT bytes the Rust harness scores) and runs
the pinned pycvvdp v0.5.4 reference at that display model, recording
the ground-truth JOD.

Output: <situations_dir>/conformance_goldens.json

    {
      "reference": "gfxdisp/ColorVideoVDP",
      "reference_version": "v0.5.4",
      "displays": [...],
      "cells": {
        "<situation>|<display>": {
          "situation": "...", "display": "...", "class": "...",
          "width": ..., "height": ...,
          "jod_ref": <float>
        },
        ...
      }
    }

Run with the isolated venv that reuses the host pycvvdp install:

    scripts/cvvdp_goldens/.venv/bin/python \
        scripts/cvvdp_goldens/build_conformance_goldens.py <situations_dir>

The script must call the reference VERBATIM — any preprocessing here
would defeat the purpose of pinning parity against the published
reference. Mirrors build_goldens.py's discipline.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "situations_dir",
        help="dir containing manifest.json + images/ from emit_situations",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: <situations_dir>/conformance_goldens.json)",
    )
    args = ap.parse_args()

    sit_dir = Path(args.situations_dir)
    manifest_path = sit_dir / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    out_path = (
        Path(args.out) if args.out else sit_dir / "conformance_goldens.json"
    )

    # Imported lazily so --help works without the dep installed.
    import pycvvdp  # noqa: F401

    ref_version = manifest["reference_version"]
    situations = manifest["situations"]
    displays = manifest["displays"]

    print(
        f"scoring {len(situations)} situations x {len(displays)} displays "
        f"= {len(situations) * len(displays)} cells with pycvvdp {ref_version}",
        file=sys.stderr,
    )

    # Cache the (image bytes) per situation so we don't re-read PNGs
    # for every display. One metric object per display (cheap to keep).
    metrics = {}
    for d in displays:
        name = d["upstream_name"]
        try:
            metrics[name] = pycvvdp.cvvdp(display_name=name, heatmap=None)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: display {name} failed to construct: {e}", file=sys.stderr)
            metrics[name] = None

    cells = {}
    t0 = time.time()
    n_done = 0
    n_total = len(situations) * len(displays)
    for s in situations:
        ref_img = np.asarray(
            Image.open(sit_dir / s["ref"]).convert("RGB")
        )
        dist_img = np.asarray(
            Image.open(sit_dir / s["dist"]).convert("RGB")
        )
        for d in displays:
            disp = d["upstream_name"]
            key = f"{s['name']}|{disp}"
            metric = metrics[disp]
            if metric is None:
                cells[key] = {
                    "situation": s["name"],
                    "display": disp,
                    "class": s["class"],
                    "width": s["width"],
                    "height": s["height"],
                    "jod_ref": None,
                    "error": "display_construct_failed",
                }
                n_done += 1
                continue
            # pycvvdp.predict signature: (test, reference, dim_order).
            # Test-first argument order — see build_goldens.py note.
            jod, _ = metric.predict(dist_img, ref_img, dim_order="HWC")
            cells[key] = {
                "situation": s["name"],
                "display": disp,
                "class": s["class"],
                "width": s["width"],
                "height": s["height"],
                "jod_ref": float(jod),
            }
            n_done += 1
            if n_done % 20 == 0 or n_done == n_total:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(
                    f"  {n_done}/{n_total} cells ({rate:.1f}/s)",
                    file=sys.stderr,
                )

    golden = {
        "reference": "gfxdisp/ColorVideoVDP",
        "reference_version": ref_version,
        "generated_unix": int(time.time()),
        "situations_manifest_sha256": sha256_file(manifest_path),
        "displays": displays,
        "n_situations": len(situations),
        "n_displays": len(displays),
        "n_cells": len(cells),
        "cells": cells,
    }

    with out_path.open("w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)
    print(f"wrote {out_path} ({len(cells)} cells)", file=sys.stderr)
    print(f"sha256: {sha256_file(out_path)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
