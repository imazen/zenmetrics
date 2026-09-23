#!/usr/bin/env python3
"""
Build pycvvdp VIDEO goldens for the cvvdp video path (v0.5.7).

Consumes the manifest + PNG frame sequences emitted by:

    cargo run -p cvvdp-conformance --bin emit_video_situations -- /scratch/cvvdpvideo

For every (situation, display) cell it loads the situation's
ref/dist PNG frame sequences (the EXACT bytes the Rust harness
scores) and runs the installed pycvvdp reference at that display
model with `dim_order="FHWC"`, recording the ground-truth JOD.

It also records per-stage dumps used by the Rust port's unit /
parity tests:

  * `temporal_filters[fps][ch]` — the FIR taps from
    `cvvdp.get_temporal_filters(fps)` (sustained A/RG/VY + transient A).
  * `stage_dumps["<situation>|<display>"]["q_per_ch"]` — the
    `[n_ch=4, n_frames, n_bands]` per-band pooled qualities from
    `stats["Q_per_ch"]` for a small set of dump fixtures.

Output: <situations_dir>/video_goldens.json — committed to
scripts/cvvdp_goldens/video_goldens.json when < 30 KB.

    {
      "reference": "gfxdisp/ColorVideoVDP",
      "reference_version": "v0.5.7",   # installed pkg, not the pin
      "temporal_filters": {"30.0": [[...], ...]},
      "cells": {"<situation>|<display>": {"jod_ref": ..., ...}},
      "stage_dumps": {...}
    }

IMPORTANT — argument order: `predict(test, reference, ...)`: the
DISTORTED video goes FIRST (see build_conformance_goldens.py; a
swapped call was a real bug here once).

Run with the pinned reference venv:

    ~/tmp/devin/venv-pycvvdp-0.5.7/bin/python \
        scripts/cvvdp_goldens/build_video_goldens.py /scratch/cvvdpvideo \
        --out scripts/cvvdp_goldens/video_goldens.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Situations whose per-stage intermediates get dumped (keep small —
# these go into the committed JSON).
DUMP_SITUATIONS = {"vid_flicker_24", "vid_temporal_noise_30"}
DUMP_DISPLAY = "standard_4k"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_pycvvdp_version() -> str:
    """`v<version>` of the installed pycvvdp distribution (PyPI name `cvvdp`,
    git installs name it `pycvvdp`)."""
    from importlib.metadata import PackageNotFoundError, version

    for dist in ("cvvdp", "pycvvdp"):
        try:
            return "v" + version(dist)
        except PackageNotFoundError:
            continue
    raise SystemExit(
        "pycvvdp is importable but no cvvdp/pycvvdp distribution metadata found"
    )


def load_clip(dir_path: Path) -> np.ndarray:
    """Load f%04d.png frames into a uint8 [F, H, W, C] array."""
    frames = sorted(dir_path.glob("f*.png"))
    if not frames:
        raise SystemExit(f"no f*.png frames in {dir_path}")
    return np.stack(
        [np.asarray(Image.open(p).convert("RGB")) for p in frames], axis=0
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "situations_dir",
        help="dir containing video_manifest.json + frames/ from emit_video_situations",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: <situations_dir>/video_goldens.json)",
    )
    args = ap.parse_args()

    sit_dir = Path(args.situations_dir)
    manifest_path = sit_dir / "video_manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    out_path = Path(args.out) if args.out else sit_dir / "video_goldens.json"

    # Imported lazily so --help works without the dep installed.
    import pycvvdp  # noqa: F401

    # Record the pycvvdp that ACTUALLY scored the cells (installed
    # package metadata — not the port's pin, not a hard-code).
    ref_version = installed_pycvvdp_version()
    port_pin = manifest["reference_version"]
    if ref_version != port_pin:
        print(
            f"NOTE: installed pycvvdp {ref_version} != port pin {port_pin}; "
            f"goldens are labelled {ref_version}",
            file=sys.stderr,
        )
    situations = manifest["situations"]
    displays = manifest["displays"]  # list of upstream_name strings

    print(
        f"scoring {len(situations)} video situations x {len(displays)} displays "
        f"= {len(situations) * len(displays)} cells with pycvvdp {ref_version}",
        file=sys.stderr,
    )

    metrics = {}
    for name in displays:
        try:
            metrics[name] = pycvvdp.cvvdp(display_name=name, heatmap=None, quiet=True)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"display {name!r} failed to construct under pycvvdp {ref_version}: {e}"
            ) from e

    # Temporal filter taps per fps used by the corpus (identical
    # across displays — they depend only on fps + parameters).
    # Taps are rounded to 10 decimals (they're ~1e-1 scale, so this
    # keeps ~1e-11 abs precision — far under the 1e-6 test gate).
    temporal_filters = {}
    fps_values = sorted({float(s["fps"]) for s in situations})
    ref_metric = metrics[displays[0]]
    for fps in fps_values:
        taps, omega = ref_metric.get_temporal_filters(fps)
        temporal_filters[f"{fps:g}"] = {
            "taps": [[round(v, 10) for v in t.tolist()] for t in taps],
            "omega": [round(v, 10) for v in omega.tolist()],
        }

    cells = {}
    stage_dumps = {}
    t0 = time.time()
    n_done = 0
    n_total = len(situations) * len(displays)
    for s in situations:
        ref_clip = load_clip(sit_dir / s["ref_dir"])
        dist_clip = load_clip(sit_dir / s["dist_dir"])
        assert ref_clip.shape[0] == s["n_frames"], s["name"]
        assert ref_clip.shape[1] == s["height"], s["name"]
        assert ref_clip.shape[2] == s["width"], s["name"]
        fps = float(s["fps"])
        for disp in displays:
            key = f"{s['name']}|{disp}"
            metric = metrics[disp]
            # pycvvdp.predict signature: (test, reference, dim_order,
            # frames_per_second). Test/distorted FIRST.
            jod, stats = metric.predict(
                dist_clip, ref_clip, dim_order="FHWC", frames_per_second=fps
            )
            cells[key] = {
                "situation": s["name"],
                "display": disp,
                "class": s["class"],
                "width": s["width"],
                "height": s["height"],
                "n_frames": s["n_frames"],
                "fps": fps,
                "jod_ref": round(float(jod), 6),
            }
            if s["name"] in DUMP_SITUATIONS and disp == DUMP_DISPLAY:
                q = stats["Q_per_ch"]  # torch [B=1, ch=4, F, bands]
                stage_dumps[key] = {
                    "q_per_ch": [round(float(v), 6) for v in q.flatten().tolist()],
                    "q_per_ch_shape": list(q.shape),
                }
            n_done += 1
            if n_done % 10 == 0 or n_done == n_total:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(
                    f"  {n_done}/{n_total} cells ({rate:.1f}/s)",
                    file=sys.stderr,
                )

    golden = {
        "reference": "gfxdisp/ColorVideoVDP",
        "reference_version": ref_version,
        "port_pinned_version": port_pin,
        "generated_unix": int(time.time()),
        "video_manifest_sha256": sha256_file(manifest_path),
        "displays": displays,
        "n_situations": len(situations),
        "n_displays": len(displays),
        "n_cells": len(cells),
        "temporal_filters": temporal_filters,
        "cells": cells,
        "stage_dumps": stage_dumps,
    }

    with out_path.open("w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)
    size_kb = out_path.stat().st_size / 1024.0
    print(f"wrote {out_path} ({len(cells)} cells, {size_kb:.1f} KB)", file=sys.stderr)
    print(f"sha256: {sha256_file(out_path)}", file=sys.stderr)
    if size_kb > 30.0:
        print(
            "NOTE: > 30 KB — do NOT commit; keep in /scratch and reference "
            "from the report",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
