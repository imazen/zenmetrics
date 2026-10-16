#!/usr/bin/env python3
"""Metadata-only indexed ladder declarations for the canonical zenfleet tools.

This module never executes, claims or uploads jobs. Source metadata must describe
pre-admitted opaque RGB8 sRGB references. Resizing is a separate canonical
Resample stage; its hashed output is supplied here as the source of each encode.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

SIZES = (32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512,
         640, 768, 1024, 1280, 1536, 2048, 4096)
EFFORTS = (*range(32), 200)
CLASSES = ("tiny", "photo", "screen", "line", "mixed")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def baseline(**overrides):
    return dict(schema="indexed-ladder-v1", joint=True, format="png", quality="best",
                dither="adaptive", priority="quality", strength=None, lambda_=None,
                sample_cap=131072, tolerance=.01, target=None, minimum=None,
                metrics=False, joint_effort=10, **overrides)


def profiles():
    """144 requested profiles. Historical aliases remain separate declarations."""
    base = baseline()
    base["lambda"] = base.pop("lambda_")
    out = []
    for (joint, fmt), quality, dither in itertools.product(
            ((False, "png"), (True, "png"), (True, "joint"), (True, "min-size")),
            ("fast", "balanced", "best"),
            ("none", "adaptive", "sierra-lite", "blue-noise", "linear", "ordered")):
        out.append(base | dict(joint=joint, format=fmt, quality=quality, dither=dither,
                               priority="compression" if fmt == "min-size" else "quality"))
    out += [base | dict(priority="compression")]
    out += [base | dict(strength=v) for v in (0., .25, .5, .75, 1.)]
    out += [base | {"lambda": v} for v in (0., .01)]
    out += [base | dict(sample_cap=0)]
    out += [base | dict(format=f, tolerance=v) for f in ("joint", "min-size") for v in (.005, .02)]
    out += [base | dict(target=v) for v in range(0, 101, 2)]
    out += [base | dict(minimum=v) for v in (0, 50, 100)]
    out += [base | dict(metrics=True)]
    out += [base | dict(format=f, joint_effort=v) for f in ("joint", "min-size") for v in (1, 22)]
    if len(out) != 144 or len({canonical(p) for p in out}) != 144:
        raise ValueError("palette profile inventory drift")
    return out


def geometry(width, height, side):
    if any(type(v) is not int or v <= 0 for v in (width, height, side)):
        raise ValueError("positive integer geometry required")
    longest = max(width, height)
    if side > longest:
        return None  # Metadata-only unattainable size, before any declaration.
    # nearest integer, ties upward; avoid floating-point host differences.
    return tuple(max(1, (2*v*side + longest)//(2*longest)) for v in (width, height))


def check_sources(sources):
    if len(sources) != 320 or len({s["source_id"] for s in sources}) != 320:
        raise ValueError("exactly 320 distinct source families required")
    families = set()
    for cls in CLASSES:
        selected = [s for s in sources if s["class"] == cls]
        if len(selected) != 64 or sum(s["role"] == "fit" for s in selected) != 50 or sum(s["role"] == "dev" for s in selected) != 14:
            raise ValueError("each class requires 50 fit and 14 development families")
    for s in sources:
        if s["family"] in families or s["colour"] != "rgb8-srgb-opaque":
            raise ValueError("source family collision or unsupported colour")
        families.add(s["family"])
        if not s.get("cluster") or len(s["source_sha256"]) != 64:
            raise ValueError("cluster and source hash bindings required")
        geometry(s["width"], s["height"], min(s["width"], s["height"]))


def requests(sources):
    """Full dense quantization inventory; streaming, no implicit budget cap."""
    check_sources(sources)
    for s, size, colors, profile in itertools.product(sources, SIZES, range(2, 257), profiles()):
        shape = geometry(s["width"], s["height"], size)
        if shape is not None:
            yield dict(source=s["source_id"], family=s["family"], role=s["role"],
                       content_class=s["class"], size=size, shape=shape, colors=colors,
                       profile=profile, quantization_id=digest([s["source_sha256"], shape, colors, profile]))


def encode_items(request, resized):
    """Emit EncodeDeclareItem rows; zenfleet-ctl owns content-addressed JobIds.

    The Resample receipt binds Mitchell B=C=1/3, geometry and source ancestry.
    No encode_fp dedup is claimed before byte-parity proof.
    """
    if (resized["source_id"] != request["source"] or tuple(resized["shape"]) != tuple(request["shape"])
            or resized["kernel"] != "mitchell-bc-1/3" or resized["colour"] != "rgb8-srgb-opaque"):
        raise ValueError("resample receipt does not bind requested source/geometry/colour")
    for effort in EFFORTS:
        cfg = request["profile"] | dict(colors=request["colors"], effort=effort)
        yield dict(image_path=resized["image_path"], codec="zenpng", q=0,
                   source_sha=resized["sha256"], hdr=False,
                   knob_tuple_json=canonical({"palette_ladder": cfg}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sources = json.loads(args.sources.read_text())["sources"]
    # Exclusive creation prevents overwriting an earlier preread manifest.
    check_sources(sources)
    with args.out.open("x") as stream:
        for request in requests(sources):
            stream.write(canonical(request) + "\n")


if __name__ == "__main__":
    main()
