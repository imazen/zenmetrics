#!/usr/bin/env python3
"""Bake a dual-head cvvdp picker JSON to ZNPR v3 binary.

Takes the JSON emitted by `train_cvvdp_picker.py`:

    {
      "model": {
        "architecture": "dual_head_leaky_relu_mlp",
        "input_dim": F,
        "trunk_hidden": [256, 256, 256],
        "heads": ["log_bytes", "cvvdp"],
        "layers": [
          { name: "trunk.0.weight", shape: [256, F], data: [...] },
          { name: "trunk.0.bias",   shape: [256],    data: [...] },
          ...
          { name: "head_bytes.weight", shape: [1, 256], data: [...] },
          { name: "head_bytes.bias",   shape: [1],      data: [...] },
          { name: "head_cvvdp.weight", shape: [1, 256], data: [...] },
          { name: "head_cvvdp.bias",   shape: [1],      data: [...] },
        ]
      },
      "Xmu", "Xsd", "X_impute",  # per-input
      "Yb_mu", "Yb_sd",          # log-bytes denorm
      "Yc_mu", "Yc_sd",          # cvvdp denorm
      "knobs", "src_feat_names",
      ...
    }

Converts to a single-head model `(F → 256 → 256 → 256 → 2)` by
stacking the two heads' weights into one `Linear(256, 2)`. Writes a
`BakeRequestJson` and shells out to `zenpredict-bake` to produce the
`.bin`.

Denormalization parameters live in metadata so runtime can recover
real-units (log_bytes, cvvdp) from the model output.

Usage:
  python3 bake_cvvdp_picker.py \\
    --in /mnt/v/.../trained_pickers/zenjpeg_cvvdp_picker.json \\
    --out /mnt/v/.../trained_pickers/zenjpeg_cvvdp_picker.bin \\
    --bake-bin /path/to/zenpredict-bake
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def schema_hash(src_feat_names: list[str], n_knobs: int, version_tag: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(version_tag.encode("utf-8"))
    h.update(b"\x00")
    for c in src_feat_names:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"||knob_onehot=")
    h.update(str(n_knobs).encode("ascii"))
    h.update(b"||q_norm")
    return int.from_bytes(h.digest(), "little")


def stack_layer_dict(layers: list[dict]) -> dict:
    """Index PyTorch state_dict-style layer list by name → tensor."""
    return {l["name"]: np.array(l["data"], dtype=np.float32).reshape(l["shape"]) for l in layers}


def build_bake_request(picker: dict, version_tag: str) -> dict:
    """Translate a dual-head picker JSON to BakeRequestJson."""
    model = picker["model"]
    state = stack_layer_dict(model["layers"])

    F = model["input_dim"]
    trunk_hidden = model["trunk_hidden"]
    n_knobs = len(picker["knobs"])
    n_src = len(picker["src_feat_names"])

    # Trunk layers
    bake_layers = []
    in_dim = F
    for li, h in enumerate(trunk_hidden):
        # PyTorch weight is (out, in); BakeRequestJson is row-major in_dim*out_dim
        # which we interpret as the (in, out) layout. Transpose.
        w = state[f"trunk.{li * 2}.weight"]  # *2 because LeakyReLU is sandwiched
        b = state[f"trunk.{li * 2}.bias"]
        assert w.shape == (h, in_dim), f"trunk.{li*2}.weight shape {w.shape} != ({h}, {in_dim})"
        w_io = w.T  # (in_dim, h)
        bake_layers.append({
            "in_dim": in_dim,
            "out_dim": h,
            "activation": "leakyrelu",
            "dtype": "f32",
            "weights": w_io.flatten().tolist(),
            "biases": b.flatten().tolist(),
        })
        in_dim = h

    # Combined output head: stack head_bytes + head_cvvdp into Linear(in_dim, 2)
    wb = state["head_bytes.weight"]  # (1, in_dim)
    bb = state["head_bytes.bias"]
    wc = state["head_cvvdp.weight"]  # (1, in_dim)
    bc = state["head_cvvdp.bias"]
    w_out = np.concatenate([wb, wc], axis=0)  # (2, in_dim)
    b_out = np.concatenate([bb, bc], axis=0)  # (2,)
    w_io = w_out.T  # (in_dim, 2)
    bake_layers.append({
        "in_dim": in_dim,
        "out_dim": 2,
        "activation": "identity",
        "dtype": "f32",
        "weights": w_io.flatten().tolist(),
        "biases": b_out.flatten().tolist(),
    })

    # Metadata: denormalisation + provenance + knob list
    metadata = [
        {"key": "zentrain.bake_name", "type": "utf8", "text": f"{picker['codec']}_cvvdp_dual_head_v1"},
        {"key": "zentrain.target_metric", "type": "utf8", "text": picker["target_metric"]},
        {"key": "zentrain.schema_version", "type": "utf8", "text": version_tag},
        {"key": "zentrain.codec", "type": "utf8", "text": picker["codec"]},
        {
            "key": "zentrain.dual_head_denorm",
            "type": "numeric",
            "f32": [picker["Yb_mu"], picker["Yb_sd"], picker["Yc_mu"], picker["Yc_sd"]],
        },
        {
            "key": "zentrain.r2_holdout",
            "type": "numeric",
            "f32": [picker["r2_bytes_holdout"], picker["r2_cvvdp_holdout"]],
        },
        {
            "key": "zentrain.knob_list",
            "type": "utf8",
            "text": json.dumps(picker["knobs"]),
        },
        {
            "key": "zentrain.x_impute",
            "type": "numeric",
            "f32": picker["X_impute"],
        },
    ]

    return {
        "schema_hash": schema_hash(picker["src_feat_names"], n_knobs, version_tag),
        "flags": 0,
        "scaler_mean": picker["Xmu"],
        "scaler_scale": picker["Xsd"],
        "layers": bake_layers,
        "metadata": metadata,
    }


def find_bake_bin(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--bake-bin {p} does not exist")
        return p
    on_path = shutil.which("zenpredict-bake")
    if on_path:
        return Path(on_path)
    # Fallback: try the workspace path
    workspace_candidates = [
        Path("/home/lilith/work/zen/zenanalyze/target/release/zenpredict-bake"),
        Path("/home/lilith/work/zen/zenanalyze/target/debug/zenpredict-bake"),
    ]
    for c in workspace_candidates:
        if c.exists():
            return c
    sys.exit("zenpredict-bake binary not found; pass --bake-bin or `cargo build -p zenpredict-bake`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bake-bin", default=None)
    ap.add_argument("--version-tag", default="zentrain.cvvdp_dual_head_v1")
    ap.add_argument("--keep-json", action="store_true",
                    help="Keep the BakeRequestJson for inspection")
    args = ap.parse_args()

    with open(args.inp) as f:
        picker = json.load(f)
    if picker.get("trivial"):
        print(f"  SKIP: input is a trivial picker (single cell) — not baking")
        return

    print(f"  picker codec: {picker['codec']}")
    print(f"  picker input_dim: {picker['model']['input_dim']}")
    print(f"  picker knobs: {len(picker['knobs'])}")
    print(f"  picker held-out R²: bytes {picker['r2_bytes_holdout']:.4f}  cvvdp {picker['r2_cvvdp_holdout']:.4f}")

    req = build_bake_request(picker, args.version_tag)
    print(f"  built BakeRequest: {len(req['layers'])} layers, schema_hash 0x{req['schema_hash']:016x}")

    bake_bin = find_bake_bin(args.bake_bin)
    print(f"  using bake binary: {bake_bin}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_json:
        req_path = args.out.with_suffix(".bake.json")
        with open(req_path, "w") as f:
            json.dump(req, f, indent=2)
        print(f"  wrote bake request JSON → {req_path}")
        run_args = [str(bake_bin), str(req_path), str(args.out)]
        tmppath = None
    else:
        # Temp file
        with tempfile.NamedTemporaryFile("w", suffix=".bake.json", delete=False) as tf:
            json.dump(req, tf)
            tmppath = Path(tf.name)
        run_args = [str(bake_bin), str(tmppath), str(args.out)]
    try:
        result = subprocess.run(run_args, capture_output=True, text=True)
        if result.returncode != 0:
            print("=== bake stdout ===")
            print(result.stdout)
            print("=== bake stderr ===")
            print(result.stderr)
            sys.exit(f"zenpredict-bake failed (exit {result.returncode})")
    finally:
        if tmppath is not None:
            tmppath.unlink(missing_ok=True)

    sz = args.out.stat().st_size
    print(f"✅ wrote {args.out} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
