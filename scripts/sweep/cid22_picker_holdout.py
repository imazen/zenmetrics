#!/usr/bin/env python3
"""CID22 holdout A/B against trained cvvdp pickers.

Pipeline:
  1. Load the picker JSON for a codec.
  2. Load CID22 features JSON (output of cid22_extract).
  3. For each (image, target_cvvdp ∈ grid), scan all (knob_tuple, q)
     candidates via PyTorch forward pass on the picker model.
  4. Find argmin pred_bytes subject to pred_cvvdp ≥ target.
  5. Emit a sweep manifest TSV / JSON that the existing sweep
     infrastructure can consume to encode + measure.

This stops short of running the encodes — that's the next step,
calling out to `zen-metrics sweep` or a vast.ai chunk. But because
the encoding is small (49 images × 6 targets × 4 codecs = 1176
encodes), this can run locally on the host's GPU in 30-60 min.

Usage:
  python3 cid22_picker_holdout.py \\
    --features /tmp/cid22_features.json \\
    --picker-dir /mnt/v/zen/zensim-training/2026-05-19/trained_pickers/ \\
    --out /tmp/cid22_picker_choices.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _torch():
    import torch
    return torch


def build_model(picker):
    torch = _torch()
    F = picker["model"]["input_dim"]
    hidden = picker["model"]["trunk_hidden"]

    class DualHead(torch.nn.Module):
        def __init__(self, F, hidden):
            super().__init__()
            layers = []
            in_dim = F
            for h in hidden:
                layers += [torch.nn.Linear(in_dim, h), torch.nn.LeakyReLU(0.1)]
                in_dim = h
            self.trunk = torch.nn.Sequential(*layers)
            self.head_bytes = torch.nn.Linear(in_dim, 1)
            self.head_cvvdp = torch.nn.Linear(in_dim, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.head_bytes(z).squeeze(-1), self.head_cvvdp(z).squeeze(-1)

    model = DualHead(F, hidden)
    state = {}
    for layer in picker["model"]["layers"]:
        state[layer["name"]] = torch.tensor(layer["data"], dtype=torch.float32).reshape(layer["shape"])
    model.load_state_dict(state)
    model.eval()
    return model


def build_src_feat(features_json: dict, src_feat_names: list[str]) -> tuple[list[str], np.ndarray]:
    """Build src_feat matrix from CID22 features JSON in the picker's
    expected column order."""
    images = sorted(features_json.keys())
    M = np.zeros((len(images), len(src_feat_names)), dtype=np.float32)
    for i, img in enumerate(images):
        row = features_json[img]
        for j, name in enumerate(src_feat_names):
            v = row.get(name)
            if v is None or not isinstance(v, (int, float)):
                M[i, j] = np.nan
            else:
                M[i, j] = float(v)
    return images, M


def predict_grid(model, picker, src_feat_row, knobs, q_grid):
    """Forward the picker over all (knob, q) for one image."""
    torch = _torch()
    n_knobs = len(knobs)
    X = []
    for k_idx in range(n_knobs):
        oh = np.zeros(n_knobs, dtype=np.float32)
        oh[k_idx] = 1.0
        for q in q_grid:
            row = np.concatenate([src_feat_row.astype(np.float32), oh, [q / 100.0]])
            X.append(row)
    X = np.stack(X, axis=0)

    Xmu = np.array(picker["Xmu"], dtype=np.float32)
    Xsd = np.array(picker["Xsd"], dtype=np.float32)
    X_impute = np.array(picker["X_impute"], dtype=np.float32)
    X = np.where(np.isnan(X), X_impute, X)
    Xn = (X - Xmu) / Xsd
    with torch.no_grad():
        pb_n, pc_n = model(torch.tensor(Xn, dtype=torch.float32))
    pb = pb_n.numpy() * picker["Yb_sd"] + picker["Yb_mu"]
    pc = pc_n.numpy() * picker["Yc_sd"] + picker["Yc_mu"]
    return pb.reshape(n_knobs, len(q_grid)), pc.reshape(n_knobs, len(q_grid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--picker-dir", required=True, type=Path)
    ap.add_argument("--codecs", nargs="+",
                    default=["zenwebp", "zenavif", "zenjxl", "zenjpeg"])
    ap.add_argument("--target-grid",
                    default="7.0,7.5,8.0,8.5,9.0,9.25,9.5,9.75,9.9")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.features) as f:
        features_json = json.load(f)
    print(f"loaded {len(features_json)} CID22 images")

    targets = [float(x) for x in args.target_grid.split(",")]

    all_choices = {}
    for codec in args.codecs:
        picker_path = args.picker_dir / f"{codec}_cvvdp_picker.json"
        print(f"\n=== {codec} ===")
        with open(picker_path) as f:
            picker = json.load(f)
        if picker.get("trivial"):
            print("  SKIP: trivial picker")
            continue

        model = build_model(picker)
        knobs = picker["knobs"]
        src_feat_names = picker["src_feat_names"]
        images, src_mat = build_src_feat(features_json, src_feat_names)

        # Choose q-grid: use 5..95 step 1 (broad)
        q_grid = list(range(5, 96))
        codec_choices = []

        for i, img in enumerate(images):
            for target in targets:
                pb, pc = predict_grid(model, picker, src_mat[i], knobs, q_grid)
                mask = pc >= target
                if not mask.any():
                    # Best-effort: maximize predicted cvvdp
                    idx = np.unravel_index(np.argmax(pc), pc.shape)
                    note = "unreachable"
                else:
                    pb_masked = np.where(mask, pb, np.inf)
                    idx = np.unravel_index(np.argmin(pb_masked), pb_masked.shape)
                    note = "ok"
                pick_knob = knobs[idx[0]]
                pick_q = q_grid[idx[1]]
                codec_choices.append({
                    "image": img,
                    "target_cvvdp": target,
                    "picked_knob": pick_knob,
                    "picked_q": pick_q,
                    "predicted_bytes": float(np.exp(pb[idx]) - 1.0),
                    "predicted_cvvdp": float(pc[idx]),
                    "note": note,
                })
        print(f"  {len(codec_choices)} choices ({len(targets)} targets × {len(images)} images)")
        all_choices[codec] = codec_choices

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_choices, f, indent=2)
    total = sum(len(v) for v in all_choices.values())
    print(f"\n✅ wrote {args.out} ({total} total choices across {len(all_choices)} codecs)")


if __name__ == "__main__":
    main()
