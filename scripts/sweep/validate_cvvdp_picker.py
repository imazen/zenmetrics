#!/usr/bin/env python3
"""Validate a trained cvvdp picker against held-out data.

For each (image, target_cvvdp ∈ grid):
  1. Run picker: scan all (knob, q) candidates, predict (bytes, cvvdp).
     Pick argmin bytes subject to predicted cvvdp ≥ target.
  2. Compare to actual measurements:
     - Did the picked (knob, q) actually meet the cvvdp target?
     - What's the bytes ratio (picked actual / oracle actual)?
     - Oracle = smallest actual-bytes measurement that meets target.

Outputs a per-(codec, target) summary table.

Usage:
  python3 validate_cvvdp_picker.py \\
    --picker-dir /mnt/v/zen/zensim-training/2026-05-19/trained_pickers/ \\
    --parquet-dir /mnt/v/zen/zensim-training/2026-05-19/per_codec/ \\
    --codec zenjpeg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

def _torch():
    import torch
    return torch


def load_picker(path: Path):
    with open(path) as f:
        p = json.load(f)
    if p.get("trivial"):
        raise RuntimeError(f"picker {path} is trivial — cannot validate")
    return p


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


def predict_grid(model, src_feat, knobs, q_grid, picker):
    """Run model on cartesian (knobs × q_grid) for one image's src_feat.
    Returns (log_bytes, cvvdp) arrays of shape (n_knobs, n_q)."""
    torch = _torch()
    n_knobs = len(knobs)
    n_q = len(q_grid)
    # Build inputs
    X = []
    for k_idx in range(n_knobs):
        onehot = np.zeros(n_knobs, dtype=np.float32)
        onehot[k_idx] = 1.0
        for q in q_grid:
            q_norm = q / 100.0
            row = np.concatenate([src_feat.astype(np.float32), onehot, [q_norm]])
            X.append(row)
    X = np.stack(X, axis=0)

    # Impute + normalise (use picker's saved vectors)
    X_impute = np.array(picker["X_impute"], dtype=np.float32)
    Xmu = np.array(picker["Xmu"], dtype=np.float32)
    Xsd = np.array(picker["Xsd"], dtype=np.float32)
    nan_mask = np.isnan(X)
    X = np.where(nan_mask, X_impute, X)
    Xn = (X - Xmu) / Xsd

    with torch.no_grad():
        pb_norm, pc_norm = model(torch.tensor(Xn, dtype=torch.float32))
    pb = pb_norm.numpy() * picker["Yb_sd"] + picker["Yb_mu"]
    pc = pc_norm.numpy() * picker["Yc_sd"] + picker["Yc_mu"]
    return pb.reshape(n_knobs, n_q), pc.reshape(n_knobs, n_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker-dir", required=True, type=Path)
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument("--codec", required=True)
    ap.add_argument(
        "--target-grid",
        default="7.0,7.5,8.0,8.5,9.0,9.25,9.5,9.75,9.9",
    )
    ap.add_argument("--max-images", type=int, default=200,
                    help="Sample at most this many images for validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    picker_path = args.picker_dir / f"{args.codec}_cvvdp_picker.json"
    parquet_path = args.parquet_dir / f"{args.codec}_training.parquet"
    print(f"loading picker {picker_path}")
    picker = load_picker(picker_path)
    model = build_model(picker)
    knobs = picker["knobs"]
    print(f"  picker: {len(knobs)} knobs, hidden={picker['model']['trunk_hidden']}")
    print(f"  held-out R²: bytes {picker['r2_bytes_holdout']:.4f}  cvvdp {picker['r2_cvvdp_holdout']:.4f}")

    print(f"\nloading per-codec parquet {parquet_path}")
    t = pq.read_table(parquet_path)
    import pyarrow.compute as pc
    t = t.filter(pc.equal(t["codec"], args.codec))
    t = t.filter(pc.is_valid(t["score_cvvdp_imazen_v0_0_1"]))
    t = t.filter(pc.is_finite(t["score_cvvdp_imazen_v0_0_1"]))

    image_paths = t["image_path"].to_pylist()
    knob_tuples = t["knob_tuple_json"].to_pylist()
    qs = np.array(t["q"].to_pylist(), dtype=np.int32)
    bytes_ = np.array(t["encoded_bytes"].to_pylist(), dtype=np.float32)
    cvvdp = np.array(t["score_cvvdp_imazen_v0_0_1"].to_pylist(), dtype=np.float32)

    src_feat_cols = [c for c in t.column_names if c.startswith("src_feat_")]
    src_feat_cols.sort(key=lambda c: int(c.split("_")[-1]))
    src_per_row = np.stack(
        [np.array(t[c].to_pylist(), dtype=np.float32) for c in src_feat_cols],
        axis=1,
    )

    # Sample images
    unique_imgs = sorted(set(image_paths))
    rng = np.random.default_rng(args.seed)
    sample = list(rng.choice(unique_imgs, size=min(args.max_images, len(unique_imgs)), replace=False))
    print(f"  validating on {len(sample)} sampled images")

    # Build a per-image -> (q-list, bytes-list, cvvdp-list, knob-list)
    by_img = {}
    src_by_img = {}
    for i, img in enumerate(image_paths):
        if img not in sample:
            continue
        by_img.setdefault(img, []).append((qs[i], bytes_[i], cvvdp[i], knob_tuples[i]))
        if img not in src_by_img:
            src_by_img[img] = src_per_row[i]

    targets = [float(x) for x in args.target_grid.split(",")]
    knob_to_idx = {k: i for i, k in enumerate(knobs)}

    # Pick a q-grid spanning the codec's measured q range
    q_min, q_max = int(qs.min()), int(qs.max())
    q_grid = list(range(q_min, q_max + 1))
    print(f"  q-grid: {q_min}..{q_max} ({len(q_grid)} points)")

    print(f"\n{'target':>7} {'covered':>10} {'hit_rate':>10} {'pick_q_p50':>12} {'oracle_q_p50':>12} {'bytes_ratio_p50':>16}")
    print("-" * 75)
    for target in targets:
        n_total = 0
        n_hit = 0
        n_covered = 0  # actual measurements include any at-or-above target
        pick_qs = []
        oracle_qs = []
        ratios = []
        for img, rows in by_img.items():
            # Oracle = min(bytes) over rows where actual cvvdp >= target
            oracle = None
            for q, b, cv, k in rows:
                if cv >= target and (oracle is None or b < oracle[1]):
                    oracle = (q, b, cv, k)
            if oracle is None:
                continue  # image unreachable
            n_covered += 1

            # Picker: scan all (knob, q), pick argmin pred_bytes s.t. pred_cvvdp >= target
            src_feat = src_by_img[img]
            pb, pc = predict_grid(model, src_feat, knobs, q_grid, picker)
            mask = pc >= target
            if not mask.any():
                # No prediction reaches target — fall back to argmax pred_cvvdp
                idx = np.unravel_index(np.argmax(pc), pc.shape)
            else:
                pb_masked = np.where(mask, pb, np.inf)
                idx = np.unravel_index(np.argmin(pb_masked), pb_masked.shape)
            pick_knob = knobs[idx[0]]
            pick_q = q_grid[idx[1]]

            # Find the actual measurement for the picked (knob, q) — if it exists.
            # The picker's choice might not be a row we actually measured, since
            # the corpus has only 1 knob per image. So we look for the closest
            # measured q in the picker's chosen knob, OR fall back to closest q
            # in any knob.
            best_actual = None
            for q, b, cv, k in rows:
                if k == pick_knob:
                    if best_actual is None or abs(q - pick_q) < abs(best_actual[0] - pick_q):
                        best_actual = (q, b, cv)
            if best_actual is None:
                # Knob not measured for this image — fall back to closest q in any
                for q, b, cv, k in rows:
                    if best_actual is None or abs(q - pick_q) < abs(best_actual[0] - pick_q):
                        best_actual = (q, b, cv)

            n_total += 1
            if best_actual[2] >= target:
                n_hit += 1
            pick_qs.append(pick_q)
            oracle_qs.append(oracle[0])
            ratios.append(best_actual[1] / oracle[1])

        if n_total == 0:
            print(f"{target:>7.2f} {n_covered:>10d} {'n/a':>10s} {'n/a':>12s} {'n/a':>12s} {'n/a':>16s}")
            continue
        hit_rate = n_hit / n_total
        print(
            f"{target:>7.2f} {n_covered:>10d} {hit_rate:>10.1%} "
            f"{int(np.median(pick_qs)):>12d} {int(np.median(oracle_qs)):>12d} "
            f"{np.median(ratios):>16.3f}"
        )


if __name__ == "__main__":
    main()
