#!/usr/bin/env python3
"""Cross-metric picker agreement analysis.

For one codec, load all 5 metric pickers (cvvdp/ssim2/butter_p3/iwssim/zensim_gpu)
and ask: for a fixed source image and a fixed quality bin, do the
five pickers pick the same (knob_tuple, q)?

The "fixed quality bin" requires aligning targets across metrics
(they live on different native scales). We use **per-metric quantile
targets**: for each picker, pick at quantiles {0.1, 0.3, 0.5, 0.7,
0.9} of the codec's measured metric distribution. This produces
five comparable "low / mid / high quality" pick rows per image per
picker.

Reports:

  - **pick agreement rate**: fraction of (image, quality_bin) cells
    where all 5 pickers pick the same (knob, q).
  - **pairwise overlap**: |intersect| / |union| of pick sets across
    each metric pair.
  - **bytes / metric Pareto distance**: average ratio of picked-bytes
    relative to the per-image oracle for each metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq


def _torch():
    import torch
    return torch


def build_model(picker):
    torch = _torch()
    F = picker["model"]["input_dim"]
    hidden = picker["model"]["trunk_hidden"]

    class DH(torch.nn.Module):
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

    m = DH(F, hidden)
    state = {}
    for layer in picker["model"]["layers"]:
        state[layer["name"]] = torch.tensor(layer["data"], dtype=torch.float32).reshape(layer["shape"])
    m.load_state_dict(state)
    m.eval()
    return m


def predict_grid(model, picker, src_feat_row, knobs, q_grid):
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
        pb_n, pm_n = model(torch.tensor(Xn, dtype=torch.float32))
    pb = pb_n.numpy() * picker["Yb_sd"] + picker["Yb_mu"]
    pm = pm_n.numpy() * picker["Yc_sd"] + picker["Yc_mu"]
    return pb.reshape(n_knobs, len(q_grid)), pm.reshape(n_knobs, len(q_grid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picker-dir", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path,
                    help="CID22 features JSON from cid22_extract.")
    ap.add_argument("--codec", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["cvvdp", "ssim2", "butter_p3", "iwssim", "zensim_gpu"])
    ap.add_argument("--quantiles", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--max-images", type=int, default=49)
    ap.add_argument("--parquet", type=Path, required=True,
                    help="Per-codec training parquet to derive per-metric "
                         "quantile targets from the codec's actual range.")
    args = ap.parse_args()

    quantiles = [float(x) for x in args.quantiles.split(",")]

    # 1. Load metric quantile targets from the parquet's measured values.
    t = pq.read_table(args.parquet, columns=["codec"] + [f"score_{m}" if not m.endswith("_p3") else "score_butteraugli_pnorm3_gpu" for m in [
        "cvvdp_imazen_v0_0_1", "ssim2_gpu", "butter_p3", "iwssim_gpu", "zensim_gpu"
    ]])
    metric_col_map = {
        "cvvdp": "score_cvvdp_imazen_v0_0_1",
        "ssim2": "score_ssim2_gpu",
        "butter_p3": "score_butteraugli_pnorm3_gpu",
        "iwssim": "score_iwssim_gpu",
        "zensim_gpu": "score_zensim_gpu",
    }
    import pyarrow as pa
    import pyarrow.compute as pc
    # cast strings to double if needed
    for c in t.column_names:
        if c.startswith("score_") and pa.types.is_string(t[c].type):
            t = t.set_column(t.column_names.index(c), c, pc.cast(t[c], pa.float64()))
    targets = {}
    for m in args.metrics:
        col = metric_col_map[m]
        vals = np.array(t[col].to_pylist(), dtype=float)
        vals = vals[np.isfinite(vals)]
        targets[m] = np.quantile(vals, quantiles)
        print(f"  {m}: quantile targets = {[f'{v:.3f}' for v in targets[m]]}")

    # 2. Load all metric pickers + their models.
    print("\nloading pickers")
    pickers = {}
    models = {}
    for m in args.metrics:
        path = args.picker_dir / f"{args.codec}_{m}_picker.json"
        if not path.exists():
            print(f"  MISSING {path}")
            continue
        with open(path) as f:
            p = json.load(f)
        pickers[m] = p
        models[m] = build_model(p)
        print(f"  loaded {m}: input_dim={p['model']['input_dim']}, knobs={len(p['knobs'])}")

    # All pickers MUST share the same knob set + src_feat names.
    knob_sets = [tuple(p["knobs"]) for p in pickers.values()]
    if len(set(knob_sets)) != 1:
        print(f"  WARN: knob_set mismatch across metric pickers; using first")
    ref = next(iter(pickers.values()))
    knobs = ref["knobs"]
    src_feat_names = ref["src_feat_names"]

    # 3. Load CID22 src features.
    with open(args.features) as f:
        feat_json = json.load(f)
    images = sorted(feat_json.keys())[: args.max_images]
    print(f"\nvalidating on {len(images)} CID22 images")

    # q-grid: 5..95 step 1
    q_grid = list(range(5, 96))

    # 4. For each image × each quantile bin → compute pick per metric.
    # Record pick agreement.
    n_bins = len(quantiles)
    agree_all = 0
    n_total = 0
    pairwise = defaultdict(lambda: [0, 0])  # (m1, m2) -> [agree, total]
    pick_q_per_metric = defaultdict(list)
    pick_knob_per_metric = defaultdict(list)
    metrics_run = list(pickers.keys())

    for img in images:
        src_row = np.array([feat_json[img].get(c, np.nan) for c in src_feat_names], dtype=np.float32)
        # Compute predictions per metric (predict once, then loop over targets).
        preds = {}
        for m in metrics_run:
            pb, pm = predict_grid(models[m], pickers[m], src_row, knobs, q_grid)
            preds[m] = (pb, pm)

        for qi, target_vec in enumerate(quantiles):
            picks = {}
            for m in metrics_run:
                pb, pm = preds[m]
                tgt = targets[m][qi]
                direction = pickers[m].get("metric_direction", "higher_better")
                if direction == "higher_better":
                    mask = pm >= tgt
                else:
                    mask = pm <= tgt
                if not mask.any():
                    # Fall back to the "closest to target" cell
                    if direction == "higher_better":
                        idx = np.unravel_index(np.argmax(pm), pm.shape)
                    else:
                        idx = np.unravel_index(np.argmin(pm), pm.shape)
                else:
                    pb_masked = np.where(mask, pb, np.inf)
                    idx = np.unravel_index(np.argmin(pb_masked), pb_masked.shape)
                picks[m] = (idx[0], q_grid[idx[1]])  # (knob_idx, q)
                pick_q_per_metric[m].append(q_grid[idx[1]])
                pick_knob_per_metric[m].append(idx[0])

            # Agreement
            n_total += 1
            unique = set(picks.values())
            if len(unique) == 1:
                agree_all += 1
            for i, m1 in enumerate(metrics_run):
                for m2 in metrics_run[i+1:]:
                    pairwise[(m1, m2)][1] += 1
                    if picks[m1] == picks[m2]:
                        pairwise[(m1, m2)][0] += 1

    # 5. Report.
    print(f"\n=== picks across {n_total} (image × quantile) cells ===")
    print(f"All-5-agree (exact (knob,q)): {agree_all}/{n_total} = {agree_all/max(1,n_total):.1%}")

    print(f"\nPairwise (knob,q) match rate:")
    print("              " + " ".join(f"{m[:8]:>9}" for m in metrics_run))
    for m1 in metrics_run:
        row = []
        for m2 in metrics_run:
            if m1 == m2:
                row.append("    1.000")
            else:
                key = (m1, m2) if (m1, m2) in pairwise else (m2, m1)
                a, t_ = pairwise[key]
                row.append(f"{a/max(1,t_):>9.3f}")
        print(f"  {m1[:10]:>10} {' '.join(row)}")

    print(f"\nq-distribution per metric (p10/p50/p90):")
    for m in metrics_run:
        arr = np.array(pick_q_per_metric[m])
        print(f"  {m:>10}: p10={np.percentile(arr,10):.0f}  p50={np.percentile(arr,50):.0f}  p90={np.percentile(arr,90):.0f}")


if __name__ == "__main__":
    main()
