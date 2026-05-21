#!/usr/bin/env python3
"""Per-codec cvvdp-target picker trainer (v2 — RD-regression design).

Critical finding (2026-05-20): the v15rc + multi-codec sweep data has
**exactly 1 knob_tuple_json per image** (different images stratified
across knob_tuples, but no per-image knob choice). The per-image only
varying axis is q. So the picker design is:

  Inputs:    src_feat (108) ⊕ knob_tuple one-hot ⊕ q (scalar normalised)
  Outputs:   predicted log(encoded_bytes), predicted cvvdp
  Training:  one row per (image, q, knob_tuple) measurement.

At inference:
  1. Compute src_feat for the source image.
  2. For each candidate (knob_tuple, q) in the picker's grid:
       predict (log_bytes, cvvdp) from the MLP.
  3. Pick argmin log_bytes subject to predicted cvvdp ≥ target_cvvdp.

This is the classical RD-regression picker. It's more honest than a
per-cell argmin pivot when the dataset doesn't actually sample
multiple (q, knob) per image — which v15rc + multi-codec don't.

Output: a single JSON per codec with both heads' weights + the
candidate (knob_tuple, q) grid. zenpredict-bake currently expects a
single-head bytes-regress; until the bake binary supports two-head
outputs, the JSON has a `--head-mode dual` marker so future bakers
know to read both heads.

Usage:
    python3 train_cvvdp_picker.py \\
        --parquet /mnt/v/.../zenjpeg_training.parquet \\
        --codec zenjpeg \\
        --out-dir /mnt/v/.../trained_models/ \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def _torch():
    import torch
    return torch


# ----------------------------------------------------------------------------
# Data loading + filtering
# ----------------------------------------------------------------------------


def load(parquet_path: Path, codec: str, metric_col: str = "score_cvvdp_imazen_v0_0_1") -> dict:
    """Load + filter parquet to per-row training records."""
    import pyarrow.compute as pc

    t = pq.read_table(parquet_path)
    print(f"  loaded {t.num_rows} rows × {t.num_columns} cols")
    # codec may be encoded as zenwebp, v12_zenwebp, v13_zenjpeg, v15rc_zenjpeg, etc.
    # Accept any codec column value whose tail matches the requested codec.
    import re
    canonical_re = re.compile(rf"^(v\d+(?:rc)?_)?{re.escape(codec)}$")
    codecs_present = set(t["codec"].to_pylist())
    matching = [c for c in codecs_present if canonical_re.match(c)]
    print(f"  matching codec aliases: {matching}")
    if not matching:
        print(f"  WARN no rows match codec={codec}")
        return None
    keep = pc.is_in(t["codec"], value_set=pa.array(matching))
    t = t.filter(keep)
    # The metric + ancillary score columns may be string-typed across runs
    # (mc18 shipped them as strings); cast every score_* column to double once.
    score_cols = [c for c in t.column_names if c.startswith("score_") or c == "zensim_score"]
    for c in score_cols:
        if pa.types.is_string(t[c].type):
            t = t.set_column(t.column_names.index(c), c, pc.cast(t[c], pa.float64()))
    t = t.filter(pc.is_valid(t[metric_col]))
    t = t.filter(pc.is_finite(t[metric_col]))
    t = t.filter(pc.is_valid(t["encoded_bytes"]))
    t = t.filter(pc.greater(t["encoded_bytes"], 0))
    print(f"  after filter codec~={codec} + non-null {metric_col} + bytes>0: {t.num_rows} rows")

    src_feat_cols = sorted(
        [c for c in t.column_names if c.startswith("src_feat_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    # Materialize columns
    qs = np.array(t["q"].to_pylist(), dtype=np.int32)
    bytes_ = np.array(t["encoded_bytes"].to_pylist(), dtype=np.float64)
    cvvdp = np.array(t[metric_col].to_pylist(), dtype=np.float32)
    knob_tuples = t["knob_tuple_json"].to_pylist()
    image_paths = t["image_path"].to_pylist()

    # src_feat matrix
    src_mat = np.stack(
        [np.array(t[c].to_pylist(), dtype=np.float32) for c in src_feat_cols],
        axis=1,
    )

    knobs = sorted(set(knob_tuples))
    knob_to_idx = {k: i for i, k in enumerate(knobs)}
    knob_idx = np.array([knob_to_idx[k] for k in knob_tuples], dtype=np.int32)

    images = sorted(set(image_paths))
    img_to_idx = {p: i for i, p in enumerate(images)}
    image_idx = np.array([img_to_idx[p] for p in image_paths], dtype=np.int32)

    print(f"  images: {len(images)}, knobs: {len(knobs)}")
    return {
        "src": src_mat,
        "q": qs,
        "bytes": bytes_,
        "cvvdp": cvvdp,
        "knob_idx": knob_idx,
        "knobs": knobs,
        "image_idx": image_idx,
        "images": images,
        "src_feat_names": src_feat_cols,
    }


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------


def train(
    data: dict,
    device: str = "cpu",
    hidden: tuple = (256, 256, 256),
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 4096,
    val_frac: float = 0.1,
    seed: int = 0,
):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    src = data["src"]
    q = data["q"]
    bytes_ = data["bytes"]
    cvvdp = data["cvvdp"]
    knob_idx = data["knob_idx"]
    image_idx = data["image_idx"]
    n_knobs = len(data["knobs"])
    n_imgs = len(data["images"])

    # Build input: src ⊕ knob_one_hot ⊕ q_scalar (normalised to [0,1])
    # q ranges 0-100 typically; normalise by /100.
    q_norm = (q.astype(np.float32) / 100.0).reshape(-1, 1)
    onehot = np.zeros((len(q), n_knobs), dtype=np.float32)
    onehot[np.arange(len(q)), knob_idx] = 1.0
    X_raw = np.concatenate([src.astype(np.float32), onehot, q_norm], axis=1)
    F = X_raw.shape[1]

    # NaN-impute src features
    X_impute = np.nanmean(X_raw, axis=0)
    X_impute = np.where(np.isnan(X_impute), 0.0, X_impute)
    nan_mask = np.isnan(X_raw)
    X_raw = np.where(nan_mask, X_impute, X_raw)

    # Z-score
    Xmu = X_raw.mean(axis=0)
    Xsd = X_raw.std(axis=0) + 1e-8
    X = (X_raw - Xmu) / Xsd

    # Targets
    Yb = np.log(bytes_ + 1.0).astype(np.float32)  # log-bytes
    Yc = cvvdp.astype(np.float32)
    # Z-score the targets too — easier optimisation, undo at inference.
    Yb_mu, Yb_sd = float(Yb.mean()), float(Yb.std() + 1e-8)
    Yc_mu, Yc_sd = float(Yc.mean()), float(Yc.std() + 1e-8)
    Yb_n = (Yb - Yb_mu) / Yb_sd
    Yc_n = (Yc - Yc_mu) / Yc_sd

    # Train/val split: HOLD OUT BY IMAGE (not by row). This is the
    # discipline from CLAUDE.md: a picker validated by held-out images
    # is meaningful; a row-shuffled split leaks per-image info.
    rng = np.random.default_rng(seed)
    img_perm = rng.permutation(n_imgs)
    n_val_imgs = max(1, int(n_imgs * val_frac))
    val_imgs = set(img_perm[:n_val_imgs].tolist())
    is_val = np.array([i in val_imgs for i in image_idx], dtype=bool)

    n_total = len(X)
    n_val = int(is_val.sum())
    n_tr = n_total - n_val
    print(f"  train rows: {n_tr}, val rows: {n_val} ({n_val_imgs}/{n_imgs} imgs held out)")

    Xtr = torch.tensor(X[~is_val], dtype=torch.float32, device=device)
    Ybtr = torch.tensor(Yb_n[~is_val], dtype=torch.float32, device=device)
    Yctr = torch.tensor(Yc_n[~is_val], dtype=torch.float32, device=device)
    Xva = torch.tensor(X[is_val], dtype=torch.float32, device=device)
    Ybva = torch.tensor(Yb_n[is_val], dtype=torch.float32, device=device)
    Ycva = torch.tensor(Yc_n[is_val], dtype=torch.float32, device=device)

    # Model: shared trunk → 2 heads (log-bytes, cvvdp)
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

    model = DualHead(F, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    n_no_imp = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        tr_loss = 0.0
        n_b = 0
        for s in range(0, Xtr.shape[0], batch_size):
            sl = perm[s : s + batch_size]
            opt.zero_grad()
            pb, pc = model(Xtr[sl])
            loss = mse(pb, Ybtr[sl]) + mse(pc, Yctr[sl])
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            n_b += 1
        tr_loss /= max(1, n_b)
        model.eval()
        with torch.no_grad():
            pb, pc = model(Xva)
            val_loss_b = mse(pb, Ybva).item()
            val_loss_c = mse(pc, Ycva).item()
            val_loss = val_loss_b + val_loss_c
        if ep % 10 == 0 or ep == epochs - 1:
            print(
                f"  ep {ep:3d}  tr {tr_loss:.4f}  "
                f"val_bytes {val_loss_b:.4f}  val_cvvdp {val_loss_c:.4f}"
            )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            n_no_imp = 0
        else:
            n_no_imp += 1
            if n_no_imp >= 30:
                print(f"  early stop at ep {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute final held-out R² for both heads (denormalised)
    model.eval()
    with torch.no_grad():
        pb, pc = model(Xva)
        pb_d = pb.cpu().numpy() * Yb_sd + Yb_mu
        pc_d = pc.cpu().numpy() * Yc_sd + Yc_mu
        tb_d = Ybva.cpu().numpy() * Yb_sd + Yb_mu
        tc_d = Ycva.cpu().numpy() * Yc_sd + Yc_mu
    r2_b = 1.0 - np.var(tb_d - pb_d) / (np.var(tb_d) + 1e-8)
    r2_c = 1.0 - np.var(tc_d - pc_d) / (np.var(tc_d) + 1e-8)
    print(f"  final held-out R²:  log-bytes {r2_b:.4f}  cvvdp {r2_c:.4f}")

    return {
        "model": model.cpu(),
        "F": F,
        "hidden": list(hidden),
        "Xmu": Xmu.tolist(),
        "Xsd": Xsd.tolist(),
        "X_impute": X_impute.tolist(),
        "Yb_mu": Yb_mu,
        "Yb_sd": Yb_sd,
        "Yc_mu": Yc_mu,
        "Yc_sd": Yc_sd,
        "best_val_mse": best_val,
        "r2_bytes_holdout": float(r2_b),
        "r2_cvvdp_holdout": float(r2_c),
    }


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------


def save_json(
    out_path: Path,
    codec: str,
    data: dict,
    trained: dict,
    metric_col: str = "score_cvvdp_imazen_v0_0_1",
    metric_dir: str = "higher_better",
):
    state = trained["model"].state_dict()
    layers = []
    for k, v in state.items():
        layers.append({"name": k, "shape": list(v.shape), "data": v.flatten().tolist()})

    out = {
        "codec": codec,
        "schema_version": "metric_dual_head_v2",
        "target_metric": metric_col,
        "metric_direction": metric_dir,
        "src_feat_names": data["src_feat_names"],
        "knobs": data["knobs"],
        "Xmu": trained["Xmu"],
        "Xsd": trained["Xsd"],
        "X_impute": trained["X_impute"],
        "Yb_mu": trained["Yb_mu"],
        "Yb_sd": trained["Yb_sd"],
        "Yc_mu": trained["Yc_mu"],
        "Yc_sd": trained["Yc_sd"],
        "best_val_mse": trained["best_val_mse"],
        "r2_bytes_holdout": trained["r2_bytes_holdout"],
        "r2_cvvdp_holdout": trained["r2_cvvdp_holdout"],
        "model": {
            "architecture": "dual_head_leaky_relu_mlp",
            "input_dim": trained["F"],
            "trunk_hidden": trained["hidden"],
            "heads": ["log_bytes", "cvvdp"],
            "layers": layers,
        },
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--metric-column", default="score_cvvdp_imazen_v0_0_1",
                    help="Which column to use as the quality target (e.g. score_ssim2_gpu, "
                         "score_butteraugli_pnorm3_gpu, score_iwssim_gpu, score_zensim_gpu).")
    ap.add_argument("--metric-direction", default="higher_better",
                    choices=["higher_better", "lower_better"],
                    help="Whether higher metric values mean better quality. "
                         "butteraugli + dssim are lower_better; everything else is higher_better.")
    ap.add_argument("--name-suffix", default=None,
                    help="Suffix appended to output filename. Defaults to a metric-derived short name.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", default="256,256,256")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hidden = tuple(int(x) for x in args.hidden.split(","))

    # Derive a short metric tag for output filenames.
    if args.name_suffix is not None:
        suffix = args.name_suffix
    else:
        # score_cvvdp_imazen_v0_0_1 → cvvdp
        # score_butteraugli_pnorm3_gpu → butter_p3
        # score_ssim2_gpu → ssim2
        s = args.metric_column.replace("score_", "").replace("_gpu", "")
        s = s.replace("_imazen_v0_0_1", "").replace("butteraugli_", "butter_")
        suffix = s

    print(f"=== loading {args.parquet} (metric={args.metric_column}, dir={args.metric_direction}) ===")
    data = load(args.parquet, args.codec, metric_col=args.metric_column)
    if data is None or len(data["knobs"]) < 1 or data["src"].shape[0] < 100:
        n_rows = 0 if data is None else data["src"].shape[0]
        n_k = 0 if data is None else len(data["knobs"])
        print(f"  SKIP: insufficient data (knobs={n_k}, rows={n_rows})")
        return

    print(f"\n=== training dual-head MLP ({args.codec} / {suffix}) ===")
    trained = train(
        data,
        device=args.device,
        hidden=hidden,
        epochs=args.epochs,
        seed=args.seed,
    )

    print("\n=== saving ===")
    out_path = args.out_dir / f"{args.codec}_{suffix}_picker.json"
    save_json(out_path, args.codec, data, trained,
              metric_col=args.metric_column, metric_dir=args.metric_direction)
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"  held-out R²:  log-bytes {trained['r2_bytes_holdout']:.4f}  metric {trained['r2_cvvdp_holdout']:.4f}")
    print("✅ done")


if __name__ == "__main__":
    main()
