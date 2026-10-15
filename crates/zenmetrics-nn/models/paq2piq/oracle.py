#!/usr/bin/env python3
"""PaQ-2-PiQ golden scores via the authors' upstream model and equivalent input ROIs.

Runs `paq2piq.model.RoIPoolModel` with the semantics of `InferenceModel.predict` from
baidut/paq2piq (MIT, pin in card.toml) against the official release checkpoint
`RoIPoolModel-fit.10.bs.120.pth` (baidut/PaQ-2-PiQ v1.0 asset; weights
CC-BY-NC-4.0 -> consent-gated in zenmetrics-nn).

Upstream eval is deterministic: ToTensor only, no resize/crop/normalize;
global score = t[0] of the 1 + 20x20 RoIPool head outputs.

Usage:
    oracle.py --upstream CHECKOUT --checkpoint PTH LADDER_DIR [LADDER_DIR2 ...] > paq2piq_goldens.tsv
Rows: stem<TAB>size<TAB>paq2piq<TAB>paq2piq_ref   (paired rows like musiq)
"""

import argparse
import sys
import os
import hashlib
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.datasets.folder import default_loader  # noqa: E402
from torchvision import transforms  # noqa: E402

CHECKPOINT_SHA256 = "5ece3dcd18e7bfc619ef6eee0c6e2c3b94350a623d62409f5cac95a2979b5b32"
BLK = (20, 20)


def get_blockwise_rois(img_h, img_w):
    """Equivalent to upstream input_block_rois(include_image=True), not verbatim."""
    a = [0, 0, img_w, img_h]
    y = np.linspace(0, img_h, num=BLK[0] + 1)
    x = np.linspace(0, img_w, num=BLK[1] + 1)
    for n in range(len(y) - 1):
        for m in range(len(x) - 1):
            a += [x[m], y[n], x[m + 1], y[n + 1]]
    return torch.tensor(a).float().view(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.environ.get("PAQ2PIQ_UPSTREAM"))
    ap.add_argument("--checkpoint", default=os.environ.get("PAQ2PIQ_CHECKPOINT"))
    ap.add_argument("ladder", nargs="+")
    a = ap.parse_args()
    if not a.upstream or not a.checkpoint:
        ap.error("--upstream and --checkpoint (or PAQ2PIQ_UPSTREAM/PAQ2PIQ_CHECKPOINT) are required")
    checkpoint = Path(a.checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == CHECKPOINT_SHA256, f"unexpected checkpoint sha256 {digest}"
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    from paq2piq.model import RoIPoolModel
    torch.manual_seed(0)
    model_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RoIPoolModel()
    model.load_state_dict(model_state["model"])
    model.eval()
    to_tensor = transforms.ToTensor()

    files = []
    for d in a.ladder:
        files += sorted(Path(d).glob("*_dist.png"))
    print("# paq2piq goldens: upstream baidut/paq2piq RoIPoolModel, t[0] global score", file=sys.stderr)
    assert files, "no *_dist.png inputs"
    print("image\tsize\tpaq2piq\tpaq2piq_ref", flush=True)
    for dist in files:
        ref = dist.with_name(dist.name.replace("_dist.png", "_ref.png"))
        image, size = dist.stem[: -len("_dist")].split("_", 1)
        row = [image, size]
        for p in (dist, ref):
            img = to_tensor(default_loader(str(p))).unsqueeze(0)
            h, w = img.shape[-2:]
            rois = get_blockwise_rois(h, w)
            model.rois = rois.view(-1, 4)
            with torch.no_grad():
                t = model(img).numpy()[0]
            row.append(f"{t[0]:.6f}")
        print("\t".join(row), flush=True)


if __name__ == "__main__":
    main()
