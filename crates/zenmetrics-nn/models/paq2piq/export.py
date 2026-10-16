#!/usr/bin/env python3
"""Export PaQ-2-PiQ (authors' RoIPoolModel) to ONNX, body + head.

`paq2piq_body.onnx`: x[1,3,h,w] -> feats[1,512,hf,wf] (resnet18 to the
last conv). `paq2piq_head.onnx`: pooled[401,512,2,2] -> score[1] (the
whole-image ROI's prediction, upstream t[0]).

The torchvision RoIPool between them runs host-side in Rust (rten has
no MaxRoiPool); this script's self-check mirrors the same quantization:
round() roi bounds / spatial_scale=1/32, +1 sizes, floor/ceil bins.

Usage:
    export.py OUT_DIR --upstream CHECKOUT --checkpoint PTH [--goldens TSV --ladder DIR ...]

Paths may also come from PAQ2PIQ_UPSTREAM and PAQ2PIQ_CHECKPOINT.
"""

import argparse
import math
import platform
import shutil
import subprocess
import sys
import os
import hashlib
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

CHECKPOINT_SHA256 = "5ece3dcd18e7bfc619ef6eee0c6e2c3b94350a623d62409f5cac95a2979b5b32"
N_ROIS = 401
SCALE = 1.0 / 32


class Body(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.body = model.body

    def forward(self, x):
        return self.body(x)


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head

    def forward(self, pooled):
        return self.head(pooled)[:1]


def block_rois(h, w):
    """Upstream get_blockwise_rois: whole image, then the 20x20 grid."""
    rois = [[0.0, 0.0, float(w), float(h)]]
    ys = np.linspace(0, h, 21)
    xs = np.linspace(0, w, 21)
    for n in range(20):
        for m in range(20):
            rois.append([xs[m], ys[n], xs[m + 1], ys[n + 1]])
    return np.asarray(rois, np.float32)


def roi_pool_np(feats, rois):
    """torchvision RoIPool(2,2,1/32) on feats[1,C,hf,wf] -> [N,C,2,2]."""
    _, c, hf, wf = feats.shape
    out = np.zeros((len(rois), c, 2, 2), np.float32)
    # torchvision RoIPool quantises roi bounds with std::round (half
    # away from zero), NOT python round (ties-to-even) — e.g. x=16.
    rnd = lambda v: int(math.floor(v + 0.5))
    for i, (x1, y1, x2, y2) in enumerate(rois):
        xs = rnd(x1 * SCALE)
        ys_ = rnd(y1 * SCALE)
        xe = rnd(x2 * SCALE)
        ye = rnd(y2 * SCALE)
        rw, rh = max(xe - xs + 1, 1), max(ye - ys_ + 1, 1)
        for ph in range(2):
            for pw in range(2):
                h0 = min(max(int(math.floor(ph * rh / 2)) + ys_, 0), hf)
                h1 = min(max(int(math.ceil((ph + 1) * rh / 2)) + ys_, 0), hf)
                w0 = min(max(int(math.floor(pw * rw / 2)) + xs, 0), wf)
                w1 = min(max(int(math.ceil((pw + 1) * rw / 2)) + xs, 0), wf)
                if h1 > h0 and w1 > w0:
                    out[i, :, ph, pw] = feats[0, :, h0:h1, w0:w1].max(axis=(1, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--upstream", default=os.environ.get("PAQ2PIQ_UPSTREAM"))
    ap.add_argument("--checkpoint", default=os.environ.get("PAQ2PIQ_CHECKPOINT"))
    ap.add_argument("--goldens", default=None)
    ap.add_argument("--ladder", action="append", default=[])
    a = ap.parse_args()
    if not a.upstream or not a.checkpoint:
        ap.error("--upstream and --checkpoint (or PAQ2PIQ_UPSTREAM/PAQ2PIQ_CHECKPOINT) are required")
    checkpoint = Path(a.checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == CHECKPOINT_SHA256, f"unexpected checkpoint sha256 {digest}"
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    from paq2piq.model import RoIPoolModel
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RoIPoolModel()
    model.load_state_dict(state["model"])
    model.eval()

    x = torch.zeros(1, 3, 480, 640)
    body, head = Body(model).eval(), Head(model).eval()
    with torch.no_grad():
        feats = body(x)
        pooled = torch.from_numpy(roi_pool_np(feats.numpy(), block_rois(480, 640)))
        ref = head(pooled).numpy()

    torch.onnx.export(body, (x,), out / "paq2piq_body.onnx", dynamo=False,
                      input_names=["x"], output_names=["feats"],
                      dynamic_axes={"x": {2: "h", 3: "w"}, "feats": {2: "hf", 3: "wf"}},
                      opset_version=16)
    torch.onnx.export(head, (pooled,), out / "paq2piq_head.onnx", dynamo=False,
                      input_names=["pooled"], output_names=["score"],
                      dynamic_axes={"pooled": {0: "n"}}, opset_version=16)
    (out / "paq2piq.meta").write_text("channels 1\nmin_size 1\n")
    import torchvision
    upstream_rev = subprocess.check_output(["git", "-C", a.upstream, "rev-parse", "HEAD"], text=True).strip()
    metadata = [f"python {platform.python_version()}", f"torch {torch.__version__}",
                f"torchvision {torchvision.__version__}", f"numpy {np.__version__}",
                f"upstream baidut/paq2piq@{upstream_rev}", f"ckpt_sha256 {digest}"]
    for name in ("paq2piq_body.onnx", "paq2piq_head.onnx"):
        metadata.append(f"sha256 {hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}")
    (out / "paq2piq_export.meta").write_text("\n".join(metadata) + "\n")
    print("wrote", (out / "paq2piq_body.onnx").stat().st_size, (out / "paq2piq_head.onnx").stat().st_size)

    import onnxruntime as ort
    sb = ort.InferenceSession(str(out / "paq2piq_body.onnx"), providers=["CPUExecutionProvider"])
    sh = ort.InferenceSession(str(out / "paq2piq_head.onnx"), providers=["CPUExecutionProvider"])

    def run(img_np):
        f = sb.run(None, {"x": img_np})[0]
        pooled = roi_pool_np(f, block_rois(img_np.shape[2], img_np.shape[3]))
        return float(sh.run(None, {"pooled": pooled})[0].flat[0]), f

    got, f = run(x.numpy())
    d = abs(got - float(ref.flat[0]))
    print(f"self-check |d| {d:.3e}")
    assert d < 1e-5

    if a.goldens:
        if Path(a.goldens).resolve() != (out / "paq2piq_goldens.tsv").resolve():
            shutil.copyfile(a.goldens, out / "paq2piq_goldens.tsv")
        from torchvision.datasets.folder import default_loader
        from torchvision import transforms
        tt = transforms.ToTensor()
        worst, n = 0.0, 0
        lines = [l for l in Path(a.goldens).read_text().splitlines() if l.strip()]
        assert lines[0] == "image\tsize\tpaq2piq\tpaq2piq_ref", "missing or invalid goldens header"
        assert len(lines) > 1, "goldens contain no pairs"
        for line in lines[1:]:
            image, size, d_g, r_g = line.split("\t")
            vals = []
            for kind in ("dist", "ref"):
                p = None
                for dd in a.ladder:
                    cand = Path(dd) / f"{image}_{size}_{kind}.png"
                    if cand.exists():
                        p = cand
                assert p, f"missing {image}_{size}_{kind}.png"
                vals.append(run(tt(default_loader(str(p))).unsqueeze(0).numpy())[0])
            dd = max(abs(vals[0] - float(d_g)), abs(vals[1] - float(r_g)))
            worst, n = max(worst, dd), n + 1
        print(f"goldens check: {n} pairs, worst |d| {worst:.3e}")
        assert worst < 5e-4, "export mismatches oracle goldens"


if __name__ == "__main__":
    main()
