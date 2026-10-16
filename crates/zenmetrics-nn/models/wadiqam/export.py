#!/usr/bin/env python3
"""Export WaDIQaM (dmaniry/deepIQA) checkpoints to ONNX.

Architecture (upstream fr_model.FRModel / nr_model.Model, top="weighted"):
a shared VGG-ish trunk — five blocks of 2x conv3x3+relu + maxpool2 — maps a
32x32 RGB patch to 512 features. FR concatenates (h - h_ref, h, h_ref) into
1536; NR uses h (512). fc1 -> relu -> fc2 gives the per-patch score y; a
parallel fc1_a -> relu -> fc2_a -> relu + 1e-6 gives the patch weight a.
The image score is sum(y*a)/sum(a), computed host-side.

`wadiqam_fr_<live|tid>_{trunk,head}.onnx`:
  trunk: x[P,3,32,32] -> h[P,512]; head: f[P,1536] -> ya[2,P]
  (split so the runtime can cache the reference's trunk features)
`wadiqam_nr_<live|tid>.onnx`:
  x[P,3,32,32] -> ya[2,P]

Weights come from the official Chainer HDF5 `.model` files (loaded with
h5py; chainer is only needed for the goldens oracle, not the export).

Usage:
    export.py OUT_DIR --weights DIR [--goldens TSV --ladder DIR ...] [--only COL,COL,...]

Emits wadiqam_<col>.onnx + wadiqam_<col>.meta + wadiqam_<col>_export.meta.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# (column suffix, mode, checkpoint file, sha256)
MODELS = {
    "wadiqam_fr_live": ("fr", "fr_live_weighted.model",
                        "18d671a0d6e11a1c6d198c8b03c516e8d1f33484f4732ad118d5cf57fdc04b63"),
    "wadiqam_fr_tid": ("fr", "fr_tid_weighted.model",
                       "942ee94414331c8541ee94dc679f67bf7375cfa0676e9e9fe6fe24718873e83c"),
    "wadiqam_nr_live": ("nr", "nr_live_weighted.model",
                        "cc0328e0fabdc41b0d7a251066bd844d362e0fae422c3114aecc774b76679b9b"),
    "wadiqam_nr_tid": ("nr", "nr_tid_weighted.model",
                       "b1e86eefc47a177897abf6c1e2a5f638b82f8682011926c2cdf4a070fba22b98"),
}
CONVS = [(3, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128),
         (128, 256), (256, 256), (256, 512), (512, 512)]
N_PATCH = 64  # self-check patch count


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        for i, (cin, cout) in enumerate(CONVS, 1):
            setattr(self, f"conv{i}", nn.Conv2d(cin, cout, 3, padding=1))

    def forward(self, x):
        h = x
        for blk in range(5):
            h = torch.relu(getattr(self, f"conv{2 * blk + 1}")(h))
            h = torch.relu(getattr(self, f"conv{2 * blk + 2}")(h))
            h = torch.max_pool2d(h, 2)
        return h.flatten(1)


class Head(nn.Module):
    """fc1 -> relu -> fc2 (score y) parallel to fc1_a -> relu -> fc2_a ->
    relu + 1e-6 (weight a), stacked `ya[2, P]`."""

    def __init__(self, fin):
        super().__init__()
        self.fc1 = nn.Linear(fin, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc1_a = nn.Linear(fin, 512)
        self.fc2_a = nn.Linear(512, 1)

    def forward(self, f):
        y = self.fc2(torch.relu(self.fc1(f)))
        a = torch.relu(self.fc2_a(torch.relu(self.fc1_a(f)))) + 1e-6
        return torch.stack((y.reshape(-1), a.reshape(-1)))


class Graph(nn.Module):
    """End-to-end reference graph: FR (x, x_ref) -> ya, NR (x) -> ya.
    NR ships as this whole graph; FR ships split into Trunk + Head so the
    runtime can cache the reference's patch features (`warm_reference`)."""

    def __init__(self, mode):
        super().__init__()
        self.trunk = Trunk()
        self.mode = mode
        self.head = Head(1536 if mode == "fr" else 512)

    def forward(self, x, x_ref=None):
        h = self.trunk(x)
        if self.mode == "fr":
            hr = self.trunk(x_ref)
            h = torch.cat([h - hr, h, hr], 1)
        return self.head(h)


def load_hdf5_weights(model: Graph, path: Path):
    import h5py
    f = h5py.File(str(path))
    with torch.no_grad():
        for i in range(1, 11):
            c = getattr(model.trunk, f"conv{i}")
            c.weight.copy_(torch.from_numpy(np.asarray(f[f"conv{i}/W"])))
            c.bias.copy_(torch.from_numpy(np.asarray(f[f"conv{i}/b"])))
        for name in ("fc1", "fc2", "fc1_a", "fc2_a"):
            lin = getattr(model.head, name)
            lin.weight.copy_(torch.from_numpy(np.asarray(f[f"{name}/W"])))
            lin.bias.copy_(torch.from_numpy(np.asarray(f[f"{name}/b"])))


def concat_fr(h, hr):
    """(h - h_ref, h, h_ref) along the feature axis, f32."""
    return np.concatenate([h - hr, h, hr], axis=1).astype(np.float32)


def pool(ya):
    """Upstream's image score: sum(y*a)/sum(a)."""
    return float(np.sum(ya[0] * ya[1]) / np.sum(ya[1]))


def extract_patches(arr):
    """deepIQA evaluate.py's stride-32 non-overlapping grid, NCHW f32 out."""
    gh = (arr.shape[0] - 32) // 32 + 1
    gw = (arr.shape[1] - 32) // 32 + 1
    ps = np.lib.stride_tricks.as_strided(
        arr, shape=(gh, gw, 32, 32, 3),
        strides=(arr.strides[0] * 32, arr.strides[1] * 32) + arr.strides)
    return np.transpose(ps.reshape((-1, 32, 32, 3)), (0, 3, 1, 2)).astype(np.float32)


def meta_path(out: Path, col: str):
    return out / f"{col}.meta"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--weights", required=True)
    p.add_argument("--goldens")
    p.add_argument("--ladder", action="append", default=[])
    p.add_argument("--only")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wdir = Path(a.weights)

    import onnxruntime as ort  # noqa: E402
    import onnx  # noqa: E402

    only = a.only.split(",") if a.only else list(MODELS)
    for col in only:
        mode, fname, sha = MODELS[col]
        src = wdir / fname
        assert src.exists(), f"missing checkpoint {src}"
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert digest == sha, f"{fname}: sha256 {digest} != pinned {sha}"

        torch.manual_seed(0)
        m = Graph(mode).eval()
        load_hdf5_weights(m, src)

        def export_onnx(mod, args, path, inputs, outputs, dyn):
            torch.onnx.export(
                mod, args, str(path), input_names=inputs, output_names=outputs,
                dynamic_axes=dyn, opset_version=17, dynamo=False)
            onnx.checker.check_model(str(path))
            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        torch.manual_seed(0)
        x = torch.randint(0, 256, (N_PATCH, 3, 32, 32), dtype=torch.float32)
        xr = torch.randint(0, 256, (N_PATCH, 3, 32, 32), dtype=torch.float32)
        with torch.no_grad():
            if mode == "fr":
                tya = m(x, xr).numpy()
            else:
                tya = m(x).numpy()

        if mode == "fr":
            # Split trunk + head: the runtime caches the reference's trunk
            # features across candidates (`warm_reference`).
            trunk_path = out / f"{col}_trunk.onnx"
            head_path = out / f"{col}_head.onnx"
            st = export_onnx(m.trunk, (x,), trunk_path, ["x"], ["h"],
                             {"x": {0: "p"}, "h": {0: "p"}})
            f = torch.randn(N_PATCH, 1536)
            sh = export_onnx(m.head, (f,), head_path, ["f"], ["ya"],
                             {"f": {0: "p"}, "ya": {1: "p"}})
            g_h = st.run(None, {"x": x.numpy()})[0]
            g_hr = st.run(None, {"x": xr.numpy()})[0]
            got = sh.run(None, {"f": concat_fr(g_h, g_hr)})[0]
            onnx_files = [trunk_path.name, head_path.name]
        else:
            onnx_path = out / f"{col}.onnx"
            sb = export_onnx(m, (x,), onnx_path, ["x"], ["ya"],
                             {"x": {0: "p"}, "ya": {1: "p"}})
            got = sb.run(None, {"x": x.numpy()})[0]
            onnx_files = [onnx_path.name]

        d = np.abs(got - tya).max()
        # Elementwise ya error is f32 reduction-order noise at WaDIQaM's
        # magnitudes (y ~ 100, a ~ 10); what ships is the pooled score
        # sum(y*a)/sum(a), so gate on that plus a loose ya sanity bound.
        ds = abs(pool(tya) - pool(got))
        print(f"{col}: onnx-vs-torch ya |d| {d:.3e}, pooled score |d| {ds:.3e}")
        assert d < 1e-2 and ds < 1e-4

        meta = {
            "channels": "3",
            "min_size": "32",
            "model": col,
            "upstream": "dmaniry/deepIQA @51dece994d73f43dbd2ea44d3f2ef581014194ad",
            "checkpoint": fname,
            "checkpoint_sha256": sha,
            "top": "weighted",
            "patch": "non-overlapping 32x32 stride-32 grid at native size (evaluate.py)",
            "onnx": ",".join(onnx_files),
            "inputs": ("trunk x[P,3,32,32]; head f[P,1536]" if mode == "fr"
                       else "x[P,3,32,32]"),
            "outputs": ("h[P,512]; ya[2,P]" if mode == "fr" else "ya[2,P]"),
            "pool": ("concat(h-h_ref, h, h_ref) host-side, then "
                     "sum(y*a)/sum(a)" if mode == "fr"
                     else "sum(y*a)/sum(a) host-side"),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        meta_path(out, col).write_text("\n".join(f"{k} {v}" for k, v in meta.items()) + "\n")

        # <col>_export.meta per ADDING_MODELS.md: environment versions plus
        # sha256 of every file this column's run wrote. Written inside the
        # loop so a partial --only run still leaves a matching receipt.
        def file_sha(p: Path):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        files = onnx_files + [f"{col}.meta"]
        gp = out / f"{col}_goldens.tsv"
        if gp.exists():
            files.append(gp.name)
        (out / f"{col}_export.meta").write_text(
            f"torch {torch.__version__}\n"
            f"python {platform.python_version()}\n"
            f"host {platform.node()}\n"
            f"checkpoint {fname}\ncheckpoint_sha256 {sha}\n"
            + "".join(f"sha256 {file_sha(out / fn)} {fn}\n" for fn in files))

    if a.goldens:
        gsrc = Path(a.goldens)
        shutil.copyfile(gsrc, out / gsrc.name)
        from PIL import Image
        lines = [l for l in gsrc.read_text().splitlines() if l.strip()]
        header = lines[0].split("\t")
        col = header[2]
        assert col in MODELS, f"goldens column {col} not a wadiqam model"
        mode = MODELS[col][0]
        if mode == "fr":
            st = ort.InferenceSession(str(out / f"{col}_trunk.onnx"),
                                      providers=["CPUExecutionProvider"])
            sh = ort.InferenceSession(str(out / f"{col}_head.onnx"),
                                      providers=["CPUExecutionProvider"])
        else:
            sb = ort.InferenceSession(str(out / f"{col}.onnx"),
                                      providers=["CPUExecutionProvider"])

        def find_png(image, size, kind):
            for dd in a.ladder:
                p = Path(dd) / f"{image}_{size}_{kind}.png"
                if p.exists():
                    return p
            raise FileNotFoundError(f"{image}_{size}_{kind}.png")

        def nr_score(png):
            x = extract_patches(np.asarray(Image.open(png).convert("RGB")))
            return pool(sb.run(None, {"x": x})[0])

        def score(image, size):
            dist = find_png(image, size, "dist")
            ref = find_png(image, size, "ref")
            if mode == "fr":
                h = st.run(None, {"x": extract_patches(
                    np.asarray(Image.open(dist).convert("RGB")))})[0]
                hr = st.run(None, {"x": extract_patches(
                    np.asarray(Image.open(ref).convert("RGB")))})[0]
                return pool(sh.run(None, {"f": concat_fr(h, hr)})[0]), None
            return nr_score(dist), nr_score(ref)

        worst, n = 0.0, 0
        for line in lines[1:]:
            f = line.split("\t")
            sd, sr = score(f[0], f[1])
            d = abs(sd - float(f[2]))
            if mode == "nr":
                d = max(d, abs(sr - float(f[3])))
            worst, n = max(worst, d), n + 1
        print(f"goldens check: {col} {n} pairs, worst |d| {worst:.3e}")
        assert worst < 5e-4, "export mismatches oracle goldens"


if __name__ == "__main__":
    main()
