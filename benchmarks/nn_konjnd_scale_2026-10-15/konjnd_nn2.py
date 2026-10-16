#!/usr/bin/env python3
"""Add ST-LPIPS (alex, vgg; shift_tolerant) and DeepDC (ImageNet input norm ON) columns to KonJND-1k pairs_scores.tsv.

Reads the existing pairs (same 1,008 pairs, same round-half-up level rule) from
pairs_scores.v1.tsv and writes pairs_scores.tsv with the new columns appended.
"""
import sys, types, os, csv, time
m = types.ModuleType("IPython"); m.embed = lambda *a, **k: None; sys.modules["IPython"] = m
sys.modules.setdefault("imp", types.ModuleType("imp"))
import numpy as np
import torch
from PIL import Image
from stlpips_pytorch import stlpips
from DeepDC_PyTorch import DeepDC

OUT = os.path.expanduser("~/tmp/konjnd-nn")
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
print(f"torch {torch.__version__} threads={torch.get_num_threads()}", flush=True)

st = {n: stlpips.LPIPS(net=n, variant="shift_tolerant", verbose=False).eval() for n in ("alex", "vgg")}
ddc = DeepDC()
# Definition in use (coordinator, 2026-10-15): upstream's ImageNet input normalisation switched on.
# This matches the JPEG AIC-4 DeepDC values.
_fe = ddc.features_extractor
_fe.use_input_norm = True
_fe.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
_fe.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
ddc.eval()
print("deepdc use_input_norm", _fe.use_input_norm, _fe.mean.flatten().tolist(), _fe.std.flatten().tolist(), flush=True)

def load(p):
    im = Image.open(p)
    if im.mode != "RGB":
        raise SystemExit(f"unexpected mode {im.mode} for {p}")
    return np.asarray(im, dtype=np.uint8)

def t_pm1(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0 * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).contiguous()

def t_01(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()

NEW = ["stlpips_alex", "stlpips_vgg", "deepdc"]

def score(ra, da):
    rl, dl = t_pm1(ra), t_pm1(da)
    out = {f"stlpips_{n}": float(mdl(rl, dl).item()) for n, mdl in st.items()}
    out["deepdc"] = float(ddc(t_01(ra), t_01(da)).item())
    return out

rows = list(csv.DictReader(open(os.path.join(OUT, "pairs_scores.v1.tsv")), delimiter="\t"))
cols = list(rows[0].keys())
print(f"pairs: {len(rows)}", flush=True)

ra = load(rows[0]["ref_path"])
print("identity check (ref vs itself):", score(ra, ra), flush=True)

t0 = time.time()
with open(os.path.join(OUT, "pairs_scores.tsv.partial"), "w") as fo:
    fo.write("\t".join(cols + NEW) + "\n")
    for i, r in enumerate(rows):
        ra, da = load(r["ref_path"]), load(r["dist_path"])
        if ra.shape != da.shape:
            raise SystemExit(f"shape mismatch {r['ref_path']} {r['dist_path']}")
        s = score(ra, da)
        fo.write("\t".join([r[c] for c in cols] + [f"{s[k]:.9g}" for k in NEW]) + "\n")
        if i % 50 == 0:
            fo.flush()
            print(f"{i+1}/{len(rows)} {time.time()-t0:.1f}s {r['src']} {r['codec']} L{r['level']} {s}", flush=True)
os.replace(os.path.join(OUT, "pairs_scores.tsv.partial"), os.path.join(OUT, "pairs_scores.tsv"))
print(f"done {len(rows)} pairs in {time.time()-t0:.1f}s", flush=True)
