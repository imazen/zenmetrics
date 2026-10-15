#!/usr/bin/env python3
"""Score KonJND-1k (reference, image-at-PJND) pairs with upstream LPIPS (alex/vgg/squeeze) and DISTS.

Level rule: round-half-up of the per-image mean PJND in subjective_ratings.csv
(the zenmetrics/zensim corrected rule, pairs-corrected-2026-09-06/konjnd.tsv).
"""
import csv, os, sys, time, math
import numpy as np
import torch
from PIL import Image
import lpips
from DISTS_pytorch import DISTS

ROOT = "/mnt/v/datasets/KonJND-1k/KonJND-1k"
OUT = os.path.expanduser("~/tmp/konjnd-nn")
CHECK = "/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06/pairs-corrected-2026-09-06/konjnd.tsv"

torch.set_grad_enabled(False)
nthreads = int(os.environ.get("OMP_NUM_THREADS", "8"))
torch.set_num_threads(nthreads)
print(f"torch {torch.__version__} threads={torch.get_num_threads()} sys.prefix={sys.prefix}", flush=True)

def load(p):
    im = Image.open(p)
    if im.mode != "RGB":
        raise SystemExit(f"unexpected mode {im.mode} for {p}")
    return np.asarray(im, dtype=np.uint8)

# --- build pairs ---
pairs = []
with open(os.path.join(ROOT, "subjective_ratings.csv"), newline="") as f:
    for r in csv.DictReader(f):
        src = r["image_id"][:-4]
        codec = r["Compression type"]
        mean = float(r["mean"])
        level = int(math.floor(mean + 0.5))
        if codec == "JPEG":
            d = f"jpeg/{src}_JPEG_{level:03d}.jpg"
        elif codec == "BPG":
            d = f"bpg/{src}_BPG_{level:03d}.png"
        else:
            raise SystemExit(f"unknown codec {codec}")
        pairs.append((src, codec, mean, level, int(r["No. of ratings"]), f"source_image/{src}.png", d))
print(f"pairs built: {len(pairs)}  JPEG={sum(p[1]=='JPEG' for p in pairs)}  BPG={sum(p[1]=='BPG' for p in pairs)}", flush=True)

# --- cross-check against the repo's corrected pair list ---
ref_names = {}
with open(CHECK) as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        ref_names[os.path.basename(r["ref_path"])[:-4]] = os.path.basename(r["dist_path"])
mism = [p for p in pairs if ref_names.get(p[0]) != os.path.basename(p[6])]
print(f"cross-check vs {CHECK}: {len(ref_names)} rows, mismatches={len(mism)}", flush=True)
for p in mism[:10]:
    print("  MISMATCH", p, ref_names.get(p[0]), flush=True)

# --- models ---
nets = {n: lpips.LPIPS(net=n, version="0.1", verbose=False).eval() for n in ("alex", "vgg", "squeeze")}
dists = DISTS().eval()

def to_lpips(a):  # uint8 HWC -> 1x3xHxW float32 in [-1,1]
    return torch.from_numpy(a.astype(np.float32) / 255.0 * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).contiguous()

def to_dists(a):  # uint8 HWC -> 1x3xHxW float32 in [0,1]
    return torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()

def score(ra, da):
    out = {}
    rl, dl = to_lpips(ra), to_lpips(da)
    for n, m in nets.items():
        out[f"lpips_{n}"] = float(m(rl, dl).item())
    out["dists"] = float(dists(to_dists(ra), to_dists(da), require_grad=False, batch_average=False).item())
    return out

# identity sanity check
ra = load(os.path.join(ROOT, pairs[0][5]))
print("identity check (ref vs itself):", score(ra, ra), flush=True)

cols = ["src", "codec", "pjnd_mean", "level", "n_ratings", "ref_path", "dist_path", "width", "height",
        "lpips_alex", "lpips_vgg", "lpips_squeeze", "dists"]
t0 = time.time()
with open(os.path.join(OUT, "pairs_scores.tsv"), "w") as fo:
    fo.write("\t".join(cols) + "\n")
    for i, (src, codec, mean, level, nr, rp, dp) in enumerate(pairs):
        ra = load(os.path.join(ROOT, rp))
        da = load(os.path.join(ROOT, dp))
        if ra.shape != da.shape:
            raise SystemExit(f"shape mismatch {rp} {ra.shape} vs {dp} {da.shape}")
        s = score(ra, da)
        row = [src, codec, f"{mean:.2f}", str(level), str(nr), os.path.join(ROOT, rp), os.path.join(ROOT, dp),
               str(ra.shape[1]), str(ra.shape[0])] + [f"{s[k]:.9g}" for k in cols[9:]]
        fo.write("\t".join(row) + "\n")
        if i % 50 == 0:
            fo.flush()
            print(f"{i+1}/{len(pairs)} {time.time()-t0:.1f}s {src} {codec} L{level} {s}", flush=True)
print(f"done {len(pairs)} pairs in {time.time()-t0:.1f}s", flush=True)
