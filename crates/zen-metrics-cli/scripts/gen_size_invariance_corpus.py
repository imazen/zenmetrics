#!/usr/bin/env python3
"""Generate the size-invariance validation corpus.

3 inputs (frymire, gradient, canon5d) x ~150 size configs (square + 1:3) x
10 distortions (JPEG q90/80/60/40/25 via PIL/libjpeg, JXL d0/1/2/3/6 via cjxl).
Each cell: resize source -> ref PNG; encode+decode -> dist PNG. Pairs keyed by
(image, WxH, distortion). Output under /mnt/v/zen/size-invariance-corpus/corpus.
"""
import os, subprocess, sys
from PIL import Image

CJXL = os.path.expanduser("~/work/jxl-efforts/libjxl/build/tools/cjxl")
DJXL = os.path.expanduser("~/work/jxl-efforts/libjxl/build/tools/djxl")
OUT = "/mnt/v/zen/size-invariance-corpus/corpus"
FRYMIRE = "/home/lilith/work/codec-corpus/imageflow/test_inputs/frymire.png"
CANON = "/home/lilith/work/codec-corpus/imageflow/test_inputs/canon_5d_srgb.jpg"

SUBSET = "--subset" in sys.argv

def size_grid():
    dims = list(range(1, 33)) + list(range(33, 64, 3)) + list(range(64, 257, 7)) + [256]
    dims = sorted(set(dims))
    cfgs = []
    for d in dims:
        cfgs.append((d, d))          # square
        cfgs.append((d, 3 * d))      # 1:3 ratio (portrait)
    return cfgs

def gen_gradient(w, h):
    import numpy as np
    xs = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    r = np.broadcast_to(xs, (h, w)).astype("float32")
    g = np.broadcast_to(ys, (h, w)).astype("float32")
    b = ((xs + ys) * 0.5).astype("float32")
    b = np.broadcast_to(b, (h, w)).astype("float32")
    arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype("uint8")
    return Image.fromarray(arr, "RGB")

def crop_to_aspect(im, w, h):
    tw, th = im.size
    target = w / h
    cur = tw / th
    if cur > target:  # too wide -> crop width
        nw = int(th * target); x0 = (tw - nw)//2
        im = im.crop((x0, 0, x0+nw, th))
    else:             # too tall -> crop height
        nh = int(tw / target); y0 = (th - nh)//2
        im = im.crop((0, y0, tw, y0+nh))
    return im

def jpeg_roundtrip(ref, q, path):
    tmp = path + ".jpg"
    ref.save(tmp, "JPEG", quality=q)
    Image.open(tmp).convert("RGB").save(path)
    os.remove(tmp)

def jxl_roundtrip(ref_png, d, path):
    tmp = path + ".jxl"
    subprocess.run([CJXL, ref_png, tmp, "-d", str(d)], capture_output=True, check=True)
    subprocess.run([DJXL, tmp, path], capture_output=True, check=True)
    Image.open(path).convert("RGB").save(path)
    os.remove(tmp)

JPEG_Q = [90, 80, 60, 40, 25]
JXL_D = [0, 1, 2, 3, 6]

def main():
    sources = {
        "frymire": Image.open(FRYMIRE).convert("RGB"),
        "gradient": "GRADIENT",
        "canon5d": Image.open(CANON).convert("RGB"),
    }
    cfgs = size_grid()
    if SUBSET:
        sources = {"frymire": sources["frymire"]}
        cfgs = [(8, 8), (32, 32), (33, 99), (64, 64), (96, 288)]
        jq, jd = [80], [1, 0]
    else:
        jq, jd = JPEG_Q, JXL_D
    total = len(sources) * len(cfgs) * (len(jq) + len(jd))
    print(f"sources={list(sources)} sizes={len(cfgs)} distortions={len(jq)+len(jd)} total_cells={total}")
    done = 0
    for name, src in sources.items():
        for (w, h) in cfgs:
            ref = gen_gradient(w, h) if src == "GRADIENT" else crop_to_aspect(src, w, h).resize((w, h), Image.LANCZOS)
            rdir = f"{OUT}/{name}/ref"; os.makedirs(rdir, exist_ok=True)
            ref_path = f"{rdir}/{w}x{h}.png"; ref.save(ref_path)
            for q in jq:
                dd = f"{OUT}/{name}/jpeg_q{q}"; os.makedirs(dd, exist_ok=True)
                p=f"{dd}/{w}x{h}.png"
                if not os.path.exists(p): jpeg_roundtrip(ref, q, p)
                done += 1
            for d in jd:
                dd = f"{OUT}/{name}/jxl_d{d}"; os.makedirs(dd, exist_ok=True)
                p=f"{dd}/{w}x{h}.png"
                if not os.path.exists(p): jxl_roundtrip(ref_path, d, p)
                done += 1
        print(f"  {name}: done ({done}/{total})")
    print(f"DONE {done} cells -> {OUT}")

if __name__ == "__main__":
    main()
