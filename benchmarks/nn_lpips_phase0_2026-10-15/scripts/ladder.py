"""Build the reference/distorted size ladder for NN-metric parity and timing.

Center-crop each source to a square, Lanczos-downscale to each size (never
upscale), and make a deterministic distortion: Gaussian blur (sigma 1.0) plus
Gaussian noise (sigma 6, seed fixed per image+size). Writes PNGs and a manifest.
"""
import hashlib, os, sys
import numpy as np
from PIL import Image, ImageFilter, ImageOps

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
rows = []
for name in sorted(os.listdir(src_dir)):
    if not name.endswith('.jpg'):
        continue
    stem = name[:-4]
    im = ImageOps.exif_transpose(Image.open(os.path.join(src_dir, name))).convert('RGB')
    w, h = im.size
    s = min(w, h)
    sq = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    for n in SIZES:
        if n > s:
            continue
        ref = sq.resize((n, n), Image.LANCZOS)
        blur = ref.filter(ImageFilter.GaussianBlur(1.0))
        rng = np.random.default_rng(int(hashlib.sha256(f'{stem}-{n}'.encode()).hexdigest()[:8], 16))
        a = np.asarray(blur, dtype=np.float32) + rng.normal(0, 6, (n, n, 3))
        dist = Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))
        for kind, img in (('ref', ref), ('dist', dist)):
            p = os.path.join(out_dir, f'{stem}_{n}_{kind}.png')
            img.save(p)
            rows.append((stem, n, kind, os.path.basename(p), hashlib.sha256(open(p, 'rb').read()).hexdigest()))
with open(os.path.join(out_dir, 'MANIFEST.tsv'), 'w') as f:
    f.write('image\tsize\tkind\tfile\tsha256\n')
    for r in rows:
        f.write('\t'.join(map(str, r)) + '\n')
print(len(rows), 'files')
