"""Parity ladder for the zenmetrics-pyiqa-ports models: 64², 256², 840×944, 1024², 2048×1536 (w×h).

Usage: ladder.py <src_dir> <out_dir>
Sources: the three 4096² imazen-26 references of zenmetrics'
benchmarks/nn_lpips_phase0_2026-10-15 ladder (`<id>_4096_ref.png`, ids 1415,
2207, 2406). Each source is centre-cropped to the target aspect, Lanczos-
downscaled (never upscaled), and distorted like that ladder: Gaussian blur
σ 1.0 plus Gaussian noise σ 6, seed sha256('<id>-<w>x<h>'). Writes PNGs,
pairs.tsv (ref_path, dist_path, distorted) and MANIFEST.tsv with sha256s.
"""
import hashlib, os, sys
import numpy as np
from PIL import Image, ImageFilter

SIZES = [(64, 64), (256, 256), (840, 944), (1024, 1024), (2048, 1536)]
IDS = ['1415', '2207', '2406']
src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
rows, pairs = [], []
for stem in IDS:
    src = Image.open(os.path.join(src_dir, f'{stem}_4096_ref.png')).convert('RGB')
    sw, sh = src.size
    for w, h in SIZES:
        # Largest centred crop with the target aspect.
        if sw * h >= sh * w:
            ch, cw = sh, sh * w // h
        else:
            cw, ch = sw, sw * h // w
        x0, y0 = (sw - cw) // 2, (sh - ch) // 2
        ref = src.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)
        blur = ref.filter(ImageFilter.GaussianBlur(1.0))
        rng = np.random.default_rng(int(hashlib.sha256(f'{stem}-{w}x{h}'.encode()).hexdigest()[:8], 16))
        a = np.asarray(blur, dtype=np.float32) + rng.normal(0, 6, (h, w, 3))
        dist = Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))
        paths = {}
        for kind, img in (('ref', ref), ('dist', dist)):
            p = os.path.join(out_dir, f'{stem}_{w}x{h}_{kind}.png')
            img.save(p)
            paths[kind] = p
            rows.append((stem, f'{w}x{h}', kind, os.path.basename(p),
                         hashlib.sha256(open(p, 'rb').read()).hexdigest()))
        pairs.append((paths['ref'], paths['dist'], os.path.basename(paths['dist'])))
with open(os.path.join(out_dir, 'MANIFEST.tsv'), 'w') as f:
    f.write('image\tsize\tkind\tfile\tsha256\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')
with open(os.path.join(out_dir, 'pairs.tsv'), 'w') as f:
    f.write('ref_path\tdist_path\tdistorted\n')
    for r in pairs:
        f.write('\t'.join(r) + '\n')
print(len(pairs), 'pairs')
