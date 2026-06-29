#!/usr/bin/env python3
"""Generate a handful of downscaled sizes per picked origin (Lanczos, downscale-only).
Output filenames lead with the origin id so origin_split.split_of() still resolves train."""
import json, os, sys
from PIL import Image
sys.path.insert(0, '/home/lilith/work/zen/zenmetrics/scripts/picker')
import origin_split

SCRATCH = '/tmp/claude-1000/-home-lilith-work-zen-zenmetrics/51b72165-bbdf-44d4-9b34-be022d2f50f5/scratchpad'
OUT = os.path.join(SCRATCH, 'sources')
os.makedirs(OUT, exist_ok=True)
picks = json.load(open(os.path.join(SCRATCH, 'picks16.json')))
SIZES = [256, 512, 768]  # longest-edge targets, downscale-only

rows = []
for r in picks:
    src = r['image_path']
    oid = r['origin_id']
    im = Image.open(src).convert('RGB')
    w, h = im.size
    longe = max(w, h)
    for tgt in SIZES:
        if tgt >= longe:
            continue  # skip upscale (dense-sampling rule)
        scale = tgt / longe
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        out_name = f'o_{oid}.scale{nw}x{nh}.png'
        out_path = os.path.join(OUT, out_name)
        im.resize((nw, nh), Image.LANCZOS).save(out_path)
        assert origin_split.split_of(out_name) == 'train', out_name
        rows.append({'origin_id': oid, 'content_class': r['content_class'],
                     'cluster': r['cluster'], 'src': src, 'out': out_path,
                     'w': nw, 'h': nh, 'longedge': tgt})

json.dump(rows, open(os.path.join(SCRATCH, 'sources.json'), 'w'), indent=2)
print(f'generated {len(rows)} source images in {OUT}')
import collections
print('per longedge:', dict(collections.Counter(x['longedge'] for x in rows)))
print('all train-split:', all(origin_split.split_of(os.path.basename(x['out'])) == 'train' for x in rows))
