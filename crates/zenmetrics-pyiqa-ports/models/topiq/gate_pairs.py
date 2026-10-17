"""Build the gate pair list for the TOPIQ ports.

usage: gate_pairs.py LADDER_DIR AIC_PAIRS_TSV OUT_DIR

LADDER_DIR is the zenmetrics NN size ladder (benchmarks/nn_lpips_phase0_2026-10-15/scripts/ladder.py:
<stem>_<n>_{ref,dist}.png plus non-square <stem>_<w>x<h>_*). Writes OUT_DIR/pairs.tsv
(`ref_path dist_path distorted`) with, per ladder image: 64², 256², 1024² and a 2048×1536 crop of
the 2048² pair (top-left, written to OUT_DIR), the non-square ladder pairs, a few identical pairs,
and then every AIC2026 crop pair (840×944) from AIC_PAIRS_TSV.
"""
import csv, os, re, sys
from PIL import Image

ladder, aic, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
rows = []
stems = sorted({n.split('_')[0] for n in os.listdir(ladder) if re.fullmatch(r'\d+_\d+_ref\.png', n)})
for s in stems:
    for n in (64, 256, 1024):
        r = os.path.join(ladder, f'{s}_{n}_ref.png')
        if os.path.exists(r):
            rows.append((r, r.replace('_ref', '_dist'), f'{s}_{n}'))
    big = os.path.join(ladder, f'{s}_2048_ref.png')
    if os.path.exists(big):
        pair = []
        for kind in ('ref', 'dist'):
            p = os.path.join(out, f'{s}_2048x1536_{kind}.png')
            if not os.path.exists(p):
                Image.open(os.path.join(ladder, f'{s}_2048_{kind}.png')).convert('RGB').crop((0, 0, 2048, 1536)).save(p)
            pair.append(p)
        rows.append((pair[0], pair[1], f'{s}_2048x1536'))
for n in sorted(os.listdir(ladder)):
    m = re.fullmatch(r'(\d+)_(\d+x\d+)_ref\.png', n)
    if m:
        r = os.path.join(ladder, n)
        rows.append((r, r.replace('_ref', '_dist'), f'{m[1]}_{m[2]}'))
# Identical pairs: the score the metric gives a perfect match.
for r, _, name in [x for x in rows if x[2].endswith('_256') or x[2].endswith('_1024')][:3]:
    rows.append((r, r, name + '_same'))
for r in csv.DictReader(open(aic), delimiter='\t'):
    rows.append((r['ref_path'], r['dist_path'], r['distorted']))
with open(os.path.join(out, 'pairs.tsv'), 'w') as f:
    f.write('ref_path\tdist_path\tdistorted\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')
print(len(rows), 'pairs')
