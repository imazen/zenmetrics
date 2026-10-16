#!/usr/bin/env python3
"""WaDIQaM KonJND JND scales.

FR columns have an image-dependent identical-pair floor (the (0,h,h)
concat is not the head's zero — ~0.2 for fr_live, ~13.7 for fr_tid), so
the JND is the pooled median |score(pair) - score(ref,ref)| per
reference: run `zenmetrics batch --metric wadiqam-fr-<live|tid>` once on
pairs_abs.tsv and once on pairs_identical.tsv, pass both TSVs.

NR columns score dist and ref in one run (`PairedNoReference` emits
<col> and <col>_ref), so the JND is the pooled median
|score(dist)-score(ref)|: pass the single pairs_abs.tsv run.

usage: konjnd.py <pairs.tsv> [<identical.tsv>] <out.tsv>
  one arg   -> NR column (paired columns inside pairs.tsv)
  two args  -> FR column (pairs_abs + pairs_identical)
"""
import csv
import statistics
import sys
from pathlib import Path

args = [Path(a) for a in sys.argv[1:]]
if len(args) == 2:
    pairs_path, ident_path, out = args[0], None, args[1]
elif len(args) == 3:
    pairs_path, ident_path, out = args
else:
    sys.exit(__doc__)

with pairs_path.open() as fh:
    pairs = list(csv.DictReader(fh, delimiter='\t'))
assert len(pairs) == 1008, f'expected 1008 KonJND pairs, got {len(pairs)}'
cols = [c for c in pairs[0] if c.startswith('wadiqam_') and not c.endswith('_ref')]
assert len(cols) == 1, f'one wadiqam column per run, got {cols}'
col = cols[0]

if ident_path is None:
    deltas = [(abs(float(r[col]) - float(r[col + '_ref'])), r['codec'])
              for r in pairs]
else:
    with ident_path.open() as fh:
        ident = {r['ref_path']: float(r[col])
                 for r in csv.DictReader(fh, delimiter='\t')}
    deltas = [(abs(float(r[col]) - ident[r['ref_path']]), r['codec'])
              for r in pairs]

pooled = statistics.median(d for d, _ in deltas)
row = [col, f'{pooled:.6f}']
row += [f'{statistics.median(d for d, c in deltas if c == codec):.6f}'
        for codec in ('JPEG', 'BPG')]
with out.open('x') as fh:
    fh.write('column\tjnd\tjpeg_jnd\tbpg_jnd\n' + '\t'.join(row) + '\n')
print(*row)
