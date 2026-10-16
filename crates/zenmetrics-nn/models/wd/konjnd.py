#!/usr/bin/env python3
"""Measure five native Cool-Chic WD JND scales on all 1,008 KonJND pairs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

source, output = map(Path, sys.argv[1:])
with source.open() as stream:
    rows = list(csv.DictReader(stream, delimiter='\t'))
assert len(rows) == 1008, f'expected all 1008 canonical KonJND pairs, got {len(rows)}'
assert len({(r['ref_path'], r['dist_path']) for r in rows}) == 1008
assert {r['codec'] for r in rows} == {'JPEG', 'BPG'}
with output.open('x') as stream:
    writer = csv.writer(stream, delimiter='\t')
    writer.writerow(['column', 'pooled_median', 'JPEG_median', 'BPG_median', 'pairs'])
    for k in range(1, 6):
        col = f'wd_s{k}'
        values = [float(r[col]) for r in rows]
        assert all(math.isfinite(v) and v >= 0 for v in values)
        medians = [statistics.median(values)]
        for codec in ('JPEG', 'BPG'):
            medians.append(statistics.median(float(r[col]) for r in rows if r['codec'] == codec))
        assert all(m > 0 for m in medians)
        writer.writerow([col, *[f'{v:.17e}' for v in medians], len(rows)])
        print(col, *medians, flush=True)
output.with_suffix('.json').write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'pairs': len(rows), 'method': 'pooled median WD at observer JND threshold; identity is zero',
    'codec_counts': {c: sum(r['codec'] == c for r in rows) for c in ('JPEG', 'BPG')}}, indent=2) + '\n')
