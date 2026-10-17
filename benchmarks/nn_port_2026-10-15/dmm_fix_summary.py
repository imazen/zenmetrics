#!/usr/bin/env python3
"""Summarize a complete nn-parity-dmm log against frozen golden scores.

Usage: dmm_fix_summary.py LOG MKL_TSV NETLIB_TSV OUTPUT_TSV
The output is evidence only; this script never changes golden tolerances.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

log, mkl_path, netlib_path, out = map(Path, sys.argv[1:])

def rows(path):
    with path.open() as f:
        return list(csv.DictReader((ln for ln in f if not ln.startswith('#')), delimiter='\t'))

mkl = {(r['image'], r['size']): float(r['dmm']) for r in rows(mkl_path)}
netlib = {(r['image'], r['size']): r for r in rows(netlib_path)}
pattern = re.compile(r'(?:^|\.\.\. )dmm (\S+) (\S+): got (\S+) netlib ', re.M)
got = {(stem, size): float(score) for stem, size, score in pattern.findall(log.read_text())}
assert got.keys() == mkl.keys() == netlib.keys(), 'Missing or extra golden/log rows'
result = []
for key in mkl:
    score = got[key]
    golden = netlib[key]
    result.append((*key, score, abs(score - mkl[key]),
                   abs(score - float(golden['dmm'])), golden['kind'],
                   float(golden['tol'])))
with out.open('w') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    writer.writerow(['image', 'size', 'dmm', 'mkl_error', 'netlib_error', 'kind', 'tolerance'])
    writer.writerows(result)
for name, ix in [('MKL', 3), ('netlib', 4)]:
    values = [r[ix] for r in result]
    print(f'{name}: n={len(values)} median={statistics.median(values):.9e} max={max(values):.9e}')
scale_path = Path(__file__).resolve().parents[2] / 'crates/zenmetrics-nn/models/jnd_scale.tsv'
scale = next(float(f[1]) for ln in scale_path.read_text().splitlines()
             if (f := ln.split('\t'))[0] == 'dmm')
gate = 1e-3 * scale
print(f'Within 1e-3 JND: {sum(r[4] <= gate for r in result)}/{len(result)}')
print('Rows exceeding existing tolerance:', [r[:2] for r in result if r[4] > r[6]])
print('Pinned rows eligible for promotion:', [r[:2] for r in result if r[5] == 'pinned' and r[4] <= gate])
