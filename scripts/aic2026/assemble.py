#!/usr/bin/env python3
"""Assemble metrics_<set>.zenmetrics.csv from per-metric batch TSVs and compare it with the published file.

Usage: assemble.py <published.csv> <scores_dir> <out.csv> <report.tsv> <worst.txt>
- Copies the published header, row order and nine metadata columns.
- Fills each mapped metric column with our score (empty where no score file).
- Recomputes every JND_<col> from OUR value with the site mapping (site_jnd_mappings.json).
- report.tsv: per column n, exact-equal at published precision, median/max |d|, median/max |d site-JND|,
  Pearson, Spearman, match (median site-JND <= 1e-3).
- worst.txt: the five worst |d site-JND| rows per non-matching column.
Origin: the aic-full-repro lane (dev, 2026-10-20); column table moved to aic_columns.py.
"""
import csv, math, statistics, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aic_columns import COLS, JNDS, META, sitejnd, mdct_jnd, write_report  # noqa: E402

PUB, SDIR, OUT, REPORT, WORST = sys.argv[1:6]

pub = list(csv.DictReader(open(PUB)))
header = list(pub[0].keys())

# load our score tables: name -> dict distorted-> {col: val}
tables = {}
def load(tag):
    path = os.path.join(SDIR, tag + '.tsv')
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path), delimiter='\t'))
    tables[tag] = {r['distorted']: r for r in rows}
    return rows

for tag in set(v[0] for v in COLS.values()) - {'@mdct_jnd'}:
    load(tag)

def pick(row, tag, col):
    t = tables.get(tag)
    if t is None or row['distorted'] not in t:
        return ''
    r = t[row['distorted']]
    meta = ('ref_path', 'dist_path', 'distorted', 'source', 'codec', 'codec_acronym',
            'distortion_level', 'bpp', 'codec_id', 'enc_setting', 'enc_value')
    if col == 'auto':
        cands = [k for k in r if k not in meta and r[k] not in ('', 'nan', 'NaN', None)]
        if not cands:
            return ''
        return r[cands[0]]
    if col not in r or r[col] in ('', 'nan', 'NaN', None):
        # versioned impl column (e.g. fsimc_imazen_v0_1_0)
        cands = [k for k in r if k not in meta and (k == col or k.startswith(col + '_'))
                 and r[k] not in ('', 'nan', 'NaN', None)]
        if not cands:
            return ''
        return r[cands[0]]
    return r[col]

ours = {}  # col -> {distorted: float}
for col, (tag, scol, _) in COLS.items():
    d = {}
    if tag == '@mdct_jnd':
        for row in pub:
            v = pick(row, 'mdctpsnr', 'auto')
            if v:
                d[row['distorted']] = mdct_jnd(float(v))
    else:
        for row in pub:
            v = pick(row, tag, scol)
            if v:
                d[row['distorted']] = float(v)
    ours[col] = d

# write the matching CSV
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for row in pub:
        out = []
        for c in header:
            if c in META:
                out.append(row[c])
            elif c in JNDS:
                base, mk = JNDS[c]
                v = ours[base].get(row['distorted'])
                out.append('' if v is None else repr(sitejnd(mk, v)))
            elif c in COLS:
                v = ours[c].get(row['distorted'])
                out.append('' if v is None else repr(v))
            else:
                out.append('')
        w.writerow(out)

write_report(pub, header, ours, REPORT, WORST)
print('wrote', OUT, REPORT, WORST)
