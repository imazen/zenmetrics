"""One JND per column from konjnd_score.py output.

usage: konjnd_summary.py SCORES_TSV METRIC[:same] [...]

`METRIC:same` uses |score(ref, dist) - score(ref, ref)| per pair (TOPIQ: higher is better and the
identical-pair score is image-dependent, like ckdn in zenmetrics' jnd_scale.tsv); plain `METRIC`
uses the score itself (MS-SWD: a distance, 0 for identical inputs). 1 JND := pooled median over
the pairs. Prints p25/median/p75/mean per codec and pooled (numpy linear percentiles).
"""
import csv, sys
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
print(f'{len(rows)} pairs')
print('| metric | codec | n | p25 | median | p75 | mean | p75/p25 |')
print('|---|---|---|---|---|---|---|---|')
for spec in sys.argv[2:]:
    m, _, mode = spec.partition(':')
    for codec in ('JPEG', 'BPG', 'pooled'):
        sel = [r for r in rows if codec == 'pooled' or r['codec'] == codec]
        v = np.array([abs(float(r[m]) - float(r[m + '_same'])) if mode == 'same' else float(r[m]) for r in sel])
        p25, med, p75 = np.percentile(v, [25, 50, 75])
        print(f'| {m} | {codec} | {len(v)} | {p25:.6f} | {med:.6f} | {p75:.6f} | {v.mean():.6f} | {p75 / p25:.3f} |')
    if mode == 'same':
        neg = sum(float(r[m]) > float(r[m + '_same']) for r in rows)
        print(f'{m}: distorted scored above its own identical-pair score in {neg} of {len(rows)} pairs')
