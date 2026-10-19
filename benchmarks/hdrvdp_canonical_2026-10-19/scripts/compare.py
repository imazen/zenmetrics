"""Strict complete-panel residuals; no fitted corrections. Results to stdout JSON.
Usage: compare.py PUBLISHED CONFIG COLUMN SCORES SCORE_FIELD [REFERENCE FIELD]
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

pub, config, column, scores, field = sys.argv[1:6]
expected = {r['distorted']: r for r in csv.DictReader(open(pub))}
rows = list(csv.DictReader(open(scores), delimiter='\t'))
assert len(rows) == len(expected) == 61, 'requires complete 61-pair panel'
assert len({r['distorted'] for r in rows}) == 61
assert {r['distorted'] for r in rows} == set(expected)
m = json.load(open(config))['mappings'][column]
def jnd(x):
    y = m['sign'] * (x - m['b'])
    return math.copysign(m['a'] * abs(y)**m['c'], y)
a = [float(r[field]) for r in rows]
b = [float(expected[r['distorted']][column]) for r in rows]
assert all(math.isfinite(x) for x in a + b)
d = [abs(x-y) for x, y in zip(a,b)]
dj = [abs(jnd(x)-jnd(y)) for x,y in zip(a,b)]
def ranks(x):
    return [st.mean([i for i,y in enumerate(sorted(x)) if y == v]) for v in x]
def corr(x,y):
    mx,my=st.mean(x),st.mean(y)
    return sum((u-mx)*(v-my) for u,v in zip(x,y))/math.sqrt(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y))
res = dict(n=61, median_raw=st.median(d), max_raw=max(d), median_jnd=st.median(dj), max_jnd=max(dj), spearman=corr(ranks(a),ranks(b)), exact_printed=sum(f'{x:.6f}' == expected[r['distorted']][column] for x,r in zip(a,rows)), scores_sha256=hashlib.sha256(Path(scores).read_bytes()).hexdigest())
if len(sys.argv) == 8:
    ref={r['distorted']:float(r[sys.argv[7]]) for r in csv.DictReader(open(sys.argv[6]),delimiter='\t')}
    delta=[abs(float(r[field])-ref[r['distorted']]) for r in rows]
    res['vs_reference_median']=st.median(delta);res['vs_reference_max']=max(delta)
print(json.dumps(res,indent=2))
