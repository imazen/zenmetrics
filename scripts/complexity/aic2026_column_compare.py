"""Compare our per-pair scores with published AIC2026 metric columns.

Usage: aic2026_column_compare.py <metrics_subset.csv> <scores_dir> <metric> <col[,col...]> [<metric> <cols>...]
<scores_dir>/<metric>.tsv is `zenmetrics batch` output keyed by `distorted`.
Prints Pearson and Spearman per column (rank ties broken by order).
"""
import csv, sys, statistics, math
if len(sys.argv) < 5 or len(sys.argv) % 2 == 0:
    sys.exit(__doc__)
pub = {r['distorted']: r for r in csv.DictReader(open(sys.argv[1]))}
D = sys.argv[2].rstrip('/') + '/'
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0]*len(x)
    for k, i in enumerate(o): r[i] = k
    return r
def pear(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
for metric, cols in [(sys.argv[i], sys.argv[i+1].split(',')) for i in range(3, len(sys.argv), 2)]:
    rows = list(csv.DictReader(open(D + metric + '.tsv'), delimiter='\t'))
    for col in cols:
        ours, theirs = [], []
        for r in rows:
            name = r.get('distorted') or r['dist_path'].rsplit('/', 1)[-1]
            score_col = [k for k in r if k not in ('ref_path', 'dist_path', 'distorted') and r[k] not in ('',)]
            v = float(r[score_col[0]]) if len(score_col) == 1 else float(r[col.lower()]) if col.lower() in r else None
            if v is None or name not in pub or pub[name][col] in ('', 'nan'): continue
            ours.append(v); theirs.append(float(pub[name][col]))
        d = [abs(a-b) for a, b in zip(ours, theirs)]
        print(f'{metric:16} vs {col:8} n={len(d)} median|d|={statistics.median(d):.3g} max|d|={max(d):.3g} '
              f'pearson={pear(ours, theirs):.5f} spearman={pear(ranks(ours), ranks(theirs)):.4f}')
