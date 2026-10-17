"""evalcol.py <published.csv> <site_config.json> <col> <log.tsv> <tag>=<scores.tsv>[:<scorecol>] ...
Appends one row per scores file to log.tsv: raw and site-JND residuals, Pearson, Spearman,
and the least-squares affine map ours->published with its JND residual.
Scores TSV: keyed by `distorted` (or dist_path basename); score column = <scorecol> or the only
numeric non-key column named score/first remaining."""
import csv, sys, json, math, statistics as st, os
pubf, cfgf, col, log = sys.argv[1:5]
pub = {r['distorted']: r for r in csv.DictReader(open(pubf))}
m = json.load(open(cfgf))['mappings'][col]
def jnd(x): return m['a'] * abs(m['sign'] * (x - m['b'])) ** m['c'] * (1 if m['sign'] * (x - m['b']) >= 0 else -1)
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def pear(a, b):
    ma, mb = st.fmean(a), st.fmean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else float('nan')
new = not os.path.exists(log)
out = open(log, 'a')
if new: out.write('tag\tcol\tn\tmed_abs\tmax_abs\tmed_jnd\tmax_jnd\tpearson\tspearman\taff_slope\taff_icpt\taff_med_jnd\taff_max_jnd\n')
for spec in sys.argv[5:]:
    tag, path = spec.split('=', 1)
    sc = None
    if ':' in path: path, sc = path.rsplit(':', 1)
    rows = list(csv.DictReader(open(path), delimiter='\t'))
    ours, theirs = [], []
    for r in rows:
        name = r.get('distorted') or os.path.basename(r['dist_path'])
        k = sc or ('score' if 'score' in r else [c for c in r if c not in ('ref_path', 'dist_path', 'distorted', 'sec')][0])
        try: v = float(r[k])
        except (ValueError, TypeError): continue
        if name not in pub or pub[name][col] in ('', 'nan') or not math.isfinite(v): continue
        ours.append(v); theirs.append(float(pub[name][col]))
    n = len(ours)
    if n < 3: print(tag, 'n<3'); continue
    d = [abs(a - b) for a, b in zip(ours, theirs)]
    dj = [abs(jnd(a) - jnd(b)) for a, b in zip(ours, theirs)]
    mo, mt = st.fmean(ours), st.fmean(theirs)
    vo = sum((x - mo) ** 2 for x in ours)
    s = sum((x - mo) * (y - mt) for x, y in zip(ours, theirs)) / vo if vo else float('nan')
    c0 = mt - s * mo
    fj = [abs(jnd(s * a + c0) - jnd(b)) for a, b in zip(ours, theirs)]
    row = [tag, col, n, st.median(d), max(d), st.median(dj), max(dj), pear(ours, theirs), pear(ranks(ours), ranks(theirs)), s, c0, st.median(fj), max(fj)]
    out.write('\t'.join(str(x) if isinstance(x, (int, str)) else f'{x:.6g}' for x in row) + '\n')
    print('\t'.join(str(x) if isinstance(x, (int, str)) else f'{x:.4g}' for x in row))
