"""wd_eval.py <published.csv> <site_config.json> <out_summary.tsv> <wd_probe_out>...
Assembles conventions from per-feature WD rows and scores each against WD_s1..WD_s5.
Convention = input, dist, floor, sigma mode, tap set, pixel layer, image scales; log2sigma = k+shift."""
import csv, sys, json, math, statistics as st, itertools, collections
pubf, cfgf, outf = sys.argv[1:4]
pub = {r['distorted']: r for r in csv.DictReader(open(pubf))}
maps = json.load(open(cfgf))['mappings']
V = collections.defaultdict(float)  # (distorted,input,dist,floor,scale,feature,sigma,log2s) -> wd
for p in sys.argv[4:]:
    for r in csv.DictReader(open(p), delimiter='\t'):
        V[(r['distorted'], r['input'], r['dist'], r['floor'], r['scale'], r['feature'], r['sigma'], int(r['log2s']))] = float(r['wd'])
names = sorted({k[0] for k in V})
def jnd(col, x):
    m = maps[col]; y = m['sign'] * (x - m['b']); return m['a'] * abs(y) ** m['c'] * (1 if y >= 0 else -1)
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
    for k, i in enumerate(o): r[i] = k
    return r
def pear(a, b):
    ma, mb = st.fmean(a), st.fmean(b); den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den
TAPS4, TAPS5 = ['tap3', 'tap8', 'tap15', 'tap22'], ['tap3', 'tap8', 'tap15', 'tap22', 'tap29']
out = open(outf, 'w')
out.write('input\tdist\tfloor\tsigma\ttaps\tpix\tscales\tcol\tlog2s\tn\tmed_abs\tmed_jnd\tmax_jnd\tpearson\tspearman\tmed_ratio\n')
FLOORS = sorted({k[3] for k in V}); INPUTS = sorted({k[1] for k in V}); DISTS = sorted({k[2] for k in V}); SMAX = max(k[7] for k in V)
for inp, dist, floor, sig, taps, pix, scales in itertools.product(INPUTS, DISTS, FLOORS, ('const', 'rescale'), (4, 5), (0, 1), ('1', '1+2+4')):
    feats = (TAPS4 if taps == 4 else TAPS5) + (['pix'] if pix else [])
    scs = scales.split('+')
    for col_k in range(1, 6):
        col = f'WD_s{col_k}'
        for s in range(0, SMAX + 1):
            ours, theirs = [], []
            for n in names:
                if pub[n][col] in ('', 'nan'): continue
                ks = [(n, inp, dist, floor, sc, f, sig, s) for sc in scs for f in feats]
                if not all(k in V for k in ks): continue
                v = sum(V[k] for k in ks)
                ours.append(v); theirs.append(float(pub[n][col]))
            if len(ours) < 3: continue
            d = [abs(a - b) for a, b in zip(ours, theirs)]; dj = [abs(jnd(col, a) - jnd(col, b)) for a, b in zip(ours, theirs)]
            ratio = st.median([b / a for a, b in zip(ours, theirs) if a > 0])
            out.write('\t'.join(map(str, (inp, dist, floor, sig, taps, pix, scales, col, s, len(d)))) + '\t' + '\t'.join(f'{x:.5g}' for x in (st.median(d), st.median(dj), max(dj), pear(ours, theirs), pear(ranks(ours), ranks(theirs)), ratio)) + '\n')
