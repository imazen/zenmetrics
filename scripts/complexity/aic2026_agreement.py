# Agreement of each published AIC2026 metric column with the preliminary subjective JND reconstruction.
# Inputs are private (AIC2026_WORKDIR/private-aic/..., aic2026/metrics_fullres.csv) and never committed; the committed output
# is per-metric aggregates only (benchmarks/iqa_complexity_2026-10-17/analysis/aic2026_agreement.json).
import csv, math, random, statistics, json, sys
import os
P = os.environ.get('AIC2026_WORKDIR', os.path.expanduser('~/tmp/iqa-cx-lead/')) + '/'
lab = list(csv.DictReader(open(P + 'private-aic/aic2026_public_split_preliminary_jnd_reconstruction.csv')))
jnd = {r['distorted']: float(r['JND_score']) for r in lab}
src = {r['distorted']: r['source'] for r in lab}
rows = list(csv.DictReader(open(P + 'aic2026/metrics_fullres.csv')))
key = lambda d: d.rsplit('.', 1)[0]
meta = {'distorted', 'source', 'codec', 'codec_acronym', 'distortion_level', 'bpp', 'codec_id', 'enc_setting', 'enc_value'}
cols = [c for c in rows[0] if c not in meta and not c.startswith('JND_')]
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def pear(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float('nan')
def srcc(a, b): return pear(ranks(a), ranks(b))
out = []
for c in cols:
    pts = []
    for r in rows:
        k = key(r['distorted'])
        if k in jnd and r[c] not in ('', 'nan', 'NaN', 'inf', '-inf'):
            try: v = float(r[c])
            except ValueError: continue
            if math.isfinite(v): pts.append((src[k], v, jnd[k]))
    if len(pts) < 100: continue
    rho = srcc([p[1] for p in pts], [p[2] for p in pts])
    sign = -1 if rho < 0 else 1
    by = {}
    for s, v, j in pts: by.setdefault(s, []).append((v, j))
    per = {s: srcc([v for v, _ in l], [j for _, j in l]) * sign for s, l in by.items() if len(l) >= 5}
    per = {s: v for s, v in per.items() if not math.isnan(v)}
    rng = random.Random(20261017); ss = sorted(per); boots = []
    for _ in range(2000):
        pick = [per[rng.choice(ss)] for _ in ss]; boots.append(statistics.fmean(pick))
    boots.sort()
    out.append(dict(column=c, n=len(pts), sources=len(per), overall_abs_srcc=abs(rho), direction='higher=worse' if rho > 0 else 'higher=better',
                    within_source_srcc=statistics.fmean(per.values()), ci_lo=boots[50], ci_hi=boots[1949]))
out.sort(key=lambda d: -d['within_source_srcc'])
json.dump(out, open(P + 'aic2026/agreement.json', 'w'), indent=1)
for d in out: print(f"{d['column']:28} n={d['n']:5} src={d['sources']:2} |rho|={d['overall_abs_srcc']:.3f} within={d['within_source_srcc']:.3f} [{d['ci_lo']:.3f},{d['ci_hi']:.3f}] {d['direction']}")
