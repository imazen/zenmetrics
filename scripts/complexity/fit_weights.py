#!/usr/bin/env python3
"""Fit operation-group weights to measured time, with family-held-out validation.

Model (per original pixel, one host / thread count / shape):
    time_ns_per_px ≈ Σ_g w_g · n_g,   w_g ≥ 0
over grouped operation counts n_g from the tallies. Fitted by non-negative
least squares on relative error (rows divided by the measured time). Groups,
not single classes, because add, mul and fma counts move together and their
separate weights are not identifiable at this sample size.

Validation holds out whole metric families (variants such as -k2..-k6 share
code, so holding out single names leaks). If an effort table is given
(family, cli_names, iterations ...), held-out errors are also reported per
effort tier (tertiles of optimization iterations), which shows whether
heavily optimized implementations sit systematically below the fit.

Baselines: each fixed weight table (scripts/complexity/weights_*.json) with
one scale factor fitted on the same training folds.

The fitted weights describe these implementations on this host, including
their vectorization; they are not intrinsic method costs.

  fit_weights.py --tallies GLOB --measured GLOB [--effort FILE.tsv] [--threads 1] [--shape 1024x1024]
"""
import argparse, glob, json, math, statistics, sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import evaluate, validate, parse_shape  # noqa: E402
from correlate import weighted  # noqa: E402

GROUPS = {
    'simple': {'add': 1, 'mul': 1, 'fma': 1, 'cmp': 1, 'abs': 1},
    'divide': {'div': 1, 'rcp': 1},
    'root': {'sqrt': 1, 'rsqrt': 1},
    'transcendental': {'exp': 1, 'log': 1, 'trig': 1},
    'power': {'pow': 1},
    'memory': {'load': 1, 'store': 1},
    'convert_int': {'cvt': 1, 'int': 1},
}


def family_of(name, effort_map):
    if name in effort_map:
        return effort_map[name]
    return name.split('-')[0]


def load(a):
    w, h = parse_shape(a.shape)
    prefer = a.prefer_counter.split(',')
    tallies = {}
    for f in glob.glob(a.tallies):
        t = json.loads(Path(f).read_text())
        if validate(t):
            continue
        r = prefer.index(t['counter']) if t['counter'] in prefer else len(prefer)
        if t['metric'] not in tallies or r < tallies[t['metric']][0]:
            tallies[t['metric']] = (r, t)
    times = {}
    for path in sorted(glob.glob(a.measured)):
        for line in Path(path).read_text().splitlines():
            if line.startswith('#') or line.startswith('metric\t') or not line.strip():
                continue
            f = line.split('\t')
            if f[1] == a.shape and f[2] == str(a.threads) and f[5] != 'error':
                times.setdefault(f[0], []).append(float(f[5]))
    effort_map, effort = {}, {}
    if a.effort:
        rows = [l.split('\t') for l in Path(a.effort).read_text().splitlines() if l and not l.startswith('#')]
        head = rows[0]
        for r in rows[1:]:
            d = dict(zip(head, r))
            for n in d['cli_names'].replace(' ', '').split(','):
                effort_map[n.strip('`')] = d['family']
            effort[d['family']] = float(d.get('iterations', 'nan') or 'nan')
    data = []
    for m, (_, t) in tallies.items():
        if m not in times:
            continue
        r = evaluate(t, w, h)
        n = {g: sum(sum(r['ops_per_px'].get(op, {}).values()) * k for op, k in ops.items()) for g, ops in GROUPS.items()}
        y = statistics.median(times[m]) * 1e6 / (w * h)
        data.append(dict(metric=m, family=family_of(m, effort_map), n=n, y=y, result=r))
    return data, effort


def fit(rows):
    X = np.array([[r['n'][g] / r['y'] for g in GROUPS] for r in rows])
    w, _ = nnls(X, np.ones(len(rows)))
    return w


def predict(w, r):
    return sum(wi * r['n'][g] for wi, g in zip(w, GROUPS))


def scale_fit(table, rows):
    xs = [weighted(r['result'], table)[0] for r in rows]
    logs = [math.log(r['y'] / x) for r, x in zip(rows, xs) if x > 0]
    return math.exp(statistics.median(logs)) if logs else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tallies', required=True)
    ap.add_argument('--measured', required=True)
    ap.add_argument('--effort')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--shape', default='1024x1024')
    ap.add_argument('--prefer-counter', default='swe2,swe,lead')
    a = ap.parse_args()
    data, effort = load(a)
    families = sorted({r['family'] for r in data})
    print(f'# n={len(data)} metrics in {len(families)} families; groups={list(GROUPS)}')
    if len(families) < 4:
        print('too few families for held-out validation')
        return
    tables = {p.stem.replace('weights_', ''): json.loads(p.read_text())['weights'] for p in sorted(HERE.glob('weights_*.json'))}
    errs = {k: [] for k in ['fit'] + list(tables)}
    rows_out = []
    for fam in families:
        train = [r for r in data if r['family'] != fam]
        test = [r for r in data if r['family'] == fam]
        wfit = fit(train)
        scales = {k: scale_fit(t, train) for k, t in tables.items()}
        for r in test:
            e = {'fit': math.log(max(predict(wfit, r), 1e-12) / r['y'])}
            for k, t in tables.items():
                x = weighted(r['result'], t)[0] * scales[k]
                e[k] = math.log(x / r['y']) if x > 0 else float('nan')
            for k, v in e.items():
                errs[k].append(abs(v))
            rows_out.append((r['metric'], fam, e))
    print('held-out |log(pred/measured)| by model (family left out):')
    for k, v in errs.items():
        v = [x for x in v if not math.isnan(x)]
        print(f'  {k:28} median {math.exp(statistics.median(v)):.2f}x  max {math.exp(max(v)):.2f}x  n={len(v)}')
    wall = fit(data)
    print('weights fitted on all rows (ns per op, this host/thread/shape, implementation-inclusive):')
    for g, wi in zip(GROUPS, wall):
        print(f'  {g:16} {wi:.4g}')
    if effort:
        vals = sorted(v for v in effort.values() if not math.isnan(v))
        if len(vals) >= 3:
            t1, t2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
            tier = lambda f: 'unknown' if math.isnan(effort.get(f, float('nan'))) else \
                'low' if effort[f] <= t1 else 'mid' if effort[f] <= t2 else 'high'
            print(f'held-out fit error by optimization-effort tier (iterations tertiles {t1:g}/{t2:g}):')
            by = {}
            for m, fam, e in rows_out:
                by.setdefault(tier(fam), []).append(e['fit'])
            for k, v in sorted(by.items()):
                print(f'  {k:8} n={len(v)} median signed log ratio {statistics.median(v):+.2f} '
                      f'(log pred/measured; positive = runs faster than the fit predicts)')
    print('metric\tfamily\theld-out ratio pred/measured (fit)')
    for m, fam, e in sorted(rows_out, key=lambda x: x[2]['fit']):
        print(f'{m}\t{fam}\t{math.exp(e["fit"]):.2f}')


if __name__ == '__main__':
    main()
