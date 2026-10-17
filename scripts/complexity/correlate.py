#!/usr/bin/env python3
"""Compare tallied per-pixel cost with measured per-call CPU time and wall time.

Inputs: tallies (scripts/complexity/evaluate.py schema), a weight table, and a
perf_deltas.py TSV measured under one host, core, thread count, mode, ingress
and revision. CPU time (task-clock) and wall time are analysed separately.
Reports n, Spearman rank correlation and Pearson correlation of log cost vs
log time, each with a bootstrap 95 % interval, and per-metric residual ratios
(measured / predicted after the single best scale factor). Nothing is
extrapolated: a metric without both a tally and a measurement is listed as
missing.

  correlate.py --tallies DIR --measured FILE.tsv --weights FILE.json --shape 1024x1024 [--out FILE.tsv]

Weight file: {"name": ..., "source": ..., "weights": {"<op>": {"<precision>|*": cost}}}; ops without a
weight are reported and excluded from the weighted sum (never assumed zero silently).
"""
import argparse, glob, json, math, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, validate, parse_shape  # noqa: E402


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x, y):
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float('nan')


def boot(x, y, f, n=2000, seed=20261017):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(len(x)) for _ in x]
        xs, ys = [x[i] for i in idx], [y[i] for i in idx]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            vals.append(f(xs, ys))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float('nan'),) * 2


def weighted(result, weights):
    total, unweighted = 0.0, set()
    for op, by_prec in result['ops_per_px'].items():
        for prec, count in by_prec.items():
            w = weights.get(op, {})
            cost = w.get(prec, w.get('*'))
            if cost is None:
                if count:
                    unweighted.add(f'{op}/{prec}')
                continue
            total += cost * count
    return total, sorted(unweighted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tallies', required=True, help='glob of tally JSON files')
    ap.add_argument('--measured', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--shape', default='1024x1024')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--prefer-counter', default='', help='comma list, first match wins per metric')
    ap.add_argument('--out')
    a = ap.parse_args()
    w, h = parse_shape(a.shape)
    wt = json.loads(Path(a.weights).read_text())
    prefer = [c for c in a.prefer_counter.split(',') if c]
    tallies = {}
    for f in sorted(glob.glob(a.tallies)):
        t = json.loads(Path(f).read_text())
        if validate(t):
            continue
        cur = tallies.get(t['metric'])
        rank = prefer.index(t['counter']) if t['counter'] in prefer else len(prefer)
        if cur is None or rank < cur[0]:
            tallies[t['metric']] = (rank, t)
    rows = {}
    for line in (l for path in sorted(glob.glob(a.measured)) for l in Path(path).read_text().splitlines()):
        if line.startswith('#') or line.startswith('metric\t') or not line.strip():
            continue
        f = line.split('\t')
        if f[1] != a.shape.replace('×', 'x') or f[5] == 'error' or f[2] != str(a.threads):
            continue
        rows.setdefault(f[0], []).append((float(f[5]), float(f[-2])))  # task_clock_ms, wall_ms per call
    P = w * h
    table, missing = [], []
    for m in sorted(set(tallies) | set(rows)):
        if m not in tallies or m not in rows:
            missing.append((m, 'no tally' if m not in tallies else 'no measurement'))
            continue
        r = evaluate(tallies[m][1], w, h)
        cost, unweighted = weighted(r, wt['weights'])
        cpu = statistics.median(x for x, _ in rows[m]) * 1e6 / P  # ns per pixel
        wall = statistics.median(y for _, y in rows[m]) * 1e6 / P
        table.append(dict(metric=m, counter=tallies[m][1]['counter'], flops_per_px=r['flops_per_px'], weighted=cost,
                          unweighted_ops=';'.join(unweighted), cpu_ns_per_px=cpu, wall_ns_per_px=wall,
                          repeats=len(rows[m])))
    out = [f'# weights={wt.get("name")} ({wt.get("source")}) shape={a.shape} measured={a.measured}']
    for key, label in [('cpu_ns_per_px', 'CPU time'), ('wall_ns_per_px', 'wall time')]:
        ok = [t for t in table if t['weighted'] > 0 and t[key] > 0]
        if len(ok) < 3:
            out.append(f'{label}: n={len(ok)} too few for correlation')
            continue
        x = [math.log(t['weighted']) for t in ok]
        y = [math.log(t[key]) for t in ok]
        sp = pearson(ranks(x), ranks(y))
        pe = pearson(x, y)
        lo_s, hi_s = boot(x, y, lambda a_, b_: pearson(ranks(a_), ranks(b_)))
        lo_p, hi_p = boot(x, y, pearson)
        k = statistics.median(b - a for a, b in zip(x, y))  # log scale factor
        for t, a_, b_ in zip(ok, x, y):
            t[key.replace('ns_per_px', 'residual')] = math.exp(b_ - a_ - k)
        flag = 'POOR' if sp < 0.7 else 'moderate' if sp < 0.9 else 'strong'
        res = [t[key.replace('ns_per_px', 'residual')] for t in ok]
        far = sorted((t['metric'] for t in ok if not 1 / 3 <= t[key.replace('ns_per_px', 'residual')] <= 3))
        out.append(f'{label}: n={len(ok)} spearman={sp:.3f} [{lo_s:.3f},{hi_s:.3f}] '
                   f'pearson_log={pe:.3f} [{lo_p:.3f},{hi_p:.3f}] -> {flag} (rank agreement over the whole cost range); '
                   f'residual spread max/min={max(res) / min(res):.1f}x; outside 3x of the fit: {", ".join(far) or "none"}')
    cols = ['metric', 'counter', 'flops_per_px', 'weighted', 'cpu_ns_per_px', 'wall_ns_per_px', 'cpu_residual',
            'wall_residual', 'repeats', 'unweighted_ops']
    lines = out + ['\t'.join(cols)] + ['\t'.join(f'{t.get(c, ""):.4g}' if isinstance(t.get(c), float) else str(t.get(c, ''))
                                                 for c in cols) for t in sorted(table, key=lambda t: t['weighted'])]
    lines += [f'# missing\t{m}\t{why}' for m, why in missing]
    text = '\n'.join(lines) + '\n'
    if a.out:
        Path(a.out).write_text(text)
    sys.stdout.write(text)


if __name__ == '__main__':
    main()
