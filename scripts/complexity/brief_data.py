#!/usr/bin/env python3
"""Collect every input of the metric-cost brief into one aggregates-only JSON file.

Joins, per metric: counted cost under each weight table, measured CPU and wall time on i265
(1024x1024, 1 and 8 threads; 3840x2160 single thread), agreement with the preliminary AIC2026
subjective JND, and the reference ("canonical") implementation with its parity status.
Then computes, per weight table and thread count, the statistics the brief reports.

Sources (the superset; each optional source is skipped with a note when absent):
  public tallies      benchmarks/iqa_complexity_2026-10-17/tallies/*.json
  public timings      benchmarks/iqa_complexity_2026-10-17/measured/*.tsv (i265)
  DVIFM counts        benchmarks/aic_refusals_2026-10-18/dvifm-counts.tsv (aggregate op counts)
  agreement           docs/paper/pareto.json, plus --agreement JSON files (aggregates only)
  canonical           scripts/dashboard/model_cards.py (the README "Metrics" table)
  private repo        --private DIR: tallies, timings and model cards read from origin/main
                      with read-only git (never the working copy)

    brief_data.py --out FILE [--private DIR] [--agreement FILE]... [--timing FILE]...

Only aggregates are written: no per-image score and no subjective value.
"""
import argparse, csv, glob, io, json, math, random, re, statistics, subprocess, sys, tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'scripts/dashboard'))
import paper  # noqa: E402  (tallies, weight tables)
from evaluate import evaluate, validate  # noqa: E402
from correlate import weighted  # noqa: E402
from scipy import stats as st  # noqa: E402

B = ROOT / 'benchmarks/iqa_complexity_2026-10-17'
W, H = 1024, 1024
HOST = 'i265'
SEED = 20261020
DVIFM = {  # aggregate-count name -> dvifmish preset (benchmarks/aic_refusals_2026-10-18/README.md)
    'dvifm-port-rgb': 'port-lap-v0.3', 'dvifm-port-y': 'port-lap-v0.2',
    'dvifm-port-2d-y': 'port-2d-v0.2', 'dvifm-port-min-y': 'port-min-v0.2'}
DVIFM_DRAFT = {'DVIFM_RGB': 'dvifm-port-rgb', 'DVIFM_Y': 'dvifm-port-y', 'DVIFM_2D_Y': 'dvifm-port-2d-y',
               'DVIFM_MIN_Y': 'dvifm-port-min-y'}
PRIVATE_COLUMNS = {'idfiqa': 'proposal-IDFIQA', 'msswd': 'MSSWD', 'topiq-fr': 'TOPIQ', 'topiq-fr-pipal': 'TOPIQ-PIPAL'}


def git_show(repo, path):
    r = subprocess.run(['git', '-C', str(repo), 'show', f'origin/main:{path}'], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def git_ls(repo):
    r = subprocess.run(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', 'origin/main'], capture_output=True, text=True)
    return r.stdout.split()


def git_rev(repo, ref):
    r = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--short=12', ref], capture_output=True, text=True)
    return r.stdout.strip() or None


def public_rev():
    r = subprocess.run(['jj', '-R', str(ROOT), 'log', '--no-graph', '-r', 'heads(::@ & ::master@origin)', '-T', 'commit_id.short(12)'],
                       capture_output=True, text=True)
    return r.stdout.strip() or git_rev(ROOT, 'HEAD') or 'unknown'


# ---------------------------------------------------------------- timings

def read_perf(text, path, rename=lambda m: m):
    """perf_deltas TSV text -> {(metric, shape, threads): [(cpu_ms, wall_ms)]}."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#') or line.startswith('metric\t'):
            continue
        f = line.split('\t')
        try:
            cpu, wall = float(f[5]), float(f[-2])
        except (ValueError, IndexError):
            continue
        if not (math.isfinite(cpu) and math.isfinite(wall)):
            continue
        out.setdefault((rename(f[0]), f[1], int(f[2])), []).append((cpu, wall))
    return out


def merge(dst, src, origin, path):
    for k, v in src.items():
        dst.setdefault(k, []).append((origin, path, v))


def dvifm_name(m):
    return 'dvifmish-' + DVIFM[DVIFM_DRAFT[m]] if m in DVIFM_DRAFT else m


def timings(private, extra):
    acc = {}
    for path in sorted(glob.glob(str(B / 'measured/*.tsv'))):
        merge(acc, read_perf(Path(path).read_text(), path, dvifm_name), 'public', Path(path).relative_to(ROOT).as_posix())
    for path in extra:
        p = Path(path).expanduser()
        merge(acc, read_perf(p.read_text(), str(p), dvifm_name), 'extra', p.name)
    if private:
        for rel in git_ls(private):
            if not (rel.startswith('benchmarks/') and rel.endswith('.tsv')) or 'rten' not in rel or 'cold' in rel:
                continue
            text = git_show(private, rel)
            if text and 'task_clock_ms_per_call' in text.split('\n', 3)[1 if text.startswith('#') else 0]:
                merge(acc, read_perf(text, rel, lambda m: m.replace('_', '-')), 'private', rel)
    # One value per (metric, shape, threads): the default-build 1-thread run is preferred over
    # the parallel build's 1-thread rows (as paper.py does); otherwise all rows pool.
    out = {}
    for k, groups in acc.items():
        if k[2] == 1 and len(groups) > 1:
            non_par = [g for g in groups if '_par_' not in g[1]]
            groups = non_par or groups
        rows = [r for g in groups for r in g[2]]
        out[k] = dict(cpu_ms=statistics.median(c for c, _ in rows), wall_ms=statistics.median(w for _, w in rows),
                      cpu_reps=[c for c, _ in rows], wall_reps=[w for _, w in rows],
                      n=len(rows), source=sorted({g[1] for g in groups}), origin=groups[0][0])
    return out


# ---------------------------------------------------------------- counts

def estimates(private):
    tables = [(stem, label, json.loads((HERE / f'weights_{stem}.json').read_text())['weights']) for stem, label in paper.TABLES]
    out = {}
    for m, (t, r) in paper.tallies().items():
        out[m] = dict(origin='public', counter=t['counter'], flops=r.get('flops_per_px'),
                      est={stem: weighted(r, wt)[0] for stem, _, wt in tables})
    if private:
        for rel in git_ls(private):
            if re.match(r'benchmarks/iqa_complexity_2026-10-17/[^/]+\.json$', rel) and not rel.endswith('.eval.json'):
                t = json.loads(git_show(private, rel))
                if validate(t) or t['metric'] in out:
                    continue
                r = evaluate(t, W, H)
                out[t['metric']] = dict(origin='private', counter=t['counter'], flops=r.get('flops_per_px'),
                                        est={stem: weighted(r, wt)[0] for stem, _, wt in tables})
    p = ROOT / 'benchmarks/aic_refusals_2026-10-18/dvifm-counts.tsv'
    if p.exists():
        for row in csv.DictReader(p.open(), delimiter='\t'):
            if row['shape'] != f'{W}x{H}':
                continue
            est = {}
            for stem, _, wt in tables:
                est[stem] = sum(float(row[op + '_per_pixel']) * w['*'] for op, w in wt.items() if op + '_per_pixel' in row and '*' in w)
            out['dvifmish-' + DVIFM[row['metric']]] = dict(origin='dvifmish', counter='aggregate-core-only', scope=row['scope'],
                                                           flops=float(row['flops_per_pixel']), est=est)
    return [dict(id=s, label=l) for s, l, _ in tables], out


# ---------------------------------------------------------------- canonical implementation

def canonical(private):
    out = {}
    try:
        import model_cards
        for c in model_cards.cards():
            cite = c.get('citation') or {}
            out[c['metric']] = dict(family=f"{c['kind']} · {c['family']}", reference=c['reference_implementation'],
                                    parity=c['parity_status'], licence=f"code {c['rust_license']}; weights {c['weights_license']}",
                                    paper=cite.get('title'), paper_url=cite.get('url'), source='README Metrics table (model_cards.py)')
    except Exception as e:  # the brief still builds; the gap is reported
        out['__error__'] = f'model_cards.py failed: {e}'
    if private:
        for rel in git_ls(private):
            if not re.match(r'models/[^/]+/card\.toml$', rel):
                continue
            card = tomllib.loads(git_show(private, rel))
            code, par, aic = card.get('code', {}), card.get('parity', {}), card.get('aic4', {})
            names = ['topiq-fr', 'topiq-fr-pipal'] if card['name'] == 'topiq' else [card['name']]
            ev = aic.get('results') or par.get('aic61_rten') or par.get('results') or par.get('gate')
            for n in names:
                if n in out:
                    continue
                out[n] = dict(family='FR · Learned', reference=(code.get('upstream', '') + ' ' + code.get('upstream_revision', '')).strip(),
                              parity='Reported results' if ev else 'Not documented', evidence=ev,
                              licence=f"code {code.get('licence', 'unknown')} (private)", paper=card.get('paper'),
                              source=f'zenmetrics-private {rel}')
    dv = Path.home() / 'work/dvifmish/README.md'
    rows = {}
    if dv.exists():
        for line in dv.read_text().splitlines():
            m = re.match(r'\| `(port-[\w.-]+)` \| ([\d.e-]+) \| ([\d.e-]+) \| ([\d.]+) → ([\d.]+) \| ([\d.]+) → ([\d.]+) \|', line)
            if m:
                rows[m[1]] = m.groups()
    for preset in DVIFM.values():
        g = rows.get(preset)
        out['dvifmish-' + preset] = dict(
            family='FR · Classical', reference=f"M. Pedzisz's C++ DVIFM {preset.removeprefix('port-')} (AIC-4 contribution)",
            parity='Reported results' if g else 'Not documented',
            evidence=(f'mean |ΔJND| {g[1]} vs his builds on 9,618 AIC2026 crops; 840×944 1 thread: C++ {g[3]} ms → '
                      f'dvifmish {g[4]} ms' if g else None),
            licence='private: publishing needs the author\'s consent', paper='DVIFM (AIC-4 proposal; BMVA talk 2026-02-04)',
            source='dvifmish README.md (port mode table)')
    return out


# ---------------------------------------------------------------- statistics

def boot_ci(xs, ys, fn, n=2000):
    rng = random.Random(SEED)
    idx = range(len(xs))
    vals = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        try:
            v = fn([xs[i] for i in pick], [ys[i] for i in pick])
        except Exception:
            continue
        if v is not None and math.isfinite(v):
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def fit_stats(pts):
    """pts: [(estimate, measured)]. Rank, log-log and linear statistics with 95 % intervals and p-values."""
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    lx = [math.log10(v) for v in x]; ly = [math.log10(v) for v in y]
    n = len(pts)
    sp = st.spearmanr(x, y)
    sp_lo, sp_hi = boot_ci(x, y, lambda a, b: st.spearmanr(a, b)[0])
    pr = st.pearsonr(lx, ly)
    z, se = math.atanh(pr[0]), 1 / math.sqrt(n - 3)
    lr = st.linregress(lx, ly)
    t = st.t.ppf(0.975, n - 2)
    k = statistics.median(b - a for a, b in zip(lx, ly))
    k_lo, k_hi = boot_ci(lx, ly, lambda a, b: statistics.median(q - p for p, q in zip(a, b)))
    res = [b - a - k for a, b in zip(lx, ly)]
    kl = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)  # least squares through the origin, linear axes
    kl_lo, kl_hi = boot_ci(x, y, lambda a, b: sum(p * q for p, q in zip(a, b)) / sum(p * p for p in a))
    lin = st.pearsonr(x, y)
    return dict(n=n, spearman=sp[0], spearman_lo=sp_lo, spearman_hi=sp_hi, spearman_p=sp[1],
                pearson_log=pr[0], pearson_log_lo=math.tanh(z - 1.96 * se), pearson_log_hi=math.tanh(z + 1.96 * se), pearson_log_p=pr[1],
                slope=lr.slope, slope_lo=lr.slope - t * lr.stderr, slope_hi=lr.slope + t * lr.stderr, slope_p=lr.pvalue,
                intercept=lr.intercept,
                scale=10 ** k, scale_lo=10 ** k_lo, scale_hi=10 ** k_hi,
                typical=10 ** statistics.median(abs(r) for r in res),
                within2=sum(1 for r in res if abs(r) <= math.log10(2)),
                scale_linear=kl, scale_linear_lo=kl_lo, scale_linear_hi=kl_hi,
                pearson_linear=lin[0], pearson_linear_p=lin[1])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--private', type=Path, help='zenmetrics-private checkout (read from origin/main with git show)')
    ap.add_argument('--agreement', action='append', default=[], help='aggregate agreement JSON: list of {column|metric, within_source_srcc|agreement, ci_lo|lo, ci_hi|hi, overall_abs_srcc|overall}')
    ap.add_argument('--timing', action='append', default=[], help='extra perf_deltas TSV measured on i265')
    a = ap.parse_args()
    notes = []
    priv = a.private if a.private and (a.private / '.git').exists() else None
    if a.private and not priv:
        notes.append(f'private repo {a.private} not found: private metrics omitted')

    tables, est = estimates(priv)
    tim = timings(priv, a.timing)
    canon = canonical(priv)
    if '__error__' in canon:
        notes.append(canon.pop('__error__'))
    pareto = json.loads((ROOT / 'docs/paper/pareto.json').read_text())
    points = json.loads((ROOT / 'docs/paper/points.json').read_text())
    kinds = {m['metric']: m['kind'] for m in points['metrics']}
    cmap = json.loads((HERE / 'aic2026_columns.json').read_text())['map']
    col_of = {v: k for k, v in cmap.items()}
    col_of.update(PRIVATE_COLUMNS)

    agree = {}
    for p in pareto['points']:
        agree[p['metric']] = dict(agreement=p['agreement'], lo=p['lo'], hi=p['hi'], overall=p['overall'], column=p['column'], source='docs/paper/pareto.json')
    for path in a.agreement:
        for d in json.loads(Path(path).expanduser().read_text()):
            col = d.get('column'); metric = d.get('metric') or next((m for m, c in col_of.items() if c == col), None)
            if not metric or metric in agree:
                continue
            agree[metric] = dict(agreement=d.get('within_source_srcc', d.get('agreement')), lo=d.get('ci_lo', d.get('lo')),
                                 hi=d.get('ci_hi', d.get('hi')), overall=d.get('overall_abs_srcc', d.get('overall')),
                                 column=col or d.get('label') or metric, source=Path(path).name)

    names = sorted(set(est) | {k[0] for k in tim} | set(agree))
    metrics = []
    for m in names:
        t = lambda shape, th: tim.get((m, shape, th))
        t1, t8, t4 = t(f'{W}x{H}', 1), t(f'{W}x{H}', 8), t('3840x2160', 1)
        px, px4 = W * H, 3840 * 2160
        e = est.get(m, {})
        origin = 'dvifmish' if m.startswith('dvifmish-') else e.get('origin') or (t1 or t8 or {}).get('origin')
        if origin in ('extra', None):
            origin = 'private' if m in PRIVATE_COLUMNS else 'public'
        kind = kinds.get(m) or ('nn-fr' if m in paper.NN_NAMES or origin == 'private' or m.endswith('-pyiqa') and m != 'cw-ssim-pyiqa' else 'classic-fr')
        metrics.append(dict(
            metric=m, origin=origin, kind=kind, column=col_of.get(m),
            counter=e.get('counter'), count_scope=e.get('scope'), flops_per_px=e.get('flops'), estimate=e.get('est'),
            cpu_1t=t1 and t1['cpu_ms'] * 1e6 / px, wall_1t=t1 and t1['wall_ms'] * 1e6 / px,
            cpu_8t=t8 and t8['cpu_ms'] * 1e6 / px, wall_8t=t8 and t8['wall_ms'] * 1e6 / px,
            cpu_1t_4k=t4 and t4['cpu_ms'] * 1e6 / px4,
            cpu_1t_reps=t1 and [v * 1e6 / px for v in t1['cpu_reps']], cpu_8t_reps=t8 and [v * 1e6 / px for v in t8['cpu_reps']],
            wall_8t_reps=t8 and [v * 1e6 / px for v in t8['wall_reps']],
            timing_sources=sorted(set((t1 or {}).get('source', []) + (t8 or {}).get('source', []))),
            agreement=agree.get(m), canonical=canon.get(m)))

    stats = []
    for tb in tables:
        for threads in (1, 8):
            for measure in ('cpu', 'wall'):
                for subset in ('all', 'classical', 'neural'):
                    key = f'{measure}_{threads}t'
                    pts = [(r['estimate'][tb['id']], r[key]) for r in metrics
                           if r['estimate'] and r.get(key) and r['estimate'].get(tb['id'], 0) > 0
                           and (subset == 'all' or (subset == 'neural') == r['kind'].startswith('nn'))]
                    if len(pts) >= 8:
                        stats.append(dict(table=tb['id'], threads=threads, measure=measure, subset=subset, **fit_stats(pts)))

    # Pareto fronts on agreement vs time, over every metric with both
    fronts = {}
    for key in ('cpu_1t', 'cpu_8t'):
        pts = sorted((r for r in metrics if r['agreement'] and r.get(key)), key=lambda r: r[key])
        front, best = [], -1
        for r in pts:
            if r['agreement']['agreement'] > best:
                front.append(r['metric']); best = r['agreement']['agreement']
        fronts[key] = front

    doc = dict(schema=1, host=HOST, shape=f'{W}x{H}', seed=SEED,
               sources=dict(public=public_rev(), private=priv and git_rev(priv, 'origin/main'),
                            dvifmish=git_rev(Path.home() / 'work/dvifmish', 'origin/main'),
                            extra_timing=[Path(p).name for p in a.timing], agreement=[Path(p).name for p in a.agreement]),
               dvifmish_versions=[dict(metric='dvifmish-' + v, preset=v, aggregate_name=k) for k, v in DVIFM.items()],
               tables=tables, metrics=metrics, stats=stats, fronts=fronts, notes=notes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, allow_nan=False))
    n1 = sum(1 for r in metrics if r['estimate'] and r['cpu_1t'])
    print(f'{a.out}: {len(metrics)} metrics, {n1} with count and 1-thread time, {len(stats)} stat rows; notes: {notes}')


if __name__ == '__main__':
    main()
