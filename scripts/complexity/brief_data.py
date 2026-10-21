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
DVIFM_LICENCE = 'code (MIT OR Apache-2.0) AND BSD-3-Clause-Clear (port of Nokia\'s Clear BSD releases)'
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
    # talk-inspired presets: our own reconstruction, so there is no reference implementation to match
    talk = re.findall(r'^\| `([a-z0-9-]+)`: ', dv.read_text(), re.M) if dv.exists() else []
    for preset in set(talk) - set(DVIFM.values()):
        out.setdefault('dvifmish-' + preset, dict(
            family='FR · Classical', reference='dvifmish talk-inspired preset (BMVA talk, 2026-02-04)',
            parity='Not applicable', licence=DVIFM_LICENCE, source='dvifmish README.md'))
    for preset in DVIFM.values():
        g = rows.get(preset)
        out['dvifmish-' + preset] = dict(
            family='FR · Classical', reference=f"M. Pedzisz's C++ DVIFM {preset.removeprefix('port-')} (AIC-4 contribution)",
            parity='Reported results' if g else 'Not documented',
            evidence=(f'mean |ΔJND| {g[1]} vs his builds on 9,618 AIC2026 crops; 840×944 1 thread: C++ {g[3]} ms → '
                      f'dvifmish {g[4]} ms' if g else None),
            licence=DVIFM_LICENCE, paper='DVIFM (AIC-4 proposal; BMVA talk 2026-02-04)',
            source='dvifmish README.md (port mode table)')
    return out


def cell_columns(cell):
    """Column names in one status-doc table cell: each backticked name, with `X1`–`X5` ranges expanded."""
    names = re.findall(r'`([^`]+)`', cell)
    if not names:
        return [cell.strip()]
    m = re.search(r'`([A-Za-z_-]+?)(\d+)`\s*[–-]\s*`\1(\d+)`', cell)
    if m:
        names += [f'{m.group(1)}{i}' for i in range(int(m.group(2)) + 1, int(m.group(3)))]
    return names


def aic_status():
    """Per published AIC2026 column: how docs/AIC2026_REPRODUCTION_STATUS.md classifies our match."""
    st = (ROOT / 'docs/AIC2026_REPRODUCTION_STATUS.md').read_text()
    def sec(a, b):
        i = st.index(a); return st[i:st.index(b, i + len(a))]
    def firsts(body):
        return [c for l in body.splitlines() if l.startswith('| ') and not l.startswith('| Column') and not set(l) <= set('|-: ')
                for c in cell_columns(l.split('|')[1])]
    out = {}
    for col in re.findall(r'`([^`]+)`', re.sub(r'\([^)]*\)', '', sec('## Reproduced', '## Explicit AIC routes'))):
        out[col] = 'reproduced'
    for col in firsts(sec('## Identified by configuration search', '## Identified: the column')):
        out.setdefault(col, 'matched by configuring the reference')
    for col in firsts(sec('## Identified: the column', '## Close but not exact')):
        out.setdefault(col, 'matched pyiqa defaults')
    for col in firsts(sec('## Close but not exact', '## Settled')):
        out.setdefault(col, 'close, minor residual')
    for col in firsts(sec('## Implemented but not reproduced', '## Identified by')):
        out.setdefault(col, 'not reproduced')
    return out


def spotcheck():
    """Primary-oracle execution coverage (latest port spot-check): metric -> 'status (passes/expected)'."""
    files = sorted(ROOT.glob('benchmarks/port_spotcheck_*/execution-coverage.tsv'))
    if not files:
        return {}, None
    out = {}
    for r in csv.DictReader(files[-1].open(), delimiter='\t'):
        out[r['metric']] = f"{r['status']} ({r['scalar_pass_cases']}/{r['expected_scalar_cases']})" if r['status'].startswith('scalar') else r['status']
    return out, files[-1].relative_to(ROOT).as_posix()


PASSED = {'pass', 'parity_pass', 'bit-identical', 'both-refused'}  # spot-check row statuses that are agreement


def recorded_diffs():
    """Per metric, the measured differences behind a parity record, so a deviation is never shown without its numbers.

    Rows come from three committed sources: the latest primary-oracle spot check (every case that is not a pass),
    native goldens whose rows are regression floors rather than 1e-3 JND matches (kind=pinned), and the
    `measured_*` lines of a model card's [parity] table."""
    out = {}
    files = sorted(ROOT.glob('benchmarks/port_spotcheck_*/execution-coverage.tsv'))
    if files:
        for r in csv.DictReader(files[-1].open(), delimiter='\t'):
            ev = files[-1].parent / (r.get('evidence') or '')
            if not r.get('evidence') or not ev.is_file():
                continue
            cases = {}
            for shard in json.loads(ev.read_text()).get('shards') or []:
                for row in csv.DictReader((ev.parent / shard['path']).open(), delimiter='\t'):
                    if row.get('status') in PASSED or row.get('passed') == 'True' or row.get('metric', r['metric']) != r['metric']:
                        continue
                    w, h = row.get('width'), row.get('height')
                    key = (row.get('pair'), f'{w}×{h}' if w else row.get('size'), row.get('status'))
                    delta = row.get('max_abs_delta') or row.get('absolute_delta') or None
                    c = cases.setdefault(key, dict(source='primary oracle', case=f'{key[0]} {key[1]}', threads=[], status=key[2],
                                                    delta=delta, bound=row.get('bound') or row.get('score_relative_bound') or None,
                                                    bound_kind=row.get('bound_kind') or ('relative' if row.get('score_relative_bound') else None),
                                                    ours=row.get('ours'), reference=row.get('canonical_first') or row.get('canonical') or None))
                    if row.get('threads') not in c['threads']:
                        c['threads'].append(row.get('threads'))
            if cases:
                out.setdefault(r['metric'], []).extend(cases.values())
    for card in sorted(ROOT.glob('crates/*/models/*/card.toml')):
        name = card.parent.name
        par = tomllib.loads(card.read_text()).get('parity') or {}
        jnd = par.get('jnd')
        jnd = float(jnd.split()[0]) if isinstance(jnd, str) and re.match(r'[\d.]+', jnd) else jnd
        for g in sorted(card.parent.glob('*goldens*.tsv')):
            rows = csv.DictReader((l for l in g.open() if not l.startswith('#')), delimiter='\t')
            if not {'kind', 'residual', 'tol'} <= set(rows.fieldnames or []):
                continue
            engine = g.stem.rsplit('_', 1)[-1] if '_netlib_' in g.stem else g.stem
            engine = {'cuda': 'ort-cuda'}.get(engine, engine)
            for row in rows:
                if row['kind'] != 'pinned':
                    continue
                out.setdefault(name, []).append(dict(
                    source=f'{engine} golden ({g.name})', case=f"image {row['image']} {row['size']}²", status='pinned regression floor',
                    delta=row['residual'], delta_jnd=f"{float(row['residual']) / jnd:.3g}" if jnd else None,
                    bound=f'{1e-3 * jnd:.4g}' if jnd else None, bound_kind='absolute (1e-3 JND)'))
        for k, v in par.items():
            if k.startswith('measured_') and isinstance(v, str):
                out.setdefault(name, []).append(dict(source=f"model card {k.removeprefix('measured_').replace('_', ' ')}", text=v))
        if isinstance(par.get('blocker'), str):
            out.setdefault(name, []).append(dict(source='model card blocker', text=par['blocker']))
    return out


def csv_evidence(metrics, routes):
    """Per metric, evidence against its published AIC2026 column beyond the full-set run: the retained 61-pair panel,
    private aggregates (site-JND where they record it) and the status-doc row that classifies the column. DVIFM port
    presets also pick up the doc rows that name them ('port mode' or the preset), since no column map lists them."""
    doc = status_rows()
    panel = {}
    for r in csv.DictReader((ROOT / 'benchmarks/dashboard/aic2026/retained-comparisons.tsv').open(), delimiter='\t'):
        col = re.split(r' / | \(', r['column'])[0].strip()
        panel[col] = dict(column=col, n=r['n'], median=r['median'] or None, max=r['maximum'] or None, max_jnd=r['max_jnd'] or None,
                          source='61-pair panel')
    for x in json.loads((ROOT / 'benchmarks/dashboard/aic2026/private-aggregates.json').read_text())['models']:
        j = re.search(r'maximum ([\d.e-]+) JND', x['published_match'])
        panel.setdefault(x['published_column'], dict(column=x['published_column'], n='61', max_jnd=j and j[1],
                                                     text=x['published_match'], source='private aggregate'))
    out = {}
    for m in metrics:
        name = m['metric']
        cols = ({m['column']} if m['column'] else set()) | {c for c, ms in routes.items() if name in ms}
        rows = {c: r for c, r in doc.items() if c in cols}
        if name.startswith('dvifmish-port-'):
            preset = name.removeprefix('dvifmish-')
            rows.update({c: r for c, r in doc.items() if c.startswith('proposal-DVIFM')
                         and any(k == 'Ours' and ('port mode' in v or preset in v) for k, v in r['cells'])})
        ev = [panel[c] for c in sorted(cols) if c in panel]
        merged = {}  # one table row can name several columns
        for c, r in sorted(rows.items()):
            text = '; '.join(f'{k}: {v}' for k, v in r['cells'] if k not in ('What to ask', 'Evidence', 'Open detail'))
            merged.setdefault(text, dict(column=[], source='status doc', section=r['section'], text=text,
                                         ours=dict(r['cells']).get('Ours')))['column'].append(c)
        ev += [dict(e, column=', '.join(e['column'])) for e in merged.values()]
        if ev:
            out[name] = ev
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


HOSTS = {'i265': 'Intel Core Ultra 7 265K', 'r7900x': 'AMD Ryzen 9 7900X', 'mac': 'Apple M4 Pro'}


def sig(v, n=4):
    return float(f'{v:.{n}g}')


_ANCESTRY = {}


def newer_source(a, b):
    """The newer of two source commits when one descends from the other, else None (equal or unrelated)."""
    if a == b:
        return None
    if (a, b) not in _ANCESTRY:
        q = lambda x, y: subprocess.run(['jj', '-R', str(ROOT), 'log', '--no-graph', '-r', f'{x} & ::{y}', '-T', 'commit_id'],
                                        capture_output=True, text=True).stdout.strip()
        _ANCESTRY[(a, b)] = b if q(a, b) else a if q(b, a) else None
    return _ANCESTRY[(a, b)]


def canonical_perf():
    """Cross-host canonical timings (benchmarks/iqa_saved_latest): preferred, qualified, ok rows only.

    Warm cells keep the published summary of their rounds (median, mean, p10, p90, n); cold cells keep
    each fresh-process repeat. Different binaries are never pooled: a (host, metric, size, threads, phase)
    cell measured by several builds uses the newest source (commit ancestry) once it has as many rows, and
    the others are counted as superseded; unrelated or equal sources keep the earlier series.
    """
    d = ROOT / 'benchmarks/iqa_saved_latest'
    if not d.exists():
        return None
    # Group rows by cell and implementation identity first; a cell measured by two builds (a remeasurement of a
    # changed metric) is never pooled: the newer source wins once it has as many cold repeats, else the older stays.
    groups, sizes = {}, {}
    for f in sorted(d.glob('*-observations.csv')):
        for r in csv.DictReader(f.open()):
            if r['status'] != 'ok' or r['preferred'] != 'True' or r['qualified_timing'] != 'True' or r['only_mode'] != 'score':
                continue
            key = (r['host'], r['metric_name'], r['size_name'], int(r['threads']), r['timing_phase'])
            sizes[r['size_name']] = (int(r['width']), int(r['height']))
            idn = (r['impl_binary_sha'], r['engine'], r['reference_sha256'], r['distorted_sha256'])
            groups.setdefault(key, {}).setdefault(idn, []).append(r)
    ident, cells, conflicts, superseded = {}, {}, 0, 0
    for key, by in groups.items():
        order = sorted(by, key=lambda i: max(r['capture_utc'] for r in by[i]))
        best = order[0]
        for cand in order[1:]:
            newer = newer_source(by[best][0]['source_commit'], by[cand][0]['source_commit'])
            if newer is None:
                conflicts += 1  # same or unrelated sources: keep the earlier series, count the other
            elif newer == by[cand][0]['source_commit'] and len(by[cand]) >= len(by[best]):
                best = cand
            superseded += 1
        ident[key] = best
        c = cells.setdefault(key, {})
        for r in by[best]:
            c['source'] = r['source_commit'][:12]; c['binary'] = r['impl_binary_sha'][:12]
            if r['timing_phase'] == 'warm':
                c.update(med=sig(float(r['median_ns']) / 1e6), mean=sig(float(r['mean_ns']) / 1e6) if r['mean_ns'] else None,
                         p10=sig(float(r['p10_ns']) / 1e6) if r['p10_ns'] else None, p90=sig(float(r['p90_ns']) / 1e6) if r['p90_ns'] else None,
                         n=int(r['n']), clean=int(r['warm_rounds_clean'] or 0), flagged=int(r['warm_rounds_flagged'] or 0),
                         rounds=[sig(float(v) / 1e6) for v in (r.get('round_means_ns') or '').split(';') if v] or None)
            else:  # cold: one row per fresh-process repeat
                c.setdefault('reps', []).append((int(r['cold_repeat'] or 0), sig(float(r['median_ns']) / 1e6)))
            for k in ('simd_tier', 'placement_policy', 'placement_cpus', 'rayon_threads_observed'):
                if r.get(k):
                    c[k] = r[k]
            if r.get('peak_rss_gib_rounded'):
                c['rss'] = max(c.get('rss', 0), float(r['peak_rss_gib_rounded']))
    # Individual warm rounds (and their gate status) from the rounds projection, matched on the same identity.
    for f in sorted(d.glob('*-rounds.csv')):
        got = {}
        for r in csv.DictReader(f.open()):
            if r['timing_phase'] != 'warm' or r['preferred'] != 'True' or r['only_mode'] != 'score' or not r['time_ns']:
                continue
            key = (r['host'], r['metric_name'], r['size_name'], int(r['threads']), 'warm')
            if key not in cells or ident.get(key) != (r['impl_binary_sha'], r['engine'], r['reference_sha256'], r['distorted_sha256']):
                continue
            got.setdefault(key, []).append((int(r['round_index']), float(r['time_ns']) / 1e6, r['gate_clean'] == 'True'))
        for key, v in got.items():
            v.sort()
            cells[key]['rounds'] = [sig(t) for _, t, _ in v]
            cells[key]['flagged_rounds'] = [i for i, (_, _, ok) in enumerate(v) if not ok] or None
    for c in cells.values():
        if c.get('reps'):
            c['reps'] = [v for _, v in sorted(c['reps'])]
    rows = [[h, m, s, t, ph, c.get('med'), c.get('mean'), c.get('p10'), c.get('p90'), c.get('n'), c.get('reps'), c.get('rounds'),
             c.get('clean'), c.get('flagged'), c.get('rss'), c.get('simd_tier'), c.get('placement_policy'), c.get('placement_cpus'), c.get('flagged_rounds'),
             c.get('source'), c.get('binary')]
            for (h, m, s, t, ph), c in sorted(cells.items())]
    cov = json.loads((d / 'coverage.json').read_text()) if (d / 'coverage.json').exists() else {}
    sync = json.loads((d / 'sync.json').read_text()) if (d / 'sync.json').exists() else {}
    retired = sync.get('retired_hosts') or {}
    coverage = {}
    for h, v in cov.items():
        if 'cells' in v:
            # Each logical cell once across a parent campaign and its reruns (publish_saved_results.cell_coverage).
            c = v['cells']
            coverage[h] = dict(ok=c['ok'], cells=c['planned'], refused=c['refused_by_design'], failed=c['failed'], pending=c['pending'])
        else:
            # Older publications: per-campaign counts overlap parents and successors, so pending is overstated.
            ok = sum(c.get('observed_cell_statuses', {}).get('ok', 0) for c in v.get('campaigns', []))
            total = sum(c.get('manifest_cells', 0) for c in v.get('campaigns', []))
            coverage[h] = dict(ok=ok, cells=total, refused=0, failed=0, pending=total - ok)
        if h in retired:
            coverage[h]['retired'] = retired[h]
    order = sorted(sizes, key=lambda s: sizes[s][0] * sizes[s][1])
    return dict(fields=['host', 'metric', 'size', 'threads', 'phase', 'median_ms', 'mean_ms', 'p10_ms', 'p90_ms', 'n', 'cold_ms', 'rounds_ms',
                        'clean', 'flagged', 'rss_gib', 'simd', 'placement', 'placement_cpus', 'flagged_rounds', 'source', 'binary'],
                rows=rows, hosts={h: HOSTS.get(h, h) for h in sorted({r[0] for r in rows})},
                sizes=[dict(name=s, w=sizes[s][0], h=sizes[s][1]) for s in order],
                coverage=coverage, synced_utc=sync.get('updated_utc'), identity_conflicts_dropped=conflicts, superseded_series=superseded,
                protocol=json.loads((d / 'protocol.json').read_text()).get('protocol') if (d / 'protocol.json').exists() else None)


# ---------------------------------------------------------------- AIC2026 reproduction and coherence checks

GH = 'https://github.com/imazen/zenmetrics'


def doc_routes():
    """Published column -> our CLI names, from the tables in docs/AIC2026_REPRODUCTION_STATUS.md (first two cells)."""
    out = {}
    for l in (ROOT / 'docs/AIC2026_REPRODUCTION_STATUS.md').read_text().splitlines():
        if not l.startswith('| ') or set(l) <= set('|-: '):
            continue
        cells = l.strip().strip('|').split('|')
        if len(cells) < 2:
            continue
        for col in cell_columns(cells[0]):
            out.setdefault(col, set()).update(re.findall(r'`([^`]+)`', cells[1]))
    return out


def plain(md):
    """Markdown cell text without code ticks, links reduced to their text."""
    md = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', md)
    return md.replace('`', '').replace('\\|', '|').strip()


def status_rows():
    """Published column -> {'section', 'cells': [(header, text), ...]} from every table in the status doc."""
    out, section, header = {}, None, None
    for l in (ROOT / 'docs/AIC2026_REPRODUCTION_STATUS.md').read_text().splitlines():
        if l.startswith('## '):
            section, header = l[3:].strip(), None
            continue
        if not l.startswith('| ') or set(l) <= set('|-: '):
            continue
        cells = [c for c in re.split(r'(?<!\\)\|', l.strip().strip('|'))]
        if cells[0].strip() == 'Column':
            header = [plain(c) for c in cells]
            continue
        if not header:
            continue
        for col in cell_columns(cells[0]):
            out.setdefault(col, dict(section=section, cells=[(header[i], plain(c)) for i, c in enumerate(cells) if i and i < len(header)]))
    return out


def changed_since(commit):
    """Files under crates/ that differ between the reproduction's source commit and the revision being built."""
    r = subprocess.run(['jj', '-R', str(ROOT), 'diff', '--from', commit, '--to', 'heads(::@ & ::master@origin)', '--name-only', 'crates'],
                       capture_output=True, text=True)
    return None if r.returncode else [l for l in r.stdout.split() if l]


def affected_metrics(files, names):
    """Changed source files -> the metrics they can affect (by file or crate name; heuristic, stated as such)."""
    out, generic = {}, []
    for f in files:
        p = f.split('/')
        if 'examples' in p or f.endswith('Cargo.toml'):
            continue
        if p[1] == 'zenmetrics-cli' and p[2:4] == ['src', 'metrics']:
            stem = Path(f).stem.replace('_', '-')
            ms = [n for n in names if n == 'psnr' or n.startswith('psnr-')] if stem in ('psnr-kernels', 'classical') else \
                 [n for n in names if n == stem or n.startswith(stem + '-')]
        elif p[1] in ('zenmetrics-math', 'zenmetrics-api', 'zenmetrics-cli', 'zenmetrics-nn'):
            generic.append(f); continue
        else:
            base = p[1].removesuffix('-iqa').removesuffix('-gpu')
            ms = [n for n in names if n == base or n.startswith(base + '-')]
        for n in ms:
            out.setdefault(n, []).append(f)
        if not ms:
            generic.append(f)
    return out, generic


def reproduction(names):
    """Committed AIC2026 reproduction sets (compare TSVs, provenance) and canonical-implementation comparisons."""
    sys.path.insert(0, str(ROOT / 'scripts/aic2026'))
    from aic_columns import COLS, REFERENCE, REF_META
    import repro_check
    out = dict(sets=[], canonical=[], routes={}, july_columns=[], checks=repro_check.run())
    cmap = json.loads((HERE / 'aic2026_columns.json').read_text())['map']
    routes = {c: set() for c in COLS}
    for c, m in cmap.items():
        routes.setdefault(c, set()).add(m)
    for c, ms in doc_routes().items():
        routes.setdefault(c, set()).update(ms)
    for c, (tag, _, _) in COLS.items():
        routes[c].add(tag)
    out['july_columns'] = [c for c in REFERENCE['columns'] if c not in REF_META and not c.startswith('JND_')]
    out['reference'] = {k: REFERENCE[k] for k in ('name', 'sha256', 'url')}
    for d in sorted(ROOT.glob('benchmarks/aic2026_*')):
        for cmp in sorted(d.glob('compare_*.tsv')):
            kind = cmp.stem.removeprefix('compare_')
            rows = list(csv.DictReader(cmp.open(), delimiter='\t'))
            rel = cmp.relative_to(ROOT).as_posix()
            if kind.startswith('canonical_'):
                out['canonical'].append(dict(set=kind.removeprefix('canonical_'), path=rel, url=f'{GH}/blob/master/{rel}', rows=rows))
                continue
            prov_p = d / f'provenance_{kind}.json'
            prov = json.loads(prov_p.read_text()) if prov_p.exists() else {}
            commit = next(iter(((prov.get('source') or {}).get('commit') or '').split()), None)
            gz = d / f'metrics_{kind}.zenmetrics.csv.gz'
            changed = changed_since(commit) if commit else None
            aff, generic = affected_metrics(changed or [], names)
            out['sets'].append(dict(
                set=kind, path=rel, url=f'{GH}/blob/master/{rel}', affected=aff, generic_changes=generic,
                csv_url=f'{GH}/raw/master/{gz.relative_to(ROOT).as_posix()}' if gz.exists() else None,
                commit=commit and commit[:12], binary=((prov.get('binary') or {}).get('sha256') or '')[:12] or None,
                generated=prov.get('generated'), changed=changed,
                rows=[r for r in rows if r['match'] not in ('info-only',)]))
    doc = status_rows()
    out['misses'] = [dict(set=st['set'], column=r['column'], cli=r['cli'], n=r['n'], median_d=r['median_d'], max_d=r['max_d'],
                          median_dJND=r['median_dJND'], max_dJND=r['max_dJND'], spearman=r['spearman'],
                          section=(doc.get(r['column']) or {}).get('section'), cells=(doc.get(r['column']) or {}).get('cells') or [])
                     for st in out['sets'] for r in st['rows'] if r['match'] == 'MISS']
    out['routes'] = {c: sorted(m for m in ms if m in names) for c, ms in routes.items()}
    out['routes'] = {c: ms for c, ms in out['routes'].items() if ms}
    return out


def coherence(metrics, perf, repro, aic, sync=None):
    """Build-time checks: names that resolve nowhere, doc claims that the data contradicts, stale or old inputs."""
    out = list(repro['checks'])
    add = lambda level, check, detail: out.append(dict(level=level, check=check, detail=detail))
    known = {m['metric'] for m in metrics}
    norm = lambda m: m.replace('_', '-')
    for m in sorted({r[1] for r in (perf or {}).get('rows', [])}):
        if m not in known and norm(m) not in known:
            add('info', 'timings: metric without a record', f'{m} is timed but has no counts, card or reproduction status')
    july = set(repro['july_columns'])
    for m in repro.get('misses', []):
        if not m['cells']:
            add('warn', 'miss without a recorded cause', f"{m['column']} misses the gate in the {m['set']} set but has no row in docs/AIC2026_REPRODUCTION_STATUS.md")
    for s in repro['sets']:
        for r in s['rows']:
            st_ = aic.get(r['column'])
            if r['match'] == 'MISS' and st_ in ('reproduced', 'matched by configuring the reference', 'matched pyiqa defaults'):
                add('warn', 'status doc vs full reproduction', f"{r['column']}: the status doc says {st_}, the {s['set']} set (route {r['cli']}) misses the 1e-3 "
                    f"site-JND gate (median {r['median_dJND']}, n {r['n']})")
            if r['match'] == 'MATCH' and st_ in ('close, minor residual', 'not reproduced'):
                add('info', 'status doc vs full reproduction', f"{r['column']}: the status doc says {st_}, the {s['set']} set matches (median {r['median_dJND']})")
        scored = {r['cli'] for r in s['rows']}
        hit = sorted(m for m in s['affected'] if m in scored)
        if hit:
            add('warn', f"{s['set']} reproduction older than the code", f"since its source commit {s['commit']}, code changed for scored routes "
                + '; '.join(f"{m} ({', '.join(Path(f).name for f in s['affected'][m])})" for m in hit) + '; rescore those columns or confirm the change is bit-exact')
        if s['generic_changes']:
            add('info', f"{s['set']} reproduction older than shared code", f"shared files changed since {s['commit']}: {', '.join(s['generic_changes'])}")
        if s['changed'] is None and s['commit']:
            add('warn', f"{s['set']} reproduction", f"could not diff crates/ against its source commit {s['commit']}")
    for c in sorted(set(aic) - july):
        if july:
            add('info', 'status doc column not in the published tables', f'{c} is classified in the status doc but is not a column of the July/drop published tables')
    cov = (perf or {}).get('coverage') or {}
    # Finished campaigns produce no new results; a quiet sync is then expected, not a stall.
    live_pending = any(c.get('pending') and not c.get('retired') for c in cov.values())
    if perf and perf.get('synced_utc') and (live_pending or not cov):
        import datetime
        age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(perf['synced_utc'])
        if age.total_seconds() > 3 * 3600:
            add('warn', 'no new timings', f"newest timing evidence is from {perf['synced_utc'][:16]} UTC, {age.total_seconds() / 3600:.0f} h before this build; "
                "the five-minute sync commits only when the campaigns produce new results, so the campaigns have produced none since")
    out += sync_checks(sync)
    if perf and perf.get('superseded_series'):
        add('info', 'timings superseded', f"{perf['superseded_series']} cell series measured by an older build are superseded by a newer build's complete series (or kept until it is complete)")
    if perf and perf.get('identity_conflicts_dropped'):
        add('warn', 'timings identity', f"{perf['identity_conflicts_dropped']} rows dropped: one cell came from more than one binary or input")
    return out


SYNC_REASON = re.compile(r'[a-z][a-z ]{2,60}( \((exit \d+|[A-Za-z]+Error|[A-Za-z]+Exception)\))?')


def sync_checks(sync, now=None):
    """The five-minute results sync's private status (fetched by the watcher) as page checks.

    Only the state, times, cycle count and the coordinator's fixed reason class reach the page; the
    diff summary and push-log tail stay private (a hygiene refusal quotes the lines it matched)."""
    import datetime
    if sync is None:
        return []
    out = []
    add = lambda level, check, detail: out.append(dict(level=level, check=check, detail=detail))
    if sync.get('state') == 'unreachable':
        add('warn', 'results sync status unavailable', 'the build could not read the five-minute sync status; whether new timings are being published is unknown')
        return out
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = lambda t: (now - datetime.datetime.fromisoformat(t)).total_seconds() / 60
    if sync.get('updated_utc') and age(sync['updated_utc']) > 20:
        add('error', 'results sync not running', f"the five-minute sync last ran at {sync['updated_utc'][:16]} UTC, {age(sync['updated_utc']):.0f} min before this build")
    if sync.get('blocked'):
        reason = sync.get('reason', '')
        reason = reason if SYNC_REASON.fullmatch(reason) else 'unclassified failure'
        since = sync.get('stalled_since') or sync.get('updated_utc')
        minutes, n = age(since), sync.get('blocked_cycles', 1)
        add('error' if minutes >= 15 else 'warn', 'results sync blocked',
            f"new timings are captured but not published: publishing has been blocked since {since[:16]} UTC "
            f"({minutes:.0f} min, {n} cycle{'s' if n != 1 else ''}); reason: {reason}")
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--private', type=Path, help='zenmetrics-private checkout (read from origin/main with git show)')
    ap.add_argument('--agreement', action='append', default=[], help='aggregate agreement JSON: list of {column|metric, within_source_srcc|agreement, ci_lo|lo, ci_hi|hi, overall_abs_srcc|overall}')
    ap.add_argument('--timing', action='append', default=[], help='extra perf_deltas TSV measured on i265')
    ap.add_argument('--sync-status', type=Path, help="the five-minute results sync's private status.json (only its state and reason class reach the page)")
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

    aic = aic_status()
    spot, spot_src = spotcheck()
    diffs = recorded_diffs()
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
            agreement=agree.get(m), canonical=canon.get(m),
            aic_match=aic.get(col_of.get(m)) if col_of.get(m) else None, spotcheck=spot.get(m), diffs=diffs.get(m) or []))

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

    perf = canonical_perf()
    repro = reproduction({m['metric'] for m in metrics} | {r[1] for r in (perf or {}).get('rows', [])})
    csv_ev = csv_evidence(metrics, repro.get('routes') or {})
    for m in metrics:
        m['csv_evidence'] = csv_ev.get(m['metric']) or []
    doc = dict(schema=1, host=HOST, shape=f'{W}x{H}', seed=SEED,
               sources=dict(public=public_rev(), private=priv and git_rev(priv, 'origin/main'),
                            dvifmish=git_rev(Path.home() / 'work/dvifmish', 'origin/main'),
                            extra_timing=[Path(p).name for p in a.timing], agreement=[Path(p).name for p in a.agreement],
                            spotcheck=spot_src),
               dvifmish_versions=[dict(metric='dvifmish-' + v, preset=v, aggregate_name=k) for k, v in DVIFM.items()],
               tables=tables, metrics=metrics, stats=stats, fronts=fronts, notes=notes, perf=perf, repro=repro,
               checks=coherence(metrics, perf, repro, aic, json.loads(a.sync_status.read_text()) if a.sync_status else None))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, allow_nan=False))
    n1 = sum(1 for r in metrics if r['estimate'] and r['cpu_1t'])
    print(f'{a.out}: {len(metrics)} metrics, {n1} with count and 1-thread time, {len(stats)} stat rows; notes: {notes}')
    for c in doc['checks']:
        print(f"check {c['level']}: {c['check']}: {c['detail']}")


if __name__ == '__main__':
    main()
