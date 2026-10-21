#!/usr/bin/env python3
"""Render the metric-cost dashboard from brief_data.json: one self-contained page plus CSV downloads.

No network requests: system fonts, inline CSS and script, charts drawn in the browser from the
embedded aggregates. Tabs: overview, speed, size scaling, counted cost, reproduction, data, method.
`index.html?print=1` lays every tab out as a printable page (brief.sh prints it to brief.pdf).

    brief_page.py --data brief_data.json --out DIR    # writes DIR/index.html, DIR/timings.csv, DIR/complexity.csv

Every number on the page comes from the data file; brief_data.py documents its sources.
"""
import argparse, csv, datetime, json, math, re
from pathlib import Path

HERE = Path(__file__).with_name('brief_page')
ROOT = Path(__file__).resolve().parents[2]
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086]

# Evidence classes used for the deviation/unchecked marks (see parity()).
AIC_OK = {'reproduced', 'matched pyiqa defaults', 'matched by configuring the reference'}
AIC_DEV = {'close, minor residual', 'not reproduced'}
PARITY_OK = {'Reported results', 'Declared gate'}
PARITY_DEV = {'Partial canonical parity'}
PARITY_NA = {'Not applicable'}  # our own design: no external reference to check against


def family(kind, origin):
    if origin == 'dvifmish': return 'dvifmish'
    if origin == 'private': return 'private'
    return 'neural' if (kind or '').startswith('nn') else 'classic'


def parity(m):
    """'deviation' when any record shows a difference, 'checked' when one shows agreement, 'own' when there is
    no external reference, else 'unchecked'."""
    aic, spot, par = m.get('aic_match'), m.get('spotcheck') or '', (m.get('canonical') or {}).get('parity')
    why = []
    if aic in AIC_DEV: why.append(f'AIC2026 column: {aic}')
    if spot.startswith('finding'): why.append(f'primary-oracle check: {spot}')
    if par in PARITY_DEV: why.append(f'reference parity: {par.lower()}')
    if why: return 'deviation', why
    if aic in AIC_OK or spot.startswith('scalar-pass') or par in PARITY_OK: return 'checked', []
    if par in PARITY_NA: return 'own', []
    return 'unchecked', []


def aic_categories(md):
    """AIC2026 column reproduction status from docs/AIC2026_REPRODUCTION_STATUS.md, by section."""
    def sec(a, b):
        s = md.index(a); e = md.index(b, s + len(a)); return md[s:e].split('\n', 1)[1]
    def firsts(body):
        return [re.sub('`', '', l.strip().strip('|').split('|')[0].strip()) for l in body.splitlines()
                if l.startswith('| ') and not l.startswith('| Column') and not set(l) <= set('|-: ')]
    repro = sorted(set(re.findall(r'`([^`]+)`', re.sub(r'\([^)]*\)', '', sec('## Reproduced', '## Explicit AIC routes')))))
    return [('Reproduced with our implementation', 'ok', repro),
            ('Matched by configuring a reference implementation', 'ok', firsts(sec('## Identified by configuration search', '## Identified: the column'))),
            ('Matched by pyiqa 0.1.16 defaults', 'ok', firsts(sec('## Identified: the column', '## Close but not exact'))),
            ('Close but not exact', 'dev', firsts(sec('## Close but not exact', '## Settled'))),
            ('Open: questions for the dataset authors', 'open', firsts(sec('## Implemented but not reproduced', '## Identified by'))),
            ('Excluded', 'none', firsts(sec('## Excluded', '## Not implemented'))),
            ('Not implemented', 'none', firsts(sec('## Not implemented', '## Reproduced')))]


def metric_meta(d):
    """Per metric: family, parity evidence and counted costs; perf-only names are aliased or described."""
    meta = {}
    for m in d['metrics']:
        c = m.get('canonical') or {}
        status, why = parity(m)
        meta[m['metric']] = {
            'fam': family(m['kind'], m['origin']), 'kind': m['kind'], 'origin': m['origin'], 'col': m.get('column'),
            'ref': c.get('reference') if c.get('reference') not in (None, 'no evidence') else None,
            'family': c.get('family'), 'parity': c.get('parity'), 'aic': m.get('aic_match'), 'spot': m.get('spotcheck'),
            'status': status, 'why': why, 'diffs': m.get('diffs') or [], 'evidence': c.get('evidence'),
            'est': {k: v for k, v in (m.get('estimate') or {}).items() if v},
            'c1': m.get('cpu_1t'), 'c8': m.get('cpu_8t')}
    for name in sorted({r[1] for r in (d.get('perf') or {}).get('rows', [])} - set(meta)):
        alias = name.replace('_', '-')
        if alias in meta:
            meta[name] = dict(meta[alias], alias=alias)
        else:
            origin = 'dvifmish' if name.startswith('dvifmish-') else 'public'
            meta[name] = {'fam': family('classic-fr', origin), 'kind': 'classic-fr', 'origin': origin, 'col': None, 'ref': None,
                          'family': None, 'parity': None, 'aic': None, 'spot': None,
                          # a timed dvifmish preset outside the port table is one of our talk-inspired designs
                          'status': 'own' if origin == 'dvifmish' and not name.startswith('dvifmish-port-') else 'unchecked', 'why': [],
                          'diffs': [], 'evidence': None, 'est': {}, 'c1': None, 'c8': None}
    repro = d.get('repro') or {}
    cols_of = {}
    for col, ms in (repro.get('routes') or {}).items():
        for m in ms:
            cols_of.setdefault(m, set()).add(col)
    for name, mm in meta.items():
        key = mm.get('alias', name)
        cols = cols_of.get(key, set()) | ({mm['col']} if mm.get('col') else set())
        in_csv = set(repro.get('july_columns') or cols)
        mm['july_cols'] = sorted(cols & in_csv)
        mm['site_cols'] = sorted(cols - in_csv)
        mm['code_changed'] = sorted({f for st in repro.get('sets', []) for f in (st.get('affected') or {}).get(key, [])})
        mm['july'] = [dict(set=st['set'], **{k: r[k] for k in ('column', 'cli', 'n', 'median_d', 'max_d', 'median_dJND', 'max_dJND', 'spearman', 'match')})
                      for st in repro.get('sets', []) for r in st['rows'] if r['cli'] == key]
        mm['canon'] = [dict(set=st['set'], **r) for st in repro.get('canonical', []) for r in st['rows'] if r.get('cli') == key]
    return meta


def cell_stats(F, r):
    """Same statistics the page draws (dash.js cellStats): centre, 95 % interval of the mean, spread, fastest."""
    if r[F['phase']] == 'warm':
        med = r[F['median_ms']]
        if med is None: return None
        rounds = r[F['rounds_ms']] or []
        n = len(rounds) or r[F['n']] or 0
        mean, ci = r[F['mean_ms']] if r[F['mean_ms']] is not None else med, None
        if len(rounds) > 1:
            mean = sum(rounds) / n
            sd = math.sqrt(sum((v - mean) ** 2 for v in rounds) / (n - 1)); h = T975[min(n - 2, 19)] * sd / math.sqrt(n)
            ci = (mean - h, mean + h)
        elif r[F['p10_ms']] is not None and r[F['p90_ms']] is not None and n > 1:
            sd = (r[F['p90_ms']] - r[F['p10_ms']]) / 2.5631; h = T975[min(n - 2, 19)] * sd / math.sqrt(n)
            ci = (mean - h, mean + h)
        return dict(median=med, mean=mean, ci=ci, lo=r[F['p10_ms']], hi=r[F['p90_ms']], fast=min(rounds) if rounds else None, n=n)
    v = sorted(r[F['cold_ms']] or [])
    if not v: return None
    n = len(v); mean = sum(v) / n
    med = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    ci = None
    if n > 1:
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)); h = T975[min(n - 2, 19)] * sd / math.sqrt(n)
        ci = (max(mean - h, v[0] / 2), mean + h)
    return dict(median=med, mean=mean, ci=ci, lo=v[0], hi=v[-1], fast=v[0], n=n)


def write_csvs(d, meta, out):
    perf = d.get('perf') or {}
    F = {f: i for i, f in enumerate(perf.get('fields', []))}
    sizes = {s['name']: s for s in perf.get('sizes', [])}
    g = lambda v: '' if v is None else f'{v:.6g}'
    with (out / 'timings.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['host', 'cpu', 'metric', 'family', 'parity_evidence', 'size', 'width', 'height', 'threads', 'phase',
                    'median_ns_per_px', 'mean_ns_per_px', 'ci95_lo_ns_per_px', 'ci95_hi_ns_per_px', 'spread_lo_ns_per_px', 'spread_hi_ns_per_px',
                    'fastest_ns_per_px', 'median_ms', 'n', 'clean_rounds', 'flagged_rounds', 'peak_rss_gib', 'simd_tier', 'placement', 'cold_runs_ms',
                    'source_commit', 'binary_sha256'])
        for r in sorted(perf.get('rows', []), key=lambda r: (r[F['host']], r[F['metric']], sizes[r[F['size']]]['w'] * sizes[r[F['size']]]['h'], r[F['threads']], r[F['phase']])):
            s = cell_stats(F, r)
            if not s: continue
            z = sizes[r[F['size']]]; k = 1e6 / (z['w'] * z['h'])
            m = meta.get(r[F['metric']], {})
            sc = lambda v: g(None if v is None else v * k)
            w.writerow([r[F['host']], perf['hosts'].get(r[F['host']], ''), r[F['metric']], m.get('fam', ''), m.get('status', ''), r[F['size']], z['w'], z['h'],
                        r[F['threads']], r[F['phase']], sc(s['median']), sc(s['mean']), sc(s['ci'] and s['ci'][0]), sc(s['ci'] and s['ci'][1]),
                        sc(s['lo']), sc(s['hi']), sc(s['fast']), g(s['median']), s['n'],
                        '' if r[F['clean']] is None else r[F['clean']], '' if r[F['flagged']] is None else r[F['flagged']],
                        g(r[F['rss_gib']]), r[F['simd']] or '', r[F['placement']] or '',
                        ';'.join(g(v) for v in (r[F['cold_ms']] or [])),
                        r[F['source']] if 'source' in F else '', r[F['binary']] if 'binary' in F else ''])
    tables = [t['id'] for t in d['tables']]
    with (out / 'complexity.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'family', 'i265_cpu_ns_per_px_1t_1024', 'i265_cpu_ns_per_px_8t_1024'] + [f'counted_{t}' for t in tables])
        for m in d['metrics']:
            mm = meta[m['metric']]
            if not (mm['est'] or mm['c1']): continue
            w.writerow([m['metric'], mm['fam'], g(mm['c1']), g(mm['c8'])] + [g(mm['est'].get(t)) for t in tables])


def undocumented(meta):
    """A deviation must come with the numbers that show it: a full-set row, a reference comparison or a recorded diff."""
    return [dict(level='warn', check='deviation without a diff list',
                 detail=f"{m}: marked as a deviation ({'; '.join(mm['why'])}) but no measured difference is recorded")
            for m, mm in sorted(meta.items())
            if mm['status'] == 'deviation' and not mm.get('alias') and not (mm['july'] or mm['canon'] or mm['diffs'])]


def build(data_path, out):
    d = json.loads(Path(data_path).read_text())
    meta = metric_meta(d)
    stats = [s for s in d['stats'] if s['measure'] == 'cpu' and s['subset'] == 'all']
    s1 = {s['table']: s for s in stats if s['threads'] == 1}
    best = min((t for t in d['tables'] if t['id'] in s1 and 'other CPU' not in t['label']), key=lambda t: s1[t['id']]['typical'])['id']
    perf = d.get('perf') or {}
    data = {
        'perf': perf, 'meta': meta, 'tables': d['tables'], 'stats': stats, 'best_table': best, 'seed': d['seed'],
        'fit_host': d.get('host'), 'fit_shape': d.get('shape'),
        'aic': aic_categories((ROOT / 'docs/AIC2026_REPRODUCTION_STATUS.md').read_text()),
        'dvifm': d.get('dvifmish_versions') or [], 'sources': d['sources'], 'notes': d.get('notes') or [],
        'repro': {k: v for k, v in (d.get('repro') or {}).items() if k != 'checks'}, 'checks': (d.get('checks') or []) + undocumented(meta)}
    out.mkdir(parents=True, exist_ok=True)
    write_csvs(d, meta, out)
    built = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    page = (HERE / 'index.html').read_text()
    for k, v in (('/*CSS*/', (HERE / 'style.css').read_text()), ('/*JS*/', (HERE / 'dash.js').read_text()),
                 ('/*DATA*/', blob), ('{{BUILT}}', built)):
        page = page.replace(k, v)
    (out / 'index.html').write_text(page)
    print(f'wrote {out}/index.html ({len(page)//1024} KiB), timings.csv, complexity.csv; default weights {best}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    build(a.data, a.out)
