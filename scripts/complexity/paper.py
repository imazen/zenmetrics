#!/usr/bin/env python3
"""Generate the draft paper: tallied cost vs measured time, figures and tables.

Everything is recomputed from committed inputs:
  tallies     benchmarks/iqa_complexity_2026-10-17/tallies/*.json
  measured    benchmarks/iqa_complexity_2026-10-17/measured/*.tsv
  weights     scripts/complexity/weights_*.json
  draft       scripts/complexity/pedzisz_2026_draft_values.json
Writes docs/paper/IQA_COMPLEXITY_VS_RUNTIME.md and docs/paper/fig/*.svg.
No plotting library: figures are plain SVG.

Halo variants (stated assumptions, not measurements):
  none    tallied cost as is
  pad     cost × (W+2·hx)(H+2·hy)/(W·H): work over the padded domain an output needs
  strip64 cost × (1 + 2·hy/64): recomputation if processed in independent 64-row strips

  paper.py [--check]
"""
import argparse, glob, json, math, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from evaluate import evaluate, validate  # noqa: E402
from correlate import weighted, pearson, ranks, boot  # noqa: E402

B = ROOT/'benchmarks/iqa_complexity_2026-10-17'
OUT = ROOT/'docs/paper'
W, H = 1024, 1024
TABLES = [  # file stem, short label
    ('flops', 'flop count'),
    ('pedzisz_2026_draft', 'Pedzisz draft weights'),
    ('i265_P_rthroughput', 'measured i265 P-core throughput'),
    ('i265_P_latency', 'measured i265 P-core latency'),
    ('llvm_tti_arrowlake_throughput', 'LLVM cost model arrowlake throughput'),
    ('llvm_tti_arrowlake_latency', 'LLVM cost model arrowlake latency'),
    ('llvm_mca_arrowlake_rthroughput', 'llvm-mca arrowlake (ADLP) throughput'),
    ('llvm_tti_znver5_throughput', 'LLVM cost model znver5 throughput (other CPU)'),
]
HALO = {'none': lambda r: 1.0,
        'pad': lambda r: (W + 2 * r['halo_x_px']) * (H + 2 * r['halo_y_px']) / (W * H),
        'strip64': lambda r: 1 + 2 * r['halo_y_px'] / 64}


def measured(threads):
    t = {}
    for path in sorted(glob.glob(str(B/'measured/*.tsv'))):
        for line in Path(path).read_text().splitlines():
            if line.startswith('#') or line.startswith('metric\t') or not line.strip():
                continue
            f = line.split('\t')
            if f[1] == f'{W}x{H}' and f[2] == str(threads) and f[5] != 'error':
                # the default-build 1-thread run is preferred over the parallel build's 1-thread rows
                key = (f[0], 'par' in path)
                t.setdefault(key, []).append((float(f[5]), float(f[-2])))
    out = {}
    for (m, par), v in t.items():
        if threads == 1 and par and (m, False) in t:
            continue
        out[m] = (statistics.median(x for x, _ in v) * 1e6 / (W * H), statistics.median(y for _, y in v) * 1e6 / (W * H))
    return out


def tallies():
    prefer = ['swe2', 'swe', 'lead']
    best = {}
    for f in glob.glob(str(B/'tallies/*.json')):
        t = json.loads(Path(f).read_text())
        if validate(t):
            continue
        r = prefer.index(t['counter']) if t['counter'] in prefer else len(prefer)
        if t['metric'] not in best or r < best[t['metric']][0]:
            best[t['metric']] = (r, t)
    return {m: (t, evaluate(t, W, H)) for m, (_, t) in best.items()}


def stats(xs, ys):
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    sp = pearson(ranks(lx), ranks(ly))
    lo, hi = boot(lx, ly, lambda a, b: pearson(ranks(a), ranks(b)))
    k = statistics.median(b - a for a, b in zip(lx, ly))
    res = [math.exp(b - a - k) for a, b in zip(lx, ly)]
    return dict(n=len(xs), spearman=sp, lo=lo, hi=hi, pearson=pearson(lx, ly), scale=math.exp(k),
                spread=max(res) / min(res), mad=math.exp(statistics.median(abs(math.log(r)) for r in res)))


def svg_scatter(points, title, xlabel, ylabel, scale):
    """points: (x, y, label). Log-log, with the fitted y = scale·x line."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    lo = min(min(xs) * scale, min(ys)) / 1.5
    hi = max(max(xs) * scale, max(ys)) * 1.5
    lox, hix = lo / scale, hi / scale
    X = lambda v: 70 + 560 * (math.log(v) - math.log(lox)) / (math.log(hix) - math.log(lox))
    Y = lambda v: 400 - 360 * (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    def ticks(a, b):
        e = math.floor(math.log10(a))
        while 10 ** e <= b:
            if 10 ** e >= a:
                yield 10 ** e
            e += 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 460" font-family="sans-serif" font-size="11">',
             '<rect width="700" height="460" fill="white"/>',
             f'<text x="350" y="20" text-anchor="middle" font-size="13">{title}</text>',
             '<path d="M70 40V400H630" fill="none" stroke="#333"/>']
    for t in ticks(lox, hix):
        parts.append(f'<path d="M{X(t):.1f} 400v5" stroke="#333"/><text x="{X(t):.1f}" y="418" text-anchor="middle">{t:g}</text>')
    for t in ticks(lo, hi):
        parts.append(f'<path d="M65 {Y(t):.1f}h5" stroke="#333"/><text x="62" y="{Y(t)+4:.1f}" text-anchor="end">{t:g}</text>')
    parts.append(f'<path d="M{X(lox):.1f} {Y(lox*scale):.1f}L{X(hix):.1f} {Y(hix*scale):.1f}" stroke="#bf5825" stroke-dasharray="5 4"/>')
    for x, y, label in points:
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="#2179ac" fill-opacity=".75"><title>{label}</title></circle>'
                     f'<text x="{X(x)+5:.1f}" y="{Y(y)-4:.1f}" font-size="9" fill="#444">{label}</text>')
    parts.append(f'<text x="350" y="448" text-anchor="middle">{xlabel}</text>')
    parts.append(f'<text x="18" y="220" text-anchor="middle" transform="rotate(-90 18 220)">{ylabel}</text></svg>')
    return '\n'.join(parts) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    tal = tallies()
    m1, m8 = measured(1), measured(8)
    files = {}
    rows = []
    for stem, label in TABLES:
        wt = json.loads((HERE/f'weights_{stem}.json').read_text())['weights']
        for halo, fac in HALO.items():
            for threads, meas in ((1, m1), (8, m8)):
                pts = []
                for m, (t, r) in sorted(tal.items()):
                    if m not in meas:
                        continue
                    est = weighted(r, wt)[0] * fac(r)
                    if est > 0:
                        pts.append((est, meas[m][0], m, meas[m][1]))
                if len(pts) < 4:
                    continue
                s_cpu = stats([p[0] for p in pts], [p[1] for p in pts])
                s_wall = stats([p[0] for p in pts], [p[3] for p in pts])
                rows.append((label, halo, threads, s_cpu, s_wall))
                if threads == 1 and halo in ('none', 'pad'):
                    name = f'fig/scatter_{stem}_{halo}.svg'
                    files[name] = svg_scatter([(p[0], p[1], p[2]) for p in pts],
                                              f'{label}, halo {halo}: estimate vs CPU time (i265, 1 thread, 1024²)',
                                              'estimated cost per pixel (table units)', 'measured CPU ns per pixel',
                                              s_cpu['scale'])
    draft = json.loads((HERE/'pedzisz_2026_draft_values.json').read_text())
    dpts = [(v['value'], m1[v['zenmetrics']][0], v['zenmetrics']) for v in draft['values']
            if v['zenmetrics'] and v['zenmetrics'] in m1]
    dstats = stats([p[0] for p in dpts], [p[1] for p in dpts]) if len(dpts) >= 4 else None
    if dstats:
        files['fig/scatter_pedzisz_draft_totals.svg'] = svg_scatter(
            dpts, 'Pedzisz draft totals (840x944, reference implementations) vs our CPU time', 'draft weighted ops per pixel',
            'measured CPU ns per pixel (zenmetrics, i265, 1 thread, 1024²)', dstats['scale'])
    n_tally = len(tal)
    lines = [
        '# Counted cost versus measured runtime for image-quality metrics (draft)', '',
        '*Generated by `scripts/complexity/paper.py` from committed tallies, measurements and weight tables; '
        'regenerate after any of them changes. Draft: the operation inventory is incomplete and single-counter '
        'tallies await independent recounts. Numbers below are what this revision of the inputs gives, nothing more.*', '',
        '## Question', '',
        'Can a source-grounded count of operations per pixel, weighted by per-operation costs, predict how long an '
        'image-quality metric takes, and which cost table predicts best? If it can, counts compare metrics '
        'independently of how much optimization each implementation received; where it cannot, the residual is '
        'implementation effect.', '',
        '## Method', '',
        'Counting follows the approach of M. Pedzisz (Nokia), *Complexity Assessment of Full-Reference Image Quality '
        'Metrics*, a JPEG AIC-4 contribution circulated as a preliminary draft dated 2026-10-07; only part of that '
        'draft was seen (screenshots), and no accuracy is claimed for the values transcribed from it. Conventions, '
        'halo and parallel-structure extensions: [IQA_COMPLEXITY.md](../IQA_COMPLEXITY.md).', '',
        f'- Tallies: {n_tally} of 115 CLI metric names, counted from zenmetrics source; measured on {len(m1)} metrics.',
        '- Measurements: i265 (Intel Core Ultra 7 265K), per-call deltas under `perf stat`, decode and setup excluded, '
        '1024×1024, one content pair, three repeats; 1 thread pinned to a P-core, and 8 threads on P-cores 0-7 from a '
        'build with each metric crate\'s threading enabled.',
        '- Weight tables: flop count; the draft\'s declared weights; measured scalar throughput and latency on the '
        'i265 P-core; LLVM 22 cost model (`print<cost-model>`) for arrowlake and, as a different-CPU control, znver5; '
        'llvm-mca per-instruction throughput (`-mcpu=arrowlake` uses the Alder Lake model).',
        '- Halo variants: none; padded domain (W+2h_x)(H+2h_y)/(WH); strip recomputation 1+2h_y/64.',
        '- Statistics: Spearman rank correlation with a 2,000-sample bootstrap 95 % interval; Pearson of logs; '
        'a single scale fitted as the median log ratio; residual spread = max/min of residual ratios; typical '
        'error = median absolute residual ratio.', '',
        '## Results', '',
        '| weights | halo | threads | n | Spearman CPU (95 %) | Pearson log CPU | typical error CPU | spread CPU | Spearman wall |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for label, halo, threads, c, w in rows:
        lines.append(f"| {label} | {halo} | {threads} | {c['n']} | {c['spearman']:.2f} ({c['lo']:.2f}–{c['hi']:.2f}) | "
                     f"{c['pearson']:.2f} | {c['mad']:.2f}× | {c['spread']:.1f}× | {w['spearman']:.2f} |")
    lines += ['', '### Figures', '']
    for name in sorted(files):
        lines.append(f'![{name}]({name})')
        lines.append('')
    if dstats:
        lines += ['### The draft\'s own totals against our measured time', '',
                  f"{dstats['n']} metrics overlap. Spearman {dstats['spearman']:.2f} ({dstats['lo']:.2f}–{dstats['hi']:.2f}), "
                  f"typical error {dstats['mad']:.2f}×, spread {dstats['spread']:.1f}×. The draft counted third-party "
                  'reference or production implementations (libvmaf, Google butteraugli, the FSIM MATLAB code, HaarPSI '
                  'Python, ColorVideoVDP PyTorch) at 840×944; DVIFM, the only one written by its author, has no '
                  'zenmetrics counterpart here. So this compares two independent things: counts of reference code, '
                  'and the runtime of zenmetrics code.', '']
    # Where the work is: separable vs 2D stencils, transforms, network layers; and full materialization.
    groups = {'separable passes': ('sep_stencil',), '2D stencils': ('stencil2d',), 'transforms': ('transform',),
              'network layers': ('conv', 'matmul'), 'pointwise / reductions': ('pointwise', 'reduction'),
              'resampling': ('downsample', 'upsample', 'resample')}
    lines += ['## Where the work is', '',
              'Share of each metric\'s flops by access class (separable 1-D passes vs 2D windows computed directly, '
              'transforms, network layers), and bytes written per pixel to fully materialized planes (`materialize` = '
              '`full`; "unknown" until the tally records it). From the tallies at 1024×1024.', '',
              '| metric | flops/px | ' + ' | '.join(groups) + ' | other | full-materialized B/px | unannotated B/px |',
              '|---|---:|' + '---:|' * (len(groups) + 3)]
    rpts = []
    wt_res = json.loads((HERE/'weights_i265_P_rthroughput.json').read_text())['weights']
    resid = []
    for m, (t, r) in sorted(tal.items(), key=lambda kv: (-kv[1][1]['flops_per_px'], kv[0])):
        tot = r['flops_per_px'] or 1
        acc = r['flops_per_px_by_access']
        shares = [sum(acc.get(k, 0) for k in ks) / tot for ks in groups.values()]
        other = max(0.0, 1 - sum(shares))
        mat = r['bytes_written_per_px_by_materialize']
        lines.append(f"| {m} | {r['flops_per_px']:.4g} | " + ' | '.join(f'{x:.0%}' for x in shares) +
                     f" | {other:.0%} | {mat.get('full', 0):.4g} | {mat.get('unknown', 0):.4g} |")
        if m in m1 and 'full' in mat:
            est = weighted(r, wt_res)[0]
            if est > 0:
                resid.append((m, mat['full'], m1[m][0] / est))
    if len(resid) >= 4:
        k = statistics.median(r for _, _, r in resid)
        files['fig/residual_vs_materialized.svg'] = svg_scatter(
            [(max(b, 0.1), r / k, m) for m, b, r in resid], 'Measured / estimated CPU time vs fully materialized bytes',
            'bytes written to full planes per pixel (0 shown at 0.1)', 'residual (measured / estimate, median 1)', 1.0)
        lines += ['', '![residual vs materialized](fig/residual_vs_materialized.svg)', '']
    else:
        lines += ['', f'Materialization is annotated for {len(resid)} measured metrics so far; the residual-vs-materialization '
                  'figure appears once at least four are annotated.', '']
    lines += ['## Limits', '',
              '- One host, one shape, one content pair. Nothing here ranks metrics across hosts.',
              '- The rank correlations are dominated by the three-decade spread between the cheapest and most expensive '
              'metrics; the residual spread is the within-range error.',
              '- Per-pixel estimates ignore vectorization, cache behaviour and threading, which the measured times include.',
              '- Halo factors are modelling assumptions, not measured overheads.',
              '- Tallies counted by one lane are provisional until the swapped recounts reconcile them.', '']
    files['IQA_COMPLEXITY_VS_RUNTIME.md'] = '\n'.join(lines)
    stale = []
    for name, text in files.items():
        p = OUT/name
        if a.check:
            if not p.exists() or p.read_text() != text:
                stale.append(name)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    if a.check and stale:
        raise SystemExit('paper stale: ' + ', '.join(stale))
    print(f'{len(files)} files; {len(rows)} result rows; tallies {n_tally}; measured {len(m1)}')


if __name__ == '__main__':
    main()
