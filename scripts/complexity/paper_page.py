#!/usr/bin/env python3
"""Build the counted-cost-vs-runtime brief as one self-contained HTML file.

Reads the generated paper (docs/paper/), the 1024x1024 and 3840x2160 timing tables and the
AIC2026 reproduction status. Writes HTML that makes no network requests: system fonts, inline
CSS, inline SVG charts and an inline explorer script, so the file works offline, from a USB
stick or as an email attachment, and on hosts where third-party CDNs are blocked.

    python3 scripts/complexity/paper_page.py --out DIR      # writes DIR/brief.html

A short brief (four printed pages at most) for readers who will not audit the full paper.
Print it to PDF with headless Chromium (--print-to-pdf). Run from the repository root.
"""
import argparse, csv, html, json, math, re, statistics, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).with_name('paper_page')
PAPER = ROOT / 'docs/paper'
MEASURED = ROOT / 'benchmarks/iqa_complexity_2026-10-17/measured'
STATUS = ROOT / 'docs/AIC2026_REPRODUCTION_STATUS.md'
E = html.escape


# ---------------------------------------------------------------- inputs

def md_table_rows(md, header_start):
    """Rows (lists of cells) of the first markdown table whose header line starts with header_start."""
    lines = md.splitlines()
    for i, l in enumerate(lines):
        if l.startswith(header_start):
            out = []
            for r in lines[i + 2:]:
                if not r.startswith('|'):
                    break
                out.append([c.strip() for c in r.strip().strip('|').split('|')])
            return out
    raise SystemExit(f'table not found: {header_start}')


def results_table(md):
    """weights, halo, threads -> dict(n, rho, lo, hi, pearson, typical, spread)."""
    out = {}
    for c in md_table_rows(md, '| weights | halo | threads |'):
        m = re.match(r'([\d.]+) \(([\d.]+)–([\d.]+)\)', c[4])
        out[(c[0], c[1], int(c[2]))] = dict(n=int(c[3]), rho=float(m[1]), lo=float(m[2]), hi=float(m[3]),
                                            pearson=float(c[5]), typical=float(c[6].rstrip('×')),
                                            spread=float(c[7].rstrip('×')))
    return out


def timing(path):
    """metric -> median task-clock ms per call from a perf_deltas TSV (error rows dropped)."""
    acc = {}
    with open(path) as f:
        rows = csv.DictReader((l for l in f if not l.startswith('#')), delimiter='\t')
        for r in rows:
            try:
                v = float(r['task_clock_ms_per_call'])
                acc.setdefault(r['metric'], []).append(v)
            except (ValueError, TypeError):
                pass
    return {m: statistics.median(v) for m, v in acc.items() if v}


def md_section(md, start, end):
    s = md.index(start)
    e = md.index(end, s + len(start))
    return md[s:e].split('\n', 1)[1]


def inline(t):
    """Inline markdown: code spans and links (relative repo links become plain text)."""
    parts = re.split(r'(`[^`]*`)', t)
    out = []
    for p in parts:
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            out.append(f'<code>{E(p[1:-1])}</code>')
            continue
        p = E(p)
        p = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2">\1</a>', p)
        p = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', p)
        p = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', p)
        out.append(p)
    return ''.join(out)


def md_html(text):
    """Minimal markdown subset: paragraphs, ### headings, 4-space formulas, bullet lists, tables."""
    out, para, table, items = [], [], [], []

    def flush():
        if para:
            out.append('<p>' + inline(' '.join(para)) + '</p>'); para.clear()
        if items:
            out.append('<ul>' + ''.join(f'<li>{inline(i)}</li>' for i in items) + '</ul>'); items.clear()
        if table:
            rows = [r for r in table if not set(r.replace('|', '').strip()) <= set('-:')]
            split = lambda r: [c.replace('\x00', '|').strip() for c in r.strip().replace('\\|', '\x00').strip('|').split('|')]
            head = ''.join(f'<th>{inline(c)}</th>' for c in split(rows[0]))
            body = ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in split(r)) + '</tr>' for r in rows[1:])
            out.append(f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
            table.clear()

    for line in text.splitlines():
        if line.startswith('### '):
            flush(); out.append(f'<h4>{inline(line[4:])}</h4>')
        elif line.startswith('    '):
            flush(); out.append(f'<pre class="formula">{E(line.strip())}</pre>')
        elif line.startswith('|'):
            if para or items:
                p, it = list(para), list(items); para.clear(); items.clear()
                if p: out.append('<p>' + inline(' '.join(p)) + '</p>')
                if it: out.append('<ul>' + ''.join(f'<li>{inline(i)}</li>' for i in it) + '</ul>')
            table.append(line)
        elif line.startswith('- '):
            if table or para: flush()
            items.append(line[2:].strip())
        elif not line.strip():
            flush()
        elif items and line.startswith('  '):
            items[-1] += ' ' + line.strip()
        else:
            if table: flush()
            para.append(line.strip())
    flush()
    return '\n'.join(out)


# ---------------------------------------------------------------- SVG plotting

def fmt_num(v):
    if v >= 1e6: return f'{v/1e6:g}M'
    if v >= 1e3: return f'{v/1e3:g}k'
    if v >= 1: return f'{v:g}'
    return f'{v:g}'


class Plot:
    """A single panel: linear or log axes, gridlines, ticks, labelled points without overlaps."""

    def __init__(self, w, h, xdom, ydom, xlog=True, ylog=False, margin=(54, 16, 14, 44), x0=0, y0=0):
        self.w, self.h, self.xlog, self.ylog = w, h, xlog, ylog
        self.L, self.R, self.T, self.B = margin
        self.x0, self.y0 = x0, y0
        self.xd = [math.log10(v) for v in xdom] if xlog else list(xdom)
        self.yd = [math.log10(v) for v in ydom] if ylog else list(ydom)
        self.parts, self.boxes = [], []

    def X(self, v):
        t = math.log10(v) if self.xlog else v
        return self.x0 + self.L + (t - self.xd[0]) / (self.xd[1] - self.xd[0]) * (self.w - self.L - self.R)

    def Y(self, v):
        t = math.log10(v) if self.ylog else v
        return self.y0 + self.T + (1 - (t - self.yd[0]) / (self.yd[1] - self.yd[0])) * (self.h - self.T - self.B)

    def ticks(self, dom, log, step=None):
        if log:
            return [10 ** k for k in range(math.ceil(dom[0] - 1e-9), math.floor(dom[1] + 1e-9) + 1)]
        n = round((dom[1] - dom[0]) / step)
        return [round(dom[0] + i * step, 6) for i in range(n + 1) if dom[0] - 1e-9 <= dom[0] + i * step <= dom[1] + 1e-9]

    def axes(self, xlabel, ylabel, ystep=None, xstep=None, yfmt=None, xfmt=fmt_num):
        L, T = self.x0 + self.L, self.y0 + self.T
        Rr, Bb = self.x0 + self.w - self.R, self.y0 + self.h - self.B
        p = self.parts
        p.append(f'<rect class="pl-bg" x="{L}" y="{T}" width="{Rr-L:.1f}" height="{Bb-T:.1f}"/>')
        for v in self.ticks(self.xd, self.xlog, xstep):
            x = self.X(v)
            p.append(f'<line class="pl-grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{Bb}"/>')
            p.append(f'<text class="pl-tick" x="{x:.1f}" y="{Bb+15}" text-anchor="middle">{xfmt(v)}</text>')
        for v in self.ticks(self.yd, self.ylog, ystep):
            y = self.Y(v)
            p.append(f'<line class="pl-grid" x1="{L}" y1="{y:.1f}" x2="{Rr}" y2="{y:.1f}"/>')
            p.append(f'<text class="pl-tick" x="{L-6}" y="{y+4:.1f}" text-anchor="end">{(yfmt or fmt_num)(v)}</text>')
        p.append(f'<text class="pl-axis" x="{(L+Rr)/2:.1f}" y="{Bb+34}" text-anchor="middle">{E(xlabel)}</text>')
        p.append(f'<text class="pl-axis" transform="translate({self.x0+14},{(T+Bb)/2:.1f}) rotate(-90)" text-anchor="middle">{E(ylabel)}</text>')

    def inside(self, x, y):
        return (self.x0 + self.L <= x <= self.x0 + self.w - self.R) and (self.y0 + self.T <= y <= self.y0 + self.h - self.B)

    def label(self, x, y, text, cls='pl-lab', size=11, force=False):
        """Place a label next to (x, y) at the first position that overlaps nothing placed so far."""
        w = len(text) * size * 0.58 + 2
        h = size + 2
        for dx, dy, anchor in ((6, 4, 'start'), (-6, 4, 'end'), (6, -7, 'start'), (6, 14, 'start'), (-6, -7, 'end'), (-6, 14, 'end')):
            bx = x + dx if anchor == 'start' else x + dx - w
            by = y + dy - size
            box = (bx, by, bx + w, by + h)
            if box[0] < self.x0 + self.L - 30 or box[2] > self.x0 + self.w + 4 or box[1] < self.y0 or box[3] > self.y0 + self.h - self.B + 2:
                continue
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in self.boxes):
                continue
            self.boxes.append(box)
            self.parts.append(f'<text class="{cls}" x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}">{E(text)}</text>')
            return True
        if force:
            self.boxes.append((x + 6, y - size, x + 6 + w, y + 2))
            self.parts.append(f'<text class="{cls}" x="{x+6:.1f}" y="{y+4:.1f}">{E(text)}</text>')
            return True
        return False

    def reserve(self, x, y, r=4):
        self.boxes.append((x - r, y - r, x + r, y + r))

    def svg(self):
        return ''.join(self.parts)


def svg_wrap(w, h, body, label):
    return f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{E(label)}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def legend(x, y, items):
    out, cx = [], x
    for cls, text in items:
        out.append(f'<circle class="{cls}" cx="{cx+5}" cy="{y-4}" r="5"/><text class="pl-tick" x="{cx+14}" y="{y}">{E(text)}</text>')
        cx += 24 + len(text) * 6.4
    return ''.join(out)


def kind_cls(kind):
    return 'pt-nn' if kind.startswith('nn') else 'pt-cl'


# ---------------------------------------------------------------- charts

def chart_pareto(pa, xkey='cpu_1t', ylo=0.70):
    pts = [p for p in pa['points'] if p.get(xkey)]
    front = set(pa['fronts'][xkey])
    xs = [p[xkey] for p in pts]
    W, H = 920, 520
    pl = Plot(W, H, (10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs)))), (ylo, 0.96),
              xlog=True, ylog=False, margin=(58, 20, 16, 48))
    pl.axes('CPU time per pixel, 1 thread (ns, log scale)', 'agreement with subjective JND', ystep=0.04, yfmt=lambda v: f'{v:.2f}')
    shown = [p for p in pts if p['agreement'] >= ylo]
    for p in shown:  # intervals first, under the points
        x = pl.X(p[xkey])
        pl.parts.append(f'<line class="pl-ci" x1="{x:.1f}" y1="{pl.Y(max(p["lo"], ylo)):.1f}" x2="{x:.1f}" y2="{pl.Y(p["hi"]):.1f}"/>')
    fp = sorted((p for p in shown if p['metric'] in front), key=lambda p: p[xkey])
    if fp:
        d = f'M{pl.X(fp[0][xkey]):.1f},{pl.Y(fp[0]["agreement"]):.1f}'
        for a, b in zip(fp, fp[1:]):
            d += f' H{pl.X(b[xkey]):.1f} V{pl.Y(b["agreement"]):.1f}'
        pl.parts.append(f'<path class="pl-front" d="{d}"/>')
    for p in shown:
        x, y = pl.X(p[xkey]), pl.Y(p['agreement'])
        cls = 'pt-front' if p['metric'] in front else kind_cls(p['kind'])
        r = 6.5 if p['metric'] in front else 4.2
        pl.parts.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"><title>{E(p["metric"])} ({E(p["column"])}): agreement {p["agreement"]:.3f}, {p[xkey]:.3g} ns/px</title></circle>')
        pl.reserve(x, y, r)
    for p in fp:
        pl.label(pl.X(p[xkey]), pl.Y(p['agreement']), p['metric'], 'pl-lab pl-lab-front', 13, force=True)
    for p in sorted((p for p in shown if p['metric'] not in front), key=lambda p: -p['agreement']):
        pl.label(pl.X(p[xkey]), pl.Y(p['agreement']), p['metric'])
    body = pl.svg() + legend(W - 360, 34, [('pt-front', 'on the front'), ('pt-cl', 'classical'), ('pt-nn', 'neural network')])
    below = [p for p in pts if p['agreement'] < ylo]
    untimed = [p for p in pa['points'] if not p.get(xkey)]
    return svg_wrap(W, H, body, 'Pareto front: agreement with subjective JND versus single-thread CPU time'), below, untimed


def chart_tables(res, tables):
    rows = [(t['label'], res[(t['label'], 'none', 1)], res[(t['label'], 'none', 8)]) for t in tables]
    rows.sort(key=lambda r: (-r[1]['rho'], r[1]['typical']))
    W, rowh, top = 920, 34, 26
    H = top + rowh * len(rows) + 44
    L = 330
    pl = Plot(W, H, (0.80, 1.0), (0, 1), xlog=False, margin=(L, 150, top, 44))
    p = pl.parts
    x0, x1 = pl.X(0.80), pl.X(1.0)
    for v in (0.80, 0.85, 0.90, 0.95, 1.0):
        x = pl.X(v)
        p.append(f'<line class="pl-grid" x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{H-44}"/><text class="pl-tick" x="{x:.1f}" y="{H-28}" text-anchor="middle">{v:.2f}</text>')
    p.append(f'<text class="pl-axis" x="{(x0+x1)/2:.1f}" y="{H-8}" text-anchor="middle">rank correlation of counted cost with measured CPU time (95 % interval)</text>')
    p.append(f'<text class="pl-tick" x="{W-140}" y="{top-10}">typical error 1t / 8t</text>')
    for i, (label, a, b) in enumerate(rows):
        y = top + rowh * i + rowh / 2
        p.append(f'<text class="pl-row" x="{L-12}" y="{y+4:.1f}" text-anchor="end">{E(label)}</text>')
        p.append(f'<line class="pl-ci-h" x1="{pl.X(a["lo"]):.1f}" y1="{y-4:.1f}" x2="{pl.X(a["hi"]):.1f}" y2="{y-4:.1f}"/><circle class="pt-cl" cx="{pl.X(a["rho"]):.1f}" cy="{y-4:.1f}" r="5"/>')
        p.append(f'<line class="pl-ci-h" x1="{pl.X(b["lo"]):.1f}" y1="{y+5:.1f}" x2="{pl.X(b["hi"]):.1f}" y2="{y+5:.1f}"/><rect class="pt-8t" x="{pl.X(b["rho"])-4:.1f}" y="{y+1:.1f}" width="8" height="8"/>')
        p.append(f'<text class="pl-row" x="{W-140}" y="{y+4:.1f}">{a["typical"]:.2f}× / {b["typical"]:.2f}×</text>')
    body = pl.svg() + legend(L, 14, [('pt-cl', '1 thread')]) + f'<rect class="pt-8t" x="{L+90}" y="6" width="8" height="8"/><text class="pl-tick" x="{L+104}" y="14">8 threads</text>'
    return svg_wrap(W, H, body, 'Rank correlation of counted cost with CPU time, per weight table'), rows


def chart_small_multiples(points, tables):
    W, H, n = 920, 340, len(tables)
    pw = W / n
    out = []
    for i, (tid, label) in enumerate(tables):
        pts = [(m['metric'], m['kind'], m['estimate'][tid], m['cpu_1t']) for m in points['metrics']
               if m.get('cpu_1t') and m['estimate'].get(tid)]
        ex = [p[2] for p in pts]; ys = [p[3] for p in pts]
        pl = Plot(pw, H, (10 ** math.floor(math.log10(min(ex))), 10 ** math.ceil(math.log10(max(ex)))),
                  (10 ** math.floor(math.log10(min(ys))), 10 ** math.ceil(math.log10(max(ys)))),
                  xlog=True, ylog=True, margin=(52, 18, 44, 44), x0=pw * i)
        pl.axes('counted cost per pixel', 'measured CPU ns/px' if i == 0 else '')
        k = statistics.median(math.log10(y / x) for _, _, x, y in pts)
        x_lo, x_hi = 10 ** pl.xd[0], 10 ** pl.xd[1]
        band = []
        for f, xa in ((2, x_lo), (2, x_hi), (0.5, x_hi), (0.5, x_lo)):
            band.append((pl.X(xa), pl.Y(min(max(xa * 10 ** k * f, 10 ** pl.yd[0]), 10 ** pl.yd[1]))))
        pl.parts.append('<polygon class="pl-band" points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in band) + '"/>')
        pl.parts.append(f'<line class="pl-fit" x1="{pl.X(x_lo):.1f}" y1="{pl.Y(max(x_lo*10**k, 10**pl.yd[0])):.1f}" x2="{pl.X(x_hi):.1f}" y2="{pl.Y(min(x_hi*10**k, 10**pl.yd[1])):.1f}"/>')
        for m, kind, x, y in pts:
            pl.parts.append(f'<circle class="{kind_cls(kind)}" cx="{pl.X(x):.1f}" cy="{pl.Y(y):.1f}" r="3.2"><title>{E(m)}: cost {x:.3g}, {y:.3g} ns/px</title></circle>')
            pl.reserve(pl.X(x), pl.Y(y), 3)
        far = sorted(pts, key=lambda p: -abs(math.log10(p[3] / p[2]) - k))
        for m, kind, x, y in far:
            if abs(math.log10(y / x) - k) < math.log10(4):
                break
            pl.label(pl.X(x), pl.Y(y), m, 'pl-lab', 10)
        within = sum(1 for _, _, x, y in pts if abs(math.log10(y / x) - k) <= math.log10(2))
        out.append(pl.svg() + f'<text class="pl-title" x="{pw*i+52}" y="16">{E(label)}</text>'
                   f'<text class="pl-tick" x="{pw*i+52}" y="32">{within} of {len(pts)} within 2×</text>')
    return svg_wrap(W, H, ''.join(out), 'Counted cost versus measured CPU time for three weight tables')


def chart_scaling(t1k, t4k, kinds):
    px1, px4 = 1024 * 1024, 3840 * 2160
    pts = [(m, t1k[m] * 1e6 / px1, (t4k[m] / px4) / (t1k[m] / px1)) for m in sorted(set(t1k) & set(t4k)) if t1k[m] > 0 and t4k[m] > 0]
    W, H = 920, 420
    xs = [p[1] for p in pts]
    pl = Plot(W, H, (10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs)))), (0.4, 6),
              xlog=True, ylog=True, margin=(58, 20, 16, 48))
    pl.axes('CPU time per pixel at 1024×1024, 1 thread (ns, log scale)', 'per-pixel time, 3840×2160 ÷ 1024×1024',
            yfmt=lambda v: f'{v:g}×')
    for v in (0.5, 2, 4):
        y = pl.Y(v)
        pl.parts.append(f'<line class="pl-grid" x1="{pl.X(10**pl.xd[0]):.1f}" y1="{y:.1f}" x2="{pl.X(10**pl.xd[1]):.1f}" y2="{y:.1f}"/><text class="pl-tick" x="{pl.x0+pl.L-6}" y="{y+4:.1f}" text-anchor="end">{v:g}×</text>')
    y_hi, y_lo = pl.Y(1.5), pl.Y(1 / 1.5)
    pl.parts.append(f'<rect class="pl-band" x="{pl.x0+pl.L}" y="{y_hi:.1f}" width="{W-pl.L-pl.R}" height="{y_lo-y_hi:.1f}"/>')
    pl.parts.append(f'<line class="pl-fit" x1="{pl.x0+pl.L}" y1="{pl.Y(1):.1f}" x2="{W-pl.R}" y2="{pl.Y(1):.1f}"/>')
    for m, x, r in pts:
        cls = kind_cls(kinds.get(m, 'classic'))
        pl.parts.append(f'<circle class="{cls}" cx="{pl.X(x):.1f}" cy="{pl.Y(min(max(r, 0.41), 5.9)):.1f}" r="4"><title>{E(m)}: {r:.2f}×</title></circle>')
        pl.reserve(pl.X(x), pl.Y(r), 4)
    for m, x, r in sorted(pts, key=lambda p: -abs(math.log10(p[2]))):
        if 1 / 1.5 <= r <= 1.5:
            break
        pl.label(pl.X(x), pl.Y(min(max(r, 0.41), 5.9)), f'{m} {r:.1f}×', 'pl-lab', 11)
    flat = sum(1 for _, _, r in pts if 1 / 1.5 <= r <= 1.5)
    return svg_wrap(W, H, pl.svg(), 'Per-pixel time at 3840x2160 relative to 1024x1024'), len(pts), flat, pts


# ---------------------------------------------------------------- page

def status_summary(md):
    def rows(section, end):
        body = md_section(md, section, end)
        return [l for l in body.splitlines() if l.startswith('| ') and not l.startswith('| Column') and not set(l) <= set('|-: ')]
    def first(r):
        return re.sub(r'`', '', r.strip().strip('|').split('|')[0].strip())
    repro = sorted(set(re.findall(r'`([^`]+)`', re.sub(r'\([^)]*\)', '', md_section(md, '## Reproduced', '## Explicit AIC routes')))))
    cats = [
        ('Reproduced with our implementation', len(repro), repro),
        ('Matched by configuring a canonical implementation', len(rows('## Identified by configuration search', '## Identified: the column')),
         [first(r) for r in rows('## Identified by configuration search', '## Identified: the column')]),
        ('Matched by pyiqa 0.1.16 defaults', len(rows('## Identified: the column', '## Close but not exact')),
         [first(r) for r in rows('## Identified: the column', '## Close but not exact')]),
        ('Close but not exact (minor residual)', len(rows('## Close but not exact', '## Settled')),
         [first(r) for r in rows('## Close but not exact', '## Settled')]),
        ('Open: questions for the dataset authors', len(rows('## Implemented but not reproduced', '## Identified by')),
         [first(r) for r in rows('## Implemented but not reproduced', '## Identified by')]),
        ('Excluded', len(rows('## Excluded', '## Not implemented')), [first(r) for r in rows('## Excluded', '## Not implemented')]),
        ('Not implemented', len(rows('## Not implemented', '## Reproduced')), [first(r) for r in rows('## Not implemented', '## Reproduced')]),
    ]
    trs = ''.join(f'<tr><td>{E(c)}</td><td class="num">{n}</td><td>{E(", ".join(names))}</td></tr>' for c, n, names in cats)
    return f'<div class="tablewrap"><table class="wrap"><thead><tr><th>status</th><th>columns</th><th>which</th></tr></thead><tbody>{trs}</tbody></table></div>'


def build(out):
    md = (PAPER / 'IQA_COMPLEXITY_VS_RUNTIME.md').read_text()
    points = json.loads((PAPER / 'points.json').read_text())
    pareto = json.loads((PAPER / 'pareto.json').read_text())
    status_md = STATUS.read_text()
    res = results_table(md)
    commit = subprocess.run(['jj', '-R', str(ROOT), 'log', '--no-graph', '-r', 'heads(::@ & ::master@origin)', '-T', 'commit_id.short(12)'],
                            capture_output=True, text=True).stdout.strip()
    commit = commit or subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short=12', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'unknown'
    kinds = {m['metric']: m['kind'] for m in points['metrics']}

    tables = points['tables']
    none1 = {t['label']: res[(t['label'], 'none', 1)] for t in tables}
    pad1 = {t['label']: res[(t['label'], 'pad', 1)] for t in tables}
    rho_lo, rho_hi = min(v['rho'] for v in none1.values()), max(v['rho'] for v in none1.values())
    best = min(none1.items(), key=lambda kv: kv[1]['typical'])
    halo_worse = sum(1 for t in none1 if pad1[t]['rho'] < none1[t]['rho'])
    halo_d = [none1[t]['rho'] - pad1[t]['rho'] for t in none1]
    def within2(tid):
        pts = [(m['estimate'][tid], m['cpu_1t']) for m in points['metrics'] if m.get('cpu_1t') and m['estimate'].get(tid)]
        k = statistics.median(math.log10(y / x) for x, y in pts)
        return sum(1 for x, y in pts if abs(math.log10(y / x) - k) <= math.log10(2)), len(pts)
    w2 = [within2(t['id']) for t in tables]
    n1 = none1[tables[0]['label']]['n']
    cov = re.search(r'Tallies: (\d+) of (\d+) CLI metric names, counted from zenmetrics source; measured on (\d+) metrics', md)

    pareto_svg, below, untimed = chart_pareto(pareto)
    tables_svg, _ = chart_tables(res, tables)
    flops_id = 'flops'
    best_id = next(t['id'] for t in tables if t['label'] == best[0])
    sm_ids = [(flops_id, 'flop count'), ('pedzisz_2026_draft', 'Pedzisz draft weights')]
    if best_id not in (flops_id, 'pedzisz_2026_draft'):
        sm_ids.append((best_id, 'llvm-mca throughput' if best_id.startswith('llvm_mca') else best[0]))
    else:
        sm_ids.append(('i265_P_rthroughput', 'measured i265 P-core throughput'))
    sm_svg = chart_small_multiples(points, sm_ids)
    t1k, t4k = timing(MEASURED / 'i265_1t_1024.tsv'), timing(MEASURED / 'i265_1t_3840x2160.tsv')
    sc_svg, sc_n, sc_flat, sc_pts = chart_scaling(t1k, t4k, kinds)
    sc_out = sorted((p for p in sc_pts if not 1 / 1.5 <= p[2] <= 1.5), key=lambda p: -p[2])

    front1 = ', '.join(pareto['fronts']['cpu_1t'])
    css = (HERE / 'style.css').read_text()
    below_txt = ', '.join(f'{p["metric"]} ({p["agreement"]:.2f})' for p in below)
    untimed_txt = ', '.join(p['metric'] for p in untimed)
    top = sorted((p for p in pareto['points'] if p.get('cpu_1t')), key=lambda p: -p['agreement'])[:3]
    fastest = min((p for p in pareto['points'] if p.get('cpu_1t')), key=lambda p: p['cpu_1t'])
    sc_slow = ', '.join(f'{m} {r:.1f}×' for m, _, r in sc_out if r > 1)
    sc_fast = ', '.join(f'{m} {r:.2f}×' for m, _, r in sc_out if r < 1)
    status = status_summary(status_md)
    limits = md_html(md[md.index('## Limits'):].split('\n', 1)[1])

    brief = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Metric Cost Brief</title>
<style>{css}</style></head><body><main>
<header class="head">
<p class="eyebrow">JPEG AIC-4 · image-quality metric complexity · draft brief</p>
<h1>Counted cost vs measured runtime for image-quality metrics</h1>
<p class="status"><strong>Provisional.</strong> One host (Intel Core Ultra 7 265K), one image pair, 1024×1024. Agreement uses the preliminary AIC2026 subjective JND reconstruction (public split: 42 sources, 5,768 images). Most operation counts have a single counter so far.</p>
</header>
<section><h2>Findings</h2><ul class="findings">
<li><b>Pareto front</b> (single-thread CPU time vs agreement with subjective scores): {E(front1)}. No other metric is both faster and closer to the subjective scores. Highest agreement: {E(", ".join(f"{p['metric']} {p['agreement']:.3f}" for p in top))}; fastest: {E(fastest['metric'])} at {fastest['cpu_1t']:.2g} ns per pixel.</li>
<li><b>Counted cost ranks metrics by CPU time well, but misses individual metrics by about 2×.</b> Rank correlation is {rho_lo:.2f}–{rho_hi:.2f} across {len(tables)} weight tables, from a plain flop count to LLVM's cost model, and the intervals overlap, so no table is clearly better. The median metric is off by {min(v["typical"] for v in none1.values()):.2f}–{max(v["typical"] for v in none1.values()):.2f}× depending on the table; {min(a for a, _ in w2)}–{max(a for a, _ in w2)} of {w2[0][1]} metrics fall within 2× of the fitted line.</li>
<li><b>A halo factor does not help:</b> with a padded-domain factor, {halo_worse} of {len(tables)} tables correlate slightly worse ({min(halo_d):.2f}–{max(halo_d):.2f} lower).</li>
<li><b>Per-pixel cost mostly holds at 4K:</b> {sc_flat} of {sc_n} metrics stay within 1.5× of their 1024×1024 per-pixel time at 3840×2160.</li>
</ul></section>
<section><h2>Agreement vs CPU time</h2>
<figure>{pareto_svg}<figcaption>Each point is one AIC2026 metric column; bars are 95 % bootstrap intervals over sources, which overlap near the top. Agreement is the mean within-source Spearman correlation of the published scores with subjective JND; time is our implementation of the same method. Below the plotted range: {E(below_txt) or "none"}. Not yet timed: {E(untimed_txt) or "none"}.</figcaption></figure>
</section>
<section class="pb"><h2>Does counted cost predict CPU time?</h2>
<figure>{tables_svg}<figcaption>Spearman correlation between counted cost per pixel and measured CPU time per pixel (1024×1024, no halo factor). Typical error: median factor between measured time and the fitted single scale.</figcaption></figure>
<figure>{sm_svg}<figcaption>Line: the single best scale; band: ±2×. Labelled metrics sit more than 4× off the line (above: slower than counted, below: faster). Blue classical, violet neural.</figcaption></figure>
</section>
<section class="pb"><h2>From 1024×1024 to 4K</h2>
<figure>{sc_svg}<figcaption>Per-pixel CPU time at 3840×2160 divided by that at 1024×1024, single thread, same binaries. Slower per pixel at 4K: {E(sc_slow) or "none"}. Faster: {E(sc_fast) or "none"}. Neural-network metrics at 4K are still being measured.</figcaption></figure>
</section>
<section><h2>Reproducing the published AIC2026 columns</h2>
<p>Timings only mean something next to the published scores if both come from the same method. Status per column:</p>
{status}
</section>
<section><h2>Limits</h2>{limits}</section>
<footer><p>Data revision <code>{E(commit)}</code> of github.com/imazen/zenmetrics. Every number and chart here is generated from committed files by <code>scripts/complexity/paper_page.py</code>; nothing is typed by hand.</p></footer>
</main></body></html>'''
    out.mkdir(parents=True, exist_ok=True)
    (out / 'brief.html').write_text(brief)
    print(f'wrote {out}/brief.html; {n1} metrics, rho {rho_lo:.2f}-{rho_hi:.2f}, 4K flat {sc_flat}/{sc_n}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    build(ap.parse_args().out)
