#!/usr/bin/env python3
"""Render the metric-cost brief from brief_data.json as one self-contained HTML file.

No network requests: system fonts, inline CSS, inline SVG charts and a small inline script.
The same file reads as a document, prints to PDF, and turns into slides for screen sharing
(open it with #slides, or press the Present button).

    brief_page.py --data brief_data.json --out DIR        # writes DIR/index.html

Every number on the page comes from the data file; brief_data.py documents its sources.
"""
import argparse, html, json, math, re
from pathlib import Path

HERE = Path(__file__).with_name('brief_page')
ROOT = Path(__file__).resolve().parents[2]
E = html.escape
ORIGIN_LABEL = {'public': 'zenmetrics', 'private': 'private port', 'dvifmish': 'dvifmish'}


# ---------------------------------------------------------------- formatting

def fmt_num(v):
    if v == 0: return '0'
    a = abs(v)
    if a >= 1e6: return f'{v/1e6:g}M'
    if a >= 1e3: return f'{v/1e3:g}k'
    if a >= 1: return f'{v:g}'
    return f'{v:.3g}'


def fp(p):
    """p-value for display."""
    if p is None: return '–'
    if p < 1e-300: return '< 1e-300'
    if p < 1e-3:
        m, e = f'{p:.1e}'.split('e')
        return f'{m}×10<sup>{int(e)}</sup>'
    return f'{p:.3f}'


def ci(v, lo, hi, d=2):
    return f'{v:.{d}f} <span class="ci">[{lo:.{d}f}, {hi:.{d}f}]</span>'


def nice_step(span, target=5):
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def ns(v):
    return '–' if v is None else (f'{v:.3g}' if v < 1000 else f'{v:,.0f}')


# ---------------------------------------------------------------- SVG

class Plot:
    """One panel: linear or log axes, gridlines, ticks, and labels placed without overlaps."""

    def __init__(self, w, h, xdom, ydom, xlog=True, ylog=False, margin=(56, 16, 16, 46)):
        self.w, self.h, self.xlog, self.ylog = w, h, xlog, ylog
        self.L, self.R, self.T, self.B = margin
        self.xd = [math.log10(v) for v in xdom] if xlog else list(xdom)
        self.yd = [math.log10(v) for v in ydom] if ylog else list(ydom)
        self.parts, self.boxes = [], []

    def X(self, v):
        t = math.log10(v) if self.xlog else v
        return self.L + (t - self.xd[0]) / (self.xd[1] - self.xd[0]) * (self.w - self.L - self.R)

    def Y(self, v):
        t = math.log10(v) if self.ylog else v
        return self.T + (1 - (t - self.yd[0]) / (self.yd[1] - self.yd[0])) * (self.h - self.T - self.B)

    def ticks(self, dom, log, step=None):
        if log:
            return [10 ** k for k in range(math.ceil(dom[0] - 1e-9), math.floor(dom[1] + 1e-9) + 1)]
        step = step or nice_step(dom[1] - dom[0])
        start = math.ceil(dom[0] / step - 1e-9) * step
        out, v = [], start
        while v <= dom[1] + 1e-9 * step:
            out.append(round(v, 10)); v += step
        return out

    def axes(self, xlabel, ylabel, ystep=None, yfmt=None, xfmt=fmt_num):
        L, T, R, B = self.L, self.T, self.w - self.R, self.h - self.B
        p = self.parts
        for v in self.ticks(self.xd, self.xlog):
            x = self.X(v)
            p.append(f'<line class="pl-grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}"/><text class="pl-tick" x="{x:.1f}" y="{B+15}" text-anchor="middle">{xfmt(v)}</text>')
        for v in self.ticks(self.yd, self.ylog, ystep):
            y = self.Y(v)
            p.append(f'<line class="pl-grid" x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}"/><text class="pl-tick" x="{L-6}" y="{y+4:.1f}" text-anchor="end">{(yfmt or fmt_num)(v)}</text>')
        p.append(f'<line class="pl-axisline" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/><line class="pl-axisline" x1="{L}" y1="{T}" x2="{L}" y2="{B}"/>')
        p.append(f'<text class="pl-axis" x="{(L+R)/2:.1f}" y="{B+36}" text-anchor="middle">{E(xlabel)}</text>')
        p.append(f'<text class="pl-axis" transform="translate(14,{(T+B)/2:.1f}) rotate(-90)" text-anchor="middle">{E(ylabel)}</text>')

    def label(self, x, y, text, cls='pl-lab', size=11, force=False):
        w, h = len(text) * size * 0.58 + 2, size + 2
        for dx, dy, anchor in ((6, 4, 'start'), (-6, 4, 'end'), (6, -7, 'start'), (6, 14, 'start'), (-6, -7, 'end'), (-6, 14, 'end')):
            bx = x + dx if anchor == 'start' else x + dx - w
            box = (bx, y + dy - size, bx + w, y + dy - size + h)
            if box[0] < self.L - 30 or box[2] > self.w + 4 or box[1] < 0 or box[3] > self.h - self.B + 2:
                continue
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in self.boxes):
                continue
            self.boxes.append(box)
            self.parts.append(f'<text class="{cls}" x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}">{E(text)}</text>')
            return True
        if force:
            self.boxes.append((x + 6, y - size, x + 6 + w, y + 2))
            self.parts.append(f'<text class="{cls}" x="{x+6:.1f}" y="{y+4:.1f}">{E(text)}</text>')
        return force

    def reserve(self, x, y, r=4):
        self.boxes.append((x - r, y - r, x + r, y + r))

    def point(self, m, x, y, r=4.2, front=False, tip=''):
        cls = 'pt-front' if front else ('pt-nn' if m['kind'].startswith('nn') else 'pt-cl')
        cx, cy = self.X(x), self.Y(y)
        t = f'<title>{E(m["metric"])}{E(tip)}</title>'
        if m['origin'] == 'private':
            s = r * 1.7
            self.parts.append(f'<rect class="{cls} pt-priv" x="{cx-s/2:.1f}" y="{cy-s/2:.1f}" width="{s:.1f}" height="{s:.1f}">{t}</rect>')
        elif m['origin'] == 'dvifmish':
            s = r * 1.35
            self.parts.append(f'<path class="{cls} pt-dv" d="M{cx:.1f},{cy-s:.1f} L{cx+s:.1f},{cy:.1f} L{cx:.1f},{cy+s:.1f} L{cx-s:.1f},{cy:.1f} Z">{t}</path>')
        else:
            self.parts.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}">{t}</circle>')
        self.reserve(cx, cy, r + 1)
        return cx, cy

    def svg(self, label, cls='chart'):
        return f'<svg class="{cls}" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{E(label)}" xmlns="http://www.w3.org/2000/svg">{"".join(self.parts)}</svg>'


def legend_html(front=True):
    items = ([('pt-front', 'circle', 'on the Pareto front')] if front else []) + [
        ('pt-cl', 'circle', 'classical'), ('pt-nn', 'circle', 'neural network'),
        ('pt-cl', 'rect', 'private port'), ('pt-cl', 'diamond', 'dvifmish (DVIFM)')]
    out = []
    for cls, shape, text in items:
        g = {'circle': f'<circle class="{cls}" cx="7" cy="7" r="5"/>', 'rect': f'<rect class="{cls}" x="2" y="2" width="10" height="10"/>',
             'diamond': f'<path class="{cls}" d="M7,1 L13,7 L7,13 L1,7 Z"/>'}[shape]
        out.append(f'<span><svg viewBox="0 0 14 14" width="14" height="14">{g}</svg>{E(text)}</span>')
    return '<div class="legend">' + ''.join(out) + '</div>'


def log_dom(vals, pad=1.0):
    return 10 ** math.floor(math.log10(min(vals)) - 1e-9), 10 ** math.ceil(math.log10(max(vals)) + 1e-9)


# ---------------------------------------------------------------- charts

def chart_pareto(d, key, ylo=0.70):
    ms = [m for m in d['metrics'] if m['agreement'] and m.get(key)]
    front = set(d['fronts'][key])
    pl = Plot(940, 540, log_dom([m[key] for m in ms]), (ylo, 0.97), xlog=True, margin=(60, 20, 16, 48))
    th = '1 thread' if key.endswith('1t') else '8 threads'
    pl.axes(f'CPU time per pixel, {th} (ns, log scale)', 'agreement with subjective JND', ystep=0.04, yfmt=lambda v: f'{v:.2f}')
    shown = [m for m in ms if m['agreement']['agreement'] >= ylo]
    for m in shown:
        a = m['agreement']; x = pl.X(m[key])
        pl.parts.append(f'<line class="pl-ci" x1="{x:.1f}" y1="{pl.Y(max(a["lo"], ylo)):.1f}" x2="{x:.1f}" y2="{pl.Y(min(a["hi"], 0.97)):.1f}"/>')
    fp_ = sorted((m for m in shown if m['metric'] in front), key=lambda m: m[key])
    if fp_:
        path = f'M{pl.X(fp_[0][key]):.1f},{pl.Y(fp_[0]["agreement"]["agreement"]):.1f}'
        for b in fp_[1:]:
            path += f' H{pl.X(b[key]):.1f} V{pl.Y(b["agreement"]["agreement"]):.1f}'
        pl.parts.append(f'<path class="pl-front" d="{path}"/>')
    for m in shown:
        a = m['agreement']
        pl.point(m, m[key], a['agreement'], 6.5 if m['metric'] in front else 4.2, m['metric'] in front,
                 f' ({a["column"]}): agreement {a["agreement"]:.3f} [{a["lo"]:.3f}, {a["hi"]:.3f}], {m[key]:.3g} ns/px')
    for m in fp_:
        pl.label(pl.X(m[key]), pl.Y(m['agreement']['agreement']), m['metric'], 'pl-lab pl-lab-front', 13, force=True)
    for m in sorted((m for m in shown if m['metric'] not in front), key=lambda m: (m['origin'] == 'public', -m['agreement']['agreement'])):
        pl.label(pl.X(m[key]), pl.Y(m['agreement']['agreement']), m['metric'])
    below = [m for m in ms if m['agreement']['agreement'] < ylo]
    return pl.svg(f'Agreement with subjective JND versus CPU time, {th}'), below


def stat_of(d, table, threads, measure='cpu', subset='all'):
    return next((s for s in d['stats'] if s['table'] == table and s['threads'] == threads and s['measure'] == measure and s['subset'] == subset), None)


def chart_coef(d, table, threads, linear):
    s = stat_of(d, table, threads)
    key = f'cpu_{threads}t'
    ms = [m for m in d['metrics'] if m['estimate'] and m.get(key) and m['estimate'].get(table, 0) > 0]
    xs = [m['estimate'][table] for m in ms]; ys = [m[key] for m in ms]
    W_, H_ = 470, 360
    if linear:
        xmax, ymax = max(xs) * 1.05, max(ys) * 1.05
        pl = Plot(W_, H_, (0, xmax), (0, ymax), xlog=False, ylog=False, margin=(62, 14, 40, 46))
        pl.axes('counted cost per pixel', f'CPU ns/px, {threads} thread{"s" if threads > 1 else ""}')
        k, lo, hi = s['scale_linear'], s['scale_linear_lo'], s['scale_linear_hi']
        end = lambda kk: (min(xmax, ymax / kk), min(xmax * kk, ymax))  # where y = kk·x leaves the plot
        poly = [(0, 0), end(hi)] + ([(xmax, ymax)] if xmax * hi > ymax and xmax * lo < ymax else []) + [end(lo)]
        pl.parts.append('<polygon class="pl-cib" points="' + ' '.join(f'{pl.X(a):.1f},{pl.Y(b):.1f}' for a, b in poly) + '"/>')
        pl.parts.append(f'<line class="pl-fit" x1="{pl.X(0):.1f}" y1="{pl.Y(0):.1f}" x2="{pl.X(end(k)[0]):.1f}" y2="{pl.Y(end(k)[1]):.1f}"/>')
        head = f'linear: y = {k:.3g}·x [{lo:.3g}, {hi:.3g}]'
        sub = f'Pearson r {s["pearson_linear"]:.2f}, p {re.sub("<[^>]+>", "", fp(s["pearson_linear_p"])).replace("×10", "e")}'
    else:
        xd, yd = log_dom(xs), log_dom(ys)
        pl = Plot(W_, H_, xd, yd, xlog=True, ylog=True, margin=(62, 14, 40, 46))
        pl.axes('counted cost per pixel (log)', f'CPU ns/px, {threads} thread{"s" if threads > 1 else ""} (log)')
        k, klo, khi = s['scale'], s['scale_lo'], s['scale_hi']
        x0, x1 = 10 ** pl.xd[0], 10 ** pl.xd[1]
        clip = lambda v: min(max(v, 10 ** pl.yd[0]), 10 ** pl.yd[1])
        band = [(x0, 2 * k), (x1, 2 * k), (x1, k / 2), (x0, k / 2)]
        pl.parts.append('<polygon class="pl-band" points="' + ' '.join(f'{pl.X(a):.1f},{pl.Y(clip(a*b)):.1f}' for a, b in band) + '"/>')
        # fitted log-log line with its slope (OLS) and the unit-slope scale line
        b, a = s['slope'], s['intercept']
        pl.parts.append(f'<line class="pl-fit" x1="{pl.X(x0):.1f}" y1="{pl.Y(clip(x0*k)):.1f}" x2="{pl.X(x1):.1f}" y2="{pl.Y(clip(x1*k)):.1f}"/>')
        pl.parts.append(f'<line class="pl-ols" x1="{pl.X(x0):.1f}" y1="{pl.Y(clip(10**(a+b*math.log10(x0)))):.1f}" x2="{pl.X(x1):.1f}" y2="{pl.Y(clip(10**(a+b*math.log10(x1)))):.1f}"/>')
        head = f'ρ {s["spearman"]:.3f} [{s["spearman_lo"]:.3f}, {s["spearman_hi"]:.3f}], slope {b:.2f} [{s["slope_lo"]:.2f}, {s["slope_hi"]:.2f}]'
        sub = f'scale {k:.3g} ns/unit [{klo:.3g}, {khi:.3g}]; {s["within2"]} of {s["n"]} within 2×'
    for m, x, y in zip(ms, xs, ys):
        pl.point(m, x, y, 3.2, tip=f': cost {x:.3g}, {y:.3g} ns/px')
    if not linear:
        k = s['scale']
        for m, x, y in sorted(zip(ms, xs, ys), key=lambda t: -abs(math.log10(t[2] / (t[1] * k)))):
            if abs(math.log10(y / (x * k))) < math.log10(4):
                break
            pl.label(pl.X(x), pl.Y(y), m['metric'], 'pl-lab', 10)
    else:
        for m, x, y in sorted(zip(ms, xs, ys), key=lambda t: -t[2])[:6]:
            pl.label(pl.X(x), pl.Y(y), m['metric'], 'pl-lab', 10)
    pl.parts.append(f'<text class="pl-title" x="{pl.L}" y="16">{E(head)}</text><text class="pl-tick" x="{pl.L}" y="31">{E(sub)}</text>')
    return pl.svg(f'Counted cost versus CPU time, {threads} thread(s), {"linear" if linear else "log"} axes', 'chart coef')


def chart_rho(d):
    rows = []
    for t in d['tables']:
        a, b = stat_of(d, t['id'], 1), stat_of(d, t['id'], 8)
        if a and b: rows.append((t['label'], a, b))
    rows.sort(key=lambda r: -r[1]['spearman'])
    W_, rowh, top, L = 940, 34, 30, 330
    H_ = top + rowh * len(rows) + 46
    lo = math.floor(min(min(a['spearman_lo'], b['spearman_lo']) for _, a, b in rows) * 20) / 20
    pl = Plot(W_, H_, (lo, 1.0), (0, 1), xlog=False, margin=(L, 20, top, 46))
    p = pl.parts
    for v in pl.ticks(pl.xd, False, 0.05):
        x = pl.X(v)
        p.append(f'<line class="pl-grid" x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{H_-46}"/><text class="pl-tick" x="{x:.1f}" y="{H_-30}" text-anchor="middle">{v:.2f}</text>')
    p.append(f'<text class="pl-axis" x="{(pl.X(lo)+pl.X(1))/2:.1f}" y="{H_-8}" text-anchor="middle">Spearman ρ of counted cost with measured CPU time, 95 % bootstrap interval</text>')
    for i, (label, a, b) in enumerate(rows):
        y = top + rowh * i + rowh / 2
        p.append(f'<text class="pl-row" x="{L-12}" y="{y+4:.1f}" text-anchor="end">{E(label)}</text>')
        p.append(f'<line class="pl-ci-h" x1="{pl.X(a["spearman_lo"]):.1f}" y1="{y-5:.1f}" x2="{pl.X(a["spearman_hi"]):.1f}" y2="{y-5:.1f}"/><circle class="pt-cl" cx="{pl.X(a["spearman"]):.1f}" cy="{y-5:.1f}" r="5"/>')
        p.append(f'<line class="pl-ci-h" x1="{pl.X(b["spearman_lo"]):.1f}" y1="{y+6:.1f}" x2="{pl.X(b["spearman_hi"]):.1f}" y2="{y+6:.1f}"/><rect class="pt-8t" x="{pl.X(b["spearman"])-4:.1f}" y="{y+2:.1f}" width="8" height="8"/>')
    p.append(f'<circle class="pt-cl" cx="{L+5}" cy="12" r="5"/><text class="pl-tick" x="{L+14}" y="16">1 thread</text><rect class="pt-8t" x="{L+90}" y="8" width="8" height="8"/><text class="pl-tick" x="{L+104}" y="16">8 threads</text>')
    return pl.svg('Rank correlation of counted cost with CPU time, per weight table')


def chart_scaling(d):
    ms = [m for m in d['metrics'] if m.get('cpu_1t') and m.get('cpu_1t_4k')]
    pts = [(m, m['cpu_1t'], m['cpu_1t_4k'] / m['cpu_1t']) for m in ms]
    pl = Plot(940, 420, log_dom([p[1] for p in pts]), (0.4, 6), xlog=True, ylog=True, margin=(60, 20, 16, 48))
    pl.axes('CPU time per pixel at 1024×1024, 1 thread (ns, log scale)', 'per-pixel time, 4K ÷ 1024²', yfmt=lambda v: f'{v:g}×')
    for v in (0.5, 2, 4):
        y = pl.Y(v)
        pl.parts.append(f'<line class="pl-grid" x1="{pl.L}" y1="{y:.1f}" x2="{pl.w-pl.R}" y2="{y:.1f}"/><text class="pl-tick" x="{pl.L-6}" y="{y+4:.1f}" text-anchor="end">{v:g}×</text>')
    pl.parts.append(f'<rect class="pl-band" x="{pl.L}" y="{pl.Y(1.5):.1f}" width="{pl.w-pl.L-pl.R}" height="{pl.Y(1/1.5)-pl.Y(1.5):.1f}"/>')
    pl.parts.append(f'<line class="pl-fit" x1="{pl.L}" y1="{pl.Y(1):.1f}" x2="{pl.w-pl.R}" y2="{pl.Y(1):.1f}"/>')
    for m, x, r in pts:
        pl.point(m, x, min(max(r, 0.41), 5.9), 4, tip=f': {r:.2f}×')
    for m, x, r in sorted(pts, key=lambda p: -abs(math.log10(p[2]))):
        if 1 / 1.5 <= r <= 1.5: break
        pl.label(pl.X(x), pl.Y(min(max(r, 0.41), 5.9)), f'{m["metric"]} {r:.1f}×')
    flat = sum(1 for _, _, r in pts if 1 / 1.5 <= r <= 1.5)
    out = sorted((p for p in pts if not 1 / 1.5 <= p[2] <= 1.5), key=lambda p: -p[2])
    return pl.svg('Per-pixel time at 3840x2160 relative to 1024x1024'), len(pts), flat, out


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def rep_ci(reps):
    """Mean and 95 % t-interval over timing repeats (run-to-run noise on one pair), or None."""
    if not reps or len(reps) < 2:
        return None
    n = len(reps); m = sum(reps) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in reps) / (n - 1))
    h = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, max(m - h, min(reps) * 0.5), m + h


def kmeans1d(xs, k):
    """Exact 1-D k-means (dynamic programming) on sorted xs; returns the cluster index of each x."""
    n = len(xs)
    pre = [0.0]; pre2 = [0.0]
    for x in xs:
        pre.append(pre[-1] + x); pre2.append(pre2[-1] + x * x)
    cost = lambda i, j: (pre2[j] - pre2[i]) - (pre[j] - pre[i]) ** 2 / (j - i)  # SS of xs[i:j]
    INF = float('inf')
    D = [[INF] * (n + 1) for _ in range(k + 1)]; arg = [[0] * (n + 1) for _ in range(k + 1)]
    D[0][0] = 0.0
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            for i in range(c - 1, j):
                v = D[c - 1][i] + cost(i, j)
                if v < D[c][j]:
                    D[c][j], arg[c][j] = v, i
    lab, j = [0] * n, n
    for c in range(k, 0, -1):
        i = arg[c][j]
        for t in range(i, j): lab[t] = c - 1
        j = i
    return lab, D[k][n]


def clusters_of(vals, max_k=7, keep=0.06):
    """Smallest k whose within-cluster sum of squares (log10 time) is at most `keep` of the total."""
    xs = sorted(math.log10(v) for v in vals)
    total = kmeans1d(xs, 1)[1] or 1.0
    for k in range(2, max_k + 1):
        lab, ss = kmeans1d(xs, k)
        if ss <= keep * total:
            break
    return k, lab, ss / total


def chart_perf(d, key):
    reps_key = key + '_reps'
    ms = sorted((m for m in d['metrics'] if m.get(key)), key=lambda m: m[key])
    k, lab, frac = clusters_of([m[key] for m in ms])
    groups = [[m for m, l in zip(ms, lab) if l == c] for c in range(k)]
    W_, LBL, LX0, LX1, RX0, RX1, rowh, head = 940, 196, 206, 560, 600, 924, 15, 40
    H_ = 34 + sum(head + rowh * len(g) + 6 for g in groups) + 30
    lo, hi = log_dom([m[key] for m in ms])
    X = lambda v: LX0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (LX1 - LX0)
    p = []
    th = '1 thread' if key.endswith('1t') else '8 threads'
    p.append(f'<text class="pl-title" x="{LX0}" y="14">all metrics, log axis (CPU ns/px, {th})</text>'
             f'<text class="pl-title" x="{RX0}" y="14">within cluster, linear axis</text>')
    e = math.ceil(math.log10(lo) - 1e-9)
    while 10 ** e <= hi * 1.0001:
        x = X(10 ** e)
        p.append(f'<line class="pl-grid" x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{H_-26}"/><text class="pl-tick" x="{x:.1f}" y="{H_-12}" text-anchor="middle">{fmt_num(10**e)}</text>'
                 f'<text class="pl-tick" x="{x:.1f}" y="32" text-anchor="middle">{fmt_num(10**e)}</text>')
        e += 1
    y = 40
    narrow = 0
    for ci_, g in enumerate(groups):
        gmax = max(max((rep_ci(m.get(reps_key)) or (0, 0, m[key]))[2], m[key]) for m in g) * 1.05
        step = nice_step(gmax, 4)
        RX = lambda v: RX0 + v / gmax * (RX1 - RX0)
        top = y
        p.append(f'<rect class="pl-cluster" x="4" y="{y:.1f}" width="{W_-8}" height="{head + rowh*len(g) - 6:.1f}" rx="4"/>')
        p.append(f'<text class="pl-row" x="10" y="{y+16:.1f}" font-weight="600">Cluster {ci_+1}: {fmt_num(float(f'{g[0][key]:.3g}'))}–{fmt_num(float(f'{g[-1][key]:.3g}'))} ns/px, {len(g)} metrics</text>')
        v = 0.0
        while v <= gmax + 1e-12:
            x = RX(v)
            p.append(f'<line class="pl-grid" x1="{x:.1f}" y1="{y+22:.1f}" x2="{x:.1f}" y2="{y+head+rowh*len(g)-10:.1f}"/><text class="pl-tick" x="{x:.1f}" y="{y+32:.1f}" text-anchor="middle">{fmt_num(v)}</text>')
            v += step
        y += head
        prev = None
        for m in g:
            yy = y + rowh / 2
            c = rep_ci(m.get(reps_key))
            cls = 'pt-nn' if m['kind'].startswith('nn') else 'pt-cl'
            tip = f'{m["metric"]}: median {m[key]:.4g} ns/px' + (f', mean {c[0]:.4g} [95 % {c[1]:.4g}, {c[2]:.4g}], n={len(m[reps_key])}' if c else '')
            p.append(f'<text class="pl-lab perf-lab" x="{LBL}" y="{yy+4:.1f}" text-anchor="end">{E(m["metric"])}</text>')
            for XX in (X, RX):
                if c:
                    p.append(f'<line class="pl-ci-h" x1="{XX(max(c[1], lo if XX is X else 0)):.1f}" y1="{yy:.1f}" x2="{XX(c[2]):.1f}" y2="{yy:.1f}"/>')
                cx = XX(m[key])
                if m['origin'] == 'private':
                    p.append(f'<rect class="{cls}" x="{cx-3.5:.1f}" y="{yy-3.5:.1f}" width="7" height="7"><title>{E(tip)}</title></rect>')
                elif m['origin'] == 'dvifmish':
                    p.append(f'<path class="{cls}" d="M{cx:.1f},{yy-4.5:.1f} L{cx+4.5:.1f},{yy:.1f} L{cx:.1f},{yy+4.5:.1f} L{cx-4.5:.1f},{yy:.1f} Z"><title>{E(tip)}</title></path>')
                else:
                    p.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{yy:.1f}" r="3.6"><title>{E(tip)}</title></circle>')
            if c and prev and prev[2] >= c[1]:
                narrow += 1
            prev = c
            y += rowh
        y += 6
    svg = f'<svg class="chart perf" viewBox="0 0 {W_} {H_}" role="img" aria-label="CPU time per metric with 95 % intervals, {th}" xmlns="http://www.w3.org/2000/svg">{"".join(p)}</svg>'
    return svg, k, frac, narrow, len(ms)


# ---------------------------------------------------------------- tables

def table(head, rows, cls=''):
    th = ''.join(f'<th>{h}</th>' for h in head)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tablewrap {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def stats_table(d, subset='all'):
    rows = []
    for t in d['tables']:
        for th in (1, 8):
            s = stat_of(d, t['id'], th, 'cpu', subset)
            if not s: continue
            rows.append([E(t['label']) if th == 1 else '', f'{th}', f'{s["n"]}', ci(s['spearman'], s['spearman_lo'], s['spearman_hi']), fp(s['spearman_p']),
                         ci(s['pearson_log'], s['pearson_log_lo'], s['pearson_log_hi']), fp(s['pearson_log_p']),
                         ci(s['slope'], s['slope_lo'], s['slope_hi']), f'{s["scale"]:.3g} <span class="ci">[{s["scale_lo"]:.3g}, {s["scale_hi"]:.3g}]</span>',
                         f'{s["typical"]:.2f}×', f'{s["within2"]}/{s["n"]}'])
    return table(['weights', 'threads', 'n', 'Spearman ρ [95 %]', 'p', 'Pearson r, log–log [95 %]', 'p', 'log–log slope [95 %]',
                  'scale, ns per unit [95 %]', 'typical error', 'within 2×'], rows, 'stats')


def status_rows(md):
    def sec(a, b):
        s = md.index(a); e = md.index(b, s + len(a)); return md[s:e].split('\n', 1)[1]
    def firsts(body):
        return [re.sub('`', '', l.strip().strip('|').split('|')[0].strip()) for l in body.splitlines()
                if l.startswith('| ') and not l.startswith('| Column') and not set(l) <= set('|-: ')]
    repro = sorted(set(re.findall(r'`([^`]+)`', re.sub(r'\([^)]*\)', '', sec('## Reproduced', '## Explicit AIC routes')))))
    cats = [('Reproduced with our implementation', repro),
            ('Matched by configuring a canonical implementation', firsts(sec('## Identified by configuration search', '## Identified: the column'))),
            ('Matched by pyiqa 0.1.16 defaults', firsts(sec('## Identified: the column', '## Close but not exact'))),
            ('Close but not exact', firsts(sec('## Close but not exact', '## Settled'))),
            ('Open: questions for the dataset authors', firsts(sec('## Implemented but not reproduced', '## Identified by'))),
            ('Excluded', firsts(sec('## Excluded', '## Not implemented'))),
            ('Not implemented', firsts(sec('## Not implemented', '## Reproduced')))]
    return table(['status', 'columns', 'which'], [[E(c), str(len(n)), E(', '.join(n))] for c, n in cats], 'wrap')


# ---------------------------------------------------------------- page

def build(data_path, out):
    d = json.loads(Path(data_path).read_text())
    css = (HERE / 'style.css').read_text()
    js = (HERE / 'brief.js').read_text()
    tables = d['tables']
    by = {m['metric']: m for m in d['metrics']}
    s1 = {t['id']: stat_of(d, t['id'], 1) for t in tables}
    s8 = {t['id']: stat_of(d, t['id'], 8) for t in tables}
    label = {t['id']: t['label'] for t in tables}
    # default view: lowest typical error among tables that model this CPU (znver5 is the other-CPU control)
    best = min((t for t in tables if s1[t['id']] and 'other CPU' not in t['label']), key=lambda t: s1[t['id']]['typical'])['id']
    rho1 = [s1[t]['spearman'] for t in s1 if s1[t]]
    p_max = max(s['spearman_p'] for s in list(s1.values()) + list(s8.values()) if s)
    slopes = [(s['slope'], s['slope_lo'], s['slope_hi']) for s in list(s1.values()) + list(s8.values()) if s]
    n_below1 = sum(1 for _, lo, hi in slopes if hi < 1)
    typ = [s1[t]['typical'] for t in s1 if s1[t]]
    w2 = [(s1[t]['within2'], s1[t]['n']) for t in s1 if s1[t]]

    perf1 = chart_perf(d, 'cpu_1t')
    perf8 = chart_perf(d, 'cpu_8t')
    par1, below1 = chart_pareto(d, 'cpu_1t')
    par8, below8 = chart_pareto(d, 'cpu_8t')
    rho_svg = chart_rho(d)
    coef = ''.join(
        f'<div class="coefset" data-table="{t["id"]}"{"" if t["id"] == best else " hidden"}>'
        f'<div class="t1 log">{chart_coef(d, t["id"], 1, False)}</div><div class="t1 lin">{chart_coef(d, t["id"], 1, True)}</div>'
        f'<div class="t8 log">{chart_coef(d, t["id"], 8, False)}</div><div class="t8 lin">{chart_coef(d, t["id"], 8, True)}</div></div>'
        for t in tables if s1[t['id']] and s8[t['id']])
    options = ''.join(f'<option value="{t["id"]}"{" selected" if t["id"] == best else ""}>{E(t["label"])}</option>' for t in tables)
    sc_svg, sc_n, sc_flat, sc_out = chart_scaling(d)

    top = sorted((m for m in d['metrics'] if m['agreement'] and m.get('cpu_1t')), key=lambda m: -m['agreement']['agreement'])[:4]
    front1 = d['fronts']['cpu_1t']
    fastest = min((m for m in d['metrics'] if m.get('cpu_1t') and m['agreement']), key=lambda m: m['cpu_1t'])
    dv = [by[v['metric']] for v in d['dvifmish_versions'] if v['metric'] in by]
    n_priv = sum(1 for m in d['metrics'] if m['origin'] == 'private')

    findings = [
        f'<b>Pareto front</b> (agreement with subjective JND vs single-thread CPU time): {E(", ".join(front1))}. No other metric is both faster and closer to the subjective scores. Highest agreement: '
        + E(', '.join(f'{m["metric"]} {m["agreement"]["agreement"]:.3f}' for m in top)) + f'. Fastest: {E(fastest["metric"])}, {fastest["cpu_1t"]:.2g} ns per pixel.',
        f'<b>Counted cost ranks metrics by CPU time:</b> Spearman ρ {min(rho1):.2f}–{max(rho1):.2f} across {len(rho1)} weight tables (p ≤ {fp(p_max)} for every table and thread count). '
        f'The intervals overlap, so no table is clearly better.',
        f'<b>But counts grow faster than time:</b> the log–log slope is below 1 for {n_below1} of {len(slopes)} table × thread fits (95 % interval excludes 1), so expensive metrics run cheaper per counted operation. '
        f'Per metric the typical error is {min(typ):.2f}–{max(typ):.2f}×, and {min(a for a, _ in w2)}–{max(a for a, _ in w2)} of {w2[0][1]} metrics fall within 2× of the fitted scale.',
        f'<b>Per-pixel cost mostly holds at 4K:</b> {sc_flat} of {sc_n} metrics stay within 1.5× of their 1024×1024 per-pixel time at 3840×2160.']
    if dv:
        best_dv = max((m for m in dv if m['agreement']), key=lambda m: m['agreement']['agreement'], default=None)
        if best_dv:
            findings.append(f'<b>DVIFM (dvifmish ports of the author\'s C++):</b> {E(best_dv["metric"].removeprefix("dvifmish-"))} reaches agreement {best_dv["agreement"]["agreement"]:.3f} '
                            f'[{best_dv["agreement"]["lo"]:.3f}, {best_dv["agreement"]["hi"]:.3f}]' + (f' at {best_dv["cpu_1t"]:.2g} ns per pixel.' if best_dv.get('cpu_1t') else '; i265 timing pending.'))

    dv_rows = [[f'<code>{E(m["metric"].removeprefix("dvifmish-"))}</code>',
                (ci(m['agreement']['agreement'], m['agreement']['lo'], m['agreement']['hi'], 3) if m['agreement'] else '–'),
                f'{m["agreement"]["overall"]:.3f}' if m['agreement'] else '–',
                ns(m.get('cpu_1t')), ns(m.get('cpu_8t')), ns(m.get('flops_per_px')),
                E((m['canonical'] or {}).get('evidence') or '–')] for m in sorted(
                    (m for m in d['metrics'] if m['metric'].startswith('dvifmish-')), key=lambda m: -(m['agreement'] or {}).get('agreement', 0))]

    canon_rows = []
    for m in sorted(d['metrics'], key=lambda m: (-(m['agreement'] or {}).get('agreement', -1), m['metric'])):
        c = m['canonical'] or {}
        if not (m.get('cpu_1t') or m['agreement']):
            continue
        canon_rows.append([f'<code>{E(m["metric"])}</code>', E(ORIGIN_LABEL.get(m['origin'], m['origin'])), E(c.get('family', '–')),
                           E(c.get('reference') or '–'), E(c.get('parity') or '–'), E(m['column'] or (m['agreement'] or {}).get('column') or '–'),
                           f'{m["agreement"]["agreement"]:.3f}' if m['agreement'] else '–', ns(m.get('cpu_1t')), ns(m.get('cpu_8t'))])

    status_md = (ROOT / 'docs/AIC2026_REPRODUCTION_STATUS.md').read_text()
    src = d['sources']
    below_txt = ', '.join(f'{m["metric"]} ({m["agreement"]["agreement"]:.2f})' for m in below1) or 'none'
    no_time = ', '.join(m['metric'] for m in d['metrics'] if m['agreement'] and not m.get('cpu_1t')) or 'none'
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>Metric Cost Brief</title>
<style>{css}</style></head><body>
<nav class="toolbar" aria-label="View"><button type="button" id="present">Present</button>
<label>Weights <select id="tbl">{options}</select></label>
<button type="button" id="threads" aria-pressed="false">8 threads</button><span class="hint">Slides: ← → move · T 1/8 threads · W weights · F fullscreen · Esc exit</span></nav>
<main>
<section class="slide title-slide">
<p class="eyebrow">JPEG AIC-4 · image-quality metric complexity · draft</p>
<h1>Counted cost vs measured runtime for image-quality metrics</h1>
<p class="status"><strong>Provisional.</strong> One host (Intel Core Ultra 7 265K), one image pair at 1024×1024 for the fits. Agreement uses the preliminary AIC2026 subjective JND reconstruction (public split: 42 sources, 5,768 images). Most operation counts have one counter so far. Includes {n_priv} private ports and {len(dv)} dvifmish versions.</p>
<ul class="findings">{"".join(f"<li>{f}</li>" for f in findings)}</ul>
</section>

<section class="slide" id="perf"><h2>CPU time per metric, with 95 % intervals</h2>
<p class="slide-note">Performance only, no agreement scores. Each row is one metric: median CPU time per pixel at 1024×1024 (i265), and a 95 % t-interval over its timing repeats, which measures run-to-run noise on one image pair, not variation across images. Clusters: exact 1-D k-means on log time, the smallest k that leaves at most 6 % of the variance within clusters. Left: one log axis for every metric. Right: each cluster on its own linear axis, so neighbours can be compared directly. Press T (or the 8 threads button) to switch thread count.</p>
<figure><div class="t1">{perf1[0]}<figcaption>1 thread: {perf1[4]} metrics in {perf1[1]} clusters ({perf1[2]*100:.1f} % of log-time variance within clusters); {perf1[3]} adjacent pairs have overlapping intervals.</figcaption></div>
<div class="t8">{perf8[0]}<figcaption>8 threads (CPU time summed over threads): {perf8[4]} metrics in {perf8[1]} clusters ({perf8[2]*100:.1f} % within); {perf8[3]} adjacent pairs overlap.</figcaption></div>{legend_html(False)}</figure>
</section>

<section class="slide" id="pareto"><h2>Agreement with subjective scores vs CPU time</h2>
<figure><div class="t1">{par1}</div><div class="t8">{par8}</div>{legend_html()}
<figcaption>Up is closer to the subjective JND (mean within-source Spearman, bars are 95 % bootstrap intervals over sources); left is cheaper. Amber: Pareto front. Agreement for published columns comes from the scores the dataset authors published; for dvifmish from our own scores on the same images. Below the plotted range: {E(below_txt)}. Not timed yet: {E(no_time)}.</figcaption></figure>
</section>

<section class="slide" id="rank"><h2>Does counted cost predict CPU time?</h2>
<figure>{rho_svg}<figcaption>Spearman correlation between counted cost per pixel (each weight table) and measured CPU time per pixel; i265, 1024×1024, n = {s1[best]["n"]} metrics at 1 thread. p-values are below 1e-40 throughout; the intervals, not the p-values, decide whether tables differ.</figcaption></figure>
</section>

<section class="slide" id="coef"><h2>The coefficient: counted cost vs time, 1 and 8 threads</h2>
<p class="slide-note">Weights: <b class="tblname">{E(label[best])}</b>. Log axes (left): dashed is the single scale (unit slope) with a ±2× band, solid is the least-squares log–log line. Linear axes (right): least squares through the origin with its 95 % bootstrap band; on linear axes the few most expensive metrics decide the fit, which is why that band is wide.</p>
<div class="coefgrid">{coef}</div>
{legend_html(False)}
</section>

<section class="slide" id="stats"><h2>Statistics per weight table</h2>
{stats_table(d)}
<p class="note">CPU time, i265, 1024×1024. Intervals: Spearman and scale by 2,000-sample bootstrap over metrics (seed {d["seed"]}); Pearson by Fisher z; slope by OLS t-interval. Typical error: median factor between measured time and the fitted scale.</p>
<details><summary>Classical and neural metrics separately</summary><h4>Classical</h4>{stats_table(d, "classical")}<h4>Neural</h4>{stats_table(d, "neural")}</details>
</section>

<section class="slide" id="scaling"><h2>From 1024×1024 to 4K</h2>
<figure>{sc_svg}<figcaption>{sc_flat} of {sc_n} metrics stay within 1.5× (shaded). Slower per pixel at 4K: {E(", ".join(f"{m['metric']} {r:.1f}×" for m, _, r in sc_out if r > 1)) or "none"}. Faster: {E(", ".join(f"{m['metric']} {r:.2f}×" for m, _, r in sc_out if r < 1)) or "none"}. Single thread, same binaries; neural metrics at 4K still pending.</figcaption></figure>
</section>

<section class="slide" id="dvifm"><h2>DVIFM versions (dvifmish)</h2>
<p>dvifmish ports M. Pedzisz's four C++ reference variants ("port mode") and adds presets reconstructed from his talk. Agreement is with the subjective JND on the AIC2026 public split, from our scores; time is i265, 1024×1024.</p>
{table(["version", "agreement [95 %]", "pooled |ρ|", "CPU ns/px 1t", "CPU ns/px 8t", "counted flops/px", "vs the author's C++"], dv_rows, "wrap")}
<p class="note">Port variants: <code>port-lap-v0.3</code> (RGB), <code>port-lap-v0.2</code> (luma), <code>port-2d-v0.2</code>, <code>port-min-v0.2</code>. Their op counts are core-only aggregates. Publishing dvifmish itself needs the author's consent; these are aggregate results.</p>
</section>

<section class="slide" id="canon"><h2>Ours vs the reference implementation</h2>
<p>Every metric on this page with its reference ("canonical") implementation and parity status, as in the zenmetrics README's Metrics table, plus agreement and time.</p>
{table(["metric", "source", "family", "reference implementation", "parity", "AIC2026 column", "agreement", "CPU ns/px 1t", "CPU ns/px 8t"], canon_rows, "wrap canon")}
</section>

<section class="slide" id="aic"><h2>Reproducing the published AIC2026 columns</h2>
{status_rows(status_md)}
</section>

<section class="slide" id="method"><h2>Method, limits and sources</h2>
<ul>
<li>Counts: operations per pixel tallied from each implementation's source, weighted by {len(tables)} cost tables (flop count, the Pedzisz draft's declared weights, scalar costs measured on this CPU, LLVM 22 cost models, llvm-mca).</li>
<li>Time: per-call CPU (task-clock) and wall deltas under <code>perf stat</code>, decode and setup excluded, 1024×1024, one content pair, three repeats; 1 thread pinned to a P-core, 8 threads on P-cores 0–7.</li>
<li>Agreement: mean within-source Spearman with the preliminary JND reconstruction; bootstrap over the 42 sources.</li>
<li>Limits: one host, one shape, one content pair; per-pixel counts ignore vectorization, caches and threading; several neural tallies await recounts; dvifmish op counts are core-only aggregates.</li>
</ul>
<p class="note">Sources: zenmetrics <code>{E(src["public"])}</code>, zenmetrics-private <code>{E(src.get("private") or "not included")}</code>, dvifmish <code>{E(src.get("dvifmish") or "–")}</code>; extra timing: {E(", ".join(src.get("extra_timing") or []) or "none")}; extra agreement: {E(", ".join(src.get("agreement") or []) or "none")}. Generated by <code>scripts/complexity/brief_data.py</code> and <code>brief_page.py</code>; this file makes no network requests.{(" Notes: " + E("; ".join(d["notes"]))) if d["notes"] else ""}</p>
</section>
</main>
<script>{js}</script></body></html>'''
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.html').write_text(page)
    print(f'wrote {out}/index.html ({len(page)//1024} KiB); default weights {best}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    build(a.data, a.out)
