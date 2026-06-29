#!/usr/bin/env python3
"""Per-ROW (image,q) overhead tail, bucketed by quality, ssim2 vs zensim.

Exposes the LOW-zq tail (where a learned/steeper metric is least settled) that the
per-image-curve average in tail.py hides. For the best-fixed config AND the baseline
cell, at each (image,q): overhead = config_bytes / (cheapest bytes among ALL cells
reaching that row's score) - 1. Bucketed q: low {30,45} / mid {60,75} / high {85,92}.
Also lists the cells with the worst low-q ssim2 rows (veto candidates).

Usage: tail_byq.py TSV CODEC
"""
import sys, json, csv, collections, math
import numpy as np
TSV, CODEC = sys.argv[1], sys.argv[2]
base_cell = 's4' if CODEC == 'avif' else 'jp3_t0_small_420'
QBUCKET = {30: 'low', 45: 'low', 60: 'mid', 75: 'mid', 85: 'high', 92: 'high'}

def load(metric):
    cur = collections.defaultdict(dict); imgs = set()
    for r in csv.DictReader(open(TSV), delimiter='\t'):
        s = r.get(metric, '')
        if not s:
            continue
        try:
            by = float(r['encoded_bytes']); sc = float(s)
        except ValueError:
            continue
        if not (by > 0 and math.isfinite(sc)):
            continue
        cur[(r['image_path'], json.loads(r['knob_tuple_json'])['cell'])][int(r['q'])] = (by, sc)
        imgs.add(r['image_path'])
    return cur, sorted(imgs)

def per_image_oracle(img_cells):
    pts = [p for c in img_cells.values() for p in c.values()]
    pts.sort(key=lambda p: p[1])
    scs = np.array([p[1] for p in pts]); by = np.array([p[0] for p in pts], float)
    sm = np.minimum.accumulate(by[::-1])[::-1]
    def q(t):
        i = np.searchsorted(scs, t, side='left')
        return float(sm[i]) if i < len(scs) else None
    return q

def best_fixed(cur, imgs):
    icells = collections.defaultdict(dict)
    for (im, cl), c in cur.items():
        icells[im][cl] = c
    orac = {im: per_image_oracle(icells[im]) for im in imgs if len(icells[im]) >= 3}
    cov = collections.Counter(cl for im in imgs for cl in icells[im] if len(icells[im][cl]) >= 3)
    cands = [c for c, n in cov.items() if n >= 0.8 * len(imgs)]
    means = {}
    for cl in cands:
        ov = []
        for im in imgs:
            if im in orac and cl in icells[im]:
                for q, (by, sc) in icells[im][cl].items():
                    o = orac[im](sc)
                    if o:
                        ov.append(by / o - 1)
        if ov:
            means[cl] = np.mean(ov)
    return min(means, key=means.get), icells, orac

def rows_by_bucket(cell, icells, orac, imgs):
    buckets = collections.defaultdict(list)
    for im in imgs:
        if im not in orac or cell not in icells[im]:
            continue
        for q, (by, sc) in icells[im][cell].items():
            o = orac[im](sc)
            if o:
                buckets[QBUCKET.get(q, '?')].append((by / o - 1) * 100)
    return buckets

def tstat(a):
    a = np.array(a)
    return (f'n{len(a):3d}  p50 {np.percentile(a,50):6.1f}  p90 {np.percentile(a,90):6.1f}  '
            f'p99 {np.percentile(a,99):7.1f}  max {a.max():7.1f}  >50%:{int((a>50).sum()):3d} '
            f'>100%:{int((a>100).sum()):2d} >200%:{int((a>200).sum()):2d}')

print(f'\n######## {CODEC.upper()}: per-ROW (image,q) overhead by quality bucket ########')
worst = {}
for m in ('score_ssim2', 'score_zensim'):
    cur, imgs = load(m)
    cstar, icells, orac = best_fixed(cur, imgs)
    print(f'\n=== target {m}  best-fixed={cstar} ===')
    for label, cell in (('BEST-FIXED', cstar), ('baseline', base_cell)):
        b = rows_by_bucket(cell, icells, orac, imgs)
        print(f'  {label} ({cell}):')
        for bk in ('low', 'mid', 'high'):
            if b.get(bk):
                print(f'    {bk:4s} q: {tstat(b[bk])}')
    # veto scan: worst low-q rows across ALL candidate cells (this metric)
    low = collections.defaultdict(list)
    cov = collections.Counter(cl for im in imgs for cl in icells[im] if len(icells[im][cl]) >= 3)
    for cl in [c for c, n in cov.items() if n >= 0.8 * len(imgs)]:
        for im in imgs:
            if im in orac and cl in icells[im]:
                for q, (by, sc) in icells[im][cl].items():
                    if QBUCKET.get(q) == 'low':
                        o = orac[im](sc)
                        if o:
                            low[cl].append((by / o - 1) * 100)
    worst[m] = sorted(((cl, float(np.percentile(v, 99)), float(np.max(v)))
                       for cl, v in low.items() if len(v) >= 10),
                      key=lambda x: -x[1])[:6]
print('\n--- worst LOW-q rows by cell (p99 overhead %) — veto candidates [ssim2] ---')
for cl, p99, mx in worst['score_ssim2']:
    print(f'    {cl:28s} p99 {p99:7.1f}%  max {mx:7.1f}%')
