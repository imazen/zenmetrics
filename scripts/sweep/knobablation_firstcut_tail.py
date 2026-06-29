#!/usr/bin/env python3
"""Oracle-gap + zensim-vs-ssim2 safety analysis.

For each image, build the per-image ORACLE Pareto hull over ALL modes_full md1 cells
(monotone by construction -> BD-rate is robust, no scrambled-curve fragility). Then
measure the iso-quality byte OVERHEAD of shipping a single FIXED config vs that oracle
(= the RD a perfect per-image picker over the whole knob space could buy). Report the
TAIL (p50/p90/p99/max, catastrophic counts >100%/>200%) per target metric, so we can
see whether the catastrophic mis-pick tail is a zensim artifact (cleaner under ssim2).

Usage: tail.py TSV CODEC   (reads both score_zensim and score_ssim2 columns)
"""
import sys, json, csv, collections, math
import numpy as np

TSV, CODEC = sys.argv[1], sys.argv[2]
base_cell = 's4' if CODEC == 'avif' else 'jp3_t0_small_420'

def load(metric):
    cur = collections.defaultdict(dict)   # (img,cell)->{q:(by,sc)}
    imgs = set()
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
        cell = json.loads(r['knob_tuple_json'])['cell']
        cur[(r['image_path'], cell)][int(r['q'])] = (by, sc)
        imgs.add(r['image_path'])
    return cur, sorted(imgs)

def make_oracle(points):
    """points: list of (by,sc). Return a query fn oracle_by(t) = min bytes among ALL
    points with score >= t (the cheapest config+q reaching quality t). Point-based,
    no interpolation -> the overhead of any member config is >=0 by construction."""
    pts = sorted(points, key=lambda p: p[1])         # score asc
    scs = np.array([p[1] for p in pts])
    by = np.array([p[0] for p in pts], float)
    suff_min = np.minimum.accumulate(by[::-1])[::-1]  # min bytes over score>=scs[i]
    def oracle_by(t):
        i = np.searchsorted(scs, t, side='left')
        return float(suff_min[i]) if i < len(scs) else None
    return oracle_by, (scs[0], scs[-1])

def bd_vs_oracle(curve, oracle_by, orange):
    """iso-quality byte overhead (%): mean over the config's q points of
    config_bytes / (cheapest bytes reaching that quality) - 1."""
    if len(curve) < 3:
        return None
    ov = []
    for q in curve:
        by, sc = curve[q]
        if sc > orange[1]:
            continue
        ob = oracle_by(sc)
        if ob and ob > 0:
            ov.append(by / ob - 1.0)
    if len(ov) < 3:
        return None
    return float(np.mean(ov)) * 100.0

def analyze(metric, exclude=frozenset()):
    cur, imgs = load(metric)
    # per-image oracle hull over ALL cells (optionally excluding some, e.g. the
    # avif speed/compute axis, to isolate the broad knobs at a fixed speed)
    oracle = {}
    img_cells = collections.defaultdict(dict)
    for (img, cell), c in cur.items():
        if cell in exclude:
            continue
        img_cells[img][cell] = c
    for img in imgs:
        pts = [p for c in img_cells[img].values() for p in c.values()]
        if len(pts) >= 5:
            oracle[img] = make_oracle(pts)
    # candidate fixed configs: cells present (>=3 q) in >=80% of images
    cov = collections.Counter()
    for img in imgs:
        for cell, c in img_cells[img].items():
            if len(c) >= 3:
                cov[cell] += 1
    thr = 0.8 * len(imgs)
    cands = [c for c, n in cov.items() if n >= thr]
    # mean overhead per candidate fixed config
    per_cell = {}
    for cell in cands:
        ov = []
        for img in imgs:
            if img in oracle and cell in img_cells[img]:
                b = bd_vs_oracle(img_cells[img][cell], *oracle[img])
                if b is not None:
                    ov.append(b)
        if len(ov) >= 0.8 * len(imgs):
            per_cell[cell] = np.array(ov)
    cstar = min(per_cell, key=lambda c: per_cell[c].mean())
    def tailstats(arr):
        return {'n': len(arr), 'mean': round(float(arr.mean()), 2),
                'p50': round(float(np.percentile(arr, 50)), 2),
                'p90': round(float(np.percentile(arr, 90)), 2),
                'p99': round(float(np.percentile(arr, 99)), 2),
                'max': round(float(arr.max()), 2),
                'n_gt100': int((arr > 100).sum()), 'n_gt200': int((arr > 200).sum())}
    return {'metric': metric, 'n_images': len(imgs), 'best_fixed_cell': cstar,
            'best_fixed_tail': tailstats(per_cell[cstar]),
            'baseline_tail': tailstats(per_cell[base_cell]) if base_cell in per_cell else None}

# avif: also isolate broad/picker knobs at fixed speed by excluding the compute axis
SCOPES = [('FULL md1 space', frozenset())]
if CODEC == 'avif':
    SCOPES.append(('broad+picker @ fixed speed s4 (no speed axis)', frozenset({'s2', 's6', 's8'})))

print(f'\n######## {CODEC.upper()}: oracle-gap overhead tail (best-fixed config vs per-image oracle) ########')
res = {}
for scope_name, excl in SCOPES:
    print(f'\n==== scope: {scope_name} ====')
    for m in ('score_zensim', 'score_ssim2'):
        r = analyze(m, excl)
        res[f'{scope_name}|{m}'] = r
        print(f'  --- target metric = {m}  (n_images={r["n_images"]}) ---')
        print(f'    best-fixed config = {r["best_fixed_cell"]}')
        for label, t in (('BEST-FIXED', r['best_fixed_tail']), ('baseline-cell', r['baseline_tail'])):
            if t:
                print(f'    {label:13s}: mean {t["mean"]:6.1f}%  p50 {t["p50"]:6.1f}  p90 {t["p90"]:6.1f}  '
                      f'p99 {t["p99"]:7.1f}  max {t["max"]:7.1f}  | >100%: {t["n_gt100"]}  >200%: {t["n_gt200"]}')
print('\nINTERPRETATION: overhead = iso-quality % extra bytes the best FIXED config spends')
print('vs the per-image oracle over the WHOLE modes_full md1 space. A fat tail (p99/max,')
print('catastrophic counts) = images where picking would buy a lot. Compare across metrics:')
print('if zensim tail >> ssim2 tail, the catastrophes are partly zensim proxy noise.')
json.dump(res, open(TSV.rsplit('.', 1)[0] + '.tail.json', 'w'), indent=2)
