#!/usr/bin/env python3
"""ROBUST per-axis content-dependent RD-value analysis (v2).

Replaces global BD-rate (fragile when a knob scrambles the quality curve) with an
ISO-Q RD metric: at each matched q, convert the value's (Δlog-bytes, Δscore) vs the
baseline into a single quality-equivalent byte delta using the baseline's local
byte-per-quality slope s_q = d(log10 bytes)/d(score):

    rd_pct(img,v) = mean_q [ 10^( Δlog10bytes_q  -  s_q · Δscore_q ) - 1 ] · 100
    (negative => v saves bytes at equal quality = better; positive = worse)

This signs quality-destroying knobs (same bytes, lower score) correctly as +rd.

Per axis it reports:
  best_fixed_savings = -min_v mean_rd[v]                 (% bytes a fixed default saves)
  picker_gain        = mean_i(rd[v*] - oracle_i)         (extra % from per-image picking)
  flip_score         = max_v min(frac_better_v, frac_worse_v)   (two-sided content-dep,
                       noise-robust: a value materially better on some content AND
                       materially worse on other content)
Verdict:
  flip_score >= 0.15                          -> PICKER (optimum moves with content)
  best_fixed >= 0.5 & v* not worse-anywhere   -> CODE=v* (universal win)
  else                                        -> CODE/DROP (inert / keep default)
"""
import sys, json, csv, collections, math
import numpy as np

TSV, SCORE_COL, CODEC = sys.argv[1], sys.argv[2], sys.argv[3]
EFF = 0.5          # % rd considered "material"
FLIP = 0.15        # flip-score threshold for content-dependence

def avif_axis(cell):
    if cell == 's4':
        return ('_baseline', 'default')
    if cell in ('s2', 's6', 's8'):
        return ('speed', cell[1:])
    t = cell[3:]
    table = {'420': ('subsampling', '420'), '422': ('subsampling', '422'),
             'bd10': ('bit_depth', '10'), 'rgb': ('color_model', 'rgb'), 'noqm': ('qm', 'off'),
             'cdef0': ('cdef', 'off'), 'cdef1': ('cdef', 'on'), 'still': ('tune_still_image', 'on'),
             'fdb1': ('fast_deblock', 'on'), 'vaq0.5': ('vaq', 'on@0.5'),
             'rdotx0': ('rdotx', 'off'), 'rdotx1': ('rdotx', 'on'), 'sgr1': ('sgr', 'on'),
             'segcx1': ('seg_complex', 'on'), 'bup1': ('encode_bottomup', 'on'),
             'lrf0': ('lrf', 'off'), 'lrf1': ('lrf', 'on'),
             'part16.64': ('partition_range', '16.64'), 'cpred1': ('complex_prediction', 'on'),
             'trel': ('trellis', 'on')}
    if t in table:
        return table[t]
    if t.startswith('vaqs'):
        return ('vaq_strength', t[4:])
    if t.startswith('sb'):
        return ('seg_boost', t[2:])
    return ('OTHER:' + t, t)

def jpeg_axis(cell):
    BASE = 'jp3_t0_small_420'
    if cell == BASE:
        return ('_baseline', 'default')
    if cell.startswith(BASE + '-'):
        f = cell[len(BASE) + 1:]
        return {'16b': ('allow_16bit', 'on'), 'noaq': ('aq', 'off'),
                'noder': ('deringing', 'off'), 'sharp': ('sharpening', 'on'),
                'blur0.4': ('pre_blur', '0.4')}.get(f, ('OTHER:' + f, f))
    toks = cell.split('_'); fam, tr, scan, sub = (toks + ['', '', '', ''])[:4]
    if fam.startswith('moz[cqd'):
        return ('chroma_quality', fam[fam.find('cqd') + 3:fam.find(']')])
    if fam == 'moz':
        return ('quant_family', 'moz')
    if fam == 'gls':
        return ('quant_family', 'gls')
    if fam == 'pw4':
        return ('pre_blur', 'pw4')
    if fam.startswith('jp3['):
        return ('chroma_dist_scales', fam[3:])
    if sub in ('422', '444'):
        return ('subsampling', sub)
    if 'xyb' in cell:
        return ('xyb_quant', 'on')
    if scan in ('base', 'prog', 'psrch', 'smsrch'):
        return ('scans', scan)
    if tr != 't0':
        if 'cpl' in tr:
            return ('aq_coupling', tr)
        if 'dc' in tr:
            return ('delta_dc', tr)
        if 'l21' in tr:
            return ('trellis_lambda2', tr)
        return ('trellis_lambda', tr.replace('tr', ''))
    return ('OTHER:' + cell, cell)

BROAD_ALL = {'avif': {'vaq', 'vaq_strength', 'seg_boost', 'cdef', 'tune_still_image',
                      'fast_deblock', 'rdotx', 'sgr', 'seg_complex', 'encode_bottomup', 'lrf',
                      'partition_range', 'complex_prediction', 'trellis'},
             'jpeg': {'trellis_lambda', 'aq_coupling', 'delta_dc', 'trellis_lambda2',
                      'quant_family', 'chroma_quality', 'chroma_dist_scales', 'scans',
                      'pre_blur', 'aq', 'deringing', 'allow_16bit', 'sharpening', 'xyb_quant'}}
axis_of = avif_axis if CODEC == 'avif' else jpeg_axis
BROAD = BROAD_ALL[CODEC]
base_cell = 's4' if CODEC == 'avif' else 'jp3_t0_small_420'
picks = {r['out']: r for r in json.load(open(
    '/tmp/claude-1000/-home-lilith-work-zen-zenmetrics/'
    '51b72165-bbdf-44d4-9b34-be022d2f50f5/scratchpad/sources.json'))}

# load: (img, cell) -> {q: (bytes, score)}
cur = collections.defaultdict(dict)
img_class = {}
images = set()
for r in csv.DictReader(open(TSV), delimiter='\t'):
    s = r.get(SCORE_COL, '')
    if not s:
        continue
    try:
        by = float(r['encoded_bytes']); sc = float(s)
    except ValueError:
        continue
    if not (by > 0 and math.isfinite(sc)):
        continue
    cell = json.loads(r['knob_tuple_json'])['cell']
    img = r['image_path']
    images.add(img)
    if img in picks:
        img_class[img] = picks[img]['content_class']
    cur[(img, cell)][int(r['q'])] = (by, sc)

def global_slope(curve):
    """ROBUST single per-image baseline slope s = d(log10 bytes)/d(score) over the
    full q range. Avoids per-q slope noise that spuriously inflates flip-scores."""
    qs = sorted(curve)
    if len(qs) < 2:
        return None
    lb = {q: math.log10(curve[q][0]) for q in qs}
    sc = {q: curve[q][1] for q in qs}
    lo, hi = qs[0], qs[-1]
    ds = sc[hi] - sc[lo]
    if abs(ds) < 1e-6:
        return None
    s = (lb[hi] - lb[lo]) / ds
    return s if s > 0 else None

def iso_q_rd(base_curve, v_curve, s):
    """mean over matched q of the quality-equivalent byte delta (negative=better).
    d clamped to [-3,3] so a quality-cratering knob is a bounded large penalty, not
    an overflow."""
    if s is None:
        return None
    shared = [q for q in v_curve if q in base_curve]
    if len(shared) < 3:
        return None
    vals = []
    for q in shared:
        bby, bsc = base_curve[q]; vby, vsc = v_curve[q]
        d = math.log10(vby) - math.log10(bby) - s * (vsc - bsc)
        d = max(-3.0, min(3.0, d))
        vals.append(10 ** d - 1.0)
    return float(np.mean(vals)) * 100.0

# axis -> value -> {img: rd_pct}
av = collections.defaultdict(lambda: collections.defaultdict(dict))
for (img, cell), curve in cur.items():
    ax, val = axis_of(cell)
    if ax == '_baseline' or ax.startswith('OTHER:'):
        continue
    base = cur.get((img, base_cell))
    if not base:
        continue
    rd = iso_q_rd(base, curve, global_slope(base))
    if rd is not None:
        av[ax][val][img] = rd

results = []
for ax, vals in av.items():
    imgs = sorted({i for v in vals for i in vals[v]})
    if not imgs:
        continue
    cand = ['default'] + sorted(vals)
    rd = lambda v, im: 0.0 if v == 'default' else vals[v].get(im, np.nan)
    mean_rd = {v: np.nanmean([rd(v, im) for im in imgs]) for v in cand}
    mean_rd = {v: m for v, m in mean_rd.items() if np.isfinite(m)}
    vstar = min(mean_rd, key=mean_rd.get)
    best_fixed = -mean_rd[vstar]
    gains, win_v = [], []
    for im in imgs:
        if not np.isfinite(rd(vstar, im)):
            continue
        bv = {v: rd(v, im) for v in cand if np.isfinite(rd(v, im))}
        wv = min(bv, key=bv.get); win_v.append(wv)
        gains.append(rd(vstar, im) - bv[wv])
    gains = np.array(gains) if gains else np.array([0.0])
    picker_gain = float(gains.mean())
    # two-sided flip per non-default value
    flip = 0.0; per_val = {}
    for v in sorted(vals):
        arr = np.array([vals[v][im] for im in imgs if im in vals[v]])
        if len(arr) == 0:
            continue
        fb = float(np.mean(arr < -EFF)); fw = float(np.mean(arr > EFF))
        per_val[v] = {'n': len(arr), 'mean_rd': round(float(arr.mean()), 3),
                      'frac_better': round(fb, 2), 'frac_worse': round(fw, 2)}
        flip = max(flip, min(fb, fw))
    cls_win = collections.defaultdict(collections.Counter)
    for im, wv in zip(imgs, win_v):
        cls_win[img_class.get(im, '?')][wv] += 1
    results.append({'axis': ax, 'broad': ax in BROAD, 'n_img': len(imgs),
                    'best_fixed_value': vstar, 'best_fixed_savings_pct': round(best_fixed, 3),
                    'picker_gain_pct': round(picker_gain, 3), 'flip_score': round(flip, 3),
                    'per_value': per_val, 'win_distribution': dict(collections.Counter(win_v)),
                    'per_class_winner': {c: dict(w) for c, w in cls_win.items()}})

def verdict(r):
    if r['flip_score'] >= FLIP:
        return 'PICKER'
    vs = r['best_fixed_value']
    if r['best_fixed_savings_pct'] >= EFF and vs != 'default' \
       and r['per_value'].get(vs, {}).get('frac_worse', 1) < 0.10:
        return f'CODE={vs}'
    return 'CODE/DROP'

for r in results:
    r['verdict'] = verdict(r)
# order: PICKER first (by flip), then by best_fixed
results.sort(key=lambda r: (r['verdict'] != 'PICKER', -r['flip_score'], -r['best_fixed_savings_pct']))

print(f'\n=== {CODEC.upper()}  metric={SCORE_COL}  n_images={len(images)}  '
      f'(EFF={EFF}% FLIP={FLIP}) ===')
print(f'{"axis":22s} {"brd":3s} {"bestfix%":>8s} {"fixval":>9s} {"picker%":>7s} {"flip":>5s}  verdict')
print('-' * 88)
for r in results:
    print(f'{r["axis"]:22s} {"Y" if r["broad"] else " ":3s} '
          f'{r["best_fixed_savings_pct"]:8.2f} {str(r["best_fixed_value"]):>9s} '
          f'{r["picker_gain_pct"]:7.2f} {r["flip_score"]:5.2f}  {r["verdict"]}')
out = {'codec': CODEC, 'metric': SCORE_COL, 'n_images': len(images),
       'eff_pct': EFF, 'flip_thresh': FLIP, 'axes': results}
p = TSV.rsplit('.', 1)[0] + f'.rdanalysis_{SCORE_COL}.json'
json.dump(out, open(p, 'w'), indent=2)
print('wrote', p)
