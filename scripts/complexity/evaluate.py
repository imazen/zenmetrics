#!/usr/bin/env python3
"""Evaluate IQA method-complexity tallies (schema iqa-complexity-tally/1).

A tally lists the stages of one metric with per-element logical operation
counts taken from source. This script turns them into per-original-pixel
totals, cumulative halo, work/span and access summaries at given shapes.
Method and conventions: docs/IQA_COMPLEXITY.md.

  evaluate.py TALLY.json [...] [--shape 1024x1024 ...] [--json OUT]
  evaluate.py --check TALLY.json [...]     # schema + expression validation only
"""
import argparse, json, math, sys
from pathlib import Path

SCHEMA = 'iqa-complexity-tally/1'
OPS = ['add', 'mul', 'fma', 'div', 'rcp', 'sqrt', 'rsqrt', 'exp', 'log', 'pow', 'trig',
       'cmp', 'abs', 'cvt', 'int', 'load', 'store']
FLOP_WEIGHT = {'add': 1, 'mul': 1, 'fma': 2, 'div': 1, 'rcp': 1, 'sqrt': 1, 'rsqrt': 1,
               'exp': 1, 'log': 1, 'pow': 1, 'trig': 1}
PRECISIONS = {'f16', 'f32', 'f64', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'mixed'}
ACCESS = {'pointwise', 'sep_stencil', 'stencil2d', 'block', 'downsample', 'upsample', 'resample',
          'transform', 'reduction', 'gather', 'histogram', 'sort', 'data_dependent', 'matmul', 'conv'}
STAGE_KEYS = {'id', 'name', 'source', 'images', 'elements', 'levels', 'precision', 'ops',
              'bytes_read', 'bytes_written', 'access', 'radius', 'scale', 'chain', 'span',
              'depends_on', 'barrier', 'serial', 'shared_with', 'notes', 'logical_layer', 'axis', 'class'}
CLASSES = {'intrinsic', 'implementation'}  # implementation: work the method does not require (strip overlap recompute, padding copies)
SAFE = {k: getattr(math, k) for k in ('ceil', 'floor', 'log2', 'log', 'sqrt', 'exp', 'pi')}
SAFE.update(min=min, max=max, abs=abs, round=round, int=int, sum=sum, range=range)


def ev(expr, env):
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str):
        raise ValueError(f'expression must be number or string, got {expr!r}')
    return float(eval(expr, {'__builtins__': {}}, {**SAFE, **env}))


def shape_env(tally, w, h):
    env = {'W': w, 'H': h, 'P': w * h}
    for k, v in tally.get('params', {}).items():
        env[k] = v
    for k, v in tally.get('derived', {}).items():
        env[k] = ev(v, env)
    return env


def levels_of(stage, env):
    n = int(ev(stage.get('levels', 1), env))
    if n < 1:
        raise ValueError(f"stage {stage['id']}: levels < 1")
    return range(n)


def validate(tally):
    errs = []
    if tally.get('schema') != SCHEMA:
        errs.append(f'schema must be {SCHEMA}')
    for k in ('metric', 'counter', 'source_commit', 'kind', 'stages', 'assumptions'):
        if k not in tally:
            errs.append(f'missing {k}')
    ids = set()
    for s in tally.get('stages', []):
        sid = s.get('id', '?')
        extra = set(s) - STAGE_KEYS
        if extra:
            errs.append(f'{sid}: unknown keys {sorted(extra)}')
        for k in ('id', 'name', 'source', 'elements', 'precision', 'ops', 'access'):
            if k not in s:
                errs.append(f'{sid}: missing {k}')
        if sid in ids:
            errs.append(f'duplicate stage id {sid}')
        ids.add(sid)
        if s.get('precision') not in PRECISIONS:
            errs.append(f"{sid}: precision {s.get('precision')!r}")
        if s.get('axis', 'xy') not in ('x', 'y', 'xy', 'none'):
            errs.append(f"{sid}: axis {s.get('axis')!r}")
        if s.get('class', 'intrinsic') not in CLASSES:
            errs.append(f"{sid}: class {s.get('class')!r}")
        if s.get('access') not in ACCESS:
            errs.append(f"{sid}: access {s.get('access')!r}")
        bad = set(s.get('ops', {})) - set(OPS)
        if bad:
            errs.append(f'{sid}: unknown op classes {sorted(bad)}')
        src = s.get('source') or []
        if not src or not all(isinstance(x, str) and ':' in x for x in src):
            errs.append(f'{sid}: source must list path:line ranges')
    for s in tally.get('stages', []):
        for d in s.get('depends_on', []):
            if d not in ids:
                errs.append(f"{s.get('id')}: depends_on unknown {d}")
    if not errs:
        try:
            evaluate(tally, 256, 256)
        except Exception as e:  # expression errors surface here
            errs.append(f'evaluation at 256x256 failed: {e}')
    return errs


def evaluate(tally, w, h):
    env = shape_env(tally, w, h)
    P = w * h
    stages = {s['id']: s for s in tally['stages']}
    totals = {(op, s['precision']): 0.0 for s in tally['stages'] for op in OPS}
    by_class = {c: 0.0 for c in CLASSES}
    per_stage, bytes_rw = [], [0.0, 0.0]
    span_of, halo_of, min_width = {}, {}, math.inf
    barriers = serial = 0
    for s in tally['stages']:  # stages are listed in dependency order
        imgs = ev(s.get('images', 1), env)
        work = {op: 0.0 for op in OPS}
        elems_total, span, halo_levels = 0.0, 0.0, []
        chain = bool(s.get('chain', False))
        for l in levels_of(s, env):
            e = {**env, 'l': l}
            n = ev(s['elements'], e)
            elems_total += n * imgs
            min_width = min(min_width, n)
            for op, c in s['ops'].items():
                work[op] += imgs * n * ev(c, e)
            bytes_rw[0] += imgs * n * ev(s.get('bytes_read', 0), e)
            bytes_rw[1] += imgs * n * ev(s.get('bytes_written', 0), e)
            sp = ev(s.get('span', 1), e)
            span = span + sp if chain else max(span, sp)
            halo_levels.append(ev(s.get('radius', 0), e) * ev(s.get('scale', 1), e))
        own = sum(halo_levels) if chain else max(halo_levels)
        axis = s.get('axis', 'xy')
        own_xy = (own if axis in ('x', 'xy') else 0.0, own if axis in ('y', 'xy') else 0.0)
        deps = s.get('depends_on', [])
        span_of[s['id']] = span + max((span_of[d] for d in deps), default=0.0)
        base = (max((halo_of[d][0] for d in deps), default=0.0), max((halo_of[d][1] for d in deps), default=0.0))
        halo_of[s['id']] = (base[0] + own_xy[0], base[1] + own_xy[1])
        barriers += bool(s.get('barrier'))
        serial += bool(s.get('serial'))
        cls = s.get('class', 'intrinsic')
        for op in OPS:
            totals[(op, s['precision'])] += work[op]
            by_class[cls] += FLOP_WEIGHT.get(op, 0) * work[op]
        per_stage.append({'id': s['id'], 'access': s['access'], 'precision': s['precision'], 'class': cls,
                          'elements_per_px': elems_total / P,
                          'ops_per_px': {op: v / P for op, v in work.items() if v},
                          'halo_px': max(halo_of[s['id']]), 'span': span_of[s['id']]})
    per_px = {}
    for (op, prec), v in totals.items():
        if v:
            per_px.setdefault(op, {})[prec] = v / P
    flops = sum(FLOP_WEIGHT.get(op, 0) * v for (op, _), v in totals.items())
    arith = sum(v for (op, _), v in totals.items() if op not in ('load', 'store', 'int', 'cvt'))
    span_total = max(span_of.values(), default=0.0)
    byt = bytes_rw[0] + bytes_rw[1]
    return {'shape': f'{w}x{h}', 'ops_per_px': per_px, 'flops_per_px': flops / P,
            'arith_ops_per_px': arith / P, 'bytes_per_px': byt / P,
            'flops_per_byte': flops / byt if byt else None,
            'halo_px': max((max(h) for h in halo_of.values()), default=0.0),
            'halo_x_px': max((h[0] for h in halo_of.values()), default=0.0),
            'halo_y_px': max((h[1] for h in halo_of.values()), default=0.0),
            'intrinsic_flops_per_px': by_class['intrinsic'] / P,
            'implementation_flops_per_px': by_class['implementation'] / P, 'span': span_total,
            'work_over_span': arith / span_total if span_total else None,
            'min_stage_width': min_width, 'barriers': barriers, 'serial_stages': serial,
            'stages': per_stage}


def parse_shape(s):
    w, h = s.lower().split('x')
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('tallies', nargs='+')
    ap.add_argument('--shape', action='append', default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    shapes = [parse_shape(s) for s in (a.shape or ['256x256', '1024x1024', '1920x1080', '3840x2160'])]
    out, failed = [], False
    for path in a.tallies:
        t = json.loads(Path(path).read_text())
        errs = validate(t)
        if errs:
            failed = True
            print(f'{path}: INVALID', *errs, sep='\n  ', file=sys.stderr)
            continue
        if a.check:
            print(f'{path}: ok ({len(t["stages"])} stages)')
            continue
        for w, h in shapes:
            r = evaluate(t, w, h)
            r.update(metric=t['metric'], counter=t['counter'], source_commit=t['source_commit'], tally=path)
            out.append(r)
            print(f"{t['metric']:<16} {t['counter']:<6} {r['shape']:>10} flops/px {r['flops_per_px']:10.1f} "
                  f"bytes/px {r['bytes_per_px']:8.1f} halo {r['halo_px']:7.1f} span {r['span']:9.1f} "
                  f"barriers {r['barriers']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
