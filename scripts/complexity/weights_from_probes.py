#!/usr/bin/env python3
"""Build weight tables for correlate.py from measured probes (probes_i265.tsv).

Two tables per core type: scalar reciprocal throughput (cycles per scalar
operation, issue-bound) and scalar latency (cycles on a dependency chain).
Each op class maps to one named scalar instruction or libm call per precision;
the mapping is printed into the table's `source`. Classes without a probe
(int, load, store with no single-instruction meaning here) get no weight, and
correlate.py lists them as unweighted rather than assuming a value.

  weights_from_probes.py PROBES.tsv --core P --kind rthroughput --out FILE.json
"""
import argparse, csv, json
from pathlib import Path

MAP = {  # op class -> {precision: probe}
    'add': {'f32': 'vaddss', 'f64': 'vaddsd'}, 'mul': {'f32': 'vmulss', 'f64': 'vmulsd'},
    'fma': {'f32': 'vfmadd231ss', 'f64': 'vfmadd231sd'}, 'div': {'f32': 'vdivss', 'f64': 'vdivsd'},
    'sqrt': {'f32': 'vsqrtss', 'f64': 'vsqrtsd'}, 'rcp': {'f32': 'vrcpss'}, 'rsqrt': {'f32': 'vrsqrtss'},
    'cmp': {'f32': 'vminss', 'f64': 'vminsd'}, 'abs': {'f32': 'vandps_x', 'f64': 'vandpd_x'},
    'cvt': {'f32': 'vcvtss2sd', 'f64': 'vcvtsd2ss'},
    'exp': {'f32': 'expf', 'f64': 'exp'}, 'log': {'f32': 'logf', 'f64': 'log'},
    'pow': {'f32': 'powf', 'f64': 'pow'}, 'trig': {'f32': 'cosf'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('probes')
    ap.add_argument('--core', default='P')
    ap.add_argument('--kind', default='rthroughput', choices=['rthroughput', 'latency'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = {(r['probe'], r['core_type'], r['kind']): float(r['cycles_median'])
            for r in csv.DictReader(open(a.probes), delimiter='\t')}
    weights, used = {}, []
    for op, by in MAP.items():
        for prec, probe in by.items():
            v = rows.get((probe, a.core, a.kind))
            if v is not None:
                weights.setdefault(op, {})[prec] = v
                used.append(f'{op}/{prec}={probe}')
    for op in weights:  # mixed-precision stages use the f32 cost when present
        weights[op].setdefault('mixed', weights[op].get('f32', next(iter(weights[op].values()))))
    Path(a.out).write_text(json.dumps({
        'name': f'i265 {a.core}-core scalar {a.kind} (measured cycles)',
        'source': f'{a.probes}; median of 5 perf cycle slopes; mapping ' + ', '.join(used),
        'weights': weights}, indent=1) + '\n')


if __name__ == '__main__':
    main()
