#!/usr/bin/env python3
"""Functional SDR driver controls. These do not certify accuracy or admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from build_benchmark_study import inventory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binaries', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--size', type=int, nargs=2, default=[256, 256], metavar=('WIDTH', 'HEIGHT'))
parser.add_argument('--metric', action='append', help='restrict controls to named cold RGB8 routes')
args = parser.parse_args()
binaries, root = args.binaries.resolve(), args.output.resolve()
w, h = args.size
if w < 1 or h < 1:
    parser.error('dimensions must be positive')
rows = [row for row in inventory()['metrics'] if row['cold_rgb8']]
if args.metric:
    unknown = set(args.metric) - {row['metric'] for row in rows}
    if unknown:
        parser.error(f'unknown cold RGB8 metrics: {sorted(unknown)}')
    rows = [row for row in rows if row['metric'] in args.metric]
root.mkdir(parents=True, exist_ok=False)
reference = bytes((x * 17 + y * 13 + c * 41) % 256
                  for y in range(h) for x in range(w) for c in range(3))
distorted = bytes(max(0, value - 7) for value in reference)
for name, data in [('reference', reference), ('distorted', distorted)]:
    (root / f'{name}-packed').write_bytes(data)
    (root / f'{name}-padded').write_bytes(b''.join(
        data[y*w*3:(y+1)*w*3] + bytes([99]*7) for y in range(h)))
env = dict(os.environ, RAYON_NUM_THREADS='1', ZENBENCH_OPERATIONS='2')
env.pop('ZENBENCH_RESULT_FILE', None)

def execute(case, label, variant):
    path = root / f'{label}.json'
    path.write_text(json.dumps(case))
    result = subprocess.run([str(binaries / variant), 'fleet', str(path)],
                            env=env, capture_output=True, timeout=120)
    (root / f'{label}-{variant}.stdout').write_bytes(result.stdout)
    (root / f'{label}-{variant}.stderr').write_bytes(result.stderr)
    return result

controls = []
for row in rows:
    previous = None
    for layout, padding in [('packed', 0), ('padded', 7)]:
        metric = row['metric']
        label = f'{metric}-{layout}'
        case = dict(schema=2, metric=metric, width=w, height=h,
                    stride_bytes=w*3+padding, reference=f'reference-{layout}',
                    distorted=f'distorted-{layout}', pixel_format='rgb8_srgb',
                    reuse='cold', threads=1, products='all', warmup_operations=0)
        observed = []
        for variant in ['latency', 'memory']:
            result = execute(case, label, variant)
            assert result.returncode == 0, (label, variant, result.stderr.decode())
            value = json.loads(result.stdout)
            products = value['driver']['correctness']['products']
            assert len(products) == 2 and products[0] == products[1], (label, products)
            assert products[0] and len({p['name'] for p in products[0]}) == len(products[0])
            assert all(type(p['bits']) is int and 0 <= p['bits'] < 2**64 for p in products[0])
            assert value['workload']['warmup_operations'] == 0
            assert value['timing_usable'] == (variant == 'latency')
            observed.append(products)
        assert observed[0] == observed[1], (label, 'instrumentation changed scalar products')
        if previous is not None:
            assert previous == observed[0], (label, 'stride changed scalar products')
        previous = observed[0]
        controls.append(dict(case=label, products=observed[0]))

# Unknown schemas/fields and unsupported product/reuse contracts must fail.
for change in [dict(schema=3), dict(products='diffmap'), dict(reuse='warm_reference'),
               dict(warmup_operations=1025), dict(unknown=True)]:
    invalid = dict(case, **change)
    result = execute(invalid, f'refusal-{len(controls)}', 'latency')
    assert result.returncode != 0, change
    controls.append(dict(refused=change))

# Schema 2 reports and executes the declared warmup count, including zero.
for count in [0, 2]:
    warm = dict(case, metric='ssim2', reuse='warm_reference', products='primary',
                warmup_operations=count)
    result = execute(warm, f'warmup-{count}', 'memory')
    assert result.returncode == 0, result.stderr.decode()
    value = json.loads(result.stdout)
    assert value['workload']['warmup_operations'] == count
    assert len(value['driver']['correctness']['score_bits']) == 2
(root / 'controls.json').write_text(json.dumps(controls, indent=2) + '\n')
print(json.dumps(dict(sdr_metrics=len(rows), width=w, height=h,
                      instrumentation_and_stride_parity='exact scalar products',
                      qualification='functional control only; no accuracy or resource admission claim')))
