#!/usr/bin/env python3
"""Measure one common native workload for an explicit scheduling scope.

Caller supplies shared heavy admission. Uses the existing native arm, including
its ten warm observations and per-process memory log. This is a scheduling
measurement, not an original-reference comparison or a qualified regression.
"""
import argparse
import csv
import json
from pathlib import Path

from compare_canonical import run_arm, sha
from prepare_spotcheck_raw import crop_rgb8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    names = plan['metrics'] + plan.get('controls', [])
    if len(names) != len(set(names)) or not names or plan['samples'] < 10:
        raise ValueError('distinct explicit routes and at least ten observations required')
    inputs = {plan['binary']: plan['binary_sha256'],
              plan['pair_manifest']: plan['pair_manifest_sha256'],
              **{plan[role]: plan[role + '_sha256'] for role in ['reference', 'distorted']}}
    def verify():
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise ValueError('native scheduling input or worker changed')
    verify()
    a.output.mkdir()
    (a.output / 'plan.json').write_bytes(a.plan.read_bytes())
    crop = plan['crop']
    pixels = {}
    for role in ['reference', 'distorted']:
        path = a.output.resolve() / (role + '.rgb')
        path.write_bytes(crop_rgb8(Path(plan[role]), plan['source_width'], plan['source_height'],
                                   plan['source_stride'], crop['left'], crop['top'],
                                   plan['width'], plan['height']))
        pixels[role] = str(path)
        inputs[str(path)] = sha(path)
    rows = []
    for name in names:
        case = dict(metric=name, width=plan['width'], height=plan['height'],
                    threads=plan['threads'], **pixels)
        observation = run_arm(plan, case, 'rust', a.output / name, plan['samples'])
        rows.append(dict(metric=name, median_ms=observation['median_ns'] / 1e6,
                         score=observation['score'], control=name in plan.get('controls', []),
                         observations=observation))
        verify()
        print('NATIVE_ORDER_CASE ' + json.dumps({k: rows[-1][k]
              for k in ['metric', 'median_ms', 'score', 'control']}), flush=True)
    with (a.output / 'observations.json').open('x') as stream:
        json.dump(dict(plan_sha256=sha(a.plan), bindings=inputs, rows=rows,
                       source_revision=plan['source_revision'],
                       qualification='One common raw crop, warm native scheduling hints; '
                                     'no canonical ratio or regression qualification'), stream, indent=2)
        stream.write('\n')
    with (a.output / 'order.tsv').open('x') as stream:
        stream.write('# Retained local native warm medians; one common raw crop. Scheduling hints only.\n')
        stream.write(f'# source={plan["source_revision"]}; width={plan["width"]}; '
                     f'height={plan["height"]}; threads={plan["threads"]}; samples={plan["samples"]}\n')
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['metric', 'historical_ms', 'source'])
        for row in sorted((r for r in rows if not r['control']),
                          key=lambda r: (r['median_ms'], r['metric'])):
            writer.writerow([row['metric'], format(row['median_ms'], '.17g'),
                             str(a.output.resolve() / row['metric'] / 'stdout.log')])


if __name__ == '__main__':
    main()
