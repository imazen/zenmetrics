#!/usr/bin/env python3
"""Expand a frozen reference plan onto explicitly hashed RGB8 input pairs."""
import argparse
import copy
import json
from pathlib import Path

from compare_canonical import sha


def prepare(source, pairs_path, output, metrics, threads):
    source = source.resolve(strict=True)
    pairs_path = pairs_path.resolve(strict=True)
    original = json.loads(source.read_text())
    pairs = json.loads(pairs_path.read_text())['pairs']
    for asset, expected in original['bindings'].items():
        if sha(asset) != expected:
            raise ValueError('bound reference artifact changed: ' + asset)
    if not pairs or not threads or any(t < 1 for t in threads):
        raise ValueError('nonempty input pairs and positive thread counts required')
    templates = {}
    for name in metrics:
        candidates = [c for c in original['cases']
                      if c['metric'] == name and c['matched_scope']]
        if not candidates:
            raise ValueError('no matched reference contract: ' + name)
        templates[name] = min(candidates, key=lambda c: c['threads'])
    plan = copy.deepcopy(original)
    plan.pop('unavailable', None)
    plan['cases'] = []
    plan['parent_plan_sha256'] = sha(source)
    plan['grid_scope'] = ('Warm scalar comparisons only; execute every declared size, '
                          'including refusals. No inferred parity or cold/map coverage.')
    bindings = plan['bindings']
    for path in (source, pairs_path, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    identifiers = set()
    for pair in pairs:
        width, height = pair['width'], pair['height']
        if width < 1 or height < 1:
            raise ValueError('positive input geometry required')
        paths = {}
        for role in ('reference', 'distorted'):
            path = (pairs_path.parent / pair[role]).resolve(strict=True)
            if path.stat().st_size != width * height * 3:
                raise ValueError('RGB8 input length differs from declared geometry: ' + str(path))
            actual = sha(path)
            if actual != pair[role + '_sha256']:
                raise ValueError('input hash mismatch: ' + str(path))
            bindings[str(path)] = actual
            paths[role] = str(path)
        for name, template in templates.items():
            for count in threads:
                case = copy.deepcopy(template)
                case.update(paths, width=width, height=height, threads=count)
                case['id'] = f'{name}-grid-{pair["pair"]}-{width}x{height}-t{count}'
                if case['id'] in identifiers:
                    raise ValueError('duplicate input/thread case: ' + case['id'])
                identifiers.add(case['id'])
                case['input_content'] = pair['content']
                plan['cases'].append(case)
    with output.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    return len(plan['cases'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('pairs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--metrics', required=True, help='Exact comma-separated names from the source plan')
    p.add_argument('--threads', required=True, help='Explicit comma-separated thread counts')
    a = p.parse_args()
    cases = prepare(a.source, a.pairs, a.output, a.metrics.split(','),
                    list(map(int, a.threads.split(','))))
    print(json.dumps(dict(plan=str(a.output), cases=cases)))


if __name__ == '__main__':
    main()
