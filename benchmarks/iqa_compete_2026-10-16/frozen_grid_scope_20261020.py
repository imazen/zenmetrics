#!/usr/bin/env python3
"""Expand a declared grid and preserve a separately reviewed timing scope.

Existing strict grid preparation stays unchanged for frozen plans. Numerical
contracts must be identical in the original and scope-reviewed source plans.
"""
import argparse
import json
from pathlib import Path

from compare_canonical import sha
from prepare_canonical_grid import prepare


def prepare_scoped_grid(source, pairs, scoped, output, metrics, threads):
    original = json.loads(source.read_text())
    reviewed = json.loads(scoped.read_text())
    if (original['binary'], original['rust_source']) != (reviewed['binary'], reviewed['rust_source']):
        raise ValueError('scope review must describe the same native worker/source')
    left = {row['id']: row for row in original['cases']}
    right = {row['id']: row for row in reviewed['cases']}
    if left.keys() != right.keys():
        raise ValueError('scope review must preserve the complete original case inventory')
    drop = {'scope', 'matched_scope', 'scope_disposition'}
    for key in left:
        if ({k: v for k, v in left[key].items() if k not in drop}
                != {k: v for k, v in right[key].items() if k not in drop}):
            raise ValueError('scope review altered a numerical/input/callable contract: '+key)
    caveats = reviewed['scope_dispositions']
    for path, expected in reviewed['bindings'].items():
        if sha(path) != expected:
            raise ValueError('scope-reviewed artifact changed: '+path)
    count = prepare(source, pairs, output, metrics, threads)
    plan = json.loads(output.read_text())
    for row in plan['cases']:
        reason = caveats.get(row['metric'])
        if reason:
            row['matched_scope'] = False
            row['scope_disposition'] = reason
            row['scope'] += '; timing scope caveat: '+reason
    plan['scope_dispositions'] = {name: caveats[name] for name in metrics if name in caveats}
    plan['scope_qualification'] = reviewed['scope_qualification']
    plan['bindings'][str(scoped.resolve())] = sha(scoped)
    plan['bindings'][str(Path(__file__).resolve())] = sha(Path(__file__))
    output.write_text(json.dumps(plan, indent=2)+'\n')
    return dict(cases=count, scope_caveated_cases=sum(not row['matched_scope'] for row in plan['cases']),
                plan=str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pairs', 'scoped', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--metrics', required=True)
    p.add_argument('--threads', required=True)
    a = p.parse_args()
    print(json.dumps(prepare_scoped_grid(a.source, a.pairs, a.scoped, a.output,
                                        a.metrics.split(','), list(map(int, a.threads.split(','))))))


if __name__ == '__main__':
    main()
