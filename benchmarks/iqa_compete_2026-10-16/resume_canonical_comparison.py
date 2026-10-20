#!/usr/bin/env python3
"""Finish an interrupted comparison without replacing any acquired block.

Stop the original coordinator before using this command. The original plan,
bindings, alternating order, numerical bounds and scoring code remain intact.
Incomplete attempt directories remain beside explicitly named recovery attempts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from compare_canonical import sha, summarize


def unused_attempt(directory, number):
    base = directory/f'block-{number:02d}'
    out = base
    index = 0
    while out.exists() or Path(str(out)+'-wrapper.log').exists():
        index += 1
        out = directory/f'{base.name}-recovery-{index}'
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--metrics')
    a = p.parse_args()
    if a.blocks < 3 or a.samples < 3:
        p.error('at least three blocks and samples required')
    root = a.output.resolve(strict=True)
    plan_path = root/'plan.json'
    plan = json.loads(plan_path.read_text())
    names = set(a.metrics.split(',')) if a.metrics else {c['metric'] for c in plan['cases']}
    rows = []
    for case in plan['cases']:
        if case['metric'] not in names:
            continue
        directory = root/case['id']
        directory.mkdir(exist_ok=True)
        blocks = []
        for number in range(a.blocks):
            candidates = sorted(directory.glob(f'block-{number:02d}*/result.json'))
            if len(candidates) > 1:
                raise ValueError('multiple completed attempts for block: '+str(candidates))
            if candidates:
                blocks.append(json.loads(candidates[0].read_text()))
                continue
            out = unused_attempt(directory, number)
            command = [sys.executable, str(Path(__file__).with_name('compare_canonical.py')),
                       str(plan_path), str(out), '--block-case', case['id'],
                       '--block-number', str(number), '--samples', str(a.samples)]
            wrapped = ['flock', str(Path.home()/'tmp/lilith-heavy.lock'), 'flock',
                       str(Path.home()/'tmp/lilith-wsl-heavy.lock'),
                       str(Path.home()/'work/claudehints/scripts/run-heavy'),
                       '--mem', '12G', '--jobs', str(case['threads']), '--', *command]
            with Path(str(out)+'-wrapper.log').open('x') as log:
                result = subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode or not (out/'result.json').exists():
                raise RuntimeError('recovery block failed: '+str(out))
            blocks.append(json.loads((out/'result.json').read_text()))
        row = dict(metric=case['metric'], case=case['id'], width=case['width'],
                   height=case['height'], threads=case['threads'], canonical=case['canonical']['label'],
                   passed=False if any(b.get('errors') or b['passed'] is False for b in blocks)
                   else None if any(b['passed'] is None for b in blocks) else True,
                   absolute_bound=case['absolute_bound'],
                   max_abs_delta=max((b['delta'] for b in blocks if 'delta' in b), default=None),
                   ratio=summarize(blocks, case['matched_scope']), matched_scope=case['matched_scope'],
                   scope=case['scope'], qualification='descriptive; no confidence interval or regression acceptance',
                   rust_source=plan['rust_source'], plan_sha256=sha(plan_path))
        target = directory/'summary.json'
        if target.exists():
            if json.loads(target.read_text()) != row:
                raise ValueError('existing summary disagrees with original blocks: '+str(target))
        else:
            with target.open('x') as stream:
                json.dump(row, stream, indent=2)
                stream.write('\n')
        rows.append(row)
        print('CASE_DONE', json.dumps(row), flush=True)
    print(json.dumps(dict(cases=len(rows), failed=sum(r['passed'] is False for r in rows))))


if __name__ == '__main__':
    main()
