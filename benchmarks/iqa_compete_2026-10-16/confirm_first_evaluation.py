#!/usr/bin/env python3
"""Confirm a resident-model first-call comparison with a fixed number of blocks."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from canonical_intervals import interval
from compare_canonical import compare, sha


def analyze(blocks, case):
    """Use every acquired block; refuse malformed or failed observations."""
    ratios = []
    for block in blocks:
        if block.get('errors') or block.get('case') != case['id']:
            raise ValueError('failed or mismatched first-call block')
        if set(block.get('arms', {})) != {'native', 'canonical'}:
            raise ValueError('both first-call arms required')
        for arm in block['arms'].values():
            if (not math.isfinite(arm['ns']) or arm['ns'] <= 0
                    or not math.isfinite(arm['score'])):
                raise ValueError('invalid first-call time or score')
        native, canonical = (block['arms'][role] for role in ['native', 'canonical'])
        _, passed = compare(native['score'], canonical['score'],
                            case['absolute_bound'], case.get('relative_bound', 0))
        if passed is False:
            raise ValueError('first-call numerical gate failed')
        ratios.append(native['ns'] / canonical['ns'])
    return interval(ratios)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('case')
    p.add_argument('output', type=Path)
    p.add_argument('--blocks', type=int, default=20)
    a = p.parse_args()
    if a.blocks < 20:
        p.error('at least twenty fixed blocks required')
    plan_bytes = a.plan.read_bytes()
    helper = Path(__file__).with_name('compare_first_evaluation.py').resolve()
    helper_hash = sha(helper)
    plan = json.loads(plan_bytes)
    case = next(c for c in plan['cases'] if c['id'] == a.case)
    a.output.mkdir()
    frozen = a.output.resolve() / 'plan.json'
    frozen.write_bytes(plan_bytes)
    blocks = []
    for number in range(a.blocks):
        command = [
            'flock', str(Path.home() / 'tmp/lilith-heavy.lock'),
            'flock', str(Path.home() / 'tmp/lilith-wsl-heavy.lock'),
            str(Path.home() / 'work/claudehints/scripts/run-heavy'),
            '--mem', '12G', '--jobs', str(case['threads']), '--',
            sys.executable, '-B',
            str(helper),
            str(frozen), str(a.output.resolve() / f'block-{number:02d}'),
            '--case', case['id'], '--block', str(number),
        ]
        with (a.output / f'block-{number:02d}.log').open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, TMPDIR=str(a.output.resolve())), check=True)
        if frozen.read_bytes() != plan_bytes:
            raise ValueError('frozen first-call plan changed')
        if sha(helper) != helper_hash:
            raise ValueError('first-call helper changed')
        block = json.loads((a.output / f'block-{number:02d}/result.json').read_text())
        if block.get('phase') != plan['phase']:
            raise ValueError('first-call phase changed')
        blocks.append(block)
    result = dict(case=case['id'], metric=case['metric'], blocks=len(blocks),
                  phase=plan['phase'], plan_sha256=sha(frozen),
                  controller_sha256=sha(__file__), helper_sha256=helper_hash,
                  block_sha256=[sha(a.output / f'block-{n:02d}/result.json')
                                for n in range(a.blocks)],
                  **analyze(blocks, case), regression_qualified=False,
                  qualification='Descriptive first-call interval after resident model construction; '
                                'no warm, process-startup or OS-cache-cold claim. All fixed blocks '
                                'retained; canonical runtime contamination not independently audited.')
    with (a.output / 'interval.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
