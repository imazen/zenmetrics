#!/usr/bin/env python3
"""Bind installed primary references and RGB inputs for comparative CPU reruns.

This first registry covers the three revised NR ports, original SSIMULACRA1,
primary SSIMULACRA2 v2.1 and NVIDIA FLIP 1.7. Missing adapters remain explicit.
"""
import argparse
import json
from pathlib import Path
import subprocess

from compare_canonical import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'pairs', 'output', 'python', 'extras', 'brisque-weights',
                 'ssim1', 'ssim2', 'source'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--metrics', default='brisque,niqe,piqe,ssimulacra1,ssim2,flip-v17')
    p.add_argument('--sizes', default='64,odd,1mp,4k')
    p.add_argument('--threads', default='1,28')
    a = p.parse_args()
    binary = str(Path(a.binary).resolve(strict=True))
    pairs_path = Path(a.pairs).resolve(strict=True)
    pairs = json.loads(pairs_path.read_text())
    if isinstance(pairs, dict):
        pairs = pairs['pairs']
    helper = Path(__file__).with_name('canonical_cpu_cell.py').resolve()
    canonical = {
        name: dict(label=label, command=[a.python, str(helper), '{metric}', '{width}', '{height}',
            '{reference}', '{distorted}', '{threads}', '{samples}', '--weights', a.brisque_weights,
            '--extra-path', a.extras])
        for name, label in [('brisque', 'PIQ 0.8.0'), ('niqe', 'BasicSR 1.4.2'), ('piqe', 'pypiqe 1.2'),
                            ('flip-v17', 'NVIDIA FLIP 1.7')]
    }
    canonical['ssimulacra1'] = dict(label='original SSIMULACRA1 / OpenCV',
        command=[a.ssim1, '{width}', '{height}', '{reference}', '{distorted}', '{threads}', '{samples}'])
    canonical['ssim2'] = dict(label='primary SSIMULACRA2 v2.1',
        command=[a.ssim2, 'ssim2', '{width}', '{height}', '{reference}', '{distorted}', '{threads}', 'score', '{samples}'])
    bounds = dict(brisque=15.138428e-3, niqe=.602329e-3, piqe=12.486448e-3,
                  ssimulacra1=None, ssim2=None, **{'flip-v17': None})
    requested = a.metrics.split(',')
    unknown = set(requested)-canonical.keys()
    if unknown:
        p.error('no registered primary adapter: '+','.join(sorted(unknown)))
    inventory = json.loads(subprocess.check_output([binary, 'inventory']))
    compiled = {r['metric_name'] for r in inventory if r['compiled']}
    if not set(requested) <= compiled:
        p.error('uncompiled metrics: '+','.join(sorted(set(requested)-compiled)))
    bindings = {str(path): sha(path) for path in [Path(binary), pairs_path, helper, Path(__file__).resolve(),
                Path(a.python).absolute(), Path(a.brisque_weights), Path(a.ssim1), Path(a.ssim2)]}
    for path in Path(a.extras).rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.npz'):
            bindings[str(path)] = sha(path)
    cases = []
    for pair in pairs:
        if pair['size_name'] not in a.sizes.split(','):
            continue
        paths = {role: str((pairs_path.parent/pair[role]).resolve(strict=True)) for role in ('reference', 'distorted')}
        for role, path in paths.items():
            if len(Path(path).read_bytes()) != pair['width']*pair['height']*3:
                raise ValueError('invalid RGB input length')
            if pair.get(role+'_sha256') and sha(path) != pair[role+'_sha256']:
                raise ValueError('input hash mismatch')
            bindings[path] = sha(path)
        for name in requested:
            for threads in map(int, a.threads.split(',')):
                cases.append(dict(id=f'{name}-{pair.get("pair", "pair")}-{pair["size_name"]}-t{threads}',
                    metric=name, width=pair['width'], height=pair['height'], threads=threads, **paths,
                    canonical=canonical[name], absolute_bound=bounds[name],
                    matched_scope=name != 'ssimulacra1', scope=(
                        'Rust RGB8 full call vs canonical RGB8 full call; immutable parameters resident'
                        if name != 'ssimulacra1' else 'original adapter stages RGB-to-BGR before timing; ratio withheld')))
    if not cases:
        p.error('no matching input sizes')
    plan = dict(schema=1, binary=binary, rust_source=a.source, bindings=bindings, cases=cases,
                unavailable={r['metric_name']: 'not covered by this explicit primary-reference registry'
                             for r in inventory if r['metric_name'] not in canonical},
                ratio_definition='Rust/canonical; median of alternating block ratios; descriptive only')
    with Path(a.output).open('x') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    print(json.dumps(dict(plan=a.output, cases=len(cases), metrics=requested)))


if __name__ == '__main__':
    main()
