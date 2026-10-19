#!/usr/bin/env python3
"""Compare frozen SSIM candidate scores and refusals, without timing claims.

Groups serialize under both host locks. Five unsuffixed names must retain
identical score bits; changed fixed-factor results are observations, not a
new canonical acceptance gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from supervision import scoring_command

BASE_NAMES = ['ssim-pyiqa', 'ssim-pyiqa-f32', 'ssimc-pyiqa-f32',
              'ms-ssim-pyiqa', 'ms-ssim-pyiqa-f32']
FIXED_NAMES = [f'ms-ssim-pyiqa-f32-k{k}' for k in range(2, 7)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def group(plan_path):
    plan = json.loads(plan_path.read_text())
    assert sha(Path(__file__)) == plan['worker_sha256']
    root = plan_path.parent
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    for name in ['OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[name] = str(plan['threads'])
    rows = []
    for metric in plan.get('metrics', BASE_NAMES + FIXED_NAMES):
        row = {'metric': metric, 'base_name': metric in BASE_NAMES}
        for arm, binary in plan['binaries'].items():
            binary = Path(binary)
            assert sha(binary) == plan['binary_sha256'][arm]
            directory = root / (metric + '-' + arm)
            directory.mkdir()
            config = dict(width=plan['input']['width'], height=plan['input']['height'],
                          reference=plan['reference'], distorted=plan['distorted'],
                          metric_name=metric, threads=plan['threads'], probe_only=True,
                          only_mode='score', timing_phase='warm')
            write(directory / 'config.json', config)
            command = scoring_command([str(binary), str(directory / 'config.json')],
                                      system='Linux', seconds=120, memory_log=directory / 'process-memory.txt')
            with (directory / 'stdout.log').open('x') as stdout, (directory / 'stderr.log').open('x') as stderr:
                result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
            probes = [json.loads(line.removeprefix('IQA_PROBE ')) for line in
                      (directory / 'stdout.log').read_text().splitlines() if line.startswith('IQA_PROBE ')]
            arm_result = {'returncode': result.returncode, 'command': command}
            if result.returncode == 0:
                assert len(probes) == 1 and probes[0]['metric_name'] == metric
                arm_result['scores'] = probes[0]['scores']
                arm_result['values'] = [struct.unpack('>d', bytes.fromhex(x[1]))[0] for x in probes[0]['scores']]
            else:
                assert not probes
                arm_result['stderr'] = (directory / 'stderr.log').read_text()
            row[arm] = arm_result
        b, c = row['baseline'], row['candidate']
        if b['returncode'] == c['returncode'] == 0:
            row['bit_identical'] = b['scores'] == c['scores']
            row['absolute_deltas'] = [abs(x - y) for x, y in zip(b['values'], c['values'], strict=True)]
        else:
            row['both_refused'] = (b['returncode'] == c['returncode'] == 1
                                   and '11x11' in b.get('stderr', '') and '11x11' in c.get('stderr', ''))
        rows.append(row)
    for role in ['reference', 'distorted']:
        assert sha(Path(plan[role])) == plan['input'][role + '_sha256']
    write(root / 'results.json', {'rows': rows, 'qualified_timing_cells': 0})
    if any(r['base_name'] and not (r.get('bit_identical') or r.get('both_refused')) for r in rows):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', type=Path)
    p.add_argument('--single-scale-fixed', action='store_true',
                   help='Collect only the ten additional fixed single-scale luma/RGB names')
    for name in ['baseline', 'candidate', 'inputs', 'output']:
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    metrics = ([f'{family}-pyiqa-f32-k{k}' for family in ['ssim', 'ssimc'] for k in range(2, 7)]
               if a.single_scale_fixed else BASE_NAMES + FIXED_NAMES)
    if a.group:
        return group(a.group.resolve(strict=True))
    if any(getattr(a, name) is None for name in ['baseline', 'candidate', 'inputs', 'output']):
        p.error('both frozen builds, inputs and fresh output required')
    builds = {name: getattr(a, name).resolve(strict=True) for name in ['baseline', 'candidate']}
    binaries = {name: root / 'target/release/iqa-compete' for name, root in builds.items()}
    receipts = {name: json.loads((root / 'build-receipt.json').read_text()) for name, root in builds.items()}
    hashes = {name: sha(binary) for name, binary in binaries.items()}
    for name, binary in binaries.items():
        assert hashes[name] == receipts[name]['binary_sha256']
        inventory = json.loads(subprocess.check_output([str(binary), 'inventory'], text=True))
        assert all(any(r['metric_name'] == metric and r['compiled'] for r in inventory) for metric in metrics)
    a.inputs = a.inputs.resolve(strict=True)
    inputs = json.loads(a.inputs.read_text())['pairs']
    a.output.mkdir()
    root = a.output.resolve()
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    identity = {'builds': receipts, 'binary_sha256': hashes, 'inputs_sha256': sha(a.inputs),
                'worker_sha256': sha(Path(__file__)), 'threads': [1, cap], 'metrics': metrics, 'qualified_timing_cells': 0}
    write(root / 'identity.json', identity)
    results = []
    for row in inputs:
        paths = {role: (a.inputs.parent / row[role]).resolve(strict=True) for role in ['reference', 'distorted']}
        for role, path in paths.items():
            assert sha(path) == row[role + '_sha256']
        for threads in [1, cap]:
            assert sha(Path(__file__)) == identity['worker_sha256']
            directory = root / f'{row["pair"]}-{row["size_name"]}-t{threads}'
            directory.mkdir()
            plan = dict(identity, input=row, threads=threads, **{role: str(path) for role, path in paths.items()},
                        binaries={name: str(path) for name, path in binaries.items()})
            write(directory / 'plan.json', plan)
            command = ['flock', str(Path.home() / 'tmp/lilith-heavy.lock'),
                       'flock', str(Path.home() / 'tmp/lilith-wsl-heavy.lock'),
                       str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', str(threads),
                       '--', sys.executable, '-B', str(Path(__file__).resolve()), '--group', str(directory / 'plan.json')]
            env = os.environ.copy()
            env['TMPDIR'] = str(directory)
            with (directory / 'wrapper.log').open('x') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            results.append({'group': directory.name, 'returncode': result.returncode})
            print('GROUP_DONE', directory.name, result.returncode, flush=True)
    write(root / 'completion.json', {'groups': results, 'qualified_timing_cells': 0})
    if any(r['returncode'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
