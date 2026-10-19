#!/usr/bin/env python3
"""Serialize primary SSIM fixed-factor observations; preserve every refusal."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

from check_fixed_factor_oracle import check_manifest
from supervision import scoring_command


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['python', 'inputs', 'references', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--factors', default='1,2,3,4')
    p.add_argument('--direct-only', action='store_true')
    a = p.parse_args()
    # Executing the resolved target of a venv's Python symlink loses its
    # pyvenv.cfg search context. Preserve the requested launcher path.
    a.python = Path(os.path.abspath(a.python))
    assert a.python.is_file()
    for name in ['inputs', 'references']:
        setattr(a, name, getattr(a, name).resolve(strict=True))
    pairs = json.loads(a.inputs.read_text())['pairs']
    references = {int(k): Path(v).resolve(strict=True) for k, v in json.loads(a.references.read_text()).items()}
    factors = [int(k) for k in a.factors.split(',')]
    assert len(factors) == len(set(factors)) and all(k in range(1, 7) for k in factors)
    assert set(references) == set(factors)
    for k, path in references.items():
        m = check_manifest(path)
        assert m['source']['name'] == 'ssim-pyiqa' and m['patch']['factor'] == k
    base = Path(__file__).resolve().parent
    helpers = {name: sha(base / name) for name in ['run_pyiqa_ssim_panel.py', 'probe_pyiqa_ssim.py',
               'check_fixed_factor_oracle.py', 'fixed_factor_oracle.py', 'supervision.py']}
    cap = max(1, len(os.sched_getaffinity(0)) - 4)
    venv_config = a.python.parent.parent / 'pyvenv.cfg'
    a.output.mkdir()
    root = a.output.resolve()
    identity = {'inputs_sha256': sha(a.inputs), 'python_sha256': sha(a.python),
                'python_launcher': str(a.python), 'python_resolved_executable': str(a.python.resolve(strict=True)),
                'venv_config_sha256': sha(venv_config) if venv_config.exists() else None,
                'reference_manifests': {k: sha(path) for k, path in references.items()},
                'helpers': helpers, 'threads': [1, cap], 'factors': factors, 'direct_only': a.direct_only,
                'qualified_timing_cells': 0}
    cases = list(itertools.product(pairs, factors, [1, cap]))
    identity['planned_cases'] = len(cases)
    write(root / 'identity.json', identity)
    dispositions = []
    for row, k, threads in cases:
        for name, digest in helpers.items():
            assert sha(base / name) == digest, 'helper changed during panel'
        assert sha(a.python) == identity['python_sha256']
        assert (sha(venv_config) if venv_config.exists() else None) == identity['venv_config_sha256']
        assert sha(references[k]) == identity['reference_manifests'][k]
        row = dict(row)
        for role in ['reference', 'distorted']:
            path = (a.inputs.parent / row[role]).resolve(strict=True)
            assert sha(path) == row[role + '_sha256']
            row[role] = str(path)
        directory = root / f'k{k}-{row["pair"]}-{row["size_name"]}-t{threads}'
        directory.mkdir()
        write(directory / 'inputs.json', {'pairs': [row]})
        command = [str(a.python), '-B', str(base / 'probe_pyiqa_ssim.py'), '--manifest', str(references[k]),
                   '--inputs', str(directory / 'inputs.json'), '--output', str(directory / 'canonical'), '--threads', str(threads)]
        if a.direct_only:
            command.append('--direct-only')
        scorer = scoring_command(command, system='Linux', seconds=300, memory_log=directory / 'process-memory.txt')
        wrapped = ['flock', str(Path.home() / 'tmp/lilith-heavy.lock'), 'flock', str(Path.home() / 'tmp/lilith-wsl-heavy.lock'),
                   str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', str(threads), '--', *scorer]
        env = dict(os.environ, TMPDIR=str(directory), HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        with (directory / 'wrapper.log').open('x') as log:
            result = subprocess.run(wrapped, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        output = directory / 'canonical/results.json'
        status = 'collected' if result.returncode == 0 and output.exists() else 'canonical-process-failure'
        disposition = {'case': directory.name, 'status': status, 'returncode': result.returncode, 'command': wrapped}
        write(directory / 'outcome.json', disposition)
        dispositions.append(disposition)
        print('CASE_DONE', len(dispositions), '/' + str(len(cases)), directory.name, status, flush=True)
    write(root / 'completion.json', {'cases': dispositions, 'qualified_timing_cells': 0})
    if any(r['returncode'] or r['status'] != 'collected' for r in dispositions):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
