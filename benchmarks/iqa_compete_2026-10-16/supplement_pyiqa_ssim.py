#!/usr/bin/env python3
"""Isolate missing conventions after completed whole-case reference attempts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from supervision import scoring_command

NAMES = ['ssim-unrounded-y', 'ssim-unrounded-rgb', 'ms-ssim-unrounded-y',
         'ssim-original-rounded-class', 'ms-ssim-original-rounded-class']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['panel', 'python', 'references', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.panel = a.panel.resolve(strict=True)
    a.python = Path(os.path.abspath(a.python))
    references = json.loads(a.references.read_text())
    completion = json.loads((a.panel / 'completion.json').read_text())
    assert len(completion['cases']) == 96
    tasks = []
    for row in completion['cases']:
        if row['status'] == 'collected':
            continue
        directory = a.panel / row['case']
        events = directory / 'canonical/events.jsonl'
        returned = {json.loads(s)['convention'] for s in events.read_text().splitlines()} if events.exists() else set()
        tasks += [(directory, name) for name in NAMES if name not in returned]
    a.output.mkdir()
    root = a.output.resolve()
    worker = Path(__file__).resolve().with_name('probe_pyiqa_ssim.py')
    identity = dict(parent_completion_sha256=sha(a.panel / 'completion.json'), parent_identity_sha256=sha(a.panel / 'identity.json'),
                    worker_sha256=sha(worker), tasks=[dict(case=path.name, convention=name) for path, name in tasks],
                    deadline_seconds=300, repetitions=2, numerical_gate='unbound; no invented acceptance threshold', qualified_timing_cells=0)
    (root / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    outcomes = []
    for source, convention in tasks:
        assert sha(worker) == identity['worker_sha256']
        factor, *_, budget = source.name.split('-')
        threads = int(budget.removeprefix('t'))
        directory = root / (source.name + '-' + convention)
        directory.mkdir()
        command = [str(a.python), '-B', str(worker), '--manifest', references[factor.removeprefix('k')],
                   '--inputs', str(source / 'inputs.json'), '--output', str(directory / 'canonical'),
                   '--threads', str(threads), '--convention', convention]
        command = scoring_command(command, system='Linux', seconds=300, memory_log=directory / 'process-memory.txt')
        command = ['flock', str(Path.home() / 'tmp/lilith-heavy.lock'), 'flock', str(Path.home() / 'tmp/lilith-wsl-heavy.lock'),
                   str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', str(threads), '--', *command]
        env = dict(os.environ, TMPDIR=str(directory), HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        with (directory / 'wrapper.log').open('x') as log:
            result = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        outcome = dict(parent_case=source.name, convention=convention, command=command, returncode=result.returncode,
                       status='collected' if result.returncode == 0 and (directory / 'canonical/results.json').exists() else 'canonical-process-failure')
        (directory / 'outcome.json').write_text(json.dumps(outcome, indent=2) + '\n')
        outcomes.append(outcome)
        print('CASE_DONE', len(outcomes), '/', len(tasks), directory.name, outcome['status'], flush=True)
    (root / 'completion.json').write_text(json.dumps(dict(cases=outcomes, qualified_timing_cells=0), indent=2) + '\n')
    if any(row['status'] != 'collected' for row in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
