#!/usr/bin/env python3
"""Freeze and execute the canonical scalar CPU protocol with explicit build provenance."""
import argparse
import csv
import fcntl
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SIZES = 'cid22,aic-small,aic-medium,aic-large,2k,4k'


def changed_metrics():
    path = HERE.parent / 'rebenchmark_since_f7668c7/order.tsv'
    with path.open() as stream:
        return ','.join(row['metric'] for row in csv.DictReader(
            (line for line in stream if not line.startswith('#')), delimiter='\t'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sizes', choices=(SIZES,))
    p.add_argument('metrics', help='all, changed, or comma-separated compiled non-NN routes')
    for name in ('binary', 'pairs', 'build-receipt', 'bundle', 'output', 'lock', 'run-heavy'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--zenbench', type=Path)
    p.add_argument('--cpu-set')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if a.bundle.exists() or a.output.exists():
        p.error('new bundle and output paths required; resume an existing campaign with iqa-durable-run')
    if bool(a.zenbench) != bool(a.cpu_set):
        p.error('placement requires both --zenbench and --cpu-set')
    metrics = changed_metrics() if a.metrics == 'changed' else a.metrics
    command = [sys.executable, str(HERE / 'prepare_durable.py'),
        '--binary', str(a.binary.resolve()), '--pairs', str(a.pairs.resolve()),
        '--build-receipt', str(a.build_receipt.resolve()), '--output', str(a.bundle.resolve()),
        '--source-commit', a.source_commit, '--resource-wrapper', str(a.run_heavy.resolve()),
        '--metrics', metrics, '--sizes', SIZES, '--modes', 'score', '--threads', '1,8', '--cold-repeats', '3']
    if a.zenbench:
        command += ['--zenbench', str(a.zenbench.resolve()), '--cpu-set', a.cpu_set]
    a.lock.parent.mkdir(parents=True, exist_ok=True)
    # Asset verification and copying also share admission; scoring takes the same lock itself.
    with a.lock.open('a+') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        subprocess.run([str(a.run_heavy.resolve()), '--mem', '12G', '--jobs', '8', '--', *command], check=True)
    if not a.prepare_only:
        subprocess.run([sys.executable, str(a.bundle.resolve() / 'durable.py'), 'run',
            str(a.bundle.resolve() / 'manifest.json'), str(a.output.resolve()),
            '--run-heavy', str(a.bundle.resolve() / 'tools/run-heavy'), '--lock', str(a.lock.resolve()),
            '--memory', '12G', '--jobs', '8', '--max-attempts', '2', '--timeout-seconds', '1100'], check=True)


if __name__ == '__main__':
    main()
