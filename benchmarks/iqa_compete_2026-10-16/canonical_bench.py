#!/usr/bin/env python3
"""Freeze and execute the canonical scalar CPU protocol with explicit build provenance."""
import argparse
import csv
import fcntl
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SIZES = 'cid22,aic-small,aic-medium,aic-large,2k,4k'
SCOPE = HERE.parent / 'rebenchmark_since_f7668c7'
CURRENT_ORDER = SCOPE / 'order_current_local40_2026-10-20.tsv'


def changed_metrics():
    with CURRENT_ORDER.open() as stream:
        rows = list(csv.DictReader(
            (line for line in stream if not line.startswith('#')), delimiter='\t'))
    names = [row['metric'] for row in rows]
    times = [float(row['historical_ms']) for row in rows]
    expected = ((SCOPE / 'existing.txt').read_text().splitlines()
                + (SCOPE / 'new.txt').read_text().splitlines())
    if (not names or len(names) != len(set(names)) or set(names) != set(expected)
            or any(not math.isfinite(t) or t <= 0 for t in times)
            or times != sorted(times)):
        raise ValueError('current changed scope must match its complete ascending measured order')
    return ','.join(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sizes', help='comma-separated subset of ' + SIZES)
    p.add_argument('metrics', help='all, changed, or comma-separated compiled non-NN routes')
    for name in ('binary', 'pairs', 'build-receipt', 'bundle', 'output', 'lock', 'run-heavy'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--zenbench', type=Path)
    p.add_argument('--cpu-set')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    selected_sizes = a.sizes.split(',')
    if (len(selected_sizes) != len(set(selected_sizes))
            or not set(selected_sizes) <= set(SIZES.split(','))):
        p.error('distinct sizes from ' + SIZES + ' required')
    if a.bundle.exists() or a.output.exists():
        p.error('new bundle and output paths required; resume an existing campaign with iqa-durable-run')
    if bool(a.zenbench) != bool(a.cpu_set):
        p.error('placement requires both --zenbench and --cpu-set')
    metrics = changed_metrics() if a.metrics == 'changed' else a.metrics
    command = [sys.executable, str(HERE / 'prepare_durable.py'),
        '--binary', str(a.binary.resolve()), '--pairs', str(a.pairs.resolve()),
        '--build-receipt', str(a.build_receipt.resolve()), '--output', str(a.bundle.resolve()),
        '--source-commit', a.source_commit, '--resource-wrapper', str(a.run_heavy.resolve()),
        '--metrics', metrics, '--sizes', a.sizes, '--modes', 'score', '--threads', '1,8', '--cold-repeats', '3']
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
