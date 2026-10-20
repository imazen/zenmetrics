#!/usr/bin/env python3
"""Execute a supplied local reference manifest, one resource-capped step at a time."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def admitted_command(manifest, command):
    locks = manifest.get('heavy_locks')
    if (not isinstance(locks, list) or not locks
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in locks)
            or len({str(Path(path).resolve()) for path in locks}) != len(locks)):
        raise ValueError('manifest requires unique absolute heavy_locks in acquisition order')
    prefix = [item for path in locks for item in ('flock', path)]
    return prefix+[str(Path.home()/'work/claudehints/scripts/run-heavy'),
                   '--mem', '12G', '--jobs', '12', '--', *command]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--metrics', nargs='+')
    p.add_argument('--limit', type=int)
    p.add_argument('--addendum', action='store_true')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    admission = admitted_command(manifest, [])
    home = Path.home()
    root = Path(manifest['scratch'])
    output = Path(manifest['output'])
    output.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    (root/'scratch').mkdir(exist_ok=True)
    binary = Path(manifest['binary'])
    if sha(binary) != manifest['binary_sha256']:
        raise ValueError('supplied binary hash mismatch')
    for path, expected in manifest['bindings'].items():
        if sha(path) != expected:
            raise ValueError('reference/driver binding changed: '+path)
    steps = []
    specifications = manifest['addendum'] if a.addendum else manifest['metrics']
    metrics = a.metrics or list(specifications)
    for metric in metrics:
        spec = specifications[metric]
        if not a.addendum:
            steps.append((metric, 'native', spec))
        if spec.get('kind'):
            steps.append((metric, 'canonical', spec))
    for metric, role, spec in steps:
        if manifest.get('execution') and not a.limit:
            command = [sys.executable, str(Path(__file__).with_name('canonical_chunks.py')),
                       str(a.manifest), metric, role]
            if a.addendum:
                command.append('--addendum')
            completed = subprocess.run(command)
            print(f'chunked {metric}.{role}: exit {completed.returncode}', flush=True)
            # A recorded scorer refusal is a completed attempt. A controller
            # failure without its receipt must be repaired, not silently skipped.
            receipt = root/'logs'/f'{metric}.{role}.receipt.json'
            if not receipt.exists():
                raise RuntimeError('chunk controller failed before retaining a receipt: '+str(receipt))
            continue
        tag = f'{metric}.{role}'+(f'.first{a.limit}' if a.limit else '')
        result = output/(tag+('.tsv' if role == 'native' or spec['kind'] == 'octave' else '.jsonl'))
        done = root/'logs'/(tag+'.receipt.json')
        if done.exists():
            prior = json.loads(done.read_text())
            if prior.get('returncode') == 0 and sha(result) == prior.get('result_sha256'):
                print('retained '+tag, flush=True)
                continue
            print('retained failed attempt '+tag, flush=True)
            continue
        pairs = Path(manifest['pairs'])
        if a.limit:
            import csv
            rows = list(csv.DictReader(pairs.open(), delimiter='\t'))
            pairs = root/f'pairs.first{a.limit}.tsv'
            with pairs.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(rows[:a.limit])
        if role == 'native':
            command = [manifest['loader'], '--library-path', manifest['runtime'], str(binary),
                       'batch', '--metric', metric, '--pairs', str(pairs), '--output', str(result),
                       '--jobs', '12', '--use-legacy-scheduler', '--bench-on-start', 'no',
                       '--gpu-runtime', 'cpu']
        else:
            driver = 'canonical_octave.py' if spec['kind'] == 'octave' else 'canonical_python.py'
            command = [spec.get('python', sys.executable), str(Path(__file__).with_name(driver)),
                       manifest['config'], metric, str(pairs), str(result)]
        wrapped = admission+command
        env = dict(os.environ, TMPDIR=str(root/'scratch'), RAYON_NUM_THREADS='1', RTEN_NUM_THREADS='12',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   ZENMETRICS_NN_MODEL_DIR=spec.get('models', manifest.get('models', '')), ZENMETRICS_NN_ENGINE='rten',
                   ZENMETRICS_NN_ACCEPT_WEIGHTS=manifest.get('accept_weights', ''),
                   TORCH_HOME=manifest.get('torch_home', str(home/'.cache/torch')),
                   SETUPTOOLS_USE_DISTUTILS='stdlib')
        logpath = root/'logs'/(tag+'.log')
        print('queued '+tag, flush=True)
        start = datetime.now(timezone.utc).isoformat()
        with logpath.open('x') as log:
            completed = subprocess.run(wrapped, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt = dict(metric=metric, role=role, command=command, wrapped=wrapped, started=start,
                       finished=datetime.now(timezone.utc).isoformat(), returncode=completed.returncode,
                       result=str(result), log=str(logpath),
                       result_sha256=sha(result) if result.exists() else None,
                       resource_receipts=[line for line in logpath.read_text(errors='replace').splitlines()
                                          if line.startswith('run-heavy:')])
        done.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
        with (root/'report.md').open('a') as report:
            report.write(f"\n{receipt['finished']}: {tag}, exit {completed.returncode}. "
                         f"Result: {result}; log: {logpath}.\n")
            for line in receipt['resource_receipts']:
                report.write('`'+line+'`\n')


if __name__ == '__main__':
    main()
