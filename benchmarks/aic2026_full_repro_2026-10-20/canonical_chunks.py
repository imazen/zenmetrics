#!/usr/bin/env python3
"""Resume exact crop identities in bounded, jointly admitted worker bundles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from canonical_run import sha


def records(path, jsonl=False, partial=False):
    data = Path(path).read_bytes()
    if data and not data.endswith(b'\n'):
        if not partial:
            raise ValueError('unterminated result record: '+str(path))
        data = data[:data.rfind(b'\n')+1]
    text = data.decode('utf-8')
    if jsonl:
        rows = [json.loads(line) for line in text.splitlines()]
        fields = None
    else:
        reader = csv.DictReader(io.StringIO(text), delimiter='\t', strict=True)
        rows, fields = list(reader), reader.fieldnames
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError('malformed result row: '+str(path))
    result = {}
    for row in rows:
        key = row['distorted']
        if key in result:
            raise ValueError('duplicate result identity: '+key)
        result[key] = row
    return result, fields


def merge(target, incoming):
    for key, row in incoming.items():
        if key in target and target[key] != row:
            raise ValueError('overlap control changed: '+key)
        target[key] = row


def scoring_environment(m, spec, root):
    # Persist only these explicit settings. Inherited credentials belong to
    # the live process environment and must never enter an artifact.
    return dict(TMPDIR=str(root/'scratch'), RAYON_NUM_THREADS='1', RTEN_NUM_THREADS='12',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                ZENMETRICS_NN_MODEL_DIR=spec.get('models', m.get('models', '')),
                ZENMETRICS_NN_ENGINE='rten', ZENMETRICS_NN_ACCEPT_WEIGHTS=m.get('accept_weights', ''),
                TORCH_HOME=m.get('torch_home', str(Path.home()/'.cache/torch')),
                SETUPTOOLS_USE_DISTUTILS='stdlib')


def execute_bundle(path):
    bundle = json.loads(path.read_text())
    def run(worker):
        with Path(worker['log']).open('x') as log:
            return subprocess.run(worker['command'], env=dict(os.environ, **bundle['env']),
                                  stdout=log, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=len(bundle['workers'])) as executor:
        codes = list(executor.map(run, bundle['workers']))
    path.with_suffix('.exit-codes.json').write_text(json.dumps(codes)+'\n')
    return int(any(code != 0 for code in codes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('metric', nargs='?')
    p.add_argument('role', nargs='?', choices=['native', 'canonical'])
    p.add_argument('--execute-bundle', action='store_true')
    p.add_argument('--addendum', action='store_true')
    args = p.parse_args()
    if args.execute_bundle:
        return execute_bundle(args.manifest)
    m = json.loads(args.manifest.read_text())
    spec = m['addendum' if args.addendum else 'metrics'][args.metric]
    root, output = Path(m['scratch']), Path(m['output'])
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    tag = args.metric+'.'+args.role
    done = root/'logs'/(tag+'.receipt.json')
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt['result_sha256'] is not None and sha(receipt['result']) != receipt['result_sha256']:
            raise ValueError('retained result changed: '+tag)
        print('retained attempt '+tag, flush=True)
        return receipt['returncode']
    jsonl = args.role == 'canonical' and spec['kind'] != 'octave'
    suffix = '.jsonl' if jsonl else '.tsv'
    final = output/(tag+suffix)
    if final.exists():
        raise ValueError('unowned final output exists: '+str(final))
    with Path(m['pairs']).open() as source:
        reader = csv.DictReader(source, delimiter='\t')
        pair_fields, pairs = reader.fieldnames, list(reader)
    expected = {row['distorted']: row for row in pairs}
    if len(expected) != len(pairs):
        raise ValueError('duplicate requested identity')
    retained, fields = {}, None
    seed = m.get('seeds', {}).get(tag)
    if seed:
        if sha(seed['path']) != seed['sha256']:
            raise ValueError('seed hash mismatch')
        retained, fields = records(seed['path'], jsonl, partial=True)
    if not retained.keys() <= expected.keys():
        raise ValueError('seed has unknown identities')
    for key, row in retained.items():
        if args.role == 'native' and any(row[k] != expected[key][k] for k in pair_fields):
            raise ValueError('seed metadata mismatch: '+key)
    pending = [row for row in pairs if row['distorted'] not in retained]
    # Re-score two retained identities as exact process-boundary controls.
    controls = [expected[key] for key in list(retained)[:2]]
    limits = m['execution']
    workers = int(limits['workers'].get(args.metric, limits['default_workers']))
    per_worker = int(limits['rows_per_worker'])
    if not 1 <= workers <= 12 or not 1 <= per_worker <= 256:
        raise ValueError('chunk resource limits outside supported range')
    env = scoring_environment(m, spec, root)
    (root/'scratch').mkdir(exist_ok=True)
    work = output/(tag+'.chunks')
    work.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    receipts, failure = [], 0
    for offset in range(0, len(pending), workers*per_worker):
        number = offset//(workers*per_worker)
        plan = work/f'bundle-{number:04d}.json'
        receipt_path = plan.with_suffix('.receipt.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            bundle = json.loads(plan.read_text())
            for worker in bundle['workers']:
                result = Path(worker['result'])
                if receipt['outputs'].get(str(result)) != (sha(result) if result.exists() else None):
                    raise ValueError('retained chunk changed: '+str(result))
        else:
            if plan.exists():
                raise ValueError('interrupted chunk requires a new explicit attempt identity: '+str(plan))
            bundle = dict(env=env, workers=[])
            for index in range(workers):
                rows = pending[offset+index*per_worker:offset+(index+1)*per_worker]
                if not rows:
                    continue
                if number == 0 and index == 0:
                    rows = controls+rows
                pair_path = work/f'bundle-{number:04d}-worker-{index}.pairs.tsv'
                with pair_path.open('x', newline='') as target:
                    writer = csv.DictWriter(target, fieldnames=pair_fields, delimiter='\t', lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(rows)
                result = pair_path.with_name(pair_path.name.replace('.pairs.tsv', suffix))
                if args.role == 'native':
                    command = [m['loader'], '--library-path', m['runtime'], m['binary'], 'batch',
                               '--metric', args.metric, '--pairs', str(pair_path), '--output', str(result),
                               '--jobs', '1', '--use-legacy-scheduler', '--bench-on-start', 'no', '--gpu-runtime', 'cpu']
                else:
                    driver = 'canonical_octave.py' if spec['kind'] == 'octave' else 'canonical_python.py'
                    command = [spec.get('python', sys.executable), str(Path(__file__).with_name(driver)),
                               m['config'], args.metric, str(pair_path), str(result)]
                bundle['workers'].append(dict(command=command, result=str(result), pairs=str(pair_path),
                                              rows=len(rows), log=str(result)+'.log'))
            plan.write_text(json.dumps(bundle, indent=2)+'\n')
            wrapped = ['flock', str(Path.home()/'tmp/lilith-heavy.lock'),
                       'flock', str(Path.home()/'tmp/lilith-wsl-heavy.lock'),
                       str(Path.home()/'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', '12', '--',
                       sys.executable, str(Path(__file__).resolve()), str(plan), '--execute-bundle']
            log = plan.with_suffix('.log')
            print(f'queued {tag} bundle {number} ({len(bundle["workers"])} workers)', flush=True)
            with log.open('x') as target:
                completed = subprocess.run(wrapped, stdout=target, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, **env))
            receipt = dict(command=wrapped, returncode=completed.returncode,
                           resource_receipts=[line for line in log.read_text(errors='replace').splitlines()
                                              if line.startswith('run-heavy:')],
                           outputs={worker['result']: sha(worker['result']) if Path(worker['result']).exists() else None
                                    for worker in bundle['workers']})
            receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        receipts.append(dict(path=str(receipt_path), **receipt))
        for worker in bundle['workers']:
            if Path(worker['result']).exists():
                rows, header = records(worker['result'], jsonl, partial=receipt['returncode'] != 0)
                requested = records(worker['pairs'])[0]
                if not rows.keys() <= requested.keys():
                    raise ValueError('worker returned unrequested identities')
                if receipt['returncode'] == 0 and rows.keys() != requested.keys():
                    raise ValueError('successful worker omitted identities')
                if args.role == 'native' and any(any(row[k] != expected[key][k] for k in pair_fields)
                                               for key, row in rows.items()):
                    raise ValueError('native worker changed metadata')
                if fields is not None and header != fields:
                    raise ValueError('worker columns differ from retained output')
                fields = header
                merge(retained, rows)
        print(f'{tag}: {len(retained)}/{len(expected)} identities retained', flush=True)
        with (root/'report.md').open('a') as report:
            report.write(f'\n{tag} bundle {number}: {len(retained)}/{len(expected)} identities retained; '
                         f'exit {receipt["returncode"]}.\n')
            report.writelines('`'+line+'`\n' for line in receipt['resource_receipts'])
        if receipt['returncode']:
            failure = receipt['returncode']
            break
    if not failure and retained.keys() != expected.keys():
        raise ValueError('incomplete successful result')
    with final.open('x', newline='') as target:
        if jsonl:
            for row in pairs:
                if row['distorted'] in retained:
                    target.write(json.dumps(retained[row['distorted']])+'\n')
        elif fields:
            writer = csv.DictWriter(target, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(retained[row['distorted']] for row in pairs if row['distorted'] in retained)
    done.write_text(json.dumps(dict(metric=args.metric, role=args.role, started=started,
        finished=datetime.now(timezone.utc).isoformat(), returncode=failure, result=str(final),
        result_sha256=sha(final), seed=seed, native_threads=1, rows=len(retained), bundles=receipts,
        resource_receipts=[line for receipt in receipts for line in receipt['resource_receipts']]), indent=2)+'\n')
    return failure


if __name__ == '__main__':
    sys.exit(main())
