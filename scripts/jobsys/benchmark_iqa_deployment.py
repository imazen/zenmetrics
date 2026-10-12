#!/usr/bin/env python3
"""Capture all qualified native IQA controls through the existing zenfleet flow.

These small cases test deployment and retain descriptive timings/memory. Their
layout/instrumentation controls do not certify metric accuracy or speed changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['binaries', 'runner', 'sdr_controls', 'hdr_controls', 'output', 'machine', 'domain']:
        parser.add_argument(name)
    parser.add_argument('--gpu-controls', type=Path)
    parser.add_argument('--gpu-sdk', type=Path)
    parser.add_argument('--gpu-only', action='store_true')
    args = parser.parse_args()
    if bool(args.gpu_controls) != bool(args.gpu_sdk) or (args.gpu_only and not args.gpu_controls):
        parser.error('GPU controls and SDK must be supplied together; --gpu-only requires both')
    binaries, runner, sdr, hdr, root = [Path(getattr(args, name)).resolve() for name in
                                      ['binaries', 'runner', 'sdr_controls', 'hdr_controls', 'output']]
    sources = [] if args.gpu_only else [(sdr, json.loads((sdr/'controls.json').read_bytes()), {}),
                                      (hdr, json.loads((hdr/'controls.json').read_bytes()), {})]
    if args.gpu_controls:
        gpu = args.gpu_controls.resolve()
        controls = json.loads((gpu/'controls.json').read_bytes())
        if controls['backend'] != 'cuda' or controls['failures'] or len(controls['controls']) != 24:
            raise ValueError('GPU deployment requires all 24 CUDA layout/instrumentation controls')
        sdk = args.gpu_sdk.resolve()
        manifest = json.loads((sdk/'manifest.json').read_bytes())
        if manifest['schema'] != 1:
            raise ValueError('unsupported scoped SDK manifest')
        runtime_files = {'sdk/manifest.json': {'source': str(sdk/'manifest.json'), 'executable': False}}
        for name, artifact in manifest['files'].items():
            path = sdk / name
            if path.resolve().is_relative_to(sdk) is False or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError('SDK file escaped or differs from its manifest')
            runtime_files['sdk/' + name] = {'source': str(path), 'executable': False}
        for name, target in manifest['aliases'].items():
            path = sdk / name
            if not path.is_symlink() or path.readlink().as_posix() != target or not path.resolve().is_relative_to(sdk):
                raise ValueError('SDK loader alias differs from its manifest')
            runtime_files['sdk/' + name] = {'source': str(path.resolve()), 'executable': False}
        sources.append((gpu, controls['controls'], runtime_files))
    root.mkdir(parents=True, exist_ok=False)
    adapter = Path(__file__).with_name('benchmark_exec.py').resolve()
    env = dict(os.environ, ZENBENCH_BIN=str(runner), ZENBENCH_WORK_DIR=str(root),
               ZEN_BLOBS_DIR=str(root/'blobs'), ZENBENCH_MACHINE=args.machine,
               ZENBENCH_DOMAIN=args.domain, ZENBENCH_COORDINATION_DIR=str(root/'coordination'))
    env.pop('ZENBENCH_BLOB_URI', None)
    env.pop('ZENBENCH_RESULT_FILE', None)
    (root/'coordination').mkdir()

    def call(label, command, timeout=120):
        with (root/f'{label}.stdout').open('wb') as out, (root/f'{label}.stderr').open('wb') as err:
            result = subprocess.run([str(a) for a in command], env=env, stdout=out, stderr=err,
                                    timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{label} failed ({result.returncode}); see {root/label}.stderr')
        return (root/f'{label}.stdout').read_bytes()

    caps = json.loads(call('capabilities', [runner, 'capabilities']))
    process = caps['runner']
    if process['process_arch'] != 'x86_64' or process['os'] != 'linux':
        raise ValueError('this deployment control requires a separately verified native Linux x86-64 runner')
    (root/'qualification.json').write_text(json.dumps(dict(
        kind='deployment_control', accuracy='not certified by these controls',
        inference='descriptive; two calls per fresh process, no speed acceptance',
        runner=caps, machine=args.machine, domain=args.domain), indent=2)+'\n')
    observations = []
    labels = {}
    for controls, records, runtime_files in sources:
        for record in records:
            label = record.get('case', '')
            if not label.endswith('-packed'):
                continue
            case_path = controls/f'{label}.json'
            case = json.loads(case_path.read_bytes())
            products = record['products']
            if len(products) != 2 or products[0] != products[1]:
                raise ValueError(f'{label}: missing repeatability qualification')
            for variant, backend in [('latency', 'os_peak'), ('memory', 'driver')]:
                cell = root/f'{label}-{variant}'
                cell.mkdir()
                files = {**runtime_files, 'driver': {'source': str(binaries/variant), 'executable': True},
                         'case.json': {'source': str(case_path), 'executable': False}}
                for key in ['reference', 'distorted']:
                    files[case[key]] = {'source': str(controls/case[key]), 'executable': False}
                plan_env = {'RAYON_NUM_THREADS': '1'}
                if runtime_files:
                    plan_env.update(CUDA_PATH='sdk', LD_LIBRARY_PATH='sdk/lib64')
                spec = dict(schema=1, target='x86_64-unknown-linux-gnu', files=files, plan=dict(
                    schema=1, program='driver', args=['fleet', 'case.json'],
                    env=plan_env, backend=backend, operations=2,
                    timeout_seconds=60, expected_correctness={'products': products}, workload=case))
                (cell/'spec.json').write_text(json.dumps(spec))
                call(f'{cell.name}-bundle', [runner, 'bundle', cell/'spec.json', cell/'bundle'])
                hashes = json.loads(call(f'{cell.name}-stage', [adapter, 'stage', cell/'bundle']))
                job = dict(kind='benchmark', **hashes, experiment=f'iqa-deployment-{cell.name}',
                           machine=args.machine, domain=args.domain, replicate=0,
                           backend=backend, timeout_seconds=60, admission_seconds=30)
                observations.append(dict(job=job, peak_mem_bytes=1024**3, threads=1))
                labels[job['experiment']] = dict(metric=case['metric'], variant=variant,
                                                 reuse=case.get('reuse', 'cold'))
    expected = (0 if args.gpu_only else 80) + (24 if args.gpu_controls else 0)
    if len(observations) != expected:
        raise ValueError(f'expected {expected} qualified observations; found {len(observations)}')
    (root/'observations.json').write_text(json.dumps(dict(schema=1, observations=observations)))
    call('declare', [binaries/'zenfleet-ctl', 'declare-benchmarks', '--spec', root/'observations.json',
                     '--out', root/'manifest.json'])
    (root/'ledger').mkdir()
    call('worker', [binaries/'zenfleet-worker', '--manifest', root/'manifest.json',
                    '--ledger-out', root/'ledger/result.parquet', '--blobs', root/'blobs',
                    '--exec', adapter, '--worker', args.machine], timeout=1200)
    rows = json.loads(call('ledger', [binaries/'benchmark_ledger_control', root/'ledger']))
    completed = [row for row in rows if row['status'] == 'done']
    if len(completed) != len(observations):
        raise ValueError(f'worker captured {len(completed)} of {len(observations)} observations; raw ledger retained')
    summary = []
    for index, row in enumerate(completed):
        recovered = root/f'recovered-{index}'
        call(f'collect-{index}', [adapter, 'collect', row['output_sha'], recovered])
        packet = json.loads((recovered/'packet.json').read_bytes())
        record = dict(**labels[packet['experiment']], status=packet['status'],
                      memory=packet['memory'], receipt_sha=row['output_sha'],
                      elapsed_seconds=packet['elapsed_seconds'])
        stdout = recovered/'stdout.txt'
        if stdout.is_file() and packet['status'] == 'complete':
            driver = json.loads(stdout.read_bytes())
            record['timing_usable'] = driver['timing_usable']
            record['workload'] = driver['workload']
            record['correctness'] = driver['driver']['correctness']
            if driver['timing_usable']:
                record['operation_ns'] = driver['operation_ns']
        summary.append(record)
    (root/'summary.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in summary))
    valid = sum(row['status'] == 'complete' for row in summary)
    print(json.dumps(dict(captured=len(summary), complete=valid, native_routes=0 if args.gpu_only else 40,
                          gpu_variants=6 if args.gpu_controls else 0,
                          machine=args.machine, domain=args.domain,
                          qualification='deployment and descriptive observations; no accuracy/speed acceptance')))
    if valid != len(observations):
        raise ValueError('some observations refused measurement; inspect retained summary and packets')


if __name__ == '__main__':
    main()
