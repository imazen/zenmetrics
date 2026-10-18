#!/usr/bin/env python3
"""Run a real NN cold/warm control through the durable process supervisor."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import fcntl
from mac_service import verify
from prepare_durable import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--metric', default='lpips-alex')
    p.add_argument('--size', default='cid22')
    p.add_argument('--lock', type=Path, default=Path.home() / 'tmp/mac-heavy.lock')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    source = a.bundle.resolve()
    with a.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify(source)
    manifest = json.loads((source / 'manifest.json').read_bytes())
    cells = [c for c in manifest['cells'] if c['metric_name'] == a.metric and
             c['size_name'] == a.size and c['threads'] == a.threads and c['cold_repeat'] == 0]
    if len(cells) != 2 or {c['timing_phase'] for c in cells} != {'cold', 'warm'}:
        raise ValueError('control requires exactly one cold and one warm scalar cell')
    used = {c[k] for c in cells for k in ('reference', 'distorted')}
    used.update(name for c in cells for name in c['asset_paths'])
    if manifest['metadata'].get('cpu_placement'):
        used.add(manifest['metadata']['cpu_placement']['zenbench'])
    manifest.update(cells=cells, input_sha256={name: manifest['input_sha256'][name] for name in used})
    manifest['metadata'].update(control_only=True, source_bundle_manifest_sha256=sha(source / 'manifest.json'))
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    bundle = root / 'bundle'
    bundle.mkdir()
    names = used | {'durable.py', 'mac_service.py', 'select_durable.py', 'build.json',
                    'inputs.meta.json', 'bin/iqa-compete', 'tools/run-heavy', 'tools/run-heavy-macos.py'}
    if (source / 'admission_recovery.py').exists():
        names.add('admission_recovery.py')
    for name in sorted(names):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    hashes = {f.relative_to(bundle).as_posix(): sha(f) for f in bundle.rglob('*') if f.is_file()}
    (bundle / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    verify(bundle)
    results = root / 'results'
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(results),
               '--run-heavy', str(bundle / 'tools/run-heavy'), '--lock', str(a.lock),
               '--memory', '12G', '--jobs', '1', '--timeout-seconds', '1100', '--max-attempts', '1']
    (root / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (root / 'controller.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((results / 'report.json').read_bytes())
    if not report['complete'] or any(c['status'] != 'ok' for c in report['cells']):
        raise RuntimeError('NN control did not produce both timings; inspect ' + str(results / 'report.json'))
    cold, warm = report['cold'][0], report['warm'][0]
    if cold['score_bits'] != warm['score_bits']:
        raise RuntimeError('NN cold/warm score bits differ')
    budgets = []
    if cold['worker_environment']['ZENMETRICS_NN_ENGINE'] == 'ort-cuda':
        for cell in report['cells']:
            log = Path(cell['attempts'][-1]['executed_command']['worker_command'][-1]).parent / 'worker.log'
            configured = [int(line.split()[1]) for line in log.read_text().splitlines()
                          if line.startswith('IQA_ORT_THREADS ')]
            if not configured or any(n != a.threads for n in configured):
                raise RuntimeError('ORT session thread budget was not explicitly configured')
            budgets.extend(configured)
    receipt = dict(metric=a.metric, engine=cold['worker_environment']['ZENMETRICS_NN_ENGINE'],
                   source_bundle_manifest_sha256=manifest['metadata']['source_bundle_manifest_sha256'],
                   report_sha256=sha(results / 'report.json'), score_bits=cold['score_bits'],
                   cold_elapsed_ns=cold['cold_elapsed_ns'], warm_median_ns=warm['median_ns'],
                   threads=a.threads, ort_intra_threads_observed=budgets or None,
                   qualification='Real worker protocol and within-engine repeat identity; not full model parity')
    (root / 'control.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
