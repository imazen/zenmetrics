#!/usr/bin/env python3
"""Freeze a self-contained CPU benchmark bundle from a built worker and hashed RGB pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--inputs-meta', type=Path, default=HERE / 'inputs.meta.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--build-receipt', type=Path, required=True, help='JSON recording actual compiler, features, source and dependency identities')
    p.add_argument('--resource-wrapper', type=Path, help='Copy an explicit run-heavy wrapper and its adjacent macOS implementation into this local bundle')
    p.add_argument('--metrics', default='all')
    p.add_argument('--sizes', default='cid22,aic-small,aic-medium,aic-large,2k,4k,8k')
    p.add_argument('--threads', default='1,8', help='Explicit worker counts, frozen in the manifest')
    p.add_argument('--cold-repeats', type=int, default=3)
    a = p.parse_args()
    counts = list(dict.fromkeys(int(n) for n in a.threads.split(',')))
    if min(counts) < 1 or a.cold_repeats < 1:
        p.error('thread counts and cold repeats must be positive')
    # Reuse the existing exact learned-model classification; no model files enter this bundle.
    from run import LEARNED
    inventory = json.loads(subprocess.check_output([str(a.binary.resolve()), 'inventory']))
    neural = LEARNED | {'ahiq', 'dmm'}
    eligible = [r['metric_name'] for r in inventory if r['compiled'] and not r['gpu'] and r['metric_name'] not in neural]
    metrics = eligible if a.metrics == 'all' else list(dict.fromkeys(a.metrics.split(',')))
    if not metrics or set(metrics) - set(eligible):
        p.error('requested metrics must be compiled non-neural CPU routes: ' + ','.join(eligible))
    pairs = {r['size_name']: r for r in json.loads(a.pairs.read_text())}
    provenance = json.loads(a.inputs_meta.read_text())
    expected = {r['size_name']: r for r in provenance['derivatives']}
    sizes = list(dict.fromkeys(a.sizes.split(',')))
    inputs = []
    for name in sizes:
        row, known = pairs[name], expected[name]
        if any(row[k] != known[k] for k in ('width', 'height', 'method')):
            raise ValueError('input geometry/provenance mismatch: ' + name)
        for role in ('reference', 'distorted'):
            path = (a.pairs.parent / row[role]).resolve()
            if sha(path) != known['sha256'][role] or path.stat().st_size != row['width'] * row['height'] * 3:
                raise ValueError('input bytes do not match frozen provenance: ' + str(path))
            inputs.append((path, name + '-' + role + '.rgb'))
    build = json.loads(a.build_receipt.read_text())
    if build.get('binary_sha256') != sha(a.binary) or build.get('source_commit') != a.source_commit:
        raise ValueError('build receipt does not identify this binary/source')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'bin').mkdir()
    (a.output / 'inputs').mkdir()
    shutil.copy2(a.binary, a.output / 'bin/iqa-compete')
    for path, name in inputs:
        shutil.copy2(path, a.output / 'inputs' / name)
    if a.resource_wrapper:
        (a.output / 'tools').mkdir()
        shutil.copy2(a.resource_wrapper, a.output / 'tools/run-heavy')
        companion = a.resource_wrapper.with_name('run-heavy-macos.py')
        if companion.is_file():
            shutil.copy2(companion, a.output / 'tools/run-heavy-macos.py')
    for name in ('durable.py', 'mac_service.py', 'select_durable.py', 'admission_recovery.py'):
        shutil.copy2(HERE / name, a.output / name)
    shutil.copy2(HERE / 'bundle.just', a.output / 'justfile')
    shutil.copy2(a.inputs_meta, a.output / 'inputs.meta.json')
    shutil.copy2(a.build_receipt, a.output / 'build.json')
    cells = []
    # Visit every method at the smallest size before moving to the expensive sizes.
    for size in sizes:
        pair = pairs[size]
        for metric in metrics:
            for threads in counts:
                for mode in ('score', 'score+map'):
                    for phase in ('cold', 'warm'):
                        for repeat in range(a.cold_repeats if phase == 'cold' else 1):
                            cell = dict(metric_name=metric, size_name=size, width=pair['width'], height=pair['height'],
                                        reference='inputs/' + size + '-reference.rgb', distorted='inputs/' + size + '-distorted.rgb',
                                        threads=threads, device='cpu', impl='zenmetrics', only_mode=mode, timing_phase=phase,
                                        cold_repeat=repeat, required_samples=1 if phase == 'cold' else 10,
                                        sample_target_ns=100_000_000, coverage_first=False, chunk_budget_ms=800_000,
                                        locked_process_accounting=True, input_method=pair['method'])
                            cells.append(cell)
    manifest = dict(schema=1, input_sha256={'inputs/' + name: sha(path) for path, name in inputs}, binary='bin/iqa-compete', binary_sha256=sha(a.output / 'bin/iqa-compete'), cells=cells,
                    metadata=dict(source_commit=a.source_commit, build_platform=platform.platform(), build_arch=platform.machine(),
                                  build_receipt_sha256=sha(a.build_receipt), features=build['features'], cold_definition='fresh process: scorer construction and first evaluation after RGB input loading; OS caches not flushed',
                                  warm_definition='10 zenbench rounds, 100 ms target per round, after initialization and warmup',
                                  input_provenance_sha256=sha(a.inputs_meta), corpus_scope='single S67 reference/distortion; scaling study, not perceptual ranking',
                                  metrics=metrics, sizes=sizes, threads=counts, inventory=inventory))
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (a.output / 'USAGE.txt').write_text('Run with Python 3: python3 durable.py run manifest.json /absolute/results --run-heavy /absolute/bundle/tools/run-heavy --lock ~/tmp/mac-heavy.lock --memory 12G --jobs 8\nUse mac_service.py install for launchd restart, status to inspect, stop to unload, start to resume. See the repository docs/IQA_DURABLE_MAC.md. This bundle is a single-pair scaling experiment, not a perceptual ranking.\n')
    hashes = {str(path.relative_to(a.output)): sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    (a.output / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(bundle=str(a.output.resolve()), cells=len(cells), metrics=len(metrics), manifest_sha256=sha(a.output / 'manifest.json'))))


if __name__ == '__main__':
    main()
