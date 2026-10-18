#!/usr/bin/env python3
"""Freeze fresh Linux CPU/NN CPU/NN CUDA families at 512, then ascending sizes."""
import argparse
import json
from pathlib import Path
import shutil
import durable
from mac_service import verify
from prepare_coverage import freeze
from prepare_durable import HERE, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cpu', type=Path)
    p.add_argument('nn_cpu', type=Path)
    p.add_argument('nn_cuda', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    a = p.parse_args()
    build = durable.load(a.build)
    if sha(a.binary) != build['binary_sha256']:
        raise ValueError('sequence build receipt differs from worker')
    sources = [('non-nn', a.cpu.resolve()), ('nn-cpu', a.nn_cpu.resolve()), ('nn-cuda', a.nn_cuda.resolve())]
    for _, source in sources:
        verify(source)
    a.output.mkdir(parents=True, exist_ok=False)
    phases = []
    for small in (True, False):
        for family, source in sources:
            cells = [c for c in durable.load(source / 'manifest.json')['cells']
                     if (c['width'] == c['height'] == 512) == small]
            cells.sort(key=lambda c: (c['width'] * c['height'], c['metric_name'], c['threads'],
                                     c['only_mode'], c['timing_phase'], c['cold_repeat']))
            name = ('512-' if small else 'larger-') + family
            target = a.output / name
            phase = freeze(source, target, cells)
            manifest = durable.load(target / 'manifest.json')
            shutil.copy2(a.binary, target / 'bin/iqa-compete')
            shutil.copy2(a.build, target / 'build.json')
            manifest['binary_sha256'] = build['binary_sha256']
            manifest['metadata'].update(source_commit=build['source_commit'],
                build_receipt_sha256=sha(a.build), features=build['features'])
            # Sources remain immutable; the new bundle freezes the current supervisor.
            for script in ('durable.py', 'admission_recovery.py', 'mac_service.py', 'select_durable.py'):
                shutil.copy2(HERE / script, target / script)
            manifest['metadata']['runner_sha256'] = {script: sha(target / script)
                for script in ('durable.py', 'admission_recovery.py', 'mac_service.py')}
            manifest['metadata']['linux_sequence_preparation_sha256'] = sha(Path(__file__))
            (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            hashes = {f.relative_to(target).as_posix(): sha(f) for f in target.rglob('*')
                      if f.is_file() and f.name != 'SHA256SUMS.json'}
            (target / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
            verify(target)
            phase.update(name=name, manifest_sha256=hashes['manifest.json'],
                         output=str(Path.home() / 'benchmark-results' / a.output.name / name))
            phases.append(phase)
            print(json.dumps(phase), flush=True)
    plan = dict(phases=phases, lock=str(a.lock.resolve()), source_manifests={name: sha(source / 'manifest.json')
        for name, source in sources}, service_sha256=sha(HERE / 'linux_service.py'))
    (a.output / 'sequence.json').write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    main()
