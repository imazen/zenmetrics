#!/usr/bin/env python3
"""Freeze a 512-first family sequence while preserving validated prior observations."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import durable
from mac_service import verify
from prepare_durable import sha


def key(cell):
    return tuple(cell.get(k) for k in ('metric_name', 'size_name', 'width', 'height', 'threads',
                 'only_mode', 'timing_phase', 'cold_repeat', 'input_method'))


def prior_completions(core, report_path):
    binary, configs, frozen = durable.manifest(core / 'manifest.json')
    saved = durable.load(report_path.parent / 'identity.json')
    if saved['frozen'] != frozen:
        raise ValueError('prior report belongs to another frozen input/binary manifest')
    identity = durable.digest(durable.encoded(saved))
    report = durable.load(report_path)
    if report['identity'] != identity or [c['cell_id'] for c in report['cells']] != [c['durable_cell_id'] for c in configs]:
        raise ValueError('prior report cell/experiment identity differs')
    completed = set()
    for config in configs:
        directory = report_path.parent / 'cells' / config['durable_cell_id']
        candidates = [durable.read_result(p) for p in directory.glob('attempt-*/result.json')]
        if not any(r['status'] in ('ok', 'unsupported') for r in candidates):
            continue
        _, records = durable.attempts(report_path.parent, config, identity)
        if any(r['status'] in ('ok', 'unsupported') for r in records):
            completed.add(key(config))
    return completed


def clone(source, target):
    if sys.platform == 'darwin':
        subprocess.run(['/bin/cp', '-cR', str(source), str(target)], check=True)
    else:
        shutil.copytree(source, target)


def freeze(source, target, cells, retained_report=None):
    if not cells:
        raise ValueError('coverage stage is empty')
    clone(source, target)
    manifest = durable.load(source / 'manifest.json')
    manifest['cells'] = cells
    manifest['metadata'].update(metrics=list(dict.fromkeys(c['metric_name'] for c in cells)),
        sizes=list(dict.fromkeys(c['size_name'] for c in cells)),
        coverage_preparation_sha256=sha(Path(__file__)),
        coverage_policy='Every family at 512 first, then larger; retained observations keep their original binary provenance')
    if retained_report:
        shutil.copy2(retained_report, target / 'retained-core-report.json')
        manifest['input_sha256']['retained-core-report.json'] = sha(retained_report)
        manifest['metadata']['retained_report_sha256'] = sha(retained_report)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    hashes = durable.load(target / 'SHA256SUMS.json')
    hashes['manifest.json'] = sha(target / 'manifest.json')
    if retained_report:
        hashes['retained-core-report.json'] = sha(target / 'retained-core-report.json')
    (target / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return dict(bundle=str(target), cells=len(cells), metrics=len(manifest['metadata']['metrics']),
                manifest_sha256=hashes['manifest.json'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('core', 'remaining', 'nn_cpu', 'nn_metal', 'prior_report', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    for name in ('core', 'remaining', 'nn_cpu', 'nn_metal'):
        setattr(a, name, getattr(a, name).resolve())
        verify(getattr(a, name))
    completed = prior_completions(a.core, a.prior_report)
    core = durable.load(a.core / 'manifest.json')
    remaining = durable.load(a.remaining / 'manifest.json')
    if core['input_sha256'] != remaining['input_sha256']:
        raise ValueError('source input hashes differ')
    classical = core['cells'] + remaining['cells']
    if len({key(c) for c in classical}) != len(classical):
        raise ValueError('classical source cells overlap')
    a.output.mkdir(parents=True, exist_ok=False)
    phases = []
    for small in (True, False):
        label = '512' if small else 'larger'
        for family, source in [('non-nn', a.remaining), ('nn-cpu', a.nn_cpu), ('nn-metal', a.nn_metal)]:
            all_cells = classical if family == 'non-nn' else durable.load(source / 'manifest.json')['cells']
            cells = [c for c in all_cells if (c['size_name'] == 'cid22') == small
                     and (family != 'non-nn' or key(c) not in completed)]
            cells.sort(key=lambda c: (c['width'] * c['height'], c['metric_name'], c['threads'],
                                     c['only_mode'], c['timing_phase'], c['cold_repeat']))
            phase = freeze(source, a.output / (label + '-' + family), cells,
                           a.prior_report if family == 'non-nn' else None)
            phase['name'] = label + '-' + family
            phases.append(phase)
            print(json.dumps(phase), flush=True)
    receipt = dict(phases=phases, retained_cells=len(completed), prior_report_sha256=sha(a.prior_report),
                   prior_binary_sha256=core['binary_sha256'], next_binary_sha256=remaining['binary_sha256'])
    (a.output / 'sequence.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
