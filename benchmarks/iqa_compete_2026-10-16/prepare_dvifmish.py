#!/usr/bin/env python3
"""Prepare new frozen dvifmish CPU supplements, without editing existing runs."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import durable
from mac_service import verify
from prepare_coverage import freeze
from prepare_durable import HERE, sha


def templates(manifest):
    selected = {}
    for cell in manifest['cells']:
        key = tuple(cell[k] for k in ('size_name', 'width', 'height', 'threads', 'only_mode', 'timing_phase', 'cold_repeat'))
        selected.setdefault(key, cell)
    return list(selected.values())


def gate(bundle, output):
    verify(bundle)
    return dict(bundle=str(bundle.resolve()), output=str(output.resolve()),
                manifest_sha256=sha(bundle / 'manifest.json'))


def prepare(a):
    verify(a.template)
    original = durable.load(a.template / 'manifest.json')
    build = durable.load(a.build)
    if sha(a.binary) != build['binary_sha256']:
        raise ValueError('build receipt differs from supplement worker')
    inventory_bytes = subprocess.check_output([str(a.binary), 'inventory'])
    inventory = json.loads(inventory_bytes)
    routes = [r for r in inventory if r['metric_name'].startswith('dvifmish-') and r['compiled']]
    if len(routes) != 6 or any(r.get('publish') is not False or not r.get('preset_sha256') for r in routes):
        raise ValueError('worker must expose all six pinned unpublished dvifmish routes')
    if build.get('inventory_sha256') != durable.digest(inventory_bytes):
        raise ValueError('build receipt differs from native inventory')
    required = {1, 8}
    if {c['threads'] for c in original['cells']} != required:
        raise ValueError('supplement requires the same explicit one/eight-worker template')
    a.output.mkdir(parents=True, exist_ok=False)
    # The original geometry/provenance is validated again by prepare_durable.
    rows = {}
    for cell in templates(original):
        rows.setdefault(cell['size_name'], dict(size_name=cell['size_name'], width=cell['width'], height=cell['height'],
            method=cell['input_method'], reference=str(a.template / cell['reference']),
            distorted=str(a.template / cell['distorted'])))
    (a.output / 'pairs.json').write_text(json.dumps(list(rows.values()), indent=2) + '\n')
    source = a.output / 'source'
    command = [sys.executable, str(HERE / 'prepare_durable.py'), '--binary', str(a.binary),
        '--pairs', str(a.output / 'pairs.json'), '--inputs-meta', str(a.template / 'inputs.meta.json'),
        '--output', str(source), '--source-commit', build['source_commit'], '--build-receipt', str(a.build),
        '--resource-wrapper', str(a.template / 'tools/run-heavy'), '--metrics', ','.join(r['metric_name'] for r in routes),
        '--sizes', ','.join(name for name, row in sorted(rows.items(), key=lambda item:item[1]['width']*item[1]['height'])),
        '--threads', '1,8', '--cold-repeats', '3']
    placement = original['metadata'].get('cpu_placement')
    if sys.platform == 'linux':
        if not placement or placement['policy'] != 'performance':
            raise ValueError('Linux supplement requires the existing checked P-core template')
        command += ['--zenbench', str(a.template / placement['zenbench']), '--cpu-set', ','.join(map(str, placement['cpus']))]
    subprocess.run(command, check=True)
    manifest = durable.load(source / 'manifest.json')
    phases = []
    predecessors = [gate(a.after_512[0], a.after_512[1]), gate(a.after_larger[0], a.after_larger[1])]
    for small, predecessor in zip((True, False), predecessors):
        name = '512-dvifmish' if small else 'larger-dvifmish'
        cells = [c for c in manifest['cells'] if (c['width'] == c['height'] == 512) == small]
        cells.sort(key=lambda c:(c['width']*c['height'],c['metric_name'],c['threads'],c['only_mode'],c['timing_phase'],c['cold_repeat']))
        target = a.output / name
        phase = freeze(source, target, cells)
        after = [predecessor]
        if phases:
            after.append(gate(Path(phases[0]['bundle']), Path(phases[0]['output'])))
        phase.update(name=name, output=str(a.results / name), after=after)
        phases.append(phase)
    plan = dict(phases=phases, lock=str(a.lock), platform=platform.system(),
        source_template_manifest_sha256=sha(a.template / 'manifest.json'), build_receipt_sha256=sha(a.build),
        preparation_sha256=sha(Path(__file__)), routes=routes,
        policy='Separate supplemental runs; original manifests/results and NN512 CPU-to-GPU dependencies unchanged')
    (a.output / 'sequence.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


def install(plan_path):
    plan = durable.load(plan_path)
    if plan['platform'] != platform.system():
        raise ValueError('supplement must be installed on its prepared native platform')
    if plan['platform'] == 'Linux':
        return subprocess.call([sys.executable, str(HERE / 'linux_service.py'), 'install', str(plan_path)])
    if plan['platform'] != 'Darwin' or sys.platform != 'darwin':
        raise ValueError('supplement must be installed on its prepared native platform')
    for phase in plan['phases']:
        bundle = Path(phase['bundle'])
        command = [sys.executable, str(bundle / 'mac_service.py'), 'install', str(bundle), phase['output'],
            '--run-heavy', str(bundle / 'tools/run-heavy'), '--lock', plan['lock'], '--memory', '12G', '--jobs', '8']
        for after in phase['after']:
            command += ['--after', after['bundle'], after['output']]
        subprocess.run(command, check=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    make = sub.add_parser('prepare')
    for name in ('template', 'binary', 'build', 'output', 'results'):
        make.add_argument(name, type=Path)
    make.add_argument('--lock', type=Path, required=True)
    make.add_argument('--after-512', nargs=2, type=Path, required=True, metavar=('BUNDLE', 'OUTPUT'))
    make.add_argument('--after-larger', nargs=2, type=Path, required=True, metavar=('BUNDLE', 'OUTPUT'))
    sub.add_parser('install').add_argument('plan', type=Path)
    a = p.parse_args()
    if a.action == 'install':
        return install(a.plan.resolve())
    for name in ('template', 'binary', 'build', 'output', 'results', 'lock'):
        setattr(a, name, getattr(a, name).resolve())
    prepare(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
