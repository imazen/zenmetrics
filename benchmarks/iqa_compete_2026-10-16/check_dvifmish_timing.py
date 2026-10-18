#!/usr/bin/env python3
"""Require exact scalar/map/cold/warm/thread identity for six dvifmish routes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from mac_service import verify
from prepare_coverage import freeze
from prepare_durable import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    a = p.parse_args()
    source = a.bundle.resolve()
    verify(source)
    manifest = json.loads((source / 'manifest.json').read_bytes())
    cells = [c for c in manifest['cells'] if c['width'] == c['height'] == 512]
    metrics = {c['metric_name'] for c in cells}
    if len(metrics) != 6 or len(cells) != 96 or any(not name.startswith('dvifmish-') for name in metrics):
        raise ValueError('control requires all six routes, one/eight workers, scalar/map and all cold/warm cells at 512')
    a.output.mkdir(parents=True, exist_ok=False)
    bundle, results = a.output / 'bundle', a.output / 'results'
    freeze(source, bundle, cells)
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(results),
        '--run-heavy', str(bundle / 'tools/run-heavy'), '--lock', str(a.lock), '--memory', '12G', '--jobs', '8',
        '--timeout-seconds', '1100', '--max-attempts', '2']
    (a.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (a.output / 'controller.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((results / 'report.json').read_bytes())
    if not report['complete'] or any(c['status'] != 'ok' for c in report['cells']):
        raise RuntimeError('dvifmish control has incomplete or refused cells')
    bits = {name:set() for name in metrics}
    rounds = 0
    for cell in report['cells']:
        row = cell['row']
        route = row.get('benchmark_route', {})
        if route.get('metric_name') != row['metric_name'] or route.get('publish') is not False:
            raise RuntimeError('worker lacks explicit dvifmish route/map provenance')
        if not route.get('preset_sha256') or 'raw distortion' not in route.get('native_map', ''):
            raise RuntimeError('worker lacks truthful native-map units or constant hash')
        bits[row['metric_name']].add(row['score_bits'])
        if row['timing_phase'] == 'warm':
            samples = row['zenbench_samples']
            if len(samples) < 10 or any(not s['gate_clean'] for s in samples):
                raise RuntimeError('control warm rounds were incomplete or contention-flagged; preserve and rerun separately')
            rounds += len(samples)
    if any(len(values) != 1 for values in bits.values()):
        raise RuntimeError('dvifmish cold/warm/scalar/map/thread score bits differ')
    receipt = dict(report_sha256=sha(results / 'report.json'), cells=len(cells), clean_warm_rounds=rounds,
        binary_sha256=manifest['binary_sha256'], scores={name:next(iter(values)) for name,values in bits.items()},
        coverage='four port presets and default talk-faithful-luma float/int; native block maps; explicit one/eight-worker budgets',
        qualification='Within-route exact library and timing-protocol controls; no new perceptual qualification or publication')
    (a.output / 'control.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
