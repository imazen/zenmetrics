#!/usr/bin/env python3
"""Require exact scalar/map/cold/warm/thread identity for six dvifmish routes."""
import argparse
import contextlib
import time
import json
from pathlib import Path
import subprocess
import shlex
import sys
from durable import locked
from mac_service import verify
from prepare_coverage import freeze
from prepare_durable import sha



@contextlib.contextmanager
def protected_window(path, settle, receipt):
    requested = time.time_ns()
    with locked(path, wait=True) as fd:
        value = dict(shared_host_lock=str(path.resolve()), requested_unix_ns=requested,
                     acquired_unix_ns=time.time_ns(), settle_seconds=settle,
                     qualification=False, policy='Outer shared host lock covers the full control; private inner lock serializes its attempts')
        receipt.write_text(json.dumps(value, indent=2) + '\n')
        try:
            time.sleep(settle)
            value['settled_unix_ns'] = time.time_ns()
            receipt.write_text(json.dumps(value, indent=2) + '\n')
            yield fd
        finally:
            value['releasing_unix_ns'] = time.time_ns()
            receipt.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--cooldown-seconds', type=int, default=0, help='Quiet interval after acquiring the shared lock, before each attempt; outside all clocks')
    p.add_argument('--protected-window', action='store_true', help='Hold the existing host heavy lock for this entire control; use a private nested attempt lock')
    p.add_argument('--window-settle-seconds', type=int, default=5, help='Quiet settle after outer admission, outside every metric clock')
    a = p.parse_args()
    if not 0 <= a.window_settle_seconds <= 60:
        p.error('window settle must be within 0..60 seconds')
    if not 0 <= a.cooldown_seconds <= 60:
        p.error('cooldown must be within 0..60 seconds')
    a.output.mkdir(parents=True, exist_ok=False)
    window = protected_window(a.lock, a.window_settle_seconds, a.output / 'protected-window.json') if a.protected_window else contextlib.nullcontext(None)
    with window as outer_fd:
        run_control(a, outer_fd)


def run_control(a, outer_fd):
    source = a.bundle.resolve()
    verify(source)
    manifest = json.loads((source / 'manifest.json').read_bytes())
    cells = [c for c in manifest['cells'] if c['width'] == c['height'] == 512]
    metrics = {c['metric_name'] for c in cells}
    if len(metrics) != 6 or len(cells) != 96 or any(not name.startswith('dvifmish-') for name in metrics):
        raise ValueError('control requires all six routes, one/eight workers, scalar/map and all cold/warm cells at 512')
    bundle, results = a.output / 'bundle', a.output / 'results'
    freeze(source, bundle, cells)
    wrapper = bundle / 'tools/run-heavy'
    if a.cooldown_seconds:
        quiet_wrapper = a.output.resolve() / 'quiet-run-heavy.sh'
        quiet_wrapper.write_text('#!/bin/sh\nset -eu\nsleep ' + str(a.cooldown_seconds) + '\nexec ' + shlex.quote(str(wrapper.resolve())) + ' \"$@\"\n')
        quiet_wrapper.chmod(0o755)
        wrapper = quiet_wrapper
    inner_lock = a.output.resolve() / 'private-attempt.lock' if a.protected_window else a.lock
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(results),
        '--run-heavy', str(wrapper), '--lock', str(inner_lock), '--memory', '12G', '--jobs', '8',
        '--timeout-seconds', '1100', '--max-attempts', '2']
    (a.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (a.output / 'controller.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       pass_fds=(outer_fd,) if outer_fd is not None else ())
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
    receipt = dict(protected_window=a.protected_window, shared_host_lock=str(a.lock.resolve()), window_settle_seconds=a.window_settle_seconds if a.protected_window else 0, cooldown_seconds=a.cooldown_seconds, wrapper_sha256=sha(wrapper), report_sha256=sha(results / 'report.json'), cells=len(cells), clean_warm_rounds=rounds,
        binary_sha256=manifest['binary_sha256'], scores={name:next(iter(values)) for name,values in bits.items()},
        coverage='four port presets and default talk-faithful-luma float/int; native block maps; explicit one/eight-worker budgets',
        qualification='Within-route exact library and timing-protocol controls; no new perceptual qualification or publication')
    (a.output / 'control.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
