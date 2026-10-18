#!/usr/bin/env python3
"""Run frozen durable phases sequentially in a restarting systemd user service."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import durable
from mac_service import verify, predecessor_complete, supervise, wait_for_predecessors


def execute(plan, plan_hash):
    if durable.file_hash(plan) != plan_hash:
        raise ValueError('service sequence changed')
    value = durable.load(plan)
    previous = None
    for phase in value['phases']:
        bundle, output = Path(phase['bundle']), Path(phase['output'])
        verify(bundle)
        if durable.file_hash(bundle / 'manifest.json') != phase['manifest_sha256']:
            raise ValueError('sequence manifest changed')
        if phase.get('after'):
            service_dir = output.with_name(output.name + '.service')
            service_dir.mkdir(parents=True, exist_ok=True)
            wait_for_predecessors(phase['after'], service_dir)
        if previous and not predecessor_complete(previous):
            raise ValueError('predecessor incomplete or controller still alive')
        gate = dict(output=str(output), manifest_sha256=phase['manifest_sha256'])
        if predecessor_complete(gate):
            previous = gate
            continue
        command = [sys.executable, str(bundle / 'durable.py'), 'run',
            str(bundle / 'manifest.json'), str(output), '--run-heavy', str(bundle / 'tools/run-heavy'),
            '--lock', value['lock'], '--memory', '12G', '--jobs', '8',
            '--timeout-seconds', '1100', '--max-attempts', '2']
        print(json.dumps(dict(starting=phase['name'], command=command)), flush=True)
        rc = supervise(command, output)
        if rc:
            return rc
        previous = gate
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['install', 'execute', 'status'])
    p.add_argument('plan', type=Path)
    p.add_argument('--sha256')
    a = p.parse_args()
    plan = a.plan.resolve()
    name = 'iqa-durable-' + hashlib.sha256(str(plan).encode()).hexdigest()[:12]
    if a.action == 'execute':
        try:
            return execute(plan, a.sha256)
        except (OSError, ValueError) as error:
            print('Permanent sequence refusal: ' + str(error), file=sys.stderr, flush=True)
            return 2
    if a.action == 'status':
        value = durable.load(plan)
        for phase in value['phases']:
            output = Path(phase['output'])
            wait = output.with_name(output.name + '.service') / 'queue-wait.json'
            print(json.dumps(dict(name=phase['name'], progress=durable.load(output / 'progress.json')
                if (output / 'progress.json').exists() else 'queued',
                waiting_on=durable.load(wait)['pending'] if wait.exists() else [])))
        return subprocess.call(['systemctl', '--user', 'status', name, '--no-pager'])
    value = durable.load(plan)
    for phase in value['phases']:
        verify(Path(phase['bundle']))
        if durable.file_hash(Path(phase['bundle']) / 'manifest.json') != phase['manifest_sha256']:
            raise ValueError('sequence manifest changed')
    command = ['systemd-run', '--user', '--unit=' + name, '--property=Restart=on-failure',
        '--property=RestartSec=10', '--property=RestartPreventExitStatus=2',
        '--property=StandardOutput=append:' + str(plan.parent / 'service.log'),
        '--property=StandardError=append:' + str(plan.parent / 'service.log'),
        '--setenv=TMPDIR=' + str(Path.home() / 'tmp'), sys.executable, str(Path(__file__).resolve()),
        'execute', str(plan), '--sha256', durable.file_hash(plan)]
    (plan.parent / 'service-command.json').write_text(json.dumps(command, indent=2) + '\n')
    return subprocess.call(command)


if __name__ == '__main__':
    sys.exit(main())
