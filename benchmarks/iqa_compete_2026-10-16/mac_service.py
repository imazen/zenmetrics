#!/usr/bin/env python3
"""Install a resumable benchmark LaunchAgent; run only while the user is logged in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys


def verify(bundle):
    for name, expected in json.loads((bundle / 'SHA256SUMS.json').read_text()).items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('bundle hash path escapes bundle')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('bundle changed: ' + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['install', 'status', 'stop', 'execute'])
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--run-heavy', type=Path)
    p.add_argument('--lock', type=Path)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--timeout-seconds', type=int, default=1100)
    a = p.parse_args()
    bundle, output = a.bundle.resolve(), a.output.resolve()
    token = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    label = 'io.imazen.iqa-durable.' + token
    domain = 'gui/' + str(os.getuid())
    service = domain + '/' + label
    plist = Path.home() / 'Library/LaunchAgents' / (label + '.plist')
    if a.action in ('status', 'stop'):
        return subprocess.call(['launchctl', 'print' if a.action == 'status' else 'bootout', service])
    if a.action == 'execute':
        try:
            verify(bundle)
            command = json.loads((output / 'service-command.json').read_text())
        except (OSError, ValueError) as e:
            print('Permanent bundle/config failure: ' + str(e), file=sys.stderr)
            return 0  # KeepAlive must not spin on integrity failure; retained stderr is authoritative.
        rc = subprocess.call(command)
        return 0 if rc in (0, 2) else rc
    if sys.platform != 'darwin':
        p.error('LaunchAgent installation requires macOS')
    if not a.run_heavy or not a.lock:
        p.error('--run-heavy and --lock are required for installation')
    verify(bundle)
    if plist.exists():
        p.error('service already installed; use status/stop, preserve its output, or select a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(output),
               '--timeout-seconds', str(a.timeout_seconds), '--max-attempts', '2', '--lock', str(a.lock.resolve()),
               '--run-heavy', str(a.run_heavy.resolve()), '--memory', '12G', '--jobs', str(a.jobs)]
    (output / 'service-command.json').write_text(json.dumps(command, indent=2) + '\n')
    plist.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(Label=label, ProgramArguments=[sys.executable, str(bundle / 'mac_service.py'), 'execute', str(bundle), str(output)],
                   RunAtLoad=True, KeepAlive=dict(SuccessfulExit=False), ThrottleInterval=30,
                   WorkingDirectory=str(bundle), StandardOutPath=str(output / 'service.stdout.log'),
                   StandardErrorPath=str(output / 'service.stderr.log'), ProcessType='Background')
    with plist.open('xb') as stream:
        plistlib.dump(payload, stream)
    subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
    print(json.dumps(dict(service=service, plist=str(plist), output=str(output))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
