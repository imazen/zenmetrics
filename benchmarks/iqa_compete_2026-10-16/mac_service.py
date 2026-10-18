#!/usr/bin/env python3
"""Install a resumable benchmark LaunchAgent; persist attempts across controller restarts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time


def verify(bundle):
    hashes = json.loads((bundle / 'SHA256SUMS.json').read_text())
    required = {'manifest.json', 'durable.py', 'mac_service.py', 'bin/iqa-compete', 'build.json', 'inputs.meta.json'}
    if not required.issubset(hashes):
        raise ValueError('bundle hash manifest lacks required files')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if not set(manifest.get('input_sha256', {})).issubset(hashes):
        raise ValueError('bundle input hashes missing')
    for name, expected in hashes.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('bundle hash path escapes bundle')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('bundle changed: ' + name)


def supervise(command, output, silence=15.0, startup=60.0):
    """A Popen child remains our identity until reaped; never kill a saved PID."""
    child = subprocess.Popen(command)
    seen = None
    deadline = time.monotonic() + startup
    while child.poll() is None:
        try:
            stamp = (output / 'controller-heartbeat').stat().st_mtime_ns
        except FileNotFoundError:
            stamp = None
        if stamp is not None and stamp != seen:
            seen = stamp
            deadline = time.monotonic() + silence
        if time.monotonic() >= deadline:
            print('Controller heartbeat stalled; killing owned controller for restart', file=sys.stderr, flush=True)
            child.kill()
            return child.wait()
        time.sleep(.1)
    return child.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['install', 'start', 'status', 'stop', 'execute'])
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--run-heavy', type=Path)
    p.add_argument('--domain', choices=['user', 'gui'], default='user', help='user supports a Mac accessed only over SSH')
    p.add_argument('--lock', type=Path)
    p.add_argument('--command-sha256')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--timeout-seconds', type=int, default=1100)
    a = p.parse_args()
    bundle, output = a.bundle.resolve(), a.output.resolve()
    service_dir = output.with_name(output.name + '.service')
    token = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    label = 'io.imazen.iqa-durable.' + token
    domain = a.domain + '/' + str(os.getuid())
    service = domain + '/' + label
    plist = Path.home() / 'Library/LaunchAgents' / (label + '.plist')
    if a.action == 'start':
        verify(bundle)
        with plist.open('rb') as stream:
            installed = plistlib.load(stream)
        expected = [sys.executable, str(bundle / 'mac_service.py'), 'execute', str(bundle), str(output)]
        if installed.get('ProgramArguments', [])[:5] != expected:
            raise ValueError('installed service identifies another bundle/output')
        return subprocess.call(['launchctl', 'bootstrap', domain, str(plist)])
    if a.action in ('status', 'stop'):
        return subprocess.call(['launchctl', 'print' if a.action == 'status' else 'bootout', service])
    if a.action == 'execute':
        try:
            verify(bundle)
            raw = (service_dir / 'service-command.json').read_bytes()
            if hashlib.sha256(raw).hexdigest() != a.command_sha256:
                raise ValueError('installed service command changed')
            command = json.loads(raw)
            if command[:5] != [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(output)]:
                raise ValueError('service command identifies another experiment')
        except (OSError, ValueError) as e:
            print('Permanent bundle/config failure: ' + str(e), file=sys.stderr)
            return 0  # KeepAlive must not spin on integrity failure; retained stderr is authoritative.
        # Hold an idle-sleep assertion only for this service process lifetime.
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        try:
            rc = supervise(command, output)
        finally:
            awake.terminate()
            awake.wait()
        return 0 if rc in (0, 2) else rc
    if sys.platform != 'darwin':
        p.error('LaunchAgent installation requires macOS')
    if not a.run_heavy or not a.lock:
        p.error('--run-heavy and --lock are required for installation')
    verify(bundle)
    if plist.exists():
        p.error('service already installed; use status/stop, preserve its output, or select a new output directory')
    service_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(bundle / 'durable.py'), 'run', str(bundle / 'manifest.json'), str(output),
               '--timeout-seconds', str(a.timeout_seconds), '--max-attempts', '2', '--lock', str(a.lock.resolve()),
               '--run-heavy', str(a.run_heavy.resolve()), '--memory', '12G', '--jobs', str(a.jobs)]
    raw = (json.dumps(command, indent=2) + '\n').encode()
    (service_dir / 'service-command.json').write_bytes(raw)
    plist.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(Label=label, ProgramArguments=[sys.executable, str(bundle / 'mac_service.py'), 'execute', str(bundle), str(output), '--command-sha256', hashlib.sha256(raw).hexdigest()],
                   LimitLoadToSessionType='Background' if a.domain == 'user' else 'Aqua',
                   RunAtLoad=True, KeepAlive=dict(SuccessfulExit=False), ThrottleInterval=30,
                   WorkingDirectory=str(bundle), StandardOutPath=str(service_dir / 'service.stdout.log'),
                   StandardErrorPath=str(service_dir / 'service.stderr.log'), ProcessType='Background')
    with plist.open('xb') as stream:
        plistlib.dump(payload, stream)
    try:
        subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
    except subprocess.CalledProcessError:
        plist.unlink()  # Only the plist created by this failed install. Retain logs/config.
        raise
    print(json.dumps(dict(service=service, plist=str(plist), output=str(output))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
