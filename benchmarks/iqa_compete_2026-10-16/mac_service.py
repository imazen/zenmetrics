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
import fcntl
import shutil
from collections import Counter


def progress(output, domain):
    service_dir = output.with_name(output.name + '.service')
    compact = json.loads((output / 'progress.json').read_bytes()) if (output / 'progress.json').exists() else {}
    counts, phases, unsupported = Counter(), Counter(), Counter()
    from durable import read_result
    for path in output.glob('cells/*/attempt-*/result.json'):
        record = read_result(path)
        counts[record['status']] += 1
        row = record.get('row') or {}
        if record['status'] == 'ok':
            phases[row['timing_phase']] += 1
        if record['status'] == 'unsupported':
            unsupported[row['metric_name']] += 1
    label = 'io.imazen.iqa-durable.' + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    status = subprocess.run(['launchctl', 'print', domain + '/' + label], capture_output=True, text=True)
    state = next((line.strip().split(' = ', 1)[1] for line in status.stdout.splitlines() if line.strip().startswith('state = ')), 'unloaded')
    after = service_dir / 'queue-wait.json'
    waiting = json.loads(after.read_bytes()).get('pending', []) if after.exists() else []
    return dict(output=str(output), progress=compact, attempt_statuses=dict(counts), successful_phases=dict(phases),
                unsupported_maps=dict(unsupported), service_state=state, waiting_on=waiting)



def _check_in_rows(value):
    if isinstance(value, list):
        return value
    return [dict(output=p['output'], progress=p['progress'] if isinstance(p['progress'], dict) else {},
                 attempt_statuses={}, service_state=value['service'], waiting_on=(p.get('wait') or {}).get('pending', []))
            for p in value['phases']]


def check_in(plan_path, directory, pane=None):
    """Read-only remote observations survive a blocked or unavailable agent prompt."""
    import datetime
    import shlex
    plan = json.loads(plan_path.read_bytes())
    command = plan.get('status_command')
    if not command:
        command = ['python3', plan['script'], 'progress', plan['phases'][0]['bundle'], plan['phases'][0]['output']]
        for phase in plan['phases'][1:]:
            command += ['--other-output', phase['output']]
    ssh_command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', plan['machine'], shlex.join(command)]
    try:
        result = subprocess.run(ssh_command, text=True, capture_output=True, timeout=60)
        rows = _check_in_rows(json.loads(result.stdout)) if result.returncode == 0 else []
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(ssh_command, 124, '', 'Mac status SSH request timed out after 60 seconds')
    except (OSError, ValueError) as error:
        result = subprocess.CompletedProcess(ssh_command, 1, '', str(error))
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt = dict(checked_utc=stamp, command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (directory / (stamp + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode:
        message = 'Mac benchmark status check failed; inspect ' + str(directory / (stamp + '.json'))
    else:
        rows = _check_in_rows(json.loads(result.stdout))
        message = '; '.join(Path(r['output']).name + ': ' +
            (str(r['progress'].get('statuses', {})) + ', attempts=' + str(r['attempt_statuses'])
             if r['progress'] else 'queued') for r in rows)
    print(message, flush=True)
    herdr = shutil.which('herdr')
    if herdr:
        subprocess.run([herdr, 'notification', 'show', 'Mac IQA benchmark check-in', '--body', message, '--sound', 'none'],
                       timeout=15, check=False)
        if pane:
            prompt = subprocess.run([herdr, 'agent', 'prompt', pane,
                'Scheduled Mac benchmark check-in: ' + message + '. Inspect retained attempts and repair blockers; report progress to the user.'],
                capture_output=True, text=True, timeout=15)
            (directory / (stamp + '.agent.json')).write_text(json.dumps(dict(returncode=prompt.returncode, stdout=prompt.stdout, stderr=prompt.stderr)) + '\n')
    if result.returncode == 0 and all(r['progress'].get('complete') for r in rows):
        subprocess.run(['systemctl', '--user', 'stop', 'iqa-mac-checkin.timer'], check=True)
    return result.returncode


def predecessor_complete(gate):
    """Require the final report for the frozen predecessor and a free controller lock."""
    output = Path(gate['output'])
    if not (output / 'report.json').exists() or not (output / 'identity.json').exists():
        return False
    saved = json.loads((output / 'identity.json').read_bytes())
    frozen = saved['frozen']
    if frozen['manifest_sha256'] != gate['manifest_sha256']:
        raise ValueError('predecessor manifest changed')
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    identity = hashlib.sha256(canonical(saved)).hexdigest()
    report = json.loads((output / 'report.json').read_bytes())
    if report.get('identity') != identity:
        raise ValueError('predecessor report identity changed')
    if not report.get('complete'):
        return False
    expected = [f'{i:06d}-' + hashlib.sha256(canonical(c)).hexdigest()[:16]
                for i, c in enumerate(frozen['manifest']['cells'])]
    cells = report.get('cells', [])
    if not expected or [c.get('cell_id') for c in cells] != expected or any(c.get('status') not in ('ok', 'unsupported', 'failed') for c in cells):
        raise ValueError('predecessor final cell inventory differs')
    with (output / 'coordinator.lock').open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def wait_for_predecessors(gates, service_dir):
    if not gates:
        return
    while True:
        pending = [g['output'] for g in gates if not predecessor_complete(g)]
        (service_dir / 'queue-wait.json').write_text(json.dumps(dict(pending=pending, checked_unix=time.time())) + '\n')
        if not pending:
            return
        time.sleep(30)


def verify(bundle):
    hashes = json.loads((bundle / 'SHA256SUMS.json').read_text())
    required = {'manifest.json', 'durable.py', 'mac_service.py', 'bin/iqa-compete', 'build.json', 'inputs.meta.json', 'select_durable.py'}
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
    p.add_argument('action', choices=['install', 'start', 'status', 'stop', 'execute', 'progress', 'check-in'])
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--run-heavy', type=Path)
    p.add_argument('--domain', choices=['user', 'gui'], default='user', help='user supports a Mac accessed only over SSH')
    p.add_argument('--lock', type=Path)
    p.add_argument('--command-sha256')
    p.add_argument('--other-output', type=Path, action='append', default=[])
    p.add_argument('--agent-pane')
    p.add_argument('--after', nargs=2, action='append', default=[], metavar=('BUNDLE', 'OUTPUT'), help='Wait for this frozen predecessor to finish before starting')
    p.add_argument('--after-sha256')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--timeout-seconds', type=int, default=1100)
    a = p.parse_args()
    if a.action == 'check-in':
        return check_in(a.bundle, a.output, a.agent_pane)
    if a.action == 'progress':
        print(json.dumps([progress(o.resolve(), a.domain + '/' + str(os.getuid())) for o in [a.output, *a.other_output]]))
        return 0
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
            gates = []
            if a.after_sha256:
                after_raw = (service_dir / 'service-after.json').read_bytes()
                if hashlib.sha256(after_raw).hexdigest() != a.after_sha256:
                    raise ValueError('predecessor configuration changed')
                gates = json.loads(after_raw)
        except (OSError, ValueError) as e:
            print('Permanent bundle/config failure: ' + str(e), file=sys.stderr)
            return 0  # KeepAlive must not spin on integrity failure; retained stderr is authoritative.
        # Hold an idle-sleep assertion only for this service process lifetime.
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        try:
            wait_for_predecessors(gates, service_dir)
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
    after_args = []
    if a.after:
        gates = []
        for prior_bundle, prior_output in a.after:
            prior_bundle = Path(prior_bundle).resolve()
            prior_output = Path(prior_output).resolve()
            if prior_output == output:
                raise ValueError('service cannot depend on itself')
            verify(prior_bundle)
            gates.append(dict(output=str(prior_output), manifest_sha256=hashlib.sha256((prior_bundle / 'manifest.json').read_bytes()).hexdigest()))
        after_raw = (json.dumps(gates, indent=2) + '\n').encode()
        (service_dir / 'service-after.json').write_bytes(after_raw)
        after_args = ['--after-sha256', hashlib.sha256(after_raw).hexdigest()]
    plist.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(Label=label, ProgramArguments=[sys.executable, str(bundle / 'mac_service.py'), 'execute', str(bundle), str(output), '--command-sha256', hashlib.sha256(raw).hexdigest(), *after_args],
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
