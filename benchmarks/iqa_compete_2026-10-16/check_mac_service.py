#!/usr/bin/env python3
"""Real macOS LaunchAgent controller-SIGKILL/restart control; retains all evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--run-heavy', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True)
    a = p.parse_args()
    if sys.platform != 'darwin':
        p.error('this control requires macOS')
    bundle, output = a.bundle.resolve(), a.output.resolve()
    if output.exists() or output.with_name(output.name + '.service').exists():
        p.error('control needs a fresh output path')
    helper = [sys.executable, str(bundle / 'mac_service.py')]
    result = subprocess.run([*helper, 'install', str(bundle), str(output), '--run-heavy', str(a.run_heavy.resolve()),
                             '--lock', str(a.lock.resolve()), '--timeout-seconds', '120'], text=True, capture_output=True, check=True)
    info = json.loads(result.stdout)
    receipt = dict(service=info, started=time.time(), controller_killed=None, restarted_service_pid=None)
    evidence = output.with_name(output.name + '.control.json')
    def save():
        evidence.write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    try:
        deadline = time.monotonic() + 180
        first_service = None
        while time.monotonic() < deadline:
            state = subprocess.run(['launchctl', 'print', info['service']], text=True, capture_output=True)
            match = re.search(r'^\s*pid = (\d+)', state.stdout, re.M)
            service_pid = int(match[1]) if match else None
            if service_pid and first_service is None:
                first_service = service_pid
            if receipt['controller_killed'] is None and service_pid and list(output.glob('cells/*/attempt-*/worker.log')):
                table = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
                for line in table.splitlines():
                    fields = line.strip().split(None, 2)
                    if len(fields) == 3 and int(fields[1]) == service_pid and str(bundle / 'durable.py') + ' run ' in fields[2]:
                        pid = int(fields[0])
                        os.kill(pid, signal.SIGKILL)
                        receipt.update(controller_killed=pid, initial_service_pid=service_pid, killed_at=time.time())
                        save()
                        break
            if receipt['controller_killed'] and service_pid and service_pid != first_service:
                receipt['restarted_service_pid'] = service_pid
            report_path = output / 'report.json'
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get('complete'):
                    if not receipt['controller_killed']:
                        raise RuntimeError('run finished before a live controller could be killed')
                    if not receipt['restarted_service_pid']:
                        raise RuntimeError('completion without observed launchd restart')
                    receipt.update(completed=time.time(), report_complete=True,
                                   cells=[dict(cell_id=c['cell_id'], status=c['status'], attempts=len(c['attempts'])) for c in report['cells']])
                    if any(c['status'] != 'ok' for c in report['cells']):
                        raise RuntimeError('control cells did not recover successfully')
                    save()
                    print(json.dumps(receipt))
                    return
            time.sleep(.05)
        raise RuntimeError('service restart control exceeded 180 seconds')
    except BaseException as error:
        receipt['error'] = str(error)
        save()
        raise
    finally:
        subprocess.run(['launchctl', 'bootout', info['service']], check=False)
        Path(info['plist']).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
