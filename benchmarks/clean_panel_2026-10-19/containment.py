"""Linux-only owned-child containment for this pilot, independent of shared helpers.

The controller is a subreaper. Its scorer subtree is serialized; no unrelated
children may be launched while execute() owns admission. PID/start-time pairs
prevent signaling a reused PID. Escaped sessions are still owned descendants.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.next')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text(); fields = raw[raw.rindex(')')+2:].split()
            result[int(path.parent.name)] = dict(state=fields[0], ppid=int(fields[1]),
                pgid=int(fields[2]), sid=int(fields[3]), start=int(fields[19]))
        except FileNotFoundError:
            pass
    return result


def subreaper():
    if not Path('/proc/self/stat').exists():
        raise RuntimeError('Linux process identity required')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot establish child subreaper')


def execute(command, output, env, receipt, timeout=120, grace=1.0):
    subreaper()
    owner = os.getpid()
    if any(p['ppid'] == owner for p in processes().values()):
        raise RuntimeError('containment requires no pre-existing controller children')
    # Fail before launch if the required PID-safe facilities are unavailable.
    probe = os.pidfd_open(owner)
    try: signal.pidfd_send_signal(probe, 0)
    finally: os.close(probe)
    child = None
    known = {}
    details = {}
    events = []
    reason, code, error = 'complete', None, None
    failures = {}
    last_remaining = None
    interrupted = []
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def interrupt(sig, frame):
        interrupted.append(sig)

    def refresh():
        current = processes()
        changed = True
        while changed:
            changed = False
            for pid, p in current.items():
                parent_owned = p['ppid'] == owner or (p['ppid'] in known and current.get(p['ppid'], {}).get('start') == known[p['ppid']])
                if pid != owner and parent_owned and known.get(pid) != p['start']:
                    known[pid] = p['start']; changed = True
        members = {pid: p for pid, p in current.items() if known.get(pid) == p['start']}
        for pid, p in members.items():
            value = dict(pid=pid, **p)
            try: value['exe'] = str(Path(f'/proc/{pid}/exe').resolve(strict=True))
            except FileNotFoundError: value['exe'] = details.get(pid, {}).get('exe')
            details[pid] = value
        return members

    def reap():
        # Popen owns its direct child's status; reap only adopted descendants.
        for pid, p in refresh().items():
            if p['state'] == 'Z' and p['ppid'] == owner and (child is None or pid != child.pid):
                try: os.waitpid(pid, os.WNOHANG)
                except ChildProcessError: pass

    def send(sig):
        for pid, p in refresh().items():
            if p['state'] != 'Z':
                try:
                    fd = os.pidfd_open(pid)
                    try:
                        # Recheck identity after acquiring a stable kernel handle.
                        if processes().get(pid, {}).get('start') == p['start']:
                            signal.pidfd_send_signal(fd, sig)
                            events.append(dict(pid=pid,start=p['start'],signal=sig))
                    finally: os.close(fd)
                except ProcessLookupError: pass

    def state(status):
        return dict(status=status, reason=reason, exit_code=code, error=error,
                    owner_pid=owner, owned=list(details.values()),
                    remaining=last_remaining, signals=events,
                    cleanup_failures=list(failures.values()), interrupted_signals=interrupted)

    # Keep an already-open diagnostic sink: receipt replacement itself may fail
    # under EMFILE. A write failure never authorizes releasing admission.
    emergency = os.open(str(receipt)+'.errors.jsonl', os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    def failure(phase, exc):
        key = (phase, type(exc).__name__, str(exc))
        item = failures.setdefault(key, dict(phase=phase,type=type(exc).__name__,message=str(exc),count=0))
        item['count'] += 1
        if item['count'] == 1:
            try: os.write(emergency, (json.dumps(item)+'\n').encode())
            except BaseException: pass  # Retain in memory; normal receipt retries.
    try:
        for sig in previous: signal.signal(sig, interrupt)
        write(receipt, state('preflight'))
    except BaseException:
        os.close(emergency)
        for sig, handler in previous.items(): signal.signal(sig, handler)
        raise  # No child has been launched.
    try:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        last_remaining = refresh()
        write(receipt, state('running'))
        deadline = time.monotonic() + timeout
        while child.poll() is None:
            last_remaining = refresh()
            if interrupted:
                reason = 'interrupted'; error = dict(type='KeyboardInterrupt',message='signal '+str(interrupted[-1])); break
            if time.monotonic() >= deadline:
                reason = 'timeout'; break
            time.sleep(.02)
        if reason == 'complete':
            code = child.returncode
            if any(p['state'] != 'Z' for p in refresh().values()):
                reason = 'orphaned-descendants'
                error = dict(type='ContainmentError',message='direct child exited with live descendants')
    except BaseException as exc:
        reason = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'launch-or-monitor-error'
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        # Nothing in teardown may unwind to panel.acquire's ordinary rejection
        # path until a fresh scan proves emptiness AND the terminal receipt is
        # durable. Errors in monitoring, signaling, reaping or writing quarantine
        # the locks held by the caller. The signal handler only records requests.
        term_deadline = time.monotonic() + grace
        while True:
            phase = 'refresh'
            try:
                last_remaining = refresh()
                phase = 'reap'
                if child is not None: child.poll()
                reap()
                phase = 'refresh-after-reap'
                last_remaining = refresh()
                if last_remaining:
                    phase = 'send'
                    send(signal.SIGTERM if time.monotonic() < term_deadline else signal.SIGKILL)
                    phase = 'blocked-receipt'
                    write(receipt, state('containment-blocked'))
                else:
                    if reason == 'timeout': code = 124
                    elif reason != 'complete': code = 130 if reason == 'interrupted' else 125
                    phase = 'terminal-receipt'
                    write(receipt, state('empty'))
                    phase = 'final-empty-verification'
                    last_remaining = refresh()
                    if not last_remaining:
                        break
            except BaseException as exc:
                last_remaining = None  # Failed observation is UNKNOWN, not empty.
                failure(phase, exc)
                try: write(receipt, state('containment-blocked'))
                except BaseException as write_error: failure('blocked-receipt', write_error)
            try: time.sleep(.05)
            except BaseException as exc: failure('retry-wait', exc)
        os.close(emergency)
        for sig, handler in previous.items(): signal.signal(sig, handler)
    if reason == 'interrupted':
        raise KeyboardInterrupt(error['message'])
    if reason == 'launch-or-monitor-error':
        raise RuntimeError(str(error))
    return code
