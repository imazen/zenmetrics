#!/usr/bin/env python3
"""Portable Unix lock/timeout bridge; use the host resource wrapper when installed."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import platform
import shutil
from supervision import resource_wrapper, timeout_command
import subprocess
import sys


def resource_executable():
    if os.environ.get('IQA_RUN_HEAVY') or shutil.which('run-heavy'):
        return Path(resource_wrapper())
    wrapper = Path.home() / 'work/claudehints/scripts/run-heavy'
    return wrapper if wrapper.is_file() and os.access(wrapper, os.X_OK) else None


def scoped(command, *, seconds, memory, jobs):
    wrapper = resource_executable()
    if platform.system() == 'Darwin' and wrapper is not None:
        # The macOS RSS guard owns the process group; preserve that containment.
        command = [*timeout_command('Darwin', seconds), *command]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), 'timeout', str(seconds), *command]
    if wrapper is not None:
        command = [str(wrapper), '--mem', memory, '--jobs', str(jobs), '--', *command]
    return command


def locked(command, path):
    return [sys.executable, str(Path(__file__).resolve()), 'lock', str(path), *command]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['lock', 'timeout'])
    p.add_argument('value')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.action == 'lock':
        path = Path(a.value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            raise SystemExit(subprocess.call(a.command, pass_fds=(guard.fileno(),)))
    inherited = tuple(int(v) for v in os.environ.get("IQA_HEAVY_LOCK_FD", "").split(",") if v)
    child = subprocess.Popen(a.command, start_new_session=True, pass_fds=inherited)
    try:
        raise SystemExit(child.wait(timeout=float(a.value)))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise SystemExit(124)


if __name__ == '__main__':
    main()
