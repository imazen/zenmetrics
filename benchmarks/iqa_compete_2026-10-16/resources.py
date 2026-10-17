#!/usr/bin/env python3
"""Portable Unix lock/timeout bridge; use the host resource wrapper when installed."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys


def scoped(command, *, seconds, memory, jobs):
    wrapper = Path.home() / 'work/claudehints/scripts/run-heavy'
    command = [sys.executable, str(Path(__file__).resolve()), 'timeout', str(seconds), *command]
    if wrapper.is_file():
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
            raise SystemExit(subprocess.call(a.command))
    child = subprocess.Popen(a.command, start_new_session=True)
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
