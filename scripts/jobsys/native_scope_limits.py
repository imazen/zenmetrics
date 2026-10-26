#!/usr/bin/env python3
"""Nested memory limits inside the native batch's existing aggregate cgroup."""
import argparse
import os
from pathlib import Path
import time

SCOPE_BYTES = 42 * 1024**3
CELL_RESERVE_BYTES = 11 * 1024**3 // 2
HEADROOM_BYTES = 1024**3


def scope_path(runtime):
    relative = (runtime / 'scope.cgroup').read_text().strip().split('::', 1)[1]
    path = Path('/sys/fs/cgroup') / relative.lstrip('/')
    if (path / 'memory.max').read_text().strip() != str(SCOPE_BYTES):
        raise ValueError('expected original 42GiB aggregate cap')
    if (path / 'memory.swap.max').read_text().strip() != '0':
        raise ValueError('expected original no-swap scope')
    return path


def member(pid, scope):
    try:
        relative = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::', 1)[1]
    except FileNotFoundError:
        return False
    path = Path('/sys/fs/cgroup') / relative.lstrip('/')
    return path == scope or scope in path.parents


def prepare(scope):
    """Move direct members to a leaf before enabling the memory controller."""
    science = scope / 'science'
    science.mkdir(exist_ok=True)
    # Children inherit their parent's leaf. Re-read to cover a concurrent fork
    # before its parent moved; never change the outer cap or kill a process.
    deadline = time.monotonic() + 10
    while (direct := (scope / 'cgroup.procs').read_text().split()):
        for pid in direct:
            try:
                (science / 'cgroup.procs').write_text(pid)
            except ProcessLookupError:
                pass
        if time.monotonic() > deadline:
            raise TimeoutError('direct scope members did not settle in science leaf')
    (scope / 'cgroup.subtree_control').write_text('+memory')
    if 'memory' not in (scope / 'cgroup.subtree_control').read_text().split():
        raise ValueError('nested memory controller not enabled')
    return science


def admission_budget(limit, slots, batch_alive, science_current, aggregate_current):
    if limit <= 0 or slots <= 0:
        raise ValueError('positive postprocessor limit and science slots required')
    reserved = max(science_current, slots * CELL_RESERVE_BYTES if batch_alive else 0)
    return (max(reserved, aggregate_current) + limit + HEADROOM_BYTES <= SCOPE_BYTES,
            reserved)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    a = p.parse_args()
    if os.geteuid() != 0:
        p.error('scope preparation requires sudo')
    scope = scope_path(a.runtime)
    if not member(os.getpid(), scope):
        raise ValueError('prepare must run inside the admitted native scope')
    prepare(scope)
    print(f'native scope prepared: {scope}; aggregate cap unchanged', flush=True)


if __name__ == '__main__':
    main()
