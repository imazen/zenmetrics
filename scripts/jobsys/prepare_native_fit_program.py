#!/usr/bin/env python3
"""Match native executor permissions to the image without changing pinned bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(program):
    entry = program / 'fit_cell_exec.py'
    meta = json.loads((program / 'build_meta.json').read_text())
    if entry.is_symlink() or not entry.is_file():
        raise ValueError('native executor must be a regular program file')
    actual = hashlib.sha256(entry.read_bytes()).hexdigest()
    if actual != meta['files']['fit_cell_exec.py']:
        raise ValueError('native executor differs from the pinned program inventory')
    old = entry.stat().st_mode & 0o777
    entry.chmod(0o755)
    return dict(executor_sha256=actual, previous_mode=oct(old), mode='0o755',
                program_bytes_unchanged=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--program', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().program)))


if __name__ == '__main__':
    main()
