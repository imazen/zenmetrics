#!/usr/bin/env python3
"""Verify a program-only image refresh against its local archive and immutable base."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def inspect(ref):
    return json.loads(subprocess.check_output(['docker', 'image', 'inspect', ref], text=True))[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--program', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('use a fresh verification receipt')
    base, image = inspect(args.base), inspect(args.image)
    if image['RootFS']['Layers'][:len(base['RootFS']['Layers'])] != base['RootFS']['Layers']:
        raise ValueError('base layers differ')
    for key in ('Entrypoint', 'Cmd', 'User', 'WorkingDir'):
        if image['Config'].get(key) != base['Config'].get(key):
            raise ValueError(f'runtime {key} differs')
    env = lambda obj: dict(s.split('=', 1) for s in obj['Config']['Env'] if '=' in s)
    current, previous = env(image), env(base)
    changed = sorted(k for k in current.keys() | previous.keys() if current.get(k) != previous.get(k))
    pin = hashlib.sha256(args.program.read_bytes()).hexdigest()
    if changed != ['ZEN_FIT_PROGRAM_SHA'] or current['ZEN_FIT_PROGRAM_SHA'] != pin:
        raise ValueError('unexpected program/environment binding')
    with tarfile.open(args.program) as archive:
        metadata = json.load(archive.extractfile('build_meta.json'))
    code = """
import sys,json,hashlib
from pathlib import Path
expected=json.load(sys.stdin)
root=Path('/opt/fleet-fits/program')
if json.loads((root/'build_meta.json').read_text()) != expected:
    raise ValueError('image program metadata differs from the supplied archive')
for name,pin in expected['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != pin:
        raise ValueError('image file differs: '+name)
executor=Path('/usr/local/bin/fit-cell-exec')
if executor.read_bytes() != (root/'fit_cell_exec.py').read_bytes() or not executor.stat().st_mode & 0o111:
    raise ValueError('launched executor differs from pinned program or is not executable')
worker=hashlib.sha256(Path('/usr/local/bin/zenfleet-worker').read_bytes()).hexdigest()
if worker != expected['worker_sha256']:
    raise ValueError('image worker differs')
print(json.dumps(dict(verified_files=len(expected['files']),worker_sha256=worker,executor_matches_program=True)))
"""
    immutable = next(ref for ref in image['RepoDigests'] if ref.startswith('ghcr.io/imazen/zenfleet-worker@sha256:'))
    proof = json.loads(subprocess.check_output([
        'docker', 'run', '--rm', '-i', '--network', 'none', '--memory', '256m',
        '--memory-swap', '256m', '--cpus', '1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '--entrypoint', 'python3', immutable, '-c', code], input=json.dumps(metadata), text=True))
    record = dict(observed_at=datetime.now(timezone.utc).isoformat(), base=args.base,
                  image=immutable, program_sha256=pin, source_commit=metadata['source_commit'],
                  base_layers_preserved=True, runtime_config_preserved=True,
                  changed_environment_keys=changed, binary_sha256=metadata['binary_sha256'], **proof)
    with args.out.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
