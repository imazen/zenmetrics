#!/usr/bin/env python3
"""Copy the three requested exports to dev and tower, retaining old bytes.

Caller supplies shared heavy admission. A receipt is written only after all
three copies on both destinations and the unchanged sources agree by SHA256.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

from canonical_run import sha


FILES = ('metrics_cropped.canonical.csv', 'metrics_cropped.zenmetrics-extra.csv',
         'provenance.canonical.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('retain existing receipt; supply a new path')
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest['output']).parent.resolve(strict=True)
    relative = root.relative_to('/mnt/v/output')
    tower_mount = Path('/mnt/tower')
    if not tower_mount.is_mount():
        raise ValueError('tower output mount is not mounted')
    tower = tower_mount/'output'/relative
    sources = [root/name for name in FILES]
    if any(path.is_symlink() or not path.is_file() for path in sources):
        raise ValueError('all three regular export files are required')
    expected = {path.name: sha(path) for path in sources}
    provenance = json.loads(sources[-1].read_text())
    for name in FILES[:2]:
        if provenance['exports'][name]['sha256'] != expected[name]:
            raise ValueError('export differs from provenance: '+name)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    tower.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ssh', 'dev', 'mkdir -p '+shlex.quote(str(root))], check=True)
    commands = []
    for target in ('dev:'+str(root)+'/', str(tower)+'/'):
        command = ['rsync', '-rlt', '--checksum', '--backup', '--suffix=.bak-'+stamp,
                   *(str(path) for path in sources), target]
        subprocess.run(command, check=True)
        commands.append(command)
    command = 'sha256sum '+' '.join(shlex.quote(str(path)) for path in sources)
    remote = subprocess.run(['ssh', 'dev', command], check=True, capture_output=True, text=True)
    observed_dev = {}
    for line in remote.stdout.splitlines():
        digest, path = line.split(maxsplit=1)
        name = Path(path.strip()).name
        if name in observed_dev:
            raise ValueError('duplicate remote export name: '+name)
        observed_dev[name] = digest
    observed_tower = {name: sha(tower/name) for name in FILES}
    observed_source = {name: sha(root/name) for name in FILES}
    if not expected == observed_dev == observed_tower == observed_source:
        raise ValueError('mirror SHA256 mismatch or changed source during transfer')
    receipt = dict(label=args.label, recorded_at=datetime.now(timezone.utc).isoformat(),
        source=str(root), dev='dev:'+str(root), tower=str(tower), sha256=expected,
        dev_sha256=observed_dev, tower_sha256=observed_tower, commands=commands,
        backup_suffix='.bak-'+stamp, manifest_sha256=sha(args.manifest), worker_sha256=sha(__file__))
    with args.receipt.open('x') as target:
        target.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
