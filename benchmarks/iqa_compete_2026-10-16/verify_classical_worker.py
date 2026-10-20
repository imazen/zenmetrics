#!/usr/bin/env python3
"""Verify a completed, unmodified pinned classical worker export against Git."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from compare_canonical import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('preserve the previous receipt; choose a new output')
    repo = Path(__file__).resolve().parents[2]
    receipt_path = a.build / 'build-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('adapters_applied') is not False:
        raise ValueError('this verifier requires an export without overlays')
    revision = receipt['source_revision']
    tree = a.build / 'work/zenmetrics'
    entries = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', revision])
    files = {}
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        meta, path = entry.split(b'\t', 1)
        mode, kind, expected = meta.decode().split()
        name = path.decode()
        if not (name.startswith(('crates/', 'benchmarks/heaptrack/drivers/cpu_profile/'))
                or name in ['Cargo.toml', 'Cargo.lock', 'ci/sibling-pins.tsv']):
            continue
        target = tree / name
        data = target.readlink().as_posix().encode() if mode == '120000' else target.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if kind != 'blob' or actual != expected:
            raise ValueError('exported tracked input differs: ' + name)
        files[name] = hashlib.sha256(data).hexdigest()
    if not files:
        raise ValueError('no tracked crate inputs verified')
    binary = a.build / 'target/release/iqa-compete'
    if sha(binary) != receipt['binary_sha256']:
        raise ValueError('compiled worker differs from its build receipt')
    inventory = subprocess.check_output([str(binary), 'inventory'])
    value = dict(source_revision=revision, source_files=files,
                 binary_sha256=sha(binary), inventory=json.loads(inventory),
                 inventory_sha256=hashlib.sha256(inventory).hexdigest(),
                 build_receipt_sha256=sha(receipt_path), verifier_sha256=sha(__file__),
                 qualification='All tracked in-tree crate inputs, CPU-profile package files, '
                               'manifests, lock and sibling pins match the named revision. '
                               'This does not independently audit external dependency source '
                               'or establish numerical/performance qualification.')
    with a.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(source_revision=revision, binary_sha256=value['binary_sha256'])))


if __name__ == '__main__':
    main()
