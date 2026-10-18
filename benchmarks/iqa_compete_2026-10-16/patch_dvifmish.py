#!/usr/bin/env python3
"""Resolve the unpublished dvifmish dependency only in an owned benchmark snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

RELATIVE = 'benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml'
REVISION = '74ec5eeae118ead266c7b66b56e56cb5f3bef372'


def apply(snapshot, original, receipt):
    manifest = snapshot / RELATIVE
    if manifest.samefile(original / RELATIVE) or (snapshot / 'Cargo.lock').samefile(original / 'Cargo.lock'):
        raise ValueError('snapshot aliases the source checkout')
    before = {name:(snapshot / name).read_bytes() for name in (RELATIVE, 'Cargo.lock')}
    source = before[RELATIVE].decode()
    if source.count('[features]\n') != 1 or source.count('[dependencies]\n') != 1 or 'dvifmish' in source:
        raise ValueError('unexpected benchmark manifest; dependency patch refused')
    source = source.replace('[features]\n', '[features]\ncompete-dvifmish = ["compete", "dep:dvifmish"]\n')
    source = source.replace('[dependencies]\n', '[dependencies]\ndvifmish = { git = "https://github.com/imazen/dvifmish", rev = "' + REVISION + '", optional = true, default-features = false }\n')
    manifest.write_text(source)
    # Root lock identity was checked before this private overlay. Never write it back.
    subprocess.run(['cargo', 'metadata', '--format-version', '1'], cwd=snapshot,
                   stdout=subprocess.DEVNULL, check=True)
    h = lambda data:hashlib.sha256(data).hexdigest()
    value = dict(adapter_sha256=h(Path(__file__).read_bytes()), dependency_revision=REVISION,
        files={name:dict(original_sha256=h(data), adapted_sha256=h((snapshot / name).read_bytes()))
               for name,data in before.items()},
        scope='Private benchmark snapshot only; root workspace resolution and root Cargo.lock unchanged; no publication')
    if receipt.exists():
        backup = receipt.with_name(receipt.stem + '-' + h(receipt.read_bytes())[:16] + '.bak.json')
        if not backup.exists():
            receipt.rename(backup)
    receipt.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('original', type=Path)
    p.add_argument('receipt', type=Path)
    a = p.parse_args()
    apply(a.snapshot, a.original, a.receipt)


if __name__ == '__main__':
    main()
