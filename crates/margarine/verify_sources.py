#!/usr/bin/env python3
"""Verify frozen upstream hashes and the exact declared local dispatch patch."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / 'SOURCE.json').read_text())
    for name, expected in manifest['sha256'].items():
        data = (root / name).read_bytes()
        patch = manifest.get('local_patches', {}).get(name)
        if patch:
            if patch['kind'] != 'avx512-v4-to-v4x':
                raise ValueError(f'unknown vendored patch: {name}')
            actual = hashlib.sha256(data).hexdigest()
            if actual != patch['sha256']:
                raise ValueError(f'patched source differs: {name}: {actual} != {patch["sha256"]}')
            data = data.replace(b'v4x', b'v4').replace(b'V4x', b'V4')
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f'vendored upstream source differs: {name}: {actual} != {expected}')
    return manifest


if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'vendor' / 'butteraugli'
    manifest = verify(root)
    print(f"Verified {len(manifest['sha256'])} Butteraugli modules from {manifest['commit']} and exact declared local patches")
