#!/usr/bin/env python3
"""Check the frozen Butteraugli modules against their recorded source hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'vendor' / 'butteraugli'
manifest = json.loads((root/'SOURCE.json').read_text())
for name, expected in manifest['sha256'].items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'vendored source differs: {name}: {actual} != {expected}')
print(f"Verified {len(manifest['sha256'])} Butteraugli modules from {manifest['commit']}")
