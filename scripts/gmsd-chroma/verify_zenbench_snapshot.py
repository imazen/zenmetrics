#!/usr/bin/env python3
"""Verify the externally frozen imazen timing harness before compilation."""
import hashlib
import json
from pathlib import Path

root = Path('/var/tmp/gmsd-chroma')
manifest = root/'zenbench_snapshot.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == '394a18b62ba4a67b6782548d781ed9bf51c57663bb2e368d336291f3c1064da6'
data = json.loads(manifest.read_text())
for record in data['files']:
    p = root/'zenbench-snapshot'/record['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256'], str(p)
print('zenbench_snapshot_sha256', hashlib.sha256(manifest.read_bytes()).hexdigest())
print('zenbench_snapshot_files', len(data['files']))
