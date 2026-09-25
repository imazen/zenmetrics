#!/usr/bin/env python3
"""Reject stale binaries or source edits since the recorded example build."""
import json
import sys
from pathlib import Path

from build_examples import ROOT, sha

record = ROOT/'build_examples_latest.json'
manifest = json.loads(record.read_text())
for filename,digest in manifest['source_sha256'].items():
    assert sha(Path(filename)) == digest, ('source changed since build',filename)
for name in sys.argv[1:]:
    assert sha(ROOT/'target/release/examples'/name) == manifest['binary_sha256'][name]
print('verified_build_manifest_sha256',sha(record))
