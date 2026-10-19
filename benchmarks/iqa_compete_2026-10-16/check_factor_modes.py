#!/usr/bin/env python3
"""Reject automatic/fixed source confusion before any Rust probe starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['canonical', 'manifest', 'binary', 'build-receipt', 'reference', 'distorted', 'evidence']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    original = json.loads(a.canonical.read_text())
    assert original['factor_mode'] == 'automatic'
    a.evidence.mkdir()
    root = a.evidence.resolve()
    checks = []
    for name, key, value in [('wrong-source-mode', 'factor_mode', 'fixed'),
                             ('unknown-mode', 'factor_mode', 'unrecognized'),
                             ('wrong-resolved-factor', 'resolved_factor', original['resolved_factor'] + 1)]:
        altered = dict(original)
        altered[key] = value
        path = root / (name + '.json')
        path.write_text(json.dumps(altered, indent=2) + '\n')
        output = root / (name + '-must-remain-absent')
        command = [sys.executable, '-B', str(Path(__file__).with_name('probe_octave_parity.py')),
                   '--canonical', str(path), '--manifest', str(a.manifest.resolve()),
                   '--binary', str(a.binary.resolve()), '--build-receipt', str(a.build_receipt.resolve()),
                   '--reference', str(a.reference.resolve()), '--distorted', str(a.distorted.resolve()),
                   '--threads', str(original['threads_requested']), '--evidence', str(output)]
        with (root / (name + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 2 and not output.exists(), name
        checks.append(name + ': rejected before output creation or Rust execution')
    (root / 'results.json').write_text(json.dumps({'controls': checks, 'count': len(checks)}, indent=2) + '\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
