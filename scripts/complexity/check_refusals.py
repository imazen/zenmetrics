#!/usr/bin/env python3
"""Real-process regression for the complexity driver's refused-input outcome.
Caller builds with NR enabled and applies the shared resource wrapper/lock.
"""
import argparse
import os
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bin', required=True)
a = p.parse_args()
for metric, phrase in [('hdrvdp', 'nits'), ('hdrvdp3', 'absolute luminance'), ('niqe', '96')]:
    result = subprocess.run([a.bin, '64x64', metric, '1'], text=True,
                            capture_output=True, timeout=30,
                            env=dict(os.environ, RAYON_NUM_THREADS='1'))
    assert result.returncode == 2, (metric, result.returncode, result.stderr)
    assert 'REFUSED ' + metric in result.stderr, (metric, result.stderr)
    assert phrase in result.stderr, (metric, result.stderr)
    assert 'panicked' not in result.stderr, (metric, result.stderr)
    print(metric, 'explicit refusal, no panic')
