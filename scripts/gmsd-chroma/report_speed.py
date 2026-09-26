#!/usr/bin/env python3
"""Fit alpha + beta*pixels to four zenbench mean latencies, retaining residuals.

Ordinary least squares uses NumPy, not a custom correlation/statistics owner.
No significance or speedup claim is inferred from this descriptive fit.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--busy', action='store_true', help='known concurrent host load')
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    cells = []
    contaminated = args.busy
    for g in data['comparisons']:
        m = re.fullmatch(r'rgb8_(64|256|1024|4096)_t(1|8)', g['group_name'])
        assert m and len(g['samples']) >= 10
        contaminated |= any(s.get('gate_clean') is not True for s in g['samples'])
        for b in g['benchmarks']:
            cells.append(dict(size=int(m[1]), threads=int(m[2]), arm=b['name'],
                              mean_ns=b['summary']['mean'], rounds=g['completed_rounds']))
    assert len(cells) == 24
    fits = []
    for arm in ['gmsd', 'mdsi', 'ms_gmsdc']:
        for threads in [1, 8]:
            selected = sorted((r for r in cells if r['arm'] == arm and r['threads'] == threads),
                              key=lambda r: r['size'])
            assert [r['size'] for r in selected] == [64, 256, 1024, 4096]
            x = np.array([r['size']**2 for r in selected], dtype=float)
            y = np.array([r['mean_ns'] for r in selected], dtype=float)
            alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(4), x]), y, rcond=None)[0]
            residuals = y-(alpha+beta*x)
            fits.append(dict(arm=arm, threads=threads, alpha_ns=float(alpha),
                             beta_ns_per_pixel=float(beta), residual_ns=residuals.tolist()))
    result = dict(label='CONTENDED' if contaminated else 'PRE_ROUND_GATES_CLEAN',
                  note='Gate checks do not monitor load inside each timed body.',
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  cells=cells, fits=fits)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
