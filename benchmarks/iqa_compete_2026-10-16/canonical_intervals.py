#!/usr/bin/env python3
"""Bootstrap alternating-block ratios without pooling inner warm samples.

These descriptive intervals quantify acquired-block variability. They do not
establish a complete runtime-closure audit or a cross-host regression decision.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from compare_canonical import sha


def interval(ratios, draws=20_000):
    if len(ratios) < 20 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('at least twenty positive finite block ratios required')
    rng = random.Random(20261020)
    estimates = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                       for _ in range(draws))
    center = statistics.median(ratios)
    lo, hi = estimates[int(draws*.025)], estimates[int(draws*.975)-1]
    return dict(ratio=center, lower_95=lo, upper_95=hi,
                relative_halfwidth=max(center-lo, hi-center)/center,
                precision_within_three_percent=max(center-lo, hi-center)/center <= .03)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    rows = []
    for summary in sorted(a.results.glob('*/summary.json')):
        row = json.loads(summary.read_text())
        paths = sorted(summary.parent.glob('block-*/result.json'))
        blocks = [json.loads(path.read_text()) for path in paths]
        if row['ratio'] is None or len(blocks) < 20:
            raise ValueError('completed twenty-block matched comparison required')
        gates = [s['gate_clean'] for b in blocks
                 for s in b['arms']['rust']['extra'].get('zenbench_samples', [])]
        result = interval([b['rust_ns']/b['canonical_ns'] for b in blocks])
        rows.append(dict(case=row['case'], blocks=len(blocks), **result,
                         width=row['width'], height=row['height'], threads=row['threads'],
                         rust_source=row['rust_source'], plan_sha256=row['plan_sha256'],
                         summary_sha256=sha(summary),
                         block_sha256=[sha(path) for path in paths],
                         native_gate_observations=len(gates),
                         native_dirty_gate_observations=sum(not gate for gate in gates),
                         native_gates_clean=bool(gates) and all(gates),
                         regression_qualified=False))
    if not rows:
        p.error('no completed comparisons')
    with a.output.open('x') as stream:
        json.dump(dict(method='20,000 bootstrap resamples of complete alternating-block ratios; median estimator; seed 20261020',
                       qualification='descriptive; runtime closure and canonical contamination not independently audited',
                       rows=rows), stream, indent=2)
        stream.write('\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
