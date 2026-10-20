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
    p.add_argument('--additional', action='append', type=Path, default=[],
                   help='Completed fixed-stage successor linked to the original plan')
    a = p.parse_args()
    rows = []
    for summary in sorted(a.results.glob('*/summary.json')):
        row = json.loads(summary.read_text())
        paths = sorted(summary.parent.glob('block-*/result.json'))
        plan_path=a.results/'plan.json'
        plan=json.loads(plan_path.read_text())
        case=next(case for case in plan['cases'] if case['id']==row['case'])
        summaries=[summary]
        for root in a.additional:
            successor=json.loads((root/'plan.json').read_text())
            if successor.get('preceding_plan_sha256')!=sha(plan_path):
                raise ValueError('additional stage is not linked to the original plan')
            other=next(c for c in successor['cases'] if c['id']==row['case'])
            if other!=case or successor['binary']!=plan['binary'] or successor['rust_source']!=plan['rust_source']:
                raise ValueError('additional stage changes the measured contract')
            path=root/row['case']/'summary.json'
            if not path.exists():
                raise ValueError('additional fixed stage is not complete')
            summaries.append(path)
            paths.extend(sorted(path.parent.glob('block-*/result.json')))
        if len(paths)!=len(set(path.resolve() for path in paths)):
            raise ValueError('duplicate acquired block')
        blocks = [json.loads(path.read_text()) for path in paths]
        if any(json.loads(path.read_text())['ratio'] is None for path in summaries) or len(blocks) < 20:
            raise ValueError('completed twenty-block matched comparison required')
        if any(block.get('errors') or block['passed'] is False for block in blocks):
            raise ValueError('failed block in completed comparison')
        gates = [s['gate_clean'] for b in blocks
                 for s in b['arms']['rust']['extra'].get('zenbench_samples', [])]
        result = interval([b['rust_ns']/b['canonical_ns'] for b in blocks])
        clean=[block for block in blocks if len(block['arms']['rust']['extra'].get('zenbench_samples', []))>=10
               and all(sample['gate_clean'] for sample in block['arms']['rust']['extra']['zenbench_samples'])]
        clean_result=interval([block['rust_ns']/block['canonical_ns'] for block in clean]) if len(clean)>=20 else None
        rows.append(dict(case=row['case'], blocks=len(blocks), **result,
                         width=row['width'], height=row['height'], threads=row['threads'],
                         rust_source=row['rust_source'], plan_sha256=row['plan_sha256'],
                         summary_sha256=sha(summary),
                         stage_summary_sha256=[sha(path) for path in summaries],
                         block_sha256=[sha(path) for path in paths],
                         native_gate_observations=len(gates),
                         native_dirty_gate_observations=sum(not gate for gate in gates),
                         native_gates_clean=bool(gates) and all(gates),
                         whole_block_clean_count=len(clean),
                         clean_block_diagnostic=clean_result,
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
