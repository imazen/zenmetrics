#!/usr/bin/env python3
"""Read acquisition progress without starting workers or qualifying results."""
import argparse
import hashlib
import json
from pathlib import Path


def status(job):
    root = Path(job['output'])
    plan_path = root / 'plan.json' if (root / 'plan.json').exists() else Path(job['plan'])
    plan = json.loads(plan_path.read_text())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--blocks', type=int, default=3)
    options.add_argument('--samples', type=int, default=5)
    options.add_argument('--metrics')
    args = options.parse_args(job.get('options', []))
    names = set(args.metrics.split(',')) if args.metrics else None
    cases = [case for case in plan['cases'] if names is None or case['metric'] in names]
    complete = blocks = errors = findings = 0
    pending = []
    for case in cases:
        directory = root / case['id']
        acquired = len(list(directory.glob('block-*/result.json')))
        blocks += acquired
        summary = directory / 'summary.json'
        if not summary.exists():
            pending.append(dict(case=case['id'], acquired_blocks=acquired))
            continue
        row = json.loads(summary.read_text())
        if row['plan_sha256'] != digest or row['rust_source'] != plan['rust_source']:
            raise ValueError('summary identity differs: ' + str(summary))
        if acquired != args.blocks:
            raise ValueError('summary block count differs: ' + str(summary))
        complete += 1
        results = [json.loads(path.read_text())
                   for path in directory.glob('block-*/result.json')]
        errors += any(block.get('errors') for block in results)
        findings += any(block['passed'] is False and not block.get('errors')
                        for block in results)
    return dict(output=str(root), completed_cases=complete, expected_cases=len(cases),
                acquired_blocks=blocks, expected_blocks=len(cases) * args.blocks,
                process_error_cases=errors, numerical_finding_cases=findings,
                pending=pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    rows = [status(job) for job in json.loads(args.manifest.read_text())['jobs']]
    totals = {key: sum(row[key] for row in rows) for key in (
        'completed_cases', 'expected_cases', 'acquired_blocks', 'expected_blocks',
        'process_error_cases', 'numerical_finding_cases')}
    print(json.dumps(dict(qualification='acquisition progress only', **totals)))
    if args.details:
        for row in rows:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
