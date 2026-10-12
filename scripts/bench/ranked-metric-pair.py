#!/usr/bin/env python3
"""Prepare an exploratory external A/B plan, or check exact wall score bits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def score_bits(directory):
    text = (Path(directory) / 'stdout.txt').read_text()
    matches = re.findall(r'bits ([0-9a-f]{16})\)', text)
    if len(matches) != 1:
        raise ValueError(f'expected exactly one score bit pattern in {directory}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('baseline')
    check.add_argument('candidate')
    plan = sub.add_parser('plan')
    for name in ['baseline', 'candidate', 'baseline_revision', 'candidate_revision', 'output']:
        plan.add_argument(name)
    plan.add_argument('--threads', type=int, default=1)
    plan.add_argument('--cpus', default='2')
    args = parser.parse_args()
    if args.command == 'check':
        before, after = score_bits(args.baseline), score_bits(args.candidate)
        if before != after:
            raise ValueError(f'score bits changed: {before} != {after}')
        print(f'exact score bits: {before}')
        return
    if args.threads < 1:
        raise ValueError('threads must be positive')
    def arm(label, path, revision):
        binary = Path(path).resolve(strict=True)
        return dict(id=label, program=str(binary),
                    args=['${width}', '${height}', '${repetitions}'],
                    revision=revision + ';sha256=' + hashlib.sha256(binary.read_bytes()).hexdigest(),
                    timeout_seconds=60)
    value = dict(
        schema=1, baseline=arm('before', args.baseline, args.baseline_revision),
        candidates=[arm('after', args.candidate, args.candidate_revision)],
        env=dict(RAYON_NUM_THREADS=str(args.threads), ZENBENCH_CPU_SET=args.cpus),
        record_env=['TMPDIR'],
        cases=[dict(id='3355x2516', fields=dict(width='3355', height='2516', repetitions='5'))],
        metric=dict(extract=dict(kind='regex', pattern=r': ([0-9.]+) ms/call', capture=1),
                    unit='ms', expected=1, divide_by=1),
        check=dict(kind='command', program=sys.executable,
                   args=[str(Path(__file__).resolve()), 'check', '${baseline}', '${candidate}'],
                   timeout_seconds=10),
        warmup='No excluded warmup; five complete metric calls, input generation outside internal timing',
        blocks=4, aa=True, deadline_seconds=180)
    with Path(args.output).open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
