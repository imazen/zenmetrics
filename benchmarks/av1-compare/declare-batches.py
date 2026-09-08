#!/usr/bin/env python3
"""Feed JSONL to canonical declaration in bounded batches; emit one job array.

Preserves the executor's 64KiB request limit. This only adapts declaration I/O;
job identities, capabilities and validation are still produced by `declare`.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('inputs', type=Path)
    args = parser.parse_args()
    first = True
    count = 0
    sys.stdout.write('[')

    def emit(batch):
        nonlocal first, count
        result = subprocess.run([str(args.binary.resolve()), 'declare'],
                                input=batch, stdout=subprocess.PIPE, check=True)
        jobs = json.loads(result.stdout)
        for job in jobs:
            if not first:
                sys.stdout.write(',\n')
            first = False
            json.dump(job, sys.stdout, separators=(',', ':'))
            count += 1
        print(f'declared {count} jobs', file=sys.stderr, flush=True)

    batch = bytearray()
    with args.inputs.open('rb') as source:
        for line in source:
            if len(line) > 65536:
                raise ValueError('one declaration exceeds the executor request bound')
            if len(batch) + len(line) > 65536:
                emit(batch)
                batch.clear()
            batch.extend(line)
    if batch:
        emit(batch)
    sys.stdout.write(']\n')


if __name__ == '__main__':
    main()
