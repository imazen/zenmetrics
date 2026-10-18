#!/usr/bin/env python3
"""Print incomplete metrics; existing error rows remain terminal attempts."""
import argparse
from pathlib import Path


def missing(path, metrics, threads=None, repeats=3, shape='3840x2160'):
    counts = dict.fromkeys(metrics, 0)
    errors = set()
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split('\t')
            if len(parts) != 15 or parts[0] not in counts or parts[1] != shape:
                continue
            if threads is not None and parts[2] != str(threads):
                continue
            if parts[5] == 'error':
                errors.add(parts[0])
            else:
                counts[parts[0]] += 1
    return [m for m in metrics if m not in errors and counts[m] < repeats]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--shape', default='3840x2160')
    args = parser.parse_args()
    print(','.join(missing(args.out, args.metrics.split(','), args.threads,
                           args.repeats, args.shape)))


if __name__ == '__main__':
    main()
