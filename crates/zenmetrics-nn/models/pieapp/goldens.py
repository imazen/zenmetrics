#!/usr/bin/env python3
"""Validate complete supported ladder coverage and format parity goldens."""
import argparse
import csv
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pairs', type=Path)
    p.add_argument('oracle', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    with a.pairs.open() as f:
        pairs = list(csv.DictReader(f, delimiter='\t'))
    with a.oracle.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    expected = {(r['ref_path'], r['dist_path']) for r in pairs}
    assert len(expected) == len(pairs) == len(rows) == 24
    assert expected == {(r['ref_path'], r['dist_path']) for r in rows}
    with a.out.open('x') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['image', 'size', 'pieapp'])
        for row in rows:
            image, size, suffix = Path(row['ref_path']).stem.split('_')
            assert suffix == 'ref' and math.isfinite(float(row['pieapp']))
            writer.writerow([image, size, row['pieapp']])


if __name__ == '__main__':
    main()
