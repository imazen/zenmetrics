#!/usr/bin/env python3
"""Procedural RGB controls for the VIFp luma coefficient precision finding.

Only input pixels are generated here; original LIVE source supplies scores.
These controls are separate from the required photographic parity grid.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args(); a.output.mkdir()
    rows = []
    for seed in range(12):
        reference, distorted = bytearray(), bytearray()
        for y in range(64):
            for x in range(64):
                for c in range(3):
                    value = 80 + (x*(17+3*c+seed) + y*(29+5*c) + c*41 + x*y*(3+seed)) % 96
                    reference.append(value)
                    distorted.append(value + ((x//8+y//8+c+seed) % 5-2)*4)
        row = dict(pair=f'control{seed}', size_name='64', width=64, height=64, seed=seed,
                   content='procedural luma precision control; not photographic coverage')
        for role, data in [('reference', reference), ('distorted', distorted)]:
            name = f'control{seed}-{role}.rgb'; (a.output / name).write_bytes(data)
            row[role] = name; row[role+'_sha256'] = hashlib.sha256(data).hexdigest()
        rows.append(row)
    (a.output / 'pairs.json').write_text(json.dumps(dict(purpose=__doc__, producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), pairs=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
