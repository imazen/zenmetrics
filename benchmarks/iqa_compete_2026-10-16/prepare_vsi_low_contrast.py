#!/usr/bin/env python3
"""Create integer-only RGB8 probes for the low-contrast VSI finding.

These procedural probes are diagnostic inputs, not additions to the study's
three-content panel. Preserve every output; a populated destination is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path


def pixels(kind, distorted):
    out = bytearray()
    for y in range(64):
        for x in range(64):
            for c, base in enumerate([148, 152, 147]):
                if kind == 'ramp':
                    value = (x + 2 * y + 7 * c) // 16 % 9 - 4
                elif kind == 'blocks':
                    value = ((x // 8) * 3 + (y // 8) * 5 + c) % 7 - 3
                else:
                    value = ((37 * x + 91 * y + x * y + 17 * c) % 9) - 4
                if distorted:
                    # Integer division toward negative infinity, in both the
                    # generator and any reproducing Rust fixture.
                    value = (value + 1) // 2
                    if (x + 3 * y + c) % 17 == 0:
                        value += 1
                out.append(base + value)
    return bytes(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir()
    pairs = []
    for kind in ['ramp', 'blocks', 'texture']:
        row = dict(pair=kind, content='procedural low-contrast RGB8 diagnostic',
                   size_name='64', width=64, height=64)
        for role, distorted in [('reference', False), ('distorted', True)]:
            data = pixels(kind, distorted)
            name = f'{kind}-{role}.rgb'
            (a.output / name).write_bytes(data)
            row[role] = name
            row[role + '_sha256'] = hashlib.sha256(data).hexdigest()
        pairs.append(row)
    (a.output / 'pairs.json').write_text(json.dumps({
        'schema': 'vsi-low-contrast-diagnostic/1',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pairs': pairs}, indent=2) + '\n')


if __name__ == '__main__':
    main()
