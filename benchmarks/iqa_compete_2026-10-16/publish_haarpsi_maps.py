#!/usr/bin/env python3
"""Publish small HaarPSI map observation shards; retain raw planes externally."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from octave_primary import sha


def tsv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode()
    assert len(data) <= 30000
    path.write_bytes(data)
    return dict(path=path.name, sha256=sha(path), bytes=len(data), rows=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.panel = args.panel.resolve(strict=True)
    done = json.loads((args.panel / 'completion.json').read_text())
    assert len(done['cases']) == 24 and done['bound'] == 'UNBOUND'
    args.output.mkdir()
    rows, pixels = [], []
    for case in done['cases']:
        directory = args.panel / f"{case['pair']}-{case['size']}-t{case['threads']}"
        for name, digest in case['files'].items():
            assert sha(directory / name) == digest
        row = {key: case[key] for key in ['pair', 'size', 'threads', 'width', 'height',
                                         'finite_pixels', 'matching_nonfinite_pixels',
                                         'nonfinite_mismatches', 'max_abs_delta', 'mean_abs_delta', 'bound']}
        row.update(qualified_ratio=False, raw_directory=str(directory),
                   raw_results_sha256=sha(directory / 'results.json'))
        rows.append(row)
        if case['nonfinite_mismatches']:
            h, w = case['height'], case['width']
            similarities = np.memmap(directory / 'similarities.f64', dtype='<f8', mode='r', shape=(3, h, w))
            weights = np.memmap(directory / 'weights.f64', dtype='<f8', mode='r', shape=(3, h, w))
            ours = np.memmap(directory / 'rust.f32', dtype='<f4', mode='r', shape=(h, w))
            with np.errstate(invalid='ignore', divide='ignore'):
                canonical = (similarities * weights).sum(axis=0) / weights.sum(axis=0)
            finite = np.isfinite(canonical) & np.isfinite(ours)
            matching = (np.isnan(canonical) & np.isnan(ours)) | (np.isinf(canonical) & (canonical == ours))
            positions = np.argwhere(~finite & ~matching)
            assert len(positions) == case['nonfinite_mismatches']
            for y, x in positions:
                pixels.append(dict(pair=case['pair'], size=case['size'], threads=case['threads'],
                                   x=int(x), y=int(y), canonical=float(canonical[y, x]), ours=str(float(ours[y, x])),
                                   primary_weight_0=float(weights[0, y, x]), primary_weight_1=float(weights[1, y, x]),
                                   primary_weight_2=float(weights[2, y, x]), bound='UNBOUND'))
    shards = []
    for pair in ['pairA', 'pairB', 'pairC']:
        selected = [row for row in rows if row['pair'] == pair]
        assert len(selected) == 8
        shards.append(tsv(args.output / ('haarpsi-map-' + pair + '.tsv'), selected))
    if pixels:
        shards.append(tsv(args.output / 'haarpsi-map-nonfinite.tsv', pixels))
    index = dict(cases=len(rows), cells_with_nonfinite_mismatches=sum(r['nonfinite_mismatches'] > 0 for r in rows),
                 max_finite_pixel_abs_delta=max(r['max_abs_delta'] for r in rows), bound='UNBOUND',
                 map_scope=done['map_scope'], raw_panel=str(args.panel),
                 completion_sha256=sha(args.panel / 'completion.json'),
                 build_receipt_sha256=sha(args.panel / 'build-receipt.json'),
                 publisher_sha256=sha(Path(__file__)), shards=shards, qualified_timing_cells=0)
    (args.output / 'haarpsi-maps.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index, indent=2))


if __name__ == '__main__':
    main()
