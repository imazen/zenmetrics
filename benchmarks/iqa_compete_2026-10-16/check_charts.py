#!/usr/bin/env python3
"""Check rendered SVGs and exact compact-summary projections against raw rows."""
import argparse
import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET

P = pathlib.Path
KEY = ['host', 'os', 'metric_name', 'impl', 'device', 'threads', 'mode', 'size_name']


def check(raw, directory, expected_rows, expected_svgs, bar_label=None):
    contents = raw.read_bytes()
    rows = [json.loads(line) for line in contents.decode().splitlines()]
    original = {tuple(row[k] for k in KEY): row for row in rows}
    assert len(original) == len(rows) == expected_rows, 'Raw row count/identity mismatch'
    summarized = []
    for path in sorted(directory.glob('*.json')):
        values = json.loads(path.read_text())
        assert isinstance(values, list), 'Expected compact row arrays: ' + str(path)
        for row in values:
            source = original[tuple(row[k] for k in KEY)]
            assert all(value == source.get(key) for key, value in row.items()), 'Changed summary: ' + str(path)
        summarized.extend(values)
    keys = [tuple(row[k] for k in KEY) for row in summarized]
    assert len(keys) == len(set(keys)) == expected_rows, 'Summary row count/identity mismatch'
    svgs = sorted(directory.glob('*.svg'))
    assert len(svgs) == expected_svgs, 'SVG count mismatch'
    bars = [path for path in svgs if path.name.startswith(('bars-', 'metric-bars-'))]
    for path in svgs:
        ET.parse(path)
    if bar_label:
        assert bars, 'No bar panels'
        for path in bars:
            assert bar_label in ''.join(ET.parse(path).getroot().itertext()), 'Missing bar label: ' + str(path)
    return dict(raw=str(raw.resolve()), raw_sha256=hashlib.sha256(contents).hexdigest(),
                rows_unchanged=len(keys), parseable_svgs=len(svgs),
                bar_panels=len(bars), required_bar_label=bar_label,
                chart_directory=str(directory.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=P)
    parser.add_argument('directory', type=P)
    parser.add_argument('--expected-rows', type=int, required=True)
    parser.add_argument('--expected-svgs', type=int, required=True)
    parser.add_argument('--bar-label')
    args = parser.parse_args()
    print(json.dumps(check(args.raw, args.directory, args.expected_rows,
                           args.expected_svgs, args.bar_label), indent=2))


if __name__ == '__main__':
    main()
