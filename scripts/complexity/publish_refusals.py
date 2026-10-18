#!/usr/bin/env python3
"""Publish a complete 126-name, six-cell execution inventory without local paths."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root/'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    enum = source.split('pub enum MetricKind {', 1)[1].split('\n}', 1)[0]
    names = re.findall(r'#\[value\(name = "([^"]+)"\)\]', enum)
    expected = {(m, s, t) for m in names for s in ('64x64', '1024x1024', '3840x2160') for t in ('1', '8')}
    rows = {}
    for path in args.input:
        for row in csv.DictReader(path.open(), delimiter='\t'):
            key = row['metric'], row['shape'], row['threads']
            if key in rows:
                raise ValueError(f'duplicate cell {key}')
            rows[key] = row
    if set(rows) != expected:
        raise ValueError(f'incomplete inventory: missing={sorted(expected-set(rows))}, extra={sorted(set(rows)-expected)}')
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ['metric', 'shape', 'threads', 'status', 'exit_code', 'cause', 'peak_rss_gib', 'min_available_mib', 'peak_load']
    counts = Counter()
    for shape in ('64x64', '1024x1024', '3840x2160'):
        for threads in ('1', '8'):
            output = args.out/f'refusals-{shape}-{threads}.tsv'
            with output.open('x') as target:
                writer = csv.DictWriter(target, fields, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                for name in names:
                    row = rows[name, shape, threads]
                    result = {k:row[k] for k in fields[:5]}
                    text = ''
                    if row['log']:
                        text = (args.logs/row['log']).read_text()
                    detail = row['detail']
                    if row['status'] == 'panic':
                        lines = text.splitlines()
                        index = next(i for i, line in enumerate(lines) if 'panicked at' in line)
                        detail = lines[index+1]
                    if row['status'] == 'timeout':
                        detail = 'One-call process exceeded the 120-second timeout; no timing qualified'
                    detail = re.sub(r'/home/[^/\s]+/tmp/aicfix-models/', 'MODEL_DIR/', detail)
                    if '/home/' in detail:
                        raise ValueError(f'unexpected local path in {name}: {detail}')
                    result['cause'] = detail
                    resource = re.search(r'peak-RSS ([0-9.]+)GiB \| min-avail ([0-9]+)MiB \| peak-load ([0-9.]+)', text)
                    for key, value in zip(fields[6:], resource.groups() if resource else ('', '', '')):
                        result[key] = value
                    if row['status'] != 'skipped by owner' and not resource:
                        raise ValueError(f'missing resource receipt for {name}/{shape}/{threads}')
                    writer.writerow(result)
                    counts[row['status']] += 1
            if output.stat().st_size > 30_000:
                raise ValueError(f'generated shard exceeds 30KB: {output}')
    print(f'{len(rows)}/{len(expected)} cells:', dict(counts))


if __name__ == '__main__':
    main()
