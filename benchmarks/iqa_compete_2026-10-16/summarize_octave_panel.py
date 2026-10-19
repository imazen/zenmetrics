#!/usr/bin/env python3
"""Verify a completed diagnostic panel and emit small numerical-result shards.

No timing ratios are acquired or inferred. Per-process RSS remains separate
from run-heavy receipts, which are retained in the source artifact manifest.
"""
import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_tsv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def receipt(path):
    rows = [line for line in path.read_text().splitlines() if line.startswith('run-heavy: done rc=')]
    if len(rows) != 1:
        raise ValueError('missing or ambiguous resource receipt: ' + str(path))
    return rows[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', a.label):
        p.error('label must contain lowercase ASCII letters, digits, or hyphens')
    root = a.panel.resolve(strict=True)
    identity = json.loads((root / 'identity.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    expected = {f'{family}-' + ('auto' if factor is None else f'k{factor}') + f'-{row["pair"]}-{row["size_name"]}-t{budget}'
                for row, family, factor, budget in itertools.product(
                    identity['input_selection'], identity['families'], identity['factors'], identity['threads'])}
    paths = list(root.glob('*/outcome.json'))
    if not summary['execution_complete'] or len(paths) != len(expected) or {path.parent.name for path in paths} != expected:
        raise ValueError('panel is incomplete or contains unexpected cases')
    records, files, statuses = [], {}, collections.Counter()
    for path in sorted(paths):
        outcome = json.loads(path.read_text())
        assert outcome['case'] == path.parent.name
        for relative, digest in outcome['artifact_sha256'].items():
            artifact = (path.parent / relative).resolve(strict=True)
            if not artifact.is_relative_to(path.parent) or sha(artifact) != digest:
                raise ValueError('panel artifact changed: ' + str(artifact))
        files[str(path.relative_to(root))] = sha(path)
        statuses[outcome['status']] += 1
        result = outcome['comparison']
        if result is None:
            raise ValueError('this numeric-shard format requires compared scalar records; retain uncomparable cases separately')
        assert result == json.loads((path.parent / 'rust/results.json').read_text())
        canonical_path = path.parent / 'canonical/results.json'
        canonical = json.loads(canonical_path.read_text())
        assert sha(canonical_path) == result['canonical_record_sha256']
        assert result['binary_sha256'] == identity['binary_sha256']
        assert result['build_receipt_sha256'] == identity['build_receipt_sha256']
        for compared in result['comparisons']:
            assert compared['max_abs_delta'] == max(abs(compared['ours'] - v) for v in compared['canonical_values'])
            assert compared['pass'] == (compared['max_abs_delta'] <= compared['bound'])
            records.append(dict(cohort=a.label, family=outcome['family'], case=outcome['case'], metric=result['metric'],
                                column=compared['column'], pair=outcome['pair'], size=outcome['size'],
                                width=outcome['width'], height=outcome['height'], factor=compared.get('factor', result['factor']),
                                factor_mode=result.get('factor_mode', 'fixed'),
                                threads=outcome['threads'], ours=repr(compared['ours']),
                                canonical_first=repr(compared['canonical_values'][0]),
                                canonical_warm=repr(compared['canonical_values'][-1]),
                                max_abs_delta=repr(compared['max_abs_delta']), bound=repr(compared['bound']),
                                passed=compared['pass'], status=outcome['status'],
                                canonical_process_rss_kbytes=canonical['process_max_rss_kbytes'],
                                rust_process_rss_kbytes=result['process_max_rss_kbytes'],
                                canonical_resource_receipt=receipt(path.parent / 'canonical-wrapper.log'),
                                rust_resource_receipt=receipt(path.parent / 'rust-wrapper.log'), qualified_ratio=False))
    if dict(statuses) != summary['counts']:
        raise ValueError('case statuses differ from completion summary')
    a.output.mkdir()
    shards = []
    groups = collections.defaultdict(list)
    for row in records:
        groups[(row['family'], row['pair'])].append(row)
    for (family, pair), rows in sorted(groups.items()):
        if not re.fullmatch(r'[A-Za-z0-9-]+', family + '-' + pair):
            raise ValueError('unsafe group filename')
        data = encode_tsv(rows)
        if len(data) > 30_000:
            raise ValueError('generated shard exceeds 30,000 bytes')
        name = f'{a.label}-{family}-{pair}.tsv'
        (a.output / name).write_bytes(data)
        shards.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    rollup = {'source_panel': str(root), 'identity_sha256': sha(root / 'identity.json'),
              'events_sha256': sha(root / 'events.jsonl'), 'summary_sha256': sha(root / 'summary.json'),
              'cases': len(paths), 'scalar_rows': len(records), 'statuses': dict(statuses),
              'qualified_timing_cells': 0, 'shards': shards, 'outcome_sha256': files}
    data = (json.dumps(rollup, indent=2) + '\n').encode()
    if len(data) > 30_000:
        raise ValueError('generated panel index exceeds 30,000 bytes')
    (a.output / (a.label + '.json')).write_bytes(data)
    print(json.dumps({key: value for key, value in rollup.items() if key != 'outcome_sha256'}))


if __name__ == '__main__':
    main()
