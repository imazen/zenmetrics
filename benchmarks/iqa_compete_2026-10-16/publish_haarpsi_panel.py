#!/usr/bin/env python3
"""Recheck retained primary Octave scalar artifacts and publish small shards."""
import argparse
import collections
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--metric', choices=['haarpsi', 'mdsi'], default='haarpsi')
    a = p.parse_args()
    root = a.panel.resolve(strict=True)
    identity = json.loads((root / 'identity.json').read_text())
    complete = json.loads((root / 'completion.json').read_text())
    assert sha(a.inputs) == identity['inputs_sha256']
    inputs = json.loads(a.inputs.read_text())['pairs']
    metrics = ['haarpsi', 'haarpsi-y'] if a.metric == 'haarpsi' else ['mdsi']
    bound = 5e-5 if a.metric == 'haarpsi' else 1e-9
    assert (identity['bound'] if a.metric == 'haarpsi' else identity['relative_bound']) == bound
    expected_source = ('ef1bed2872fd1f74842afdccfd1d6b0e3a93bc5479b14bf845460f08dec8ee4a'
                       if a.metric == 'haarpsi' else '4dff198b28e57b39b20877c7bcb75af2c6278c95f9f321362a2046ee3e37351b')
    assert identity['source_sha256'] == expected_source
    expected = {f'{metric}-{row["pair"]}-{row["size_name"]}-t{threads}'
                for row, metric, threads in itertools.product(inputs, metrics, identity['threads'])}
    actual = {path.parent.name for path in root.glob('*/results.json')}
    completed = complete['cases'] if isinstance(complete['cases'], int) else len(complete['cases'])
    assert len(expected) == completed == 12 * len(metrics) * len(identity['threads']) and actual == expected
    receipts = [s for s in root.with_suffix('.log').read_text().splitlines() if s.startswith('run-heavy: done rc=')]
    assert len(receipts) == 1 and 'rc=0 ' in receipts[0]
    records = collections.defaultdict(list)
    for case in sorted(expected):
        directory = root / case
        result = json.loads((directory / 'results.json').read_text())
        assert result['canonical_returncode'] == result['rust_returncode'] == 0
        canonical = [float(s.split()[2]) for s in (directory / 'canonical-stdout.log').read_text().splitlines()
                     if s.startswith(('COLD ', 'SAMPLE '))]
        probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory / 'rust-stdout.log').read_text().splitlines()
                  if s.startswith('IQA_PROBE ')]
        assert len(canonical) == 2 and len(probes) == 1
        assert probes[0]['metric_name'] == result['metric'] and len(probes[0]['scores']) == 1
        ours = struct.unpack('>d', bytes.fromhex(probes[0]['scores'][0][1]))[0]
        assert ours == result['ours'] and canonical == result['reference_values']
        delta = max(abs(ours - value) for value in canonical)
        assert delta == result['max_abs_delta']
        compared = delta
        if a.metric == 'mdsi':
            compared = max(abs(ours - value) / max(abs(value), 1e-300) for value in canonical)
            assert compared == result['max_relative_delta'] and result['relative_bound'] == bound
        else:
            assert result['bound'] == bound
        assert result['status'] == ('pass' if compared <= bound else 'numeric-finding')
        for role in ['canonical', 'rust']:
            rss = [int(s.split(':', 1)[1]) for s in (directory / (role + '-memory.txt')).read_text().splitlines()
                   if s.strip().startswith('Maximum resident set size (kbytes):')]
            assert rss == [result[role + '_process_rss_kbytes']]
        row = {key: result[key] for key in ['metric', 'pair', 'size', 'width', 'height', 'threads', 'ours',
                                          'max_abs_delta', 'status', 'canonical_process_rss_kbytes',
                                          'rust_process_rss_kbytes']}
        row.update(bound=bound, bound_kind='absolute' if a.metric == 'haarpsi' else 'relative',
                   max_relative_delta=result.get('max_relative_delta', ''))
        row.update(canonical_first=canonical[0], canonical_warm=canonical[1],
                   panel_resource_receipt=receipts[0], qualified_ratio=False, map_status='missing')
        records[result['pair']].append(row)
    manifest = root / 'publication-files.sha256.jsonl'
    paths = sorted(path for path in root.rglob('*') if path.is_file() and path != manifest)
    manifest_data = ''.join(json.dumps(dict(path=str(path.relative_to(root)), sha256=sha(path))) + '\n' for path in paths)
    if manifest.exists():
        assert manifest.read_text() == manifest_data
    else:
        manifest.write_text(manifest_data)
    a.output.mkdir()
    shards = []
    for pair, rows in sorted(records.items()):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        data = stream.getvalue().encode()
        assert len(data) <= 30_000
        name = 'primary-' + a.metric + '-' + pair + '.tsv'
        (a.output / name).write_bytes(data)
        shards.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    index = dict(source_panel=str(root), identity=identity, identity_sha256=sha(root / 'identity.json'),
                 completion_sha256=sha(root / 'completion.json'), artifact_manifest_sha256=sha(manifest),
                 wrapper_log_sha256=sha(root.with_suffix('.log')), cases=completed,
                 statuses=dict(collections.Counter(row['status'] for rows in records.values() for row in rows)),
                 max_abs_delta=max(row['max_abs_delta'] for rows in records.values() for row in rows),
                 qualified_timing_cells=0, shards=shards)
    assert index['statuses'].get('pass', 0) == complete['passes']
    if a.metric == 'mdsi':
        index['max_relative_delta'] = max(row['max_relative_delta'] for rows in records.values() for row in rows)
    (a.output / ('primary-' + a.metric + '.json')).write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index))


if __name__ == '__main__':
    main()
