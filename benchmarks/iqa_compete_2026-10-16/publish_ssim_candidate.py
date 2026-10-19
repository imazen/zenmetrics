#!/usr/bin/env python3
"""Verify frozen before/after SSIM probe outputs and publish small TSVs."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss(path):
    lines = [s.split(':', 1)[1].strip() for s in path.read_text().splitlines()
             if s.strip().startswith('Maximum resident set size (kbytes):')]
    assert len(lines) == 1
    return int(lines[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    root = a.input.resolve(strict=True)
    identity = json.loads((root / 'identity.json').read_text())
    complete = json.loads((root / 'completion.json').read_text())
    assert len(complete['groups']) == 24 and all(r['returncode'] == 0 for r in complete['groups'])
    rows, counts = collections.defaultdict(list), collections.Counter()
    for group in complete['groups']:
        directory = root / group['group']
        plan = json.loads((directory / 'plan.json').read_text())
        assert plan['worker_sha256'] == identity['worker_sha256']
        receipts = [s for s in (directory / 'wrapper.log').read_text().splitlines() if s.startswith('run-heavy: done rc=')]
        assert len(receipts) == 1
        for result in json.loads((directory / 'results.json').read_text())['rows']:
            metric = result['metric']
            row = dict(pair=plan['input']['pair'], size=plan['input']['size_name'],
                       width=plan['input']['width'], height=plan['input']['height'], threads=plan['threads'], metric=metric)
            for arm in ['baseline', 'candidate']:
                data = result[arm]
                location = directory / (metric + '-' + arm)
                row[arm + '_returncode'] = data['returncode']
                row[arm + '_process_rss_kbytes'] = rss(location / 'process-memory.txt')
                if data['returncode'] == 0:
                    probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in
                              (location / 'stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                    assert len(probes) == 1 and probes[0]['scores'] == data['scores']
                    assert len(data['scores']) == 1
                    row[arm + '_bits'] = data['scores'][0][1]
                    row[arm + '_value'] = repr(data['values'][0])
                else:
                    assert (location / 'stderr.log').read_text() == data['stderr']
                    row[arm + '_bits'] = row[arm + '_value'] = ''
            if 'absolute_deltas' in result:
                delta = abs(result['baseline']['values'][0] - result['candidate']['values'][0])
                assert result['absolute_deltas'] == [delta]
                row['absolute_delta'] = repr(delta)
                status = 'bit-identical' if result['bit_identical'] else 'changed-score'
            else:
                row['absolute_delta'] = ''
                status = 'both-refused' if result.get('both_refused') else 'changed-refusal'
            row['status'] = status
            row['base_name'] = result['base_name']
            row['group_resource_receipt'] = receipts[0]
            row['qualified_ratio'] = False
            counts[('base' if result['base_name'] else 'fixed') + ':' + status] += 1
            rows[row['pair']].append(row)
    a.output.mkdir()
    outputs = []
    for pair, records in sorted(rows.items()):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
        data = stream.getvalue().encode()
        assert len(data) <= 30_000
        name = 'ssim-fixed-before-after-' + pair + '.tsv'
        (a.output / name).write_bytes(data)
        outputs.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    manifest = root / 'publication-files.sha256.jsonl'
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    with manifest.open('x') as f:
        for path in paths:
            f.write(json.dumps({'path': str(path.relative_to(root)), 'sha256': sha(path)}) + '\n')
    index = {'rows': sum(map(len, rows.values())), 'statuses': dict(counts), 'outputs': outputs,
             'input_root': str(root), 'artifact_manifest_sha256': sha(manifest), 'qualified_timing_cells': 0}
    (a.output / 'ssim-fixed-before-after.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index))


if __name__ == '__main__':
    main()
