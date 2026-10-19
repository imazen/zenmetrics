#!/usr/bin/env python3
"""Join retained primary conventions to frozen SSIM baseline/candidate probes.

No external acceptance bound is invented. The primary rows and candidate
behavior matrix remain separately source-bound observations.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['primary', 'supplement', 'multiscale', 'single-scale', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--additional-primary', type=Path)
    a = p.parse_args()
    roots = {name: getattr(a, name).resolve(strict=True) for name in ['primary', 'supplement', 'multiscale', 'single_scale']}
    if a.additional_primary:
        roots['additional_primary'] = a.additional_primary.resolve(strict=True)
    inputs, probes = {}, {}
    for label in ['multiscale', 'single_scale']:
        root = roots[label]
        done = json.loads((root / 'completion.json').read_text())
        assert len(done['groups']) == 24 and all(row['returncode'] == 0 for row in done['groups'])
        for group in done['groups']:
            directory = root / group['group']
            plan = json.loads((directory / 'plan.json').read_text())
            key = (plan['input']['pair'], plan['input']['size_name'], plan['threads'])
            if key in inputs:
                assert inputs[key] == plan['input']
            inputs[key] = plan['input']
            for row in json.loads((directory / 'results.json').read_text())['rows']:
                assert (*key, row['metric']) not in probes
                for arm in ['baseline', 'candidate']:
                    value = row[arm]
                    location = directory / (row['metric'] + '-' + arm)
                    actual = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (location / 'stdout.log').read_text().splitlines() if s.startswith('IQA_PROBE ')]
                    if value['returncode'] == 0:
                        assert len(actual) == 1 and actual[0]['scores'] == value['scores']
                        v = struct.unpack('>d', bytes.fromhex(value['scores'][0][1]))[0]
                        assert value['values'] == [v]
                    else:
                        assert not actual and (location / 'stderr.log').read_text() == value['stderr']
                probes[(*key, row['metric'])] = row
    done = json.loads((roots['primary'] / 'completion.json').read_text())
    assert len(done['cases']) == 96
    primary, dispositions = {}, {}
    for case in done['cases']:
        dispositions[case['case']] = case
        directory = roots['primary'] / case['case'] / 'canonical'
        events = directory / 'events.jsonl'
        records = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        for row in records:
            key = (row['pair'], row['size'], row['threads'], row['factor'], row['convention'])
            assert key not in primary
            primary[key] = (row, events, 'original-panel')
    extra_done = {'cases': []}
    if a.additional_primary:
        extra_root = roots['additional_primary']
        extra_identity = json.loads((extra_root / 'identity.json').read_text())
        assert extra_identity['factors'] == [5, 6] and extra_identity['direct_only']
        extra_done = json.loads((extra_root / 'completion.json').read_text())
        assert len(extra_done['cases']) == 48
        for case in extra_done['cases']:
            events = extra_root / case['case'] / 'canonical/events.jsonl'
            records = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
            for row in records:
                key = (row['pair'], row['size'], row['threads'], row['factor'], row['convention'])
                assert key not in primary and row['factor'] in [5, 6]
                primary[key] = (row, events, 'additional-k5-k6-panel')
    supplement = json.loads((roots['supplement'] / 'completion.json').read_text())
    for case in supplement['cases']:
        directory = roots['supplement'] / (case['parent_case'] + '-' + case['convention'])
        events = directory / 'canonical/events.jsonl'
        if events.exists():
            records = [json.loads(line) for line in events.read_text().splitlines()]
            assert len(records) == 1
            row = records[0]
            key = (row['pair'], row['size'], row['threads'], row['factor'], row['convention'])
            assert key not in primary
            primary[key] = (row, events, 'isolated-supplement')
    grouped, counts, maxima = collections.defaultdict(list), collections.Counter(), collections.defaultdict(float)
    conventions = {'ssim-unrounded-y': 'ssim-pyiqa-f32', 'ssim-unrounded-rgb': 'ssimc-pyiqa-f32',
                   'ms-ssim-unrounded-y': 'ms-ssim-pyiqa-f32', 'ssim-original-rounded-class': 'ssim-pyiqa',
                   'ms-ssim-original-rounded-class': 'ms-ssim-pyiqa'}
    for (pair, size, threads), input_row in sorted(inputs.items()):
        for factor in ([1, 2, 3, 4, 5, 6] if a.additional_primary else [1, 2, 3, 4]):
            for convention, base_name in conventions.items():
                if factor > 1 and 'original-rounded-class' in convention:
                    continue  # No corresponding rounded fixed CLI name exists.
                name = base_name if factor == 1 else base_name + f'-k{factor}'
                probe = probes[(pair, size, threads, name)]
                reference = primary.get((pair, size, threads, factor, convention))
                record = dict(metric=name, convention=convention, pair=pair, size=size, threads=threads, factor=factor,
                              canonical_status='not-returned', canonical='', canonical_source='', baseline='', candidate='',
                              baseline_returncode=probe['baseline']['returncode'], candidate_returncode=probe['candidate']['returncode'],
                              baseline_abs_delta='', candidate_abs_delta='', bound='UNBOUND', numerical_pass='UNQUALIFIED', qualified_ratio=False)
                for arm in ['baseline', 'candidate']:
                    if probe[arm]['returncode'] == 0:
                        record[arm] = probe[arm]['values'][0]
                if reference:
                    row, events, source = reference
                    assert row['reference_sha256'] == input_row['reference_sha256'] and row['distorted_sha256'] == input_row['distorted_sha256']
                    record['canonical_status'] = row['status']
                    record['canonical_source'] = source
                    if row['status'] == 'scored':
                        assert len(row['values']) == 2 and row['score_bits'][0] == row['score_bits'][1]
                        assert [struct.pack('>d', v).hex() for v in row['values']] == row['score_bits']
                        record['canonical'] = row['values'][0]
                        for arm in ['baseline', 'candidate']:
                            if probe[arm]['returncode'] == 0:
                                delta = abs(record[arm] - record['canonical'])
                                record[arm + '_abs_delta'] = delta
                                maxima[name + ':' + arm] = max(maxima[name + ':' + arm], delta)
                    elif row['status'] == 'canonical-error':
                        record['canonical_status'] += ':' + row['error_type']
                counts[(name, record['canonical_status'], 'scored' if record['candidate_returncode'] == 0 else 'refused')] += 1
                grouped[(pair, 'multiscale' if name.startswith('ms-') else 'single')].append(record)
    a.output.mkdir()
    outputs = []
    for (pair, family), rows in sorted(grouped.items()):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        data = stream.getvalue().encode()
        assert len(data) <= 30_000
        name = f'primary-pyiqa-{family}-{pair}.tsv'
        (a.output / name).write_bytes(data)
        outputs.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    artifacts = []
    for label, root in roots.items():
        manifest = root / 'joined-files.sha256.jsonl'
        paths = sorted(path for path in root.rglob('*') if path.is_file() and path != manifest)
        text = ''.join(json.dumps(dict(path=str(path.relative_to(root)), sha256=sha(path))) + '\n' for path in paths)
        if manifest.exists():
            assert manifest.read_text() == text
        else:
            manifest.write_text(text)
        artifacts.append(dict(cohort=label, root=str(root), manifest_sha256=sha(manifest)))
    index = dict(rows=sum(map(len, grouped.values())), outputs=outputs, artifacts=artifacts, max_abs_deltas=dict(maxima),
                 counts=[dict(metric=k[0], canonical_status=k[1], candidate_status=k[2], cases=v) for k, v in sorted(counts.items())],
                 primary_attempts=len(done['cases']) + len(extra_done['cases']),
                 primary_process_failures=sum(r['returncode'] != 0 for r in done['cases'] + extra_done['cases']),
                 supplemental_attempts=len(supplement['cases']), gate='UNBOUND; no pass inferred', qualified_timing_cells=0)
    (a.output / 'primary-pyiqa-ssim.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index))


if __name__ == '__main__':
    main()
