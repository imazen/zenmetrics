#!/usr/bin/env python3
"""Index published execution evidence without converting observations to passes."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open(), delimiter='\t'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    base = a.directory
    inventory = sum((rows(p) for p in sorted(base.glob('inventory-source-[0-9][0-9].tsv'))), [])
    assert len(inventory) == 127 and len({r['metric'] for r in inventory}) == 127
    evidence = collections.defaultdict(list)
    manifests = {}
    selections = [('runner-image212.json', 'fsim'), ('runner-image2201.json', 'vsi'),
                  ('primary-automatic.json', None), ('primary-gmsd.json', None),
                  ('primary-mdsi.json', None), ('primary-haarpsi.json', None),
                  ('approved-pyiqa-ssim.json', None)]
    if (base / 'primary-ssim2.json').exists():
        selections.append(('primary-ssim2.json', None))
    for index_name, family in selections:
        path = base / index_name
        index = json.loads(path.read_text()); manifests[index_name] = sha(path)
        for shard in index.get('shards', index.get('outputs', [])):
            assert sha(base / shard['path']) == shard['sha256']
            for row in rows(base / shard['path']):
                if family is not None and row['family'] != family:
                    continue
                metric = row.get('metric', 'gmsd')
                evidence[metric].append((index_name, row))
    timing = rows(base / 'haarpsi-paired-prefix.tsv')
    output = []
    excluded = {'pieapp', 'pieapp-pyiqa', 'aimos'}
    for entry in inventory:
        metric = entry['metric']; cases = collections.defaultdict(list)
        for index, row in evidence[metric]:
            cases[(row['pair'], row['size'], row['threads'])].append((index, row))
        assert len(cases) <= 24
        accepted = sum(all(row.get('passed') == 'True' or row.get('status') == 'pass' or row.get('candidate_status') == 'pass' for _, row in values)
                       for values in cases.values())
        refused = sum(all(row.get('candidate_status') == 'matching-size-refusal' for _, row in values)
                      for values in cases.values())
        if metric in excluded:
            status = 'excluded:proprietary-unreleased' if metric == 'aimos' else 'excluded:owner'
        elif not cases:
            status = 'missing'
        elif accepted == len(cases):
            status = 'scalar-pass' if len(cases) == 24 else 'partial-scalar-pass'
        elif accepted + refused == len(cases) and refused:
            status = 'scalar-pass-with-size-refusals'
        elif metric == 'gmsd':
            status = 'finding:canonical-odd-ASAN'
        else:
            status = 'unbound' if metric.startswith(('ssim-pyiqa', 'ssimc-pyiqa', 'ms-ssim-pyiqa')) or metric == 'ssim2' else 'finding'
        maps = 'missing'
        if metric == 'gmsd':
            maps = '18-bit-identical;6-canonical-crash'
        elif metric in {'mdsi', 'psnrhvs', 'psnrhvs-y', 'ssim2', 'fsim', 'vsi', 'fsim-nodownscale', 'vsi-nodownscale'} or metric.startswith(('fsim-k', 'vsi-k')):
            maps = 'N/A-primary-scalar-API' if cases else 'missing'
        output.append(dict(metric=metric, phase='excluded' if metric in excluded else 'learned' if metric == 'wd' or entry['family'] == 'nn' else 'classical',
                           scalar_cases=len(cases), expected_scalar_cases=0 if metric in excluded else 24,
                           scalar_pass_cases=accepted, matching_size_refusals=refused, status=status, maps=maps,
                           retained_timing_cases=sum(r['metric'] == metric for r in timing),
                           expected_timing_cases=0 if metric in excluded else 48, qualified_timing_cases=0,
                           evidence=','.join(sorted({index for index, _ in evidence[metric]}))))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(output[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader(); writer.writerows(output)
    data = stream.getvalue().encode(); assert len(data) <= 30_000
    (base / 'execution-coverage.tsv').write_bytes(data)
    value = dict(inventory_rows=len(output), included=sum(r['phase'] != 'excluded' for r in output),
                 scalar_observed_names=sum(r['scalar_cases'] > 0 for r in output),
                 qualified_timing_cells=0, status_counts=dict(collections.Counter(r['status'] for r in output)),
                 source_indexes=manifests, timing_prefix_sha256=sha(base / 'haarpsi-paired-prefix.tsv'),
                 coverage_sha256=sha(base / 'execution-coverage.tsv'), coverage_bytes=len(data))
    (base / 'execution-coverage.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
