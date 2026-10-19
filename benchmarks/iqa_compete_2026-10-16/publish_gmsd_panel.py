#!/usr/bin/env python3
"""Recheck completed GMSD artifact hashes and export small scalar/map shards."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    root = a.panel.resolve(strict=True)
    identity = json.loads((root / 'identity.json').read_text())
    manifest = root / 'files.sha256.jsonl'
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        path = (root / row['path']).resolve(strict=True)
        assert path.is_relative_to(root) and sha(path) == row['sha256']
    completion = json.loads((root / 'completion.json').read_text())
    assert len(completion['cases']) == 24
    rows = collections.defaultdict(list)
    seen = set()
    for case in completion['cases']:
        key = (case['pair'], case['size'], case['threads'])
        assert key not in seen
        seen.add(key)
        directory = root / f'{case["pair"]}-{case["size"]}-t{case["threads"]}'
        assert json.loads((directory / 'results.json').read_text()) == case
        assert sha(directory / 'rust.f32') == case['rust_map_sha256']
        row = {k: case[k] for k in ['pair', 'size', 'width', 'height', 'threads', 'status', 'ours', 'mean_gms',
                                    'canonical_returncode', 'canonical_instrumented', 'canonical_process_rss_kbytes',
                                    'rust_map_process_rss_kbytes', 'rust_cli_process_rss_kbytes']}
        for key in ['canonical', 'absolute_delta', 'relative_delta', 'canonical_mean', 'mean_absolute_delta', 'map_bit_identical']:
            row[key] = case.get(key, '')
        if case['canonical_returncode']:
            assert case['canonical_instrumented'] and case['status'] == 'canonical-process-failure'
            assert 'AddressSanitizer: heap-buffer-overflow' in (directory / 'canonical-stderr.log').read_text()
            row['finding'] = 'primary C odd-geometry buffer overflow; original input retained'
        else:
            assert sha(directory / 'canonical.f32') == case['canonical_map_sha256']
            delta = abs(case['ours'] - case['canonical'])
            assert delta == case['absolute_delta']
            assert delta / max(abs(case['canonical']), 1e-300) == case['relative_delta']
            maps_equal = (directory / 'canonical.f32').read_bytes() == (directory / 'rust.f32').read_bytes()
            assert maps_equal == case['map_bit_identical']
            passed = maps_equal and case['relative_delta'] <= identity['score_relative_bound'] and case['mean_absolute_delta'] < identity['mean_absolute_bound']
            assert case['status'] == ('pass' if passed else 'numeric-finding')
            row['finding'] = ''
        row['score_relative_bound'] = identity['score_relative_bound']
        row['mean_absolute_bound'] = identity['mean_absolute_bound']
        row['qualified_ratio'] = False
        rows[case['pair']].append(row)
    assert seen == {(pair, size, threads) for pair in ['pairA', 'pairB', 'pairC'] for size in ['64', 'odd', '1mp', '4k'] for threads in identity['threads']}
    a.output.mkdir()
    shards = []
    for pair, records in sorted(rows.items()):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
        data = stream.getvalue().encode()
        assert len(data) <= 30_000
        name = 'primary-gmsd-' + pair + '.tsv'
        (a.output / name).write_bytes(data)
        shards.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    index = dict(source_panel=str(root), identity=identity, cases=len(seen),
                 statuses=dict(collections.Counter(case['status'] for case in completion['cases'])),
                 identity_sha256=sha(root / 'identity.json'), completion_sha256=sha(root / 'completion.json'),
                 artifact_manifest_sha256=sha(manifest), wrapper_log_sha256=sha(root.with_suffix('.log')), shards=shards,
                 qualified_timing_cells=0)
    (a.output / 'primary-gmsd.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index))


if __name__ == '__main__':
    main()
