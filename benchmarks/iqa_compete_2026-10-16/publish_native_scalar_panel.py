#!/usr/bin/env python3
"""Recompute the SSIMULACRA2 scalar joins from original process artifacts.

Preserves unassigned bounds and independently checks scalar bits and RSS.
"""
import argparse
import collections
import csv
import io
import json
from pathlib import Path
import struct

from octave_primary import sha, process_max_rss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('panel', type=Path)
    p.add_argument('inputs', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    root = a.panel.resolve(strict=True)
    identity = json.loads((root / 'identity.json').read_text())
    assert identity['metric'] == 'ssim2' and identity['bound'] == 'UNBOUND'
    assert sha(a.inputs) == identity['inputs_sha256']
    inputs = {(r['pair'], r['size_name']): r for r in json.loads(a.inputs.read_text())['pairs']}
    complete = json.loads((root / 'completion.json').read_text())
    expected = {(pair, size, threads) for pair, size in inputs for threads in identity['threads']}
    assert len(complete['cases']) == len(expected) == 24
    receipts = [s for s in root.with_suffix('.log').read_text().splitlines() if s.startswith('run-heavy: done rc=')]
    assert len(receipts) == 1 and 'rc=0 ' in receipts[0]
    records, actual = collections.defaultdict(list), set()
    for path in sorted(root.glob('*/results.json')):
        result = json.loads(path.read_text()); directory = path.parent
        key = (result['pair'], result['size'], result['threads'])
        assert key in expected and key not in actual
        actual.add(key)
        assert result['canonical_returncode'] == result['rust_returncode'] == 0
        assert result['status'] == 'UNQUALIFIED' and result['bound'] == 'UNBOUND'
        canonical = [float(s.split()[2]) for s in (directory / 'canonical-stdout.log').read_text().splitlines()
                     if s.startswith(('COLD ', 'SAMPLE '))]
        probes = [json.loads(s.removeprefix('IQA_PROBE ')) for s in (directory / 'rust-stdout.log').read_text().splitlines()
                  if s.startswith('IQA_PROBE ')]
        assert len(canonical) == 2 and canonical[0] == canonical[1] and len(probes) == 1
        assert probes[0]['metric_name'] == 'ssim2' and len(probes[0]['scores']) == 1
        ours = struct.unpack('>d', bytes.fromhex(probes[0]['scores'][0][1]))[0]
        delta = abs(ours-canonical[0])
        assert ours == result['ours'] and canonical[0] == result['canonical'] and delta == result['max_abs_delta']
        for role in ['rust', 'canonical']:
            assert process_max_rss(directory / (role + '-memory.txt')) == result[role + '_process_rss_kbytes']
        row = {name: result[name] for name in ['metric', 'pair', 'size', 'width', 'height', 'threads', 'ours',
               'canonical', 'max_abs_delta', 'bound', 'status', 'rust_process_rss_kbytes', 'canonical_process_rss_kbytes']}
        row.update(qualified_ratio=False, panel_resource_receipt=receipts[0])
        records[result['pair']].append(row)
    assert actual == expected
    manifest = root / 'publication-files.sha256.jsonl'
    content = ''.join(json.dumps(dict(path=str(p.relative_to(root)), sha256=sha(p)))+'\n'
                      for p in sorted(root.rglob('*')) if p.is_file() and p != manifest)
    if manifest.exists():
        assert manifest.read_text() == content
    else:
        manifest.write_text(content)
    a.output.mkdir()
    outputs = []
    for pair, rows in sorted(records.items()):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        name = 'primary-ssim2-' + pair + '.tsv'; data = stream.getvalue().encode()
        assert len(data) <= 30_000
        (a.output / name).write_bytes(data)
        outputs.append(dict(path=name, sha256=sha(a.output / name), bytes=len(data)))
    index = dict(panel=str(root), identity=identity, identity_sha256=sha(root / 'identity.json'),
                 completion_sha256=sha(root / 'completion.json'), artifact_manifest_sha256=sha(manifest),
                 wrapper_sha256=sha(root.with_suffix('.log')), outputs=outputs, cases=len(actual),
                 max_abs_delta=max(row['max_abs_delta'] for rows in records.values() for row in rows), qualified_timing_cells=0)
    (a.output / 'primary-ssim2.json').write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps(dict(cases=len(actual), max_abs_delta=index['max_abs_delta'])))


if __name__ == '__main__':
    main()
