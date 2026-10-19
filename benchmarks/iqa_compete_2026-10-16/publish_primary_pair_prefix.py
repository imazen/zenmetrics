#!/usr/bin/env python3
"""Publish immutable completed timing cases, explicitly retaining partial status.

Reads outcome files rather than a controller's concurrently rewritten summary.
Checks packet hashes, full blocks, resource receipts and each arm's measured RSS.
No iterate-mode ratio becomes a confirmed performance result.
"""
import argparse
import csv
import gzip
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
    p.add_argument('--profiles', type=Path, required=True)
    a = p.parse_args()
    panel = a.panel.resolve(strict=True)
    identity = json.loads((panel / 'identity.json').read_text())
    records, artifacts, profiles = [], [], []
    for path in sorted(panel.glob('*/outcome.json')):
        outcome = json.loads(path.read_text())
        if outcome['status'] != 'exploratory-complete':
            continue
        row = dict(outcome['row'])
        packet_path = path.parent / 'attempt/packet.json.gz'
        assert sha(packet_path) == outcome['packet_sha256']
        packet = json.loads(gzip.decompress(packet_path.read_bytes()))
        assert packet['complete'] and packet['mode'] == 'iterate'
        summary = next(s for s in packet['summaries'] if s['arm'] == 'rust')
        assert summary['complete_blocks'] == packet['plan']['blocks']
        receipts = [s for s in (path.parent / 'wrapper.log').read_text().splitlines()
                    if s.startswith('run-heavy: done rc=')]
        assert receipts == [outcome['resource_receipt']] and 'rc=0 ' in receipts[0]
        memories = {'rust': [], 'canonical': []}
        for invocation in packet['invocations']:
            arm_path = Path(invocation['output']) / 'arm.json'
            arm = json.loads(arm_path.read_text())
            rss_path = arm_path.parent / 'process-memory.txt'
            values = [int(s.split(':', 1)[1]) for s in rss_path.read_text().splitlines()
                      if s.strip().startswith('Maximum resident set size (kbytes):')]
            assert values == [arm['process_rss_kbytes']]
            memories[arm['arm']].extend(values)
        for role in memories:
            assert max(memories[role]) == row[role.replace('rust', 'ours') + '_max_process_rss_kbytes']
        assert not row['qualified_ratio'] and row['qualification'] == 'exploratory'
        row.update(paired_blocks=summary['complete_blocks'], resource_receipt=receipts[0])
        records.append(row)
        artifacts.append(dict(case=outcome['case'], outcome_sha256=sha(path), packet_sha256=sha(packet_path),
                              wrapper_sha256=sha(path.parent / 'wrapper.log')))
        if row['ratio'] > 1.08:
            profile_path = a.profiles / outcome['case'] / 'profile.json'
            profile = json.loads(profile_path.read_text())
            assert profile['outcome_sha256'] == sha(path) and profile['returncode'] == 0
            for filename, digest in profile['files'].items():
                assert sha(Path(filename)) == digest
            profiles.append(dict(case=outcome['case'], path=str(profile_path.resolve()), sha256=sha(profile_path)))
    assert records
    a.output.mkdir()
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
    data = stream.getvalue().encode()
    assert len(data) <= 30_000, 'publish separately named shards rather than truncating'
    (a.output / 'summary.tsv').write_bytes(data)
    index = dict(panel=str(panel), completed_cases=len(records), planned_cases=identity['cases'],
                 partial=len(records) != identity['cases'], qualified_timing_cells=0,
                 identity_sha256=sha(panel / 'identity.json'), artifacts=artifacts, profiles=profiles,
                 summary_sha256=sha(a.output / 'summary.tsv'), summary_bytes=len(data))
    (a.output / 'haarpsi-paired-prefix.json').write_text(json.dumps(index, indent=2) + '\n')
    assert (a.output / 'haarpsi-paired-prefix.json').stat().st_size <= 30_000
    print(json.dumps(dict(completed=len(records), planned=identity['cases'], summary_bytes=len(data))))


if __name__ == '__main__':
    main()
