#!/usr/bin/env python3
"""Read-only current40 selection audit. Standard library only; launches nothing."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(manifest_path, comparative_path, output):
    manifest = json.loads(manifest_path.read_text())
    comparative = json.loads(comparative_path.read_text())
    assert len(manifest['metric_plans']) == 40 and len(comparative['cases']) == 80
    output.mkdir()
    records = []
    for metric, filename in manifest['metric_plans'].items():
        path = Path(filename)
        plan = json.loads(path.read_text())
        candidates = [row for row in plan['cases'] if row['metric'] == metric and row['matched_scope']]
        selected = min(candidates, key=lambda row: row['threads'])
        actual = [row for row in comparative['cases'] if row['metric'] == metric]
        assert len(actual) == 2 and {row['threads'] for row in actual} == {1, 8}
        for row in actual:
            expected = copy.deepcopy(selected)
            expected['threads'] = row['threads']
            expected['id'] = f'{metric}-changed-{selected["width"]}x{selected["height"]}-t{row["threads"]}'
            assert expected == row, metric + ': comparative case differs from preparer selection'
        command = selected['canonical']['command']
        entry = next((Path(arg) for arg in command[:2] if arg.endswith('.py')), None)
        entry_record = None
        if entry:
            bound = plan['bindings'].get(str(entry), plan['bindings'].get(str(entry.resolve())))
            entry_record = dict(path=str(entry), sha256=digest(entry), expected_sha256=bound,
                                matches_bound=digest(entry) == bound)
        record = dict(metric=metric, plan=str(path), plan_sha256=digest(path),
            selected_case_id=selected['id'], selected_thread=selected['threads'],
            width=selected['width'], height=selected['height'],
            absolute_bound=selected['absolute_bound'], relative_bound=selected.get('relative_bound', 0),
            scope=selected.get('scope'), canonical=selected['canonical'],
            canonical_command_sha256=hashlib.sha256(json.dumps(command, separators=(',', ':')).encode()).hexdigest(),
            python_entry=entry_record,
            comparative_cases=[dict(id=row['id'], threads=row['threads']) for row in actual],
            threads_placeholder='{threads}' in command,
            source_binding_candidates={key:value for key,value in plan['bindings'].items()
                if key.endswith(('.py', '.cpp', '.c')) and '/site-packages/' not in key},
            status='source_boundary_review_pending')
        # Large transitive source inventories are already retained in their plan;
        # each selection shard stays small and records only adapter source here.
        if len(json.dumps(record).encode()) >= 29000:
            record['source_binding_candidates'] = {key:value for key,value in record['source_binding_candidates'].items()
                if Path(key).name.startswith(('canonical_', 'first_boundary_')) or key.endswith('.cpp')}
        encoded = json.dumps(record, indent=2)+'\n'
        assert len(encoded.encode()) < 30000, metric + ': selection shard exceeds size contract'
        (output/(metric+'.json')).write_text(encoded)
        records.append(dict(metric=metric, selection=metric+'.json', entry_matches_bound=(None if entry_record is None else entry_record['matches_bound'])))
    index = dict(manifest_sha256=digest(manifest_path), comparative_sha256=digest(comparative_path),
                 metrics=40, cases=80, scorer_launched=False, records=records)
    (output/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    return index


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'comparative', 'output'):parser.add_argument(name, type=Path)
    args=parser.parse_args()
    result=collect(args.manifest,args.comparative,args.output)
    print(json.dumps({key:result[key] for key in ('metrics','cases','scorer_launched','manifest_sha256','comparative_sha256')}))
