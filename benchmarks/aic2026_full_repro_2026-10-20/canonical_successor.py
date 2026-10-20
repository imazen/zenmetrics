#!/usr/bin/env python3
"""Prepare a scheduling successor from completed bundles, retaining prior artifacts.

Run only after the predecessor controller has stopped at an admitted bundle
boundary. Scorer sources, configuration, cohort and worker environments stay
identical. Completed identities are carried forward without invoking scorers.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

from canonical_chunks import continuation_records, merge, records, scoring_environment
from canonical_run import admitted_command, sha


def prepare(predecessor, scratch, output, freeze, locks, rows_per_worker):
    old = json.loads(predecessor.read_text())
    if not 1 <= rows_per_worker <= 256:
        raise ValueError('rows_per_worker outside supported range')
    if any(path.exists() for path in (scratch, output, freeze)):
        raise ValueError('successor paths must be new; retain all previous artifacts')
    metadata = Path(old['scratch'])/'metrics_cropped.csv'
    metadata_sha = sha(metadata)
    source = Path(old['driver_freeze'])
    here = Path(__file__).parent
    relative = here.relative_to(next(p for p in here.parents if (p/'justfile').exists()))
    # All existing bound bytes must still agree before deriving a successor.
    for path, digest in old['bindings'].items():
        if sha(path) != digest:
            raise ValueError('predecessor binding changed: '+path)
    if sha(old['binary']) != old['binary_sha256']:
        raise ValueError('predecessor binary changed')
    new = copy.deepcopy(old)
    new.update(scratch=str(scratch), output=str(output), driver_freeze=str(freeze), heavy_locks=locks)
    new['execution'].update(rows_per_worker=rows_per_worker,
                            environment_root=old['execution'].get('environment_root', old['scratch']))
    admitted_command(new, [])
    for spec in [*old['metrics'].values(), *old['addendum'].values()]:
        if scoring_environment(old, spec, Path(old['scratch'])) != scoring_environment(new, spec, scratch):
            raise ValueError('successor changed scorer environment')
    shutil.copytree(source, freeze, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('canonical_chunks.py', 'canonical_run.py', 'canonical_campaign.py'):
        shutil.copyfile(here/name, freeze/relative/name)
    for name in ('canonical_python.py', 'canonical_octave.py'):
        if sha(source/relative/name) != sha(freeze/relative/name):
            raise ValueError('successor changed scorer source')
    for path in freeze.rglob('*'):
        if path.is_file():
            new['bindings'][str(path)] = sha(path)
    scratch.mkdir(parents=True)
    (scratch/'logs').mkdir()
    shutil.copyfile(metadata, scratch/'metrics_cropped.csv')
    if sha(scratch/'metrics_cropped.csv') != metadata_sha:
        raise ValueError('metadata copy changed')
    new['bindings'][str(scratch/'metrics_cropped.csv')] = metadata_sha
    output.mkdir(parents=True)
    with Path(old['pairs']).open() as stream:
        pairs = list(csv.DictReader(stream, delimiter='\t'))
    expected = {row['distorted'] for row in pairs}
    if len(pairs) != old['cohort_rows'] or len(expected) != len(pairs):
        raise ValueError('cohort identity mismatch')
    new['seeds'] = copy.deepcopy(old.get('seeds', {}))
    new['continuations'] = {}
    retained = {}
    for metric in [*old['metrics'], *old['addendum']]:
        for role in ('native', 'canonical'):
            tag = metric+'.'+role
            done = Path(old['scratch'])/'logs'/(tag+'.receipt.json')
            if done.exists():
                receipt = json.loads(done.read_text())
                result = Path(receipt['result'])
                if sha(result) != receipt['result_sha256']:
                    raise ValueError('completed arm changed: '+tag)
                destination = output/result.name
                shutil.copyfile(result, destination)
                receipt['result'] = str(destination)
                receipt['schedule_predecessor_receipt'] = dict(path=str(done), sha256=sha(done))
                (scratch/'logs'/done.name).write_text(json.dumps(receipt, indent=2)+'\n')
                new['seeds'].pop(tag, None)
                retained[tag] = receipt.get('rows')
                continue
            chunks = Path(old['output'])/(tag+'.chunks')
            receipts = sorted(chunks.glob('bundle-*.receipt.json'))
            previous = old.get('continuations', {}).get(tag)
            if not receipts and not previous:
                continue
            jsonl = role == 'canonical' and old['metrics'].get(metric, old['addendum'].get(metric))['kind'] != 'octave'
            combined, fields = continuation_records(previous, jsonl) if previous else ({}, None)
            bindings = copy.deepcopy(previous['receipts']) if previous else []
            for path in receipts:
                receipt = json.loads(path.read_text())
                bindings.append(dict(path=str(path), sha256=sha(path)))
                if receipt['returncode'] != 0:
                    raise ValueError('incomplete bundle cannot become a scheduling continuation')
                for result, digest in receipt['outputs'].items():
                    if digest is None or sha(result) != digest:
                        raise ValueError('completed bundle output changed')
                    incoming, header = records(result, jsonl)
                    if fields is not None and fields != header:
                        raise ValueError('bundle columns differ')
                    fields = header
                    merge(combined, incoming)
            if not combined.keys() <= expected:
                raise ValueError('bundle has unknown identities')
            suffix = '.jsonl' if jsonl else '.tsv'
            carry = output/(tag+'.carry'+suffix)
            with carry.open('x', newline='') as stream:
                if jsonl:
                    for row in pairs:
                        if row['distorted'] in combined:
                            stream.write(json.dumps(combined[row['distorted']])+'\n')
                else:
                    writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(combined[row['distorted']] for row in pairs if row['distorted'] in combined)
            continuation = dict(path=str(carry), sha256=sha(carry), rows=len(combined),
                receipts=bindings)
            continuation_records(continuation, jsonl)
            new['seeds'].pop(tag, None)
            new['continuations'][tag] = continuation
            retained[tag] = len(combined)
    new['schedule_predecessor'] = dict(path=str(predecessor), sha256=sha(predecessor),
        rows_per_worker=old['execution']['rows_per_worker'], retained_rows=retained,
        reason='owner-directed scheduling change; completed identities are never re-scored',
        worker_sha256=sha(__file__), pairs_sha256=sha(old['pairs']))
    target = scratch/'manifest.json'
    target.write_text(json.dumps(new, indent=2)+'\n')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('predecessor', type=Path)
    p.add_argument('scratch', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('freeze', type=Path)
    p.add_argument('--locks', nargs='+', required=True)
    p.add_argument('--rows-per-worker', type=int, default=32)
    a = p.parse_args()
    print(prepare(a.predecessor, a.scratch, a.output, a.freeze, a.locks, a.rows_per_worker))


if __name__ == '__main__':
    main()
