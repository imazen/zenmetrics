#!/usr/bin/env python3
"""Report partial residuals from completed, hash-checked reference bundles.

The native arm must be complete. Keep this report outside git: it contains
the five worst reference rows. A partial result never qualifies the cohort.
"""
import argparse
import csv
import json
from pathlib import Path

from canonical_chunks import continuation_records, merge, records
from canonical_compare import compare
from canonical_provenance import json_safe
from canonical_run import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('metric')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('retain previous partial report; choose a new path')
    manifest = json.loads(args.manifest.read_text())
    spec = manifest['metrics'][args.metric]
    root, work = Path(manifest['scratch']), Path(manifest['output'])
    native_receipt = root/'logs'/(args.metric+'.native.receipt.json')
    native = json.loads(native_receipt.read_text())
    if native['returncode'] != 0 or native['rows'] != manifest['cohort_rows']:
        raise ValueError('complete native arm required')
    if sha(native['result']) != native['result_sha256']:
        raise ValueError('native result changed')
    with Path(manifest['pairs']).open() as source:
        metadata = csv.DictReader(source, delimiter='\t').fieldnames
    own, fields = records(native['result'])
    columns = [field for field in fields if field not in metadata]
    if spec.get('native_prefix'):
        columns = [field for field in columns if field.startswith(spec['native_prefix'])]
    if not columns:
        raise ValueError('native score column missing')
    reference, sources = {}, {str(native_receipt): sha(native_receipt)}
    continuation = manifest.get('continuations', {}).get(args.metric+'.canonical')
    prior_bundles = 0
    if continuation:
        reference, _ = continuation_records(continuation, spec['kind'] != 'octave')
        sources[continuation['path']] = continuation['sha256']
        for binding in continuation['receipts']:
            sources[binding['path']] = binding['sha256']
        prior_bundles = len(continuation['receipts'])
    chunks = work/(args.metric+'.canonical.chunks')
    receipts = sorted(chunks.glob('bundle-*.receipt.json'))
    if not receipts and not reference:
        raise ValueError('no completed reference bundles')
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt['returncode'] != 0:
            raise ValueError('reference bundle failed: '+str(path))
        sources[str(path)] = sha(path)
        for result, digest in receipt['outputs'].items():
            if digest is None or sha(result) != digest:
                raise ValueError('reference chunk changed: '+result)
            sources[result] = digest
            rows, _ = records(result, jsonl=spec['kind'] != 'octave')
            merge(reference, rows)
    if not reference.keys() <= own.keys():
        raise ValueError('unknown reference identity')
    values = {key: row.get('value') for key, row in reference.items() if not row.get('error')}
    summary, worst = compare(args.metric, spec['reference'],
        {key: row[columns[0]] for key, row in own.items()}, values, spec.get('gate'))
    report = dict(scope='partial; completed reference bundles only', cohort_rows=manifest['cohort_rows'],
        full_cohort_qualification='pending', reference_rows=len(reference), bundles=prior_bundles+len(receipts),
        reference_errors=sum(bool(row.get('error')) for row in reference.values()),
        comparison=summary, worst_five=worst, files=sources,
        manifest_sha256=sha(args.manifest), worker_sha256=sha(__file__))
    with args.output.open('x') as target:
        target.write(json.dumps(json_safe(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
