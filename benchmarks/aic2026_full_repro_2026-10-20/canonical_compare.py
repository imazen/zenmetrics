#!/usr/bin/env python3
"""Join evaluation scores by distorted identity; never join by completion order."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


FIELDS = 'cli reference n exact_eq median_d max_d median_rel max_rel pearson spearman gate match'.split()


def unique(rows):
    result = {}
    for row in rows:
        key = row['distorted']
        if key in result:
            raise ValueError('duplicate distorted identity: '+key)
        result[key] = row
    return result


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def compare(metric, label, ours, reference, gate, gate_operator='<='):
    if gate_operator not in ('<', '<='):
        raise ValueError('unsupported comparison bound operator: '+gate_operator)
    from scipy.stats import pearsonr, spearmanr
    cases = []
    for identity in ours.keys() & reference.keys():
        x, y = finite(ours[identity]), finite(reference[identity])
        if x is not None and y is not None:
            delta = abs(x-y)
            relative = delta/abs(y) if y else (0. if delta == 0 else math.inf)
            cases.append(dict(distorted=identity, ours=x, reference=y, delta=delta, relative=relative))
    n = len(cases)
    result = dict.fromkeys(FIELDS, '')
    result.update(cli=metric, reference=label, n=n, exact_eq=sum(c['ours'] == c['reference'] for c in cases),
                  gate='none' if gate is None else f'abs{gate_operator}{gate:.17g}', match='no-gate')
    if n:
        result.update(median_d=statistics.median(c['delta'] for c in cases),
                      max_d=max(c['delta'] for c in cases),
                      median_rel=statistics.median(c['relative'] for c in cases),
                      max_rel=max(c['relative'] for c in cases))
        if gate is not None:
            inside = result['max_d'] < gate if gate_operator == '<' else result['max_d'] <= gate
            result['match'] = 'MATCH' if inside else 'MISS'
        xs, ys = [c['ours'] for c in cases], [c['reference'] for c in cases]
        if n >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
            result.update(pearson=float(pearsonr(xs, ys).statistic), spearman=float(spearmanr(xs, ys).statistic))
    elif gate is not None:
        # No finite comparisons is not evidence of a numerical match.
        result['match'] = 'MISS'
    return result, sorted(cases, key=lambda c: (-c['delta'], c['distorted']))[:5]


def backup(path):
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path.rename(path.with_name(path.name+'.bak-'+stamp))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--summary', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    root = Path(manifest['scratch'])
    out = Path(manifest['output'])
    published = list(csv.DictReader((root/'metrics_cropped.csv').open()))
    published_by_id = unique(published)
    if len(published) != 9618:
        raise ValueError('expected every July crop row')
    metadata = list(published[0])[:9]
    metrics = list(manifest['metrics'])
    summaries, statuses, worst = [], {}, {}
    native, canonical = {}, {}
    for metric, spec in manifest['metrics'].items():
        native[metric], canonical[metric] = {}, {}
        filename = out/(metric+'.native.tsv')
        if filename.exists():
            with filename.open() as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
                headers = reader.fieldnames or []
            indexed = unique(rows)
            score_columns = [name for name in headers
                             if name not in ('ref_path', 'dist_path', *metadata)]
            prefix = spec.get('native_prefix')
            if prefix:
                score_columns = [name for name in score_columns if name.startswith(prefix)]
            if rows and not score_columns:
                raise ValueError('no score column: '+metric)
            for identity, row in indexed.items():
                if identity not in published_by_id or any(row[k] != published_by_id[identity][k] for k in metadata):
                    raise ValueError('native metadata differs from July input: '+identity)
                native[metric][identity] = row[score_columns[0]]
        filename = out/(metric+'.canonical.'+('tsv' if spec.get('kind') == 'octave' else 'jsonl'))
        errors = []
        if filename.exists():
            rows = (list(csv.DictReader(filename.open(), delimiter='\t')) if filename.suffix == '.tsv'
                    else [json.loads(line) for line in filename.read_text().splitlines()])
            for identity, row in unique(rows).items():
                if identity not in published_by_id:
                    raise ValueError('unknown canonical identity: '+identity)
                if not row.get('error'):
                    canonical[metric][identity] = row.get('value')
                else:
                    errors.append(dict(distorted=identity, error=row['error']))
        summary, cases = compare(metric, spec['reference'], native[metric], canonical[metric],
                                 spec.get('gate'), spec.get('gate_operator', '<='))
        summaries.append(summary)
        worst[metric] = cases
        statuses[metric] = dict(native_rows=len(native[metric]), canonical_rows=len(canonical[metric]),
                                reference_errors=len(errors), first_errors=errors[:5],
                                missing=spec.get('missing'), comparison=summary)
    for label, values in [('canonical', canonical), ('zenmetrics-extra', native)]:
        path = out.parent/f'metrics_cropped.{label}.csv'
        backup(path)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metadata+metrics, lineterminator='\n')
            writer.writeheader()
            for row in published:
                emitted = {k: row[k] for k in metadata}
                for metric in metrics:
                    value = values[metric].get(row['distorted'])
                    emitted[metric] = value if finite(value) is not None else ''
                writer.writerow(emitted)
    backup(a.summary)
    with a.summary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    for name, data in [('status.json', statuses), ('worst-five.json', worst)]:
        path = root/name
        backup(path)
        path.write_text(json.dumps(data, indent=2, allow_nan=True)+'\n')
    print(json.dumps(statuses, indent=2))


if __name__ == '__main__':
    main()
