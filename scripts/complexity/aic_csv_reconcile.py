#!/usr/bin/env python3
"""Compare AIC CSV revisions by image key; preserve conflicting values in named columns."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import statistics
import gzip

META = {'distorted', 'source', 'codec', 'codec_acronym', 'distortion_level',
        'bpp', 'codec_id', 'enc_setting', 'enc_value'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key(name):
    # Only the documented crop-prefix alias is normalized; no fuzzy joining.
    name = Path(name.replace('\\', '/')).name
    return name[4:] if name.startswith('PTC_') else name


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        columns = r.fieldnames
        rows = list(r)
    if not columns or not {'source', 'distorted'} <= set(columns) or len(set(columns)) != len(columns):
        raise ValueError('unique header with source/distorted required: ' + str(path))
    by = {}
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('ragged CSV: ' + str(path))
        k = key(row['distorted'])
        if k in by:
            raise ValueError('ambiguous normalized image key: ' + k)
        by[k] = row
    return {'path': str(path), 'sha256': sha(path), 'columns': columns, 'rows': by}


def compare(a, b):
    shared = sorted(set(a['rows']) & set(b['rows']))
    stats = {}
    for col in sorted(set(a['columns']) & set(b['columns'])):
        exact = numeric_equal = numeric_changed = text_changed = 0
        delta, samples = [], []
        for k in shared:
            av, bv = a['rows'][k][col], b['rows'][k][col]
            if av == bv or col in ('source', 'distorted') and key(av) == key(bv):
                exact += 1
                continue
            try:
                x, y = Decimal(av), Decimal(bv)
                finite = x.is_finite() and y.is_finite()
            except InvalidOperation:
                finite = False
            if finite:
                if x == y:
                    numeric_equal += 1
                    continue
                numeric_changed += 1
                d = abs(x - y)
                delta.append(float(d))
            else:
                text_changed += 1
            if len(samples) < 3:
                samples.append({'image': k, 'left': av, 'right': bv})
        stats[col] = {'equal': exact, 'representation_only': numeric_equal,
                      'numeric_changed': numeric_changed, 'text_changed': text_changed,
                      'median_abs_delta_changed': statistics.median(delta) if delta else None,
                      'max_abs_delta': max(delta) if delta else None, 'samples': samples}
    source_mismatches = [k for k in shared if key(a['rows'][k]['source']) != key(b['rows'][k]['source'])]
    return {'left_rows': len(a['rows']), 'right_rows': len(b['rows']), 'shared_keys': len(shared),
            'left_only_keys': sorted(set(a['rows']) - set(b['rows'])),
            'right_only_keys': sorted(set(b['rows']) - set(a['rows'])),
            'source_mismatches': source_mismatches,
            'left_only_columns': [c for c in a['columns'] if c not in b['columns']],
            'right_only_columns': [c for c in b['columns'] if c not in a['columns']],
            'columns': stats}


def combine(a, b, path, view):
    columns = ['view', 'distorted_key', 'source_key'] + ['july::' + c for c in a['columns']] + ['drop::' + c for c in b['columns']]
    with path.open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, columns, lineterminator='\n')
        w.writeheader()
        for k in sorted(set(a['rows']) | set(b['rows'])):
            left, right = a['rows'].get(k, {}), b['rows'].get(k, {})
            if left and right and key(left['source']) != key(right['source']):
                raise ValueError('reference identity mismatch: ' + k)
            row = {'view': view, 'distorted_key': k, 'source_key': key((left or right)['source'])}
            row.update({'july::' + c: v for c, v in left.items()})
            row.update({'drop::' + c: v for c, v in right.items()})
            w.writerow(row)


def union_table(a, b, view, ptc_keys):
    columns = a['columns'] + [c for c in b['columns'] if c not in a['columns']]
    rows = []
    for k in sorted(set(a['rows']) | set(b['rows'])):
        left, right = a['rows'].get(k, {}), b['rows'].get(k, {})
        merged = dict(left)
        for col, value in right.items():
            if col in left and col not in ('source', 'distorted') and value != left[col]:
                try:
                    equal = Decimal(value) == Decimal(left[col])
                except InvalidOperation:
                    equal = False
                if not equal:
                    raise ValueError('union would hide conflicting cell: ' + view + '/' + k + '/' + col)
            merged[col] = value
        merged['view'] = view
        merged['ptc_subset'] = '1' if view == 'cropped' and k in ptc_keys else '0'
        rows.append(merged)
    return columns, rows


def write_csv(path, columns, rows):
    with path.open('x', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, columns, lineterminator='\n'); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['july-cropped', 'july-fullres', 'drop-cropped', 'drop-fullres', 'ptc', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('choose a new output directory; previous artifacts are preserved')
    inputs = {name: load(getattr(a, name)) for name in ['july_cropped', 'july_fullres', 'drop_cropped', 'drop_fullres', 'ptc']}
    report = {'schema': 'aic-csv-reconcile/1', 'inputs': {k: {x: v[x] for x in ['path', 'sha256', 'columns']} for k, v in inputs.items()}, 'comparisons': {}}
    for left, right in [('july_cropped', 'drop_cropped'), ('july_fullres', 'drop_fullres'), ('ptc', 'drop_cropped'), ('ptc', 'drop_fullres')]:
        r = compare(inputs[left], inputs[right])
        report['comparisons'][left + '__' + right] = r
        if r['source_mismatches']:
            raise ValueError('ambiguous reference joins: ' + str(r['source_mismatches'][:5]))
    a.output.mkdir(parents=True)
    union_rows = []
    union_columns = []
    ptc_keys = set(inputs['ptc']['rows'])
    ptc_comparison = report['comparisons']['ptc__drop_cropped']
    ptc_exact = not ptc_comparison['left_only_keys'] and all(not s['numeric_changed'] and not s['text_changed'] for s in ptc_comparison['columns'].values())
    if not ptc_exact:
        raise ValueError('PTC table is not an exact cropped subset; refusing guessed subset membership')
    for view in ['cropped', 'fullres']:
        path = a.output / ('combined_' + view + '.csv')
        combine(inputs['july_' + view], inputs['drop_' + view], path, view)
        # Re-read and verify every original string: combining never reformats scores.
        with path.open(newline='') as f:
            combined = {r['distorted_key']: r for r in csv.DictReader(f)}
        for version in ['july', 'drop']:
            for k, row in inputs[version + '_' + view]['rows'].items():
                for col, value in row.items():
                    if combined[k][version + '::' + col] != value:
                        raise ValueError('combined value changed: ' + k + '/' + col)
        cols, union = union_table(inputs['july_' + view], inputs['drop_' + view], view, ptc_keys)
        union_rows.extend(union)
        for col in cols:
            if col not in union_columns:
                union_columns.append(col)
        union_path = a.output / ('union_' + view + '.csv')
        write_csv(union_path, ['view', 'ptc_subset'] + cols, union)
        report.setdefault('outputs', {})[union_path.name] = {'bytes': union_path.stat().st_size, 'sha256': sha(union_path), 'rows': len(union), 'columns': len(cols) + 2}
        report.setdefault('outputs', {})[path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path), 'rows': len(combined)}
    combined_path = a.output / 'combined.csv'
    write_csv(combined_path, ['view', 'ptc_subset'] + union_columns, union_rows)
    with combined_path.open('rb') as source, (a.output / 'combined.csv.gz').open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as zipped:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                zipped.write(block)
    report['outputs']['combined.csv'] = {'bytes': combined_path.stat().st_size, 'sha256': sha(combined_path), 'rows': len(union_rows), 'columns': len(union_columns) + 2}
    report['outputs']['combined.csv.gz'] = {'bytes': (a.output / 'combined.csv.gz').stat().st_size, 'sha256': sha(a.output / 'combined.csv.gz')}
    report['ptc_view'] = 'Verified exact subset of drop_cropped; membership is marked only for cropped rows.'
    (a.output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, r in report['comparisons'].items():
        changes = {c: s['numeric_changed'] + s['text_changed'] for c, s in r['columns'].items() if s['numeric_changed'] or s['text_changed']}
        print(json.dumps({'comparison': name, 'left_rows': r['left_rows'], 'right_rows': r['right_rows'], 'shared_keys': r['shared_keys'], 'changed': changes,
                          'left_only_columns': r['left_only_columns'], 'right_only_columns': r['right_only_columns']}))


if __name__ == '__main__':
    main()
