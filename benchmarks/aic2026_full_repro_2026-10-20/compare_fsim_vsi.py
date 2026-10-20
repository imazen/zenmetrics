#!/usr/bin/env python3
"""Compare CPU PyIQA FSIM/FSIMc/VSI with the cropped published union."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import statistics
import subprocess
import sys

from canonical_compare import backup, finite, unique
from canonical_run import sha


def published_crops(path, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError('published reference SHA256 mismatch')
    with Path(path).open(encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if 'view' not in (reader.fieldnames or []):
            raise ValueError('published union requires its explicit view column')
        return unique([row for row in reader if row['view'] == 'cropped'])


def same_reference_metadata(published, original, metadata):
    for key in metadata:
        if published[key] == original[key]:
            continue
        if key != 'bpp':
            return False
        # The union removes trailing decimal zeroes from some bpp strings.
        # Compare exact decimal values; exported metadata still comes from July.
        try:
            left, right = Decimal(published[key]), Decimal(original[key])
            if not left.is_finite() or not right.is_finite() or left != right:
                return False
        except InvalidOperation:
            return False
    return True


def site_jnd(mapping, value):
    return mapping['a'] * abs(mapping['sign']*(value-mapping['b']))**mapping['c']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('aggregate', type=Path)
    p.add_argument('--published-reference', type=Path, required=True)
    p.add_argument('--reference-sha256', required=True)
    p.add_argument('--july-diff', type=Path, default=Path(__file__).resolve().parents[2]/'scripts/aic2026/july_diff.py')
    a = p.parse_args()
    from scipy.stats import pearsonr, spearmanr
    manifest = json.loads(a.manifest.read_text())
    root, output = Path(manifest['scratch']), Path(manifest['output'])
    july = list(csv.DictReader((root/'metrics_cropped.csv').open()))
    published = published_crops(a.published_reference, a.reference_sha256)
    original = unique(july)
    native_rows = list(csv.DictReader(Path(manifest['published_native']).open()))
    native = unique(native_rows)
    metadata = list(july[0])[:9]
    if len(july) != 9618 or original.keys() != published.keys() or published.keys() != native.keys():
        raise ValueError('expected identical complete crop identities')
    for identity, row in native.items():
        if any(row[k] != original[identity][k] for k in metadata):
            raise ValueError('native metadata mismatch: '+identity)
        if not same_reference_metadata(published[identity], original[identity], metadata):
            raise ValueError('published metadata mismatch: '+identity)
    mappings = json.loads(Path(manifest['site_mappings']).read_text())
    results, worst, columns = [], {}, {}
    for tag, column in [('pyiqa-fsim', 'FSIM'), ('pyiqa-fsimc', 'FSIMc'), ('pyiqa-vsi', 'VSI')]:
        path = output/(tag+'.canonical.jsonl')
        records = unique([json.loads(line) for line in path.read_text().splitlines()])
        if records.keys() != published.keys():
            raise ValueError('incomplete PyIQA cohort: '+tag)
        columns[tag] = {key: row.get('value') for key, row in records.items() if not row.get('error')}
        comparisons = [('pyiqa', 'published', columns[tag], {k: r[column] for k, r in published.items()}),
                       ('native', 'published', {k: r[column] for k, r in native.items()},
                        {k: r[column] for k, r in published.items()}),
                       ('pyiqa', 'native', columns[tag], {k: r[column] for k, r in native.items()})]
        for left, right, xs, ys in comparisons:
            rows = []
            for identity in published:
                x, y = finite(xs.get(identity)), finite(ys.get(identity))
                if x is not None and y is not None:
                    rows.append(dict(distorted=identity, left=x, right=y, delta=abs(x-y),
                                     delta_site_jnd=abs(site_jnd(mappings[column], x)-site_jnd(mappings[column], y))))
            if not rows:
                raise ValueError('no finite comparisons: '+tag)
            x, y = [r['left'] for r in rows], [r['right'] for r in rows]
            result = dict(column=column, left=left, right=right, n=len(rows),
                          median_d=statistics.median(r['delta'] for r in rows),
                          max_d=max(r['delta'] for r in rows),
                          median_dJND=statistics.median(r['delta_site_jnd'] for r in rows),
                          max_dJND=max(r['delta_site_jnd'] for r in rows),
                          pearson=float(pearsonr(x, y).statistic), spearman=float(spearmanr(x, y).statistic))
            results.append(result)
            worst[f'{column}:{left}:{right}'] = sorted(rows, key=lambda r: (-r['delta_site_jnd'], r['distorted']))[:5]
    backup(a.aggregate)
    with a.aggregate.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    path = output.parent/'metrics_cropped.pyiqa-fsim-vsi.csv'
    backup(path)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata+list(columns), lineterminator='\n')
        writer.writeheader()
        for row in july:
            writer.writerow({**{k: row[k] for k in metadata},
                             **{tag: values.get(row['distorted'], '') for tag, values in columns.items()}})
    path = root/'fsim-vsi-worst-five.json'
    backup(path)
    path.write_text(json.dumps(worst, indent=2)+'\n')
    # Supply published metric column names to the authoritative comparison
    # script, keeping this derived table outside the repository.
    path = output.parent/'metrics_cropped.pyiqa-fsim-vsi.published.csv'
    backup(path)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata+['FSIM', 'FSIMc', 'VSI'], lineterminator='\n')
        writer.writeheader()
        for row in july:
            identity = row['distorted']
            writer.writerow({**{k: row[k] for k in metadata},
                             **{column: columns[tag].get(identity, '') for tag, column in
                                [('pyiqa-fsim', 'FSIM'), ('pyiqa-fsimc', 'FSIMc'), ('pyiqa-vsi', 'VSI')]}})
    commands = []
    for label, scored in [('pyiqa', path), ('native', Path(manifest['published_native']))]:
        plan = root/f'fsim-vsi-{label}-published-routes.json'
        backup(plan)
        routes = ({'FSIM': 'pyiqa-fsim', 'FSIMc': 'pyiqa-fsimc', 'VSI': 'pyiqa-vsi'}
                  if label == 'pyiqa' else {'FSIM': 'fsim-y', 'FSIMc': 'fsim', 'VSI': 'vsi'})
        plan.write_text(json.dumps(dict(columns={column: dict(route=route) for column, route in routes.items()}),
                                   indent=2)+'\n')
        command = [sys.executable, str(a.july_diff), str(scored), str(a.published_reference),
                   str(root/f'fsim-vsi-{label}-vs-published.tsv'),
                   str(root/f'fsim-vsi-{label}-vs-published.worst.txt'), '--view', 'cropped', '--plan', str(plan)]
        subprocess.run(command, check=True)
        commands.append(command)
    path = root/'fsim-vsi-reference.json'
    backup(path)
    path.write_text(json.dumps(dict(
        url='https://zentrain.r2.imazen.org/aic2026/csv/july-drop-2026-10-20/combined.csv',
        path=str(a.published_reference), sha256=a.reference_sha256, view='cropped',
        identity='distorted', rows=len(published), july_diff_sha256=sha(a.july_diff),
        commands=commands), indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
