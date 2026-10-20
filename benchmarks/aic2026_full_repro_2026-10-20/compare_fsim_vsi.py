#!/usr/bin/env python3
"""Compare CPU PyIQA FSIM/FSIMc/VSI with July and the retained native full set."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from canonical_compare import backup, finite, unique


def site_jnd(mapping, value):
    return mapping['a'] * abs(mapping['sign']*(value-mapping['b']))**mapping['c']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('aggregate', type=Path)
    a = p.parse_args()
    from scipy.stats import pearsonr, spearmanr
    manifest = json.loads(a.manifest.read_text())
    root, output = Path(manifest['scratch']), Path(manifest['output'])
    july = list(csv.DictReader((root/'metrics_cropped.csv').open()))
    published = unique(july)
    native_rows = list(csv.DictReader(Path(manifest['published_native']).open()))
    native = unique(native_rows)
    metadata = list(july[0])[:9]
    if len(july) != 9618 or published.keys() != native.keys():
        raise ValueError('expected identical complete crop identities')
    for identity, row in native.items():
        if any(row[k] != published[identity][k] for k in metadata):
            raise ValueError('native metadata mismatch: '+identity)
    mappings = json.loads(Path(manifest['site_mappings']).read_text())
    results, worst, columns = [], {}, {}
    for tag, column in [('pyiqa-fsim', 'FSIM'), ('pyiqa-fsimc', 'FSIMc'), ('pyiqa-vsi', 'VSI')]:
        path = output/(tag+'.canonical.jsonl')
        records = unique([json.loads(line) for line in path.read_text().splitlines()])
        if records.keys() != published.keys():
            raise ValueError('incomplete PyIQA cohort: '+tag)
        columns[tag] = {key: row.get('value') for key, row in records.items() if not row.get('error')}
        comparisons = [('pyiqa', 'july', columns[tag], {k: r[column] for k, r in published.items()}),
                       ('native', 'july', {k: r[column] for k, r in native.items()},
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
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
