#!/usr/bin/env python3
"""Compact measured AIC probe artifacts without repeating inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    for name in ('probe', 'native', 'site', 'config', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    receipt = json.loads((args.probe/'receipt.json').read_text())
    selection = json.loads((args.probe/'selection_summary.json').read_text())
    winner = list(csv.DictReader((args.probe/'best_full61.tsv').open(), delimiter='\t'))
    site = list(csv.DictReader(args.site.open()))
    native = {Path(r['dist_path']).name:r for r in csv.DictReader(args.native.open(), delimiter='\t')}
    mappings = json.loads(args.config.read_text())['mappings']
    assert len(winner) == 305 and len(native) == len(site) == receipt['checked_pairs'] == 61
    with (args.out/'aic_predictions.tsv').open('w') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['image']+[f'{kind}_s{k}' for kind in ('native','winner','published') for k in range(1,6)])
        for row in site:
            image = row['distorted']
            found = [r for r in winner if r['image'] == image]
            assert len(found) == 5
            writer.writerow([image]+[native[image][f'wd_s{k}'] for k in range(1,6)]+[r['predicted'] for r in found]+[row[f'WD_s{k}'] for k in range(1,6)])
    summary = []
    for convention in ('native','winner'):
        for k in range(1,6):
            errors, jnds = [], []
            m = mappings[f'WD_s{k}']
            for row in site:
                image = row['distorted']; published = float(row[f'WD_s{k}'])
                predicted = float(native[image][f'wd_s{k}']) if convention == 'native' else float(next(r['predicted'] for r in winner if r['image']==image and r['column']==f'WD_s{k}'))
                errors.append(abs(predicted-published))
                jnds.append(abs(m['a']*max(0,predicted)**m['c']-m['a']*max(0,published)**m['c']))
            summary.append([convention,f'WD_s{k}',61,statistics.median(errors),max(errors),statistics.median(jnds),max(jnds)])
    with (args.out/'aic_residuals.tsv').open('w') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['convention','column','pairs','median_absolute','max_absolute','median_site_jnd','max_site_jnd'])
        writer.writerows(summary)
    with (args.out/'aic_selection.tsv').open('w') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['convention','aggregation','max_site_jnd','median_site_jnd'])
        for row in sorted(selection['candidates'],key=lambda r:(r['max_jnd'],r['median_jnd'])):
            writer.writerow([row['config']['name'],row['aggregation'],row['max_jnd'],row['median_jnd']])
    receipt['external_artifacts'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.probe.iterdir() if p.is_file()}
    receipt['selection_combinations'] = len(selection['candidates'])
    receipt['original_clock'] = json.loads((args.probe.parent/'clock.json').read_text())
    receipt['coverage_note'] = 'Selection on first three of 61; full verification includes those same three, not an independent held-out panel. No fitted coefficients.'
    (args.out/'aic_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for row in summary:print('\t'.join(map(str,row)))


if __name__ == '__main__':
    main()
