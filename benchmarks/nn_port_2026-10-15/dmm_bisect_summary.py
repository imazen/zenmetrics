#!/usr/bin/env python3
"""Compare complete native bisection panels against the measured baseline.

Acceptance requires all 48 frozen limits, no per-row residual increase, and
bit-identical CUDA repeat scores. Missing observations never count as passes.
"""
import argparse
import csv
from pathlib import Path
import statistics


def read(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    assert len(rows) == 24, path
    result = {(r['image'], r['size']): r for r in rows}
    assert len(result) == 24, path
    return result


def panel(directory, name):
    result = {}
    first = {}
    for engine in ['rten', 'ort-cuda']:
        rows = read(directory / f'{name}-{engine}.tsv')
        for key, row in rows.items():
            result[(engine, *key)] = row
        if engine == 'ort-cuda':
            first = rows
    repeated = read(directory / f'{name}-ort-cuda-repeat.tsv')
    assert first.keys() == repeated.keys()
    unstable = [key for key in first
                if float(first[key]['dmm']).hex() != float(repeated[key]['dmm']).hex()]
    return result, unstable


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--partial', action='store_true')
args = parser.parse_args()
manifest = Path(__file__).with_name('dmm_bisect_sources_2026-10-17.tsv')
with manifest.open() as stream:
    sources = list(csv.DictReader(stream, delimiter='\t'))
lines = ['# DMM individual-change acceptance, 2026-10-17', '',
         'Each candidate starts from the frozen solver, with one numerical change. '
         'Acceptance requires all 48 frozen row limits, no increase in any row residual '
         'against the measured baseline, and identical complete CUDA repeat scores.', '',
         '| Change | Source | Within frozen limits | Improved / equal / worse residuals | Median / max netlib error | CUDA repeats | Decision |',
         '|---|---|---:|---:|---:|---:|---|']
pending = []
baseline = None
resources = []
notes = []
for entry in sources:
    name, revision = entry['change'], entry['commit']
    try:
        rows, unstable = panel(args.directory, name)
    except FileNotFoundError:
        if not args.partial:
            raise
        pending.append(name)
        lines.append(f'| {name} | `{revision[:12]}` | unmeasured | unmeasured | unmeasured | unmeasured | pending |')
        continue
    if name == 'baseline':
        baseline = rows
    if baseline is None:
        raise AssertionError('Baseline must be complete before candidate comparison')
    runner = (args.directory / f'{name}.runner.log').read_text()
    receipts = [line for line in runner.splitlines() if line.startswith('run-heavy: done ')]
    assert len(receipts) == 1, (name, 'missing or ambiguous resource receipt')
    resources.append(f'{name}: {receipts[0]}')
    controls_pass = all(f'PANEL_PASS {engine} {revision}' in runner
                        for engine in ['rten', 'ort-cuda', 'ort-cuda-repeat'])
    assert rows.keys() == baseline.keys()
    for key in rows:
        assert rows[key]['kind'] == baseline[key]['kind'], (name, key, 'classification changed')
        assert float(rows[key]['tolerance']) == float(baseline[key]['tolerance']), (name, key, 'limit changed')
    errors = [float(row['netlib_error']) for row in rows.values()]
    passed = sum(float(row['netlib_error']) <= float(row['tolerance']) for row in rows.values())
    improved = sum(float(rows[key]['netlib_error']) < float(baseline[key]['netlib_error']) for key in rows)
    worse = sum(float(rows[key]['netlib_error']) > float(baseline[key]['netlib_error']) for key in rows)
    equal = 48 - improved - worse
    decision = 'baseline' if name == 'baseline' else ('keep' if passed == 48 and worse == 0 and not unstable and controls_pass else 'reject')
    if name == 'baseline':
        assert passed == 48 and not unstable and controls_pass, 'Frozen baseline itself must qualify and reproduce'
    lines.append(f'| {name} | `{revision[:12]}` | {passed}/48 | {improved} / {equal} / {worse} | '
                 f'{statistics.median(errors):.9e} / {max(errors):.9e} | {24-len(unstable)}/24 identical | {decision} |')
    if unstable:
        notes.append(f'Unstable CUDA observations for {name}: {unstable!r}. Repeat before treating them as findings.')
    if not controls_pass:
        notes.append(f'{name}: native panel controls did not all pass; this candidate cannot be retained.')
if pending:
    lines[2:2] = ['**Missing:** ' + ', '.join(pending) + '.', '']
for note in notes:
    lines.extend(['', note])
lines.extend(['', 'Resource receipts for the complete three-panel runs (parity, not speed benchmarks):',
              '', '```text', *resources, '```'])
args.output.write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
