#!/usr/bin/env python3
"""Require exact binary64 parity between native JSON and score/compare/batch TSV."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['binary', 'reference', 'distorted', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir()
    calls = []
    metrics = ['psnr', 'flip-v17', 'ws-hv']
    for metric in metrics:
        result = {}
        for fmt in ['json', 'tsv']:
            command = [str(a.binary), 'score', '--reference', str(a.reference), '--distorted', str(a.distorted), '--metric', metric, '--output', fmt]
            child = subprocess.run(command, capture_output=True, text=True, check=True)
            (a.output / (metric + '.' + fmt)).write_text(child.stdout)
            calls.append(command)
            result[fmt] = json.loads(child.stdout)['scores'] if fmt == 'json' else {k: float(v) for k, v in next(csv.DictReader(io.StringIO(child.stdout), delimiter='\t')).items()}
        assert result['json'] == result['tsv'], metric
        pairs = a.output / (metric + '.pairs.tsv')
        with pairs.open('w', newline='') as stream:
            w = csv.writer(stream, delimiter='\t', lineterminator='\n'); w.writerow(['ref_path', 'dist_path']); w.writerow([str(a.reference), str(a.distorted)])
        batch = a.output / (metric + '.batch.tsv')
        command = [str(a.binary), 'batch', '--metric', metric, '--pairs', str(pairs), '--output', str(batch), '--jobs', '1']
        subprocess.run(command, capture_output=True, text=True, check=True); calls.append(command)
        with batch.open() as stream:
            row = next(csv.DictReader(stream, delimiter='\t'))
        for col, value in result['json'].items():
            assert float(row[col]) == value, (metric, col, row[col], value)
    outputs = {}
    for fmt in ['json', 'tsv']:
        command = [str(a.binary), 'compare', '--reference', str(a.reference), '--variant', str(a.distorted), '--output', fmt]
        for metric in metrics:
            command.extend(['--metric', metric])
        child = subprocess.run(command, capture_output=True, text=True, check=True); calls.append(command)
        (a.output / ('compare.' + fmt)).write_text(child.stdout)
        outputs[fmt] = json.loads(child.stdout)['results'][0]['scores'] if fmt == 'json' else next(csv.DictReader(io.StringIO(child.stdout), delimiter='\t'))
    for col, value in outputs['json'].items():
        assert float(outputs['tsv'][col]) == value, (col, value, outputs['tsv'][col])
    (a.output / 'receipt.json').write_text(json.dumps({'commands': calls, 'scores': outputs['json'], 'all_equal': True}, indent=2) + '\n')
    print('score, compare and batch retain exact JSON binary64 scores for psnr, flip-v17 and ws-hv')


if __name__ == '__main__':
    main()
