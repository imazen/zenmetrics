#!/usr/bin/env python3
"""Check a real default-scheduler PaQ-2-PiQ CLI score against the small oracle pair."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

binary, models, ladder = map(Path, sys.argv[1:])
env = dict(os.environ, ZENMETRICS_USE_LEGACY_SCHEDULER='0',
           ZENMETRICS_NN_MODEL_DIR=str(models), ZENMETRICS_NN_ENGINE='rten',
           ZENMETRICS_NN_ACCEPT_WEIGHTS='paq2piq')
command = [str(binary), '--bench-on-start', 'no', 'score', '--metric', 'paq2piq',
           '--reference', str(ladder / '1415_32_ref.png'),
           '--distorted', str(ladder / '1415_32_dist.png'), '--output', 'json']
result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
print(result.stderr, file=sys.stderr, end='')
print(result.stdout, end='')
assert '[orchestrator] enabled' in result.stderr + result.stdout
scores = json.loads(result.stdout)
assert scores['metric'] == 'paq2piq'
with (models / 'paq2piq_goldens.tsv').open() as stream:
    oracle = next(row for row in csv.DictReader(stream, delimiter='\t')
                  if row['image'] == '1415' and row['size'] == '32')
with (Path(__file__).resolve().parent.parent / 'jnd_scale.tsv').open() as stream:
    jnd = next(float(row.split('\t')[1]) for row in stream if row.startswith('paq2piq\t'))
for column in ('paq2piq', 'paq2piq_ref'):
    error = abs(scores['scores'][column] - float(oracle[column]))
    assert error <= 1e-3 * jnd, (column, error, jnd)
print('default-scheduler PaQ-2-PiQ CLI matches both oracle columns')
