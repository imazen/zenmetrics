#!/usr/bin/env python3
"""Assert default-scheduler CLI wiring for all five WD columns."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

binary, models, ladder = map(Path, sys.argv[1:])
env = dict(os.environ, ZENMETRICS_USE_LEGACY_SCHEDULER='0', ZENMETRICS_NN_MODEL_DIR=str(models), ZENMETRICS_NN_ENGINE='rten')
command = [str(binary), '--bench-on-start', 'no', 'score', '--metric', 'wd',
           '--reference', str(ladder/'1415_64_ref.png'), '--distorted', str(ladder/'1415_64_dist.png'), '--output', 'json']
result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
print(result.stderr, file=sys.stderr, end='');print(result.stdout, end='')
assert '[orchestrator] enabled' in result.stderr + result.stdout
actual = json.loads(result.stdout)
assert actual['metric'] == 'wd'
with (models/'wd_goldens.tsv').open() as stream:
    oracle = next(r for r in csv.DictReader(stream, delimiter='\t') if r['image']=='1415' and r['size']=='64')
scales = {}
for line in (Path(__file__).resolve().parent.parent/'jnd_scale.tsv').read_text().splitlines():
    if line.startswith('wd_s'):
        fields=line.split('\t');scales[fields[0]]=float(fields[1])
assert set(actual['scores']) == {f'wd_s{k}' for k in range(1,6)}
for col, value in actual['scores'].items():
    assert abs(value-float(oracle[col])) <= 1e-3*scales[col], (col,value,oracle[col])
print('default-scheduler WD CLI matches all five oracle columns')
