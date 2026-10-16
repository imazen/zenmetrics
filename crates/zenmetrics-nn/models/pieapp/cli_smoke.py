#!/usr/bin/env python3
"""Exercise PieAPP consent refusal and the actual default-scheduler CLI."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

binary,models,ladder=map(Path,sys.argv[1:])
env=dict(os.environ,ZENMETRICS_USE_LEGACY_SCHEDULER='0',ZENMETRICS_NN_MODEL_DIR=str(models),ZENMETRICS_NN_ENGINE='rten')
env.pop('ZENMETRICS_NN_ACCEPT_WEIGHTS',None)
command=[str(binary),'--bench-on-start','no','score','--metric','pieapp','--reference',str(ladder/'1415_64_ref.png'),'--distorted',str(ladder/'1415_64_dist.png'),'--output','json']
refused=subprocess.run(command,env=env,capture_output=True,text=True)
assert refused.returncode!=0
assert 'noncommercial' in (refused.stderr+refused.stdout).lower()
assert 'PerceptualImageError/blob/master/LICENSE.txt' in refused.stderr+refused.stdout
env['ZENMETRICS_NN_ACCEPT_WEIGHTS']='pieapp'
result=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
print(result.stderr,file=sys.stderr,end='');print(result.stdout,end='')
assert '[orchestrator] enabled' in result.stderr+result.stdout
actual=json.loads(result.stdout)
assert actual['metric']=='pieapp' and set(actual['scores'])=={'pieapp'}
with (models/'pieapp_goldens.tsv').open() as f:
    wanted=next(r for r in csv.DictReader(f,delimiter='\t') if r['image']=='1415' and r['size']=='64')
scales=Path(__file__).resolve().parent.parent/'jnd_scale.tsv'
jnd=next(float(line.split('\t')[1]) for line in scales.read_text().splitlines() if line.startswith('pieapp\t'))
assert abs(actual['scores']['pieapp']-float(wanted['pieapp']))<=1e-3*jnd
print('PieAPP consent refusal and default-scheduler oracle check pass')
