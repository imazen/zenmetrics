"""Run only documented demo CLI options on the complete qualified ladder."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ('python', 'upstream', 'pairs', 'entry_scores', 'out', 'logs'):
    p.add_argument('--'+name, required=True)
a = p.parse_args()
logs = Path(a.logs)
logs.mkdir(parents=True, exist_ok=True)
with open(a.entry_scores) as f:
    official = {r['dist_path']: float(r['pieapp']) for r in csv.DictReader(f, delimiter='\t')}
with open(a.pairs) as f:
    pairs = list(csv.DictReader(f, delimiter='\t'))
assert len(pairs) == 24
rows=[]
for index, row in enumerate(pairs):
    command=[a.python, 'test_PieAPP_PT.py', '--ref_path', row['ref_path'], '--A_path', row['dist_path'], '--sampling_mode', 'dense', '--gpu_id', '0']
    result=subprocess.run(command,cwd=a.upstream,capture_output=True,text=True)
    log=result.stdout+'\n'+result.stderr
    (logs/f'{index:02}.log').write_text(log)
    if result.returncode:
        raise RuntimeError((row,result.returncode,log[-1000:]))
    # The documented CLI prints "PieAPP score ... = VALUE".
    matches=re.findall(r'(?im)^.*PieAPP.*(?:=|:)\s*([-+0-9.eE]+)\s*$',result.stdout)
    if len(matches)!=1:
        raise ValueError(('unrecognized CLI output',result.stdout))
    score=float(matches[0]);entry=official[row['dist_path']]
    item=dict(key=Path(row['dist_path']).stem,cli=score,entry=entry,absolute_difference=abs(score-entry),jnd_difference=abs(score-entry)/0.5189487912492419)
    rows.append(item);print(json.dumps(item),flush=True)
with open(a.out,'x') as f: json.dump(dict(pairs=len(rows),worst=max(rows,key=lambda r:r['absolute_difference']),rows=rows),f,indent=2);f.write('\n')
