#!/usr/bin/env python3
"""Profile a complete first-evaluation probe; construction and controls included."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from compare_canonical import sha
from haarpsi_primary import execute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan',type=Path);p.add_argument('case');p.add_argument('output',type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text());case=next(c for c in plan['cases'] if c['id']==a.case)
    def verify():
        for path,expected in plan['bindings'].items():
            if sha(path)!=expected:raise ValueError('binding changed: '+path)
    verify();a.output.mkdir();root=a.output.resolve()
    env=dict(os.environ,**case['native'].get('environment',{}),TMPDIR=str(root),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',LC_ALL='C')
    for key in ['RAYON_NUM_THREADS','RTEN_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]=str(case['threads'])
    command=[part.format_map(dict(case,samples=1,output=str(root))) for part in case['native']['command']]
    profile=root/'perf.data'
    perf=['perf','record','-F','199','-e','cycles:P','--call-graph','dwarf,8192','-o',str(profile),'--',*command]
    rc,wrapped=execute(perf,env,root,'native',1100)
    if rc:raise ValueError('profile failed; retained logs: '+str(root))
    report=root/'report.txt'
    with report.open('x') as stream:subprocess.run(['perf','report','--stdio','--no-children','-i',str(profile)],stdout=stream,check=True)
    verify()
    value=dict(metric=case['metric'],case=case['id'],threads=case['threads'],plan_sha256=sha(a.plan),worker_sha256=sha(__file__),command=wrapped,
        profile_sha256=sha(profile),report_sha256=sha(report),
        scope='Complete native probe process, including immutable-model priming, first evaluation and second score-stability control; sampled shares do not isolate the timed first evaluation')
    (root/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    print('FIRST_EVALUATION_PROFILE',case['id'],flush=True)


if __name__=='__main__':main()
