#!/usr/bin/env python3
"""Contain each primary child separately, releasing shared locks between calls."""
import json
import os
from pathlib import Path
import subprocess
import sys

from octave_primary import sha


def execute(command, env, directory, role, seconds):
    directory=Path(directory).resolve(strict=True)
    script=Path(__file__).resolve()
    plan=directory/(role+'-scope-plan.json')
    fields=['TMPDIR','LC_ALL','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','INCFLAGS']
    record=dict(command=command,environment={k:env[k] for k in fields if k in env},directory=str(directory),role=role,seconds=seconds,
        worker_sha256=sha(script),execute_helper_sha256=sha(script.with_name('haarpsi_primary.py')))
    with plan.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    jobs=min(max(1,len(os.sched_getaffinity(0))-4),max(1,int(env.get('RAYON_NUM_THREADS','1'))))
    wrapper=['flock',str(Path.home()/'tmp/lilith-heavy.lock'),'flock',str(Path.home()/'tmp/lilith-wsl-heavy.lock'),
        str(Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','12G','--jobs',str(jobs),'--',sys.executable,'-B',str(script),str(plan)]
    with (directory/(role+'-wrapper.log')).open('x') as f:
        result=subprocess.run(wrapper,env=env,stdout=f,stderr=subprocess.STDOUT)
    completed=directory/(role+'-scope-result.json')
    assert completed.is_file(), 'child scope did not produce a result; preserve wrapper log'
    out=json.loads(completed.read_text())
    assert out['plan_sha256']==sha(plan) and result.returncode==0
    return out['returncode'],out['command']


def main():
    plan=Path(sys.argv[1]).resolve(strict=True);p=json.loads(plan.read_text())
    script=Path(__file__).resolve()
    assert sha(script)==p['worker_sha256'] and sha(script.with_name('haarpsi_primary.py'))==p['execute_helper_sha256']
    from haarpsi_primary import execute as child_execute
    env=dict(os.environ,**p['environment'])
    rc,command=child_execute(p['command'],env,Path(p['directory']),p['role'],p['seconds'])
    with (Path(p['directory'])/(p['role']+'-scope-result.json')).open('x') as f:
        f.write(json.dumps(dict(returncode=rc,command=command,plan_sha256=sha(plan)),indent=2)+'\n')


if __name__=='__main__':main()
