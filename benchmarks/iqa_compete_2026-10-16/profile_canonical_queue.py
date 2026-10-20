#!/usr/bin/env python3
"""Profile pending flagged cases while preserving completed recordings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index',type=Path);a=p.parse_args()
    failures=[];home=Path.home()
    for job in json.loads(a.index.read_text())['profiles']:
        root=Path(job['output'])
        if root.exists():
            if not (root/'receipt.json').exists():raise ValueError('interrupted profile retained: '+str(root))
            print(json.dumps(dict(preserved_completed_profile=str(root))),flush=True)
            continue
        command=['flock',str(home/'tmp/lilith-heavy.lock'),'flock',str(home/'tmp/lilith-wsl-heavy.lock'),
                 str(home/'work/claudehints/scripts/run-heavy'),'--mem','12G','--jobs',str(job['threads']),'--',
                 'env','TMPDIR='+str(root.parent),'RAYON_NUM_THREADS='+str(job['threads']),'CUDA_VISIBLE_DEVICES=',
                 sys.executable,'-B',str(Path(__file__).with_name('profile_canonical_warm.py')),
                 job['binary'],job['config'],str(root)]
        with (root.parent/'profile-wrapper.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(profile=str(root),exit_code=result.returncode)),flush=True)
        if result.returncode:failures.append(str(root))
    if failures:raise SystemExit('Failed profiles retained: '+', '.join(failures))


if __name__=='__main__':main()
