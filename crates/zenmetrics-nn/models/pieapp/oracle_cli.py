#!/usr/bin/env python3
"""Execute documented official PieAPP CLI without importing/reading its source."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    for name in ('python','upstream','reference','distorted','log'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--sampling',choices=('dense','sparse'),default='dense')
    a=ap.parse_args()
    result=subprocess.run([a.python,'test_PieAPP_PT.py','--ref_path',str(Path(a.reference).resolve()),'--A_path',str(Path(a.distorted).resolve()),'--sampling_mode',a.sampling,'--gpu_id','0'],cwd=a.upstream,capture_output=True,text=True)
    combined=result.stdout+'\n'+result.stderr
    Path(a.log).write_text(combined)
    if result.returncode:
        print('Official black-box CLI failed:',combined.strip().splitlines()[-1],file=sys.stderr)
    else:
        print(result.stdout,end='')
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
