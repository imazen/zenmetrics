#!/usr/bin/env python3
"""Real-child controls for contained primary call status, deadlines and RSS."""
import argparse
import json
import os
from pathlib import Path
import sys

from contained_primary import execute
from octave_primary import process_max_rss, sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir();root=a.output.resolve();rows=[]
    for name,code,seconds,expected in [('success',"print('retained-success')",10,0),
            ('nonzero',"import sys;print('retained-error',file=sys.stderr);sys.exit(7)",10,7),
            ('deadline','import time;time.sleep(30)',1,124)]:
        directory=root/name;directory.mkdir()
        env=dict(os.environ,TMPDIR=str(directory),RAYON_NUM_THREADS='1')
        rc,command=execute([sys.executable,'-c',code],env,directory,'control',seconds)
        assert rc==expected
        rss=process_max_rss(directory/'control-memory.txt');assert rss>0
        if name=='success':assert (directory/'control-stdout.log').read_text()=='retained-success\n'
        if name=='nonzero':assert (directory/'control-stderr.log').read_text()=='retained-error\n'
        row=dict(case=name,returncode=rc,command=command,max_rss_kbytes=rss,
            files={p.name:sha(p) for p in directory.iterdir() if p.is_file()})
        rows.append(row);print('CONTROL',name,rc,rss,flush=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
