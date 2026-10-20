#!/usr/bin/env python3
"""Apply a separately hash-bound benchmark worker environment/configuration.

The generic comparison runner deliberately masks GPU visibility. This opt-in
wrapper exposes exactly the requested native device/engine and records any
GPU-runtime config addition beside the original cell configuration.
"""
import json
import os
from pathlib import Path
import sys


def main():
    settings=json.loads(Path(sys.argv[1]).read_text())
    args=sys.argv[2:]
    env=dict(os.environ)
    for name,value in settings.get('environment',{}).items():
        env[name]=str(value)
    env['RTEN_NUM_THREADS']=os.environ.get('RAYON_NUM_THREADS','1')
    if args and args[0]!='inventory' and settings.get('gpu_runtime'):
        if len(args)!=1:raise ValueError('one cell configuration required')
        original=Path(args[0]);config=json.loads(original.read_text())
        config['gpu_runtime']=settings['gpu_runtime']
        path=original.with_name('runtime-config.json')
        with path.open('x') as f:json.dump(config,f)
        args=[str(path)]
    os.execve(settings['binary'],[settings['binary'],*args],env)


if __name__=='__main__':main()
