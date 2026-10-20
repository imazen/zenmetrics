#!/usr/bin/env python3
"""Original deepIQA Chainer classes with the existing CPU-only oracle shim."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('metric',choices=['wadiqam-fr-live','wadiqam-fr-tid','wadiqam-nr-live','wadiqam-nr-tid'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--setup-only',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if os.environ.get('SETUPTOOLS_USE_DISTUTILS')!='stdlib':
        env=dict(os.environ,SETUPTOOLS_USE_DISTUTILS='stdlib')
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    import numpy as np
    import chainer
    from chainer import serializers
    original=sys.argv;sys.argv=[cfg['oracle'],cfg['upstream']]
    try:
        spec=importlib.util.spec_from_file_location('original_wadiqam_oracle',cfg['oracle'])
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    finally:sys.argv=original
    mode,dataset=a.metric.split('-')[1:]
    chainer.global_config.train=False;chainer.global_config.cudnn_deterministic=True
    if mode=='fr':
        from fr_model import FRModel
        model=FRModel(top='weighted')
    else:
        from nr_model import Model
        model=Model(top='weighted')
    serializers.load_hdf5(str(Path(cfg['weights'])/f'{mode}_{dataset}_weighted.model'),model)
    chainer.cuda.cupy=np
    if a.setup_only:
        print(json.dumps(dict(metric=a.metric,setup='ready',chainer=chainer.__version__,numpy=np.__version__)))
        return
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    for i in range(a.samples+1):
        start=time.perf_counter_ns()
        if mode=='fr':value=oracle.score_image(model,d,r);reference=None
        else:value=oracle.score_image(model,d);reference=oracle.score_image(model,r)
        elapsed=time.perf_counter_ns()-start
        print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)
        if reference is not None:print(f'PAIR_REFERENCE {reference:.17g}',flush=True)


if __name__=='__main__':main()
