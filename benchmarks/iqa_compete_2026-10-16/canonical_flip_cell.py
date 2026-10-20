#!/usr/bin/env python3
"""NVIDIA FLIP 1.7 error map with the declared f64 mean convention."""
import argparse
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    a=p.parse_args()
    import numpy as np
    import flip_evaluator
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    def score():
        x,y=r.astype(np.float32)/255,d.astype(np.float32)/255
        error,_,_=flip_evaluator.evaluate(x,y,'LDR',inputsRGB=True,applyMagma=False,computeMeanError=False)
        return float(error.mean(dtype=np.float64))
    for i in range(a.samples+1):
        start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
        print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
