#!/usr/bin/env python3
"""pycvvdp 0.5.7 CPU reference with explicit standard_fhd display geometry."""
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
    import torch
    import pycvvdp
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    model=pycvvdp.cvvdp(display_name='standard_fhd',device=torch.device('cpu'),quiet=True)
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    def score():
        reference=torch.from_numpy(r.copy()).float()/255
        distorted=torch.from_numpy(d.copy()).float()/255
        quality,stats=model.predict(distorted,reference,dim_order='HWC')
        return float(quality.item())
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
