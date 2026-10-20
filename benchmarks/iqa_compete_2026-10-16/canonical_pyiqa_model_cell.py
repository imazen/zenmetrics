#!/usr/bin/env python3
"""Resident named PyIQA models; RGB tensor conversion is inside each call."""
import argparse
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric',choices=['ahiq-pyiqa','wadiqam-pyiqa','topiq-fr','topiq-fr-pipal','cw-ssim-pyiqa','dmm-pyiqa'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    a=p.parse_args()
    import numpy as np
    import torch
    import pyiqa
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    name={'ahiq-pyiqa':'ahiq','wadiqam-pyiqa':'wadiqam_fr','topiq-fr':'topiq_fr',
          'topiq-fr-pipal':'topiq_fr-pipal','cw-ssim-pyiqa':'cw_ssim','dmm-pyiqa':'dmm'}[a.metric]
    model=pyiqa.create_metric(name,device='cpu',as_loss=False)
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns()
            reference=torch.from_numpy(r.copy()).permute(2,0,1).unsqueeze(0).contiguous().float()/255
            distorted=torch.from_numpy(d.copy()).permute(2,0,1).unsqueeze(0).contiguous().float()/255
            value=float(model(distorted,reference).flatten()[0].item())
            del reference,distorted
            elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
