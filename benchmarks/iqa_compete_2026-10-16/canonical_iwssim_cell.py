#!/usr/bin/env python3
"""Original Python-IW-SSIM with explicit modern torch.eig API compatibility.

The author body stays unchanged. PyTorch's installed migration guidance maps
torch.eig to torch.linalg.eig; the adapter restores the old real/imag columns
and requires real eigenvectors for this covariance-matrix use.
"""
import argparse
from pathlib import Path
import time

from canonical_classical_cell import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    a=p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    def legacy_eig(matrix,eigenvectors=False):
        if eigenvectors:
            values,vectors=torch.linalg.eig(matrix)
            if torch.count_nonzero(values.imag).item() or torch.count_nonzero(vectors.imag).item():
                raise ValueError('complex covariance eigensystem needs an actual legacy runtime')
            return torch.view_as_real(values),vectors.real
        values=torch.linalg.eigvals(matrix)
        return torch.view_as_real(values),torch.empty(0,dtype=matrix.dtype)
    torch.eig=legacy_eig
    module=load(a.source/'IW_SSIM_PyTorch.py','primary_iwssim')
    gray=load(a.source/'utils.py','primary_gray').rgb2gray
    model=module.IW_SSIM(use_cuda=False,use_double=False)
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    def score():return float(model.test(gray(r),gray(d)).item())
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
