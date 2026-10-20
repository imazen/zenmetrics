#!/usr/bin/env python3
"""Original LPIPS, ST-LPIPS, DISTS and DeepDC CPU model callables."""
import argparse
import ast
from pathlib import Path
import sys
import time
import types


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric');p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--deepdc-convention',type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    if a.metric.startswith('lpips-'):
        import lpips
        model=lpips.LPIPS(net=a.metric.split('-')[1],version='0.1',verbose=False).eval()
    elif a.metric.startswith('stlpips-'):
        debug=types.ModuleType('IPython');debug.embed=lambda *args,**kwargs:None
        sys.modules.setdefault('IPython',debug)
        from stlpips_pytorch import stlpips
        model=stlpips.LPIPS(net=a.metric.split('-')[1],variant='shift_tolerant',verbose=False).eval()
    elif a.metric=='dists':
        from DISTS_pytorch import DISTS
        model=DISTS().eval()
    elif a.metric=='deepdc':
        sys.modules.setdefault('imp',types.ModuleType('imp'))
        from DeepDC_PyTorch import DeepDC
        # Reuse the native export's explicit ImageNet-input convention. The
        # reference model and distance-correlation body remain unmodified.
        tree=ast.parse(a.deepdc_convention.read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='with_input_norm']
        if len(nodes)!=1:raise ValueError('one explicit DeepDC input convention required')
        namespace={'torch':torch}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.deepdc_convention),'exec'),namespace)
        model=namespace['with_input_norm'](DeepDC()).eval()
    else:raise ValueError('unregistered original neural model')
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns()
            x,y=[torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).contiguous().float()/255 for rgb in [r,d]]
            if 'lpips-' in a.metric:x,y=x*2-1,y*2-1
            value=float(model(x,y).flatten()[0].item())
            del x,y
            elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
