#!/usr/bin/env python3
"""Candidate first CUDA evaluator: construction and scalar readback included; controls required."""
import argparse
import sys
sys.path.insert(0,'/home/lilith/work/zen/zenmetrics/benchmarks/iqa_compete_2026-10-16')
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric',choices=['cvvdp','iwssim']);p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--cvvdp-package',type=Path);p.add_argument('--iwssim-source',type=Path)
    p.add_argument('--disable-tf32',action='store_true');p.add_argument('--extra-path',action='append',type=Path,default=[])
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='0'
    import numpy as np
    import torch
    sys.path.extend(str(path.resolve(strict=True)) for path in a.extra_path)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    if a.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    deferred=[]
    if a.metric=='cvvdp':
        if a.cvvdp_package is None:raise ValueError('explicit original CVVDP package required')
        package=a.cvvdp_package.resolve(strict=True)
        spec=importlib.util.spec_from_file_location('pycvvdp',package/'__init__.py',submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec);sys.modules['pycvvdp']=module;spec.loader.exec_module(module)
        from pycvvdp import utils
        if os.environ.get('CVVDP_PATH'):
            raise ValueError('explicit default pycvvdp parameters required')
        directory=Path(utils.__file__).resolve().parent/'vvdp_data'
        files={p.name:str(p) for p in directory.glob('*.json')}
        resident={str(p):p.read_text() for p in directory.glob('*.json')}
        def resident_find(cls,name,config_paths):
            if config_paths or name not in files:
                raise ValueError('unexpected pycvvdp parameter lookup')
            return files[name]
        def resident_json(path):
            if str(path) not in resident:
                raise ValueError('unexpected pycvvdp JSON read')
            return json.loads(resident[str(path)])
        utils.config_files.find=classmethod(resident_find)
        utils.json2dict=resident_json
        model=None
        def score():
            nonlocal model
            if model is None:
                model=module.cvvdp(display_name='standard_fhd',device=torch.device('cuda'),quiet=True)
            reference=torch.from_numpy(r.copy()).float()/255
            distorted=torch.from_numpy(d.copy()).float()/255
            quality,stats=model.predict(distorted,reference,dim_order='HWC')
            return float(quality.item())
    else:
        if a.iwssim_source is None:raise ValueError('explicit original Python-IW-SSIM source required')
        from canonical_classical_cell import load
        def legacy_eig(matrix,eigenvectors=False):
            if eigenvectors:
                values,vectors=torch.linalg.eig(matrix)
                deferred.extend((values.imag,vectors.imag))
                return torch.view_as_real(values),vectors.real
            values=torch.linalg.eigvals(matrix)
            return torch.view_as_real(values),torch.empty(0,dtype=matrix.dtype,device=matrix.device)
        torch.eig=legacy_eig
        module=load(a.iwssim_source/'IW_SSIM_PyTorch.py','primary_iwssim_cuda')
        gray=load(a.iwssim_source/'utils.py','primary_gray_cuda').rgb2gray
        model=None
        def score():
            nonlocal model
            if model is None:
                model=module.IW_SSIM(use_cuda=True,use_double=False)
            return float(model.test(gray(r),gray(d)).item())
    if torch.cuda.is_initialized():raise ValueError('CUDA initialized before the first evaluator clock')
    with torch.inference_mode():
        for i in range(a.samples+1):
            if i:torch.cuda.synchronize()
            start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
            for view in deferred:
                if torch.count_nonzero(view).item():raise ValueError('complex covariance eigensystem needs a legacy runtime')
            deferred.clear()
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)
            if i==0:
                print('CANONICAL '+json.dumps(dict(metric=a.metric,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(0),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    synchronization='Pre-call synchronization excluded; .item() materializes each scalar; eigenvalue reality checks deferred until after the timer')),flush=True)


if __name__=='__main__':main()
