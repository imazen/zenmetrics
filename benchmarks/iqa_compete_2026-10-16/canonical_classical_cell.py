#!/usr/bin/env python3
"""Explicit installed classical references; resident RGB conversion is timed."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import time


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric');p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    p.add_argument('--source',type=Path)
    a=p.parse_args()
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    cv2.setNumThreads(a.threads)
    r,d=[np.fromfile(f,np.uint8).reshape(a.height,a.width,3) for f in [a.reference,a.distorted]]
    if a.metric in ['nlpd','nlpd-iqa']:
        sys.path.insert(0,str(a.source))
        module=load(a.source/('nlp.py' if a.metric=='nlpd' else 'NLPD.py'),'primary_nlpd')
        model=module.LaplacianPyramid(6,dims=3) if a.metric=='nlpd' else module.NLPD(channels=1)
        def tensor(rgb):
            x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).contiguous().float()/255
            if a.metric=='nlpd-iqa':
                x=(.2126*x[:,:1]+.7152*x[:,1:2]+.0722*x[:,2:3]).clamp(0,1)
                x=torch.round(x*255)/255
            return x
        def score():
            x,y=tensor(r),tensor(d)
            return float((model.compare(x,y) if a.metric=='nlpd' else model(x,y,as_loss=False)).item())
    elif a.metric=='iwssim':
        module=load(a.source/'IW_SSIM_PyTorch.py','primary_iwssim')
        gray=load(a.source/'utils.py','primary_gray').rgb2gray
        model=module.IW_SSIM(use_cuda=False,use_double=False)
        def score():return float(model.test(gray(r),gray(d)).item())
    elif a.metric=='ws-hv':
        book=json.loads(a.source.read_text())
        source='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
        nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
        namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.source),'exec'),namespace)
        function=namespace['IQA_WS_HV']
        def score():return float(function(r,d))
    elif a.metric in ['ssim','ssimc']:
        from skimage.metrics import structural_similarity
        def score():
            _,m=structural_similarity(r.astype(np.float64),d.astype(np.float64),data_range=255,
                channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
            if a.metric=='ssimc':m=np.maximum(m[5:-5,5:-5],0)
            return float(m.mean())
    elif a.metric=='ciede2000':
        from skimage.color import rgb2lab,deltaE_ciede2000
        def score():return float(deltaE_ciede2000(rgb2lab(r),rgb2lab(d)).mean())
    elif a.metric.startswith('psnr'):
        from skimage.metrics import peak_signal_noise_ratio
        def plane(rgb):
            x=rgb.astype(np.float64)
            if a.metric=='psnr':return x
            if a.metric=='psnr-y':return .2126*x[:,:,0]+.7152*x[:,:,1]+.0722*x[:,:,2]
            if a.metric=='psnr-y601':return .299*x[:,:,0]+.587*x[:,:,1]+.114*x[:,:,2]
            x=rgb.astype(np.float32)
            y=16+(65.481*x[:,:,0]+128.553*x[:,:,1]+24.966*x[:,:,2])/255
            cb=128+(-37.797*x[:,:,0]-74.203*x[:,:,1]+112*x[:,:,2])/255
            cr=128+(112*x[:,:,0]-93.786*x[:,:,1]-18.214*x[:,:,2])/255
            # Positive code values: floor(x+.5) matches the declared u8 ingress.
            codes=[np.floor(v+.5).clip(0,255).astype(np.float64) for v in [y,cb,cr]]
            return codes
        def score():
            x,y=plane(r),plane(d)
            if a.metric in ['psnr','psnr-y','psnr-y601']:
                return float(peak_signal_noise_ratio(x,y,data_range=255))
            if a.metric=='psnr-ycbcr611':
                mse=sum(w*np.mean((u-v)**2) for w,u,v in zip([6,1,1],x,y))/8
                return float(10*np.log10(255**2/mse))
            index={'psnr-y-studio601':0,'psnr-cb':1,'psnr-cr':2}[a.metric]
            return float(peak_signal_noise_ratio(x[index],y[index],data_range=255))
    else:raise ValueError('unregistered classical reference: '+a.metric)
    with torch.inference_mode():
        for i in range(a.samples+1):
            start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
            print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
