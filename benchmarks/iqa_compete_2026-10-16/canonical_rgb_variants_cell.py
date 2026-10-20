#!/usr/bin/env python3
"""Original scikit-image metric bodies with explicit RGB/edge conventions.

The caller supplies the declared precise D65 Lab ingress for CIEDE2000 and
whole-sample mirror edges for the full-map SSIM variant. SSIMc uses valid maps.
"""
import argparse
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric',choices=['ciede2000','ssim','ssimc'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    a=p.parse_args()
    import numpy as np
    from skimage.color import deltaE_ciede2000
    from skimage.metrics import structural_similarity
    import skimage.metrics._structural_similarity as ssim_module
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    if a.metric=='ssim':
        original=ssim_module.gaussian
        def mirror(image,**kwargs):
            kwargs['mode']='mirror'
            return original(image,**kwargs)
        ssim_module.gaussian=mirror
    def lab(rgb):
        x=rgb.astype(np.float64)/255
        linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
        matrix=[[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]
        xyz=np.stack([sum(weight*linear[:,:,i] for i,weight in enumerate(row)) for row in matrix],axis=2)
        normalized=xyz/np.array([.95047,1.,1.08883])
        delta=6/29
        values=np.where(normalized>delta**3,np.cbrt(normalized),normalized/(3*delta**2)+4/29)
        return np.stack([116*values[:,:,1]-16,500*(values[:,:,0]-values[:,:,1]),
                         200*(values[:,:,1]-values[:,:,2])],axis=2)
    def score():
        if a.metric=='ciede2000':return float(deltaE_ciede2000(lab(r),lab(d)).mean())
        _,image=structural_similarity(r.astype(np.float64),d.astype(np.float64),data_range=255,
            channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
        if a.metric=='ssimc':image=np.maximum(image[5:-5,5:-5],0)
        return float(image.mean())
    for i in range(a.samples+1):
        start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
        print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
