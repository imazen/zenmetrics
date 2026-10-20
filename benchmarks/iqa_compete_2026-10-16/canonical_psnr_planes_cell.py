#!/usr/bin/env python3
"""scikit-image PSNR with explicitly named float64 studio-601 plane ingress.

The single-plane routes compute only the requested plane. The weighted route
uses the declared 6:1:1 MSE definition; it has no separate author software.
"""
import argparse
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metric',choices=['psnr-cb','psnr-cr','psnr-ycbcr611'])
    p.add_argument('width',type=int);p.add_argument('height',type=int)
    p.add_argument('reference',type=Path);p.add_argument('distorted',type=Path)
    p.add_argument('threads',type=int);p.add_argument('samples',type=int)
    a=p.parse_args()
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio
    r,d=[np.fromfile(path,np.uint8).reshape(a.height,a.width,3) for path in [a.reference,a.distorted]]
    tables=[(16.,[65.481,128.553,24.966]),(128.,[-37.797,-74.203,112.]),(128.,[112.,-93.786,-18.214])]
    def plane(rgb,index):
        x=rgb.astype(np.float64);offset,weights=tables[index]
        value=offset+(weights[0]*x[:,:,0]+weights[1]*x[:,:,1]+weights[2]*x[:,:,2])/255
        return np.floor(value+.5).clip(0,255)
    def score():
        if a.metric!='psnr-ycbcr611':
            index=1 if a.metric=='psnr-cb' else 2
            return float(peak_signal_noise_ratio(plane(r,index),plane(d,index),data_range=255))
        mses=[np.mean((plane(r,index)-plane(d,index))**2) for index in range(3)]
        return float(10*np.log10(255**2/((6*mses[0]+mses[1]+mses[2])/8)))
    for i in range(a.samples+1):
        start=time.perf_counter_ns();value=score();elapsed=time.perf_counter_ns()-start
        print(('COLD' if i==0 else 'SAMPLE')+f' {elapsed} {value:.17g}',flush=True)


if __name__=='__main__':main()
