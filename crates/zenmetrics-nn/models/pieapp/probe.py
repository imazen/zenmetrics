#!/usr/bin/env python3
"""Historical revision-2 input/layout probe; not the canonical parity oracle.

Preserves the initial absolute-weight hypothesis, since falsified by public
operator controls. Current qualification uses network.forward instead.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--reference',required=True)
    ap.add_argument('--distorted',required=True)
    a=ap.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(16)
    spec=importlib.util.spec_from_file_location('pieapp_reconstruction',Path(__file__).with_name('network.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    print('head affine',state['ref_score_subtract.weight'].tolist(),state['ref_score_subtract.bias'].tolist(),flush=True)
    r=torch.from_numpy(np.array(Image.open(a.reference).convert('RGB'),dtype=np.float32).transpose(2,0,1).copy()).unsqueeze(0)
    d=torch.from_numpy(np.array(Image.open(a.distorted).convert('RGB'),dtype=np.float32).transpose(2,0,1).copy()).unsqueeze(0)
    assert r.shape==d.shape and min(r.shape[-2:])>=64
    def axis(n):
        v=list(range(0,n-63,6))
        if v[-1]!=n-64:v.append(n-64)
        return v
    positions=[(y,x) for y in axis(r.shape[-2]) for x in axis(r.shape[-1])]
    r=torch.cat([r[:,:,y:y+64,x:x+64] for y,x in positions],0).contiguous()
    d=torch.cat([d[:,:,y:y+64,x:x+64] for y,x in positions],0).contiguous()
    print('patch count',len(positions),flush=True)
    for channel_last in (False,True):
        model=module.PieAPP(state,channel_last=channel_last)
        for norm in ('u8','01','pm1'):
            for bgr in (False,True):
                x,y=r,d
                if bgr:x,y=x[:,[2,1,0]],y[:,[2,1,0]]
                if norm=='01':x,y=x/255,y/255
                if norm=='pm1':x,y=x/127.5-1,y/127.5-1
                scores=[]; weights=[]
                for at in range(0,len(x),16):
                    error,weight=model.raw(x[at:at+16].contiguous(),y[at:at+16].contiguous())
                    scores.extend(((error/100)*model.ref_score_subtract.weight+model.ref_score_subtract.bias).flatten().tolist())
                    weights.extend((weight.abs()+1e-6).flatten().tolist())
                value=sum(a*b for a,b in zip(scores,weights))/sum(weights)
                print('candidate',channel_last,norm,bgr,'score',format(value,'.17e'),'weight_range',min(weights),max(weights),flush=True)
        del model

if __name__=='__main__':main()
