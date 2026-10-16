#!/usr/bin/env python
"""Opaque call to the same public compute_score entry point as the release CLI.

No official model source or intermediate features are inspected. Only its
published patch-error/weight outputs and public call signature are observed.
"""
from __future__ import print_function
import argparse
import inspect
import os
import sys
import cv2
import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser()
    for name in ('upstream','checkpoint','reference','distorted'):
        ap.add_argument('--'+name,required=True)
    a=ap.parse_args();sys.path.insert(0,a.upstream)
    from model.PieAPPv0pt1_PT import PieAPP
    print('public constructor',inspect.getargspec(PieAPP.__init__) if hasattr(inspect,'getargspec') else inspect.signature(PieAPP.__init__))
    print('public compute_score',inspect.getargspec(PieAPP.compute_score) if hasattr(inspect,'getargspec') else inspect.signature(PieAPP.compute_score))
    state=torch.load(a.checkpoint,map_location='cpu')
    r=cv2.imread(a.reference).astype(np.float32)
    d=cv2.imread(a.distorted).astype(np.float32)
    h,w=r.shape[:2];assert r.shape==d.shape and min(h,w)>=64
    grids={ 'regular':[(y,x) for y in range(0,h-63,6) for x in range(0,w-63,6)],
        'last':[(y,x) for y in sorted(set(list(range(0,h-63,6))+[h-64])) for x in sorted(set(list(range(0,w-63,6))+[w-64]))] }
    for name,positions in sorted(grids.items()):
        model=PieAPP(1,len(positions));model.load_state_dict(state);model.eval()
        outputs=[]
        for start in range(0,len(positions),len(positions)):
            points=positions[start:start+len(positions)]
            rp=torch.from_numpy(np.stack([r[y:y+64,x:x+64].transpose(2,0,1) for y,x in points]).copy())
            dp=torch.from_numpy(np.stack([d[y:y+64,x:x+64].transpose(2,0,1) for y,x in points]).copy())
            if torch.__version__.startswith('0.'):
                rp=torch.autograd.Variable(rp,volatile=True);dp=torch.autograd.Variable(dp,volatile=True)
            value=model.compute_score(dp,rp)
            if not isinstance(value,(tuple,list)):value=[value]
            values=[v.data.cpu().numpy().copy() for v in value]
            if not outputs:outputs=[[] for _ in values]
            for column,v in zip(outputs,values):column.extend(v.reshape(-1).tolist())
        print('grid',name,'patches',len(positions),'outputs',len(outputs))
        for i,column in enumerate(outputs):print('output',i,'length',len(column),'first',column[:3],'sum',sum(column),'mean',np.mean(column))
        if len(outputs)==2 and len(outputs[0])==len(outputs[1]) and sum(outputs[1])!=0:
            print('ratio_of_sums',sum(outputs[0])/sum(outputs[1]))
            print('weighted_mean',sum(x*y for x,y in zip(*outputs))/sum(outputs[1]))

if __name__=='__main__':main()
