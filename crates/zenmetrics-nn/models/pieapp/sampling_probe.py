#!/usr/bin/env python3
"""Historical revision-2 sampling hypothesis search, not canonical parity.

This preserves the initial absolute-weight hypothesis, since falsified by
public operator controls. Use network.forward and the revision-3 qualification
workflow for current scores; do not treat this historical search as evidence
for the signed-weight implementation.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','reference','distorted','out'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--target',type=float,required=True)
    a=ap.parse_args();torch.set_grad_enabled(False);torch.set_num_threads(16)
    spec=importlib.util.spec_from_file_location('paper_network',Path(__file__).with_name('network.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.PieAPP(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    def load(p):return torch.from_numpy(np.array(Image.open(p).convert('RGB'),dtype=np.float32)[:,:,::-1].transpose(2,0,1).copy()).unsqueeze(0)
    r,d=load(a.reference),load(a.distorted);assert r.shape==d.shape
    h,w=r.shape[-2:];points=[(y,x) for y in range(h-63) for x in range(w-63)]
    scores=np.empty((h-63,w-63),dtype=np.float32);weights=np.empty_like(scores)
    for start in range(0,len(points),32):
        batch=points[start:start+32]
        rp=torch.cat([r[:,:,y:y+64,x:x+64] for y,x in batch],0).contiguous()
        dp=torch.cat([d[:,:,y:y+64,x:x+64] for y,x in batch],0).contiguous()
        e,v=model.raw(rp,dp)
        e=((e/100)*model.ref_score_subtract.weight+model.ref_score_subtract.bias).flatten().tolist();v=(v.abs()+1e-6).flatten().tolist()
        for (y,x),s,t in zip(batch,e,v):scores[y,x]=s;weights[y,x]=t
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    np.savez_compressed(out/'all_positions.npz',scores=scores,weights=weights)
    ranked=[]
    def record(name,ys,xs):
        if not len(ys) or not len(xs):return
        es=scores[np.ix_(ys,xs)].astype(float);ws=weights[np.ix_(ys,xs)].astype(float)
        value=float((es*ws).sum()/ws.sum());ranked.append((abs(value-a.target),name,len(ys)*len(xs),value))
    for step in range(1,65):
        ys=np.arange(0,h-63,step);xs=np.arange(0,w-63,step)
        record('regular-'+str(step),ys,xs)
        record('last-'+str(step),np.unique(np.r_[ys,h-64]),np.unique(np.r_[xs,w-64]))
    for count in range(1,66):
        for mode,fun in [('floor',np.floor),('round',np.rint),('ceil',np.ceil)]:
            record('linspace-{}-{}'.format(count,mode),fun(np.linspace(0,h-64,count)).astype(int),fun(np.linspace(0,w-64,count)).astype(int))
    for step in range(2,33):
        for oy in range(step):
            for ox in range(step):record('offset-{}-{}-{}'.format(step,oy,ox),np.arange(oy,h-63,step),np.arange(ox,w-63,step))
    ranked.sort()
    with (out/'ranking.tsv').open('w') as f:
        f.write('absolute_error\tgrid\tpatches\tpredicted\n')
        for row in ranked:f.write('\t'.join(map(str,row))+'\n')
    for row in ranked[:30]:print(row,flush=True)

if __name__=='__main__':main()
