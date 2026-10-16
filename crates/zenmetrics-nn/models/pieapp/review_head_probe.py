"""Compare head conventions to opaque public patch errors and weights.

Official model internals are never inspected; raw head outputs below belong
only to our independently written reconstruction.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
for name in ('upstream','checkpoint','pairs','out'):
    p.add_argument('--'+name,required=True)
a=p.parse_args()
sys.path.insert(0,a.upstream)
from model.PieAPPv0pt1_PT import PieAPP as Official
spec=importlib.util.spec_from_file_location('independent_pieapp',Path(__file__).with_name('network.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
state=torch.load(a.checkpoint,map_location='cpu')
ours=module.PieAPP(state).cuda().eval()
oracle=Official(1,1).cuda().eval();oracle.load_state_dict(state)
import csv
with open(a.pairs) as f: pairs=list(csv.DictReader(f,delimiter='\t'))[:3]
rows=[]
for pair in pairs:
    images=[cv2.imread(pair[key])[:64,:64].transpose(2,0,1).copy()[None].astype(np.float32) for key in ('ref_path','dist_path')]
    r,d=[torch.from_numpy(v).cuda() for v in images]
    public=oracle.compute_score(d,r)
    error,weight=ours.raw(r,d)
    target_error=float(public[1].reshape(-1)[0]);target_weight=float(public[2].reshape(-1)[0])
    hypotheses={}
    for divisor in (1,10,100,255):
        value=(error/divisor)*ours.ref_score_subtract.weight+ours.ref_score_subtract.bias
        hypotheses[f'divisor_{divisor}']=dict(prediction=float(value.reshape(-1)[0]),absolute_error=abs(float(value.reshape(-1)[0])-target_error))
    for name,value in [('signed',weight),('abs',weight.abs()),('signed_epsilon',weight+1e-6),('abs_epsilon',weight.abs()+1e-6)]:
        hypotheses[name]=dict(prediction=float(value.reshape(-1)[0]),absolute_error=abs(float(value.reshape(-1)[0])-target_weight))
    row=dict(pair=Path(pair['dist_path']).name,public_error=target_error,public_weight=target_weight,raw_error=float(error.reshape(-1)[0]),raw_weight=float(weight.reshape(-1)[0]),hypotheses=hypotheses)
    rows.append(row);print(json.dumps(row),flush=True)
with open(a.out,'x') as f:json.dump(rows,f,indent=2);f.write('\n')
