"""Measure public weights under oracle-only bias/projection controls.

The earlier review_weight_control.py preserves the falsified abs hypothesis.
This measurement reads no official source or feature internals and writes no
checkpoint: only public state loading and public output observations are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser(description=__doc__)
for name in ('upstream','checkpoint','out'):p.add_argument('--'+name,required=True)
a=p.parse_args()
path=Path(a.checkpoint);digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest=='0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0'
state=torch.load(a.checkpoint,map_location='cpu')
sys.path.insert(0,a.upstream)
from model.PieAPPv0pt1_PT import PieAPP
model=PieAPP(1,1).cuda().eval()
image=torch.zeros((1,3,64,64),device='cuda')
rows=[]
for mode in ('bias','projection'):
    for value in (-2.0,-1.0,-0.25,0.0,0.25,1.0,2.0):
        control=state.copy()
        control['fc2_weight.weight']=torch.zeros_like(state['fc2_weight.weight'])
        control['fc2_weight.bias']=torch.full_like(state['fc2_weight.bias'],value)
        if mode=='projection':
            control['fc1_weight.weight']=torch.zeros_like(state['fc1_weight.weight'])
            control['fc1_weight.bias']=torch.ones_like(state['fc1_weight.bias'])
            control['fc2_weight.weight']=torch.full_like(state['fc2_weight.weight'],value/512)
            control['fc2_weight.bias']=torch.zeros_like(state['fc2_weight.bias'])
        model.load_state_dict(control)
        first=model.compute_score(image,image);repeat=model.compute_score(image,image)
        assert torch.equal(first[2],repeat[2]),'controlled weight changed on repeat'
        rows.append(dict(mode=mode,forced_raw_weight=value,public_weight=float(first[2].reshape(-1)[0]),public_weight_repeat=float(repeat[2].reshape(-1)[0]),signed_epsilon=float(torch.tensor(value,dtype=torch.float32)+1e-6),abs_epsilon=float(torch.tensor(value,dtype=torch.float32).abs()+1e-6)))
assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
receipt=dict(checkpoint_sha256=digest,torch_version=torch.__version__,cuda_version=torch.version.cuda,intervention='Oracle-only in-memory parameter copies; bias and projection separately controlled; original file and production checkpoint untouched',rows=rows)
with open(a.out,'x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
