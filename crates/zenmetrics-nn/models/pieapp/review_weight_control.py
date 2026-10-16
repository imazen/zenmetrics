"""Opaque operator identification via test-only checkpoint interventions.

Tensor keys/shapes come from checkpoint metadata. No official internals are
read: only public load_state_dict and compute_score are used. The checkpoint
file and production export are never modified.
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
path=Path(a.checkpoint)
digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest=='0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0'
state=torch.load(a.checkpoint,map_location='cpu')
sys.path.insert(0,a.upstream)
from model.PieAPPv0pt1_PT import PieAPP
model=PieAPP(1,1).cuda().eval()
image=torch.zeros((1,3,64,64),device='cuda')
rows=[]
for value in (-2.0,-1.0,-0.25,0.0,0.25,1.0,2.0):
    control=state.copy()
    control['fc2_weight.weight']=torch.zeros_like(state['fc2_weight.weight'])
    control['fc2_weight.bias']=torch.full_like(state['fc2_weight.bias'],value)
    model.load_state_dict(control)
    first=model.compute_score(image,image)
    repeated=model.compute_score(image,image)
    assert torch.equal(first[2],repeated[2]),'controlled public weight changed on repeat'
    got=float(first[2].reshape(-1)[0])
    predicted=float((torch.tensor(value,dtype=torch.float32).abs()+1e-6))
    assert got==predicted,(value,got,predicted)
    rows.append(dict(forced_raw_weight=value,public_weight=got,public_weight_repeat=float(repeated[2].reshape(-1)[0]),abs_epsilon_prediction=predicted,signed_epsilon_prediction=float(torch.tensor(value,dtype=torch.float32)+1e-6)))
assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
receipt=dict(checkpoint_sha256=digest,torch_version=torch.__version__,cuda_version=torch.version.cuda,intervention='Oracle-only in-memory parameter copies: zero fc2_weight.weight and set fc2_weight.bias; unmodified source program; original file and production model untouched',rows=rows)
with open(a.out,'x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
