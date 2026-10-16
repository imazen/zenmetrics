"""Test weight-sign hypotheses using our reconstruction and opaque oracle."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser(description=__doc__)
for name in ('upstream','checkpoint','out'):p.add_argument('--'+name,required=True)
a=p.parse_args()
spec=importlib.util.spec_from_file_location('independent_pieapp',Path(__file__).with_name('network.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
state=torch.load(a.checkpoint,map_location='cpu')
model=module.PieAPP(state).cuda().eval()
for parameter in model.parameters():parameter.requires_grad_(False)
final=model.fc2_weight
receipt=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),final_weight_min=float(final.weight.min()),final_weight_max=float(final.weight.max()),final_bias=float(final.bias.reshape(-1)[0]))
if receipt['final_weight_min']>=0 and receipt['final_bias']>=0:
    receipt['conclusion']='Final layer has nonnegative weights and bias after a ReLU; raw weights are nonnegative for every input, so abs is observationally redundant for this checkpoint.'
else:
    torch.manual_seed(20261016)
    r=torch.randint(0,256,(1,3,64,64),device='cuda').float()
    d=r.clone().requires_grad_(True)
    with torch.enable_grad():
        optimizer=torch.optim.Adam([d],lr=5.0)
        for step in range(100):
            optimizer.zero_grad()
            _,weight=model.raw(r,d)
            weight.sum().backward();optimizer.step()
            with torch.no_grad():d.clamp_(0,255)
            if float(weight.detach().min()) < -1:break
    d=d.detach().round()
    _,raw=model.raw(r,d)
    receipt.update(steps=step+1,rounded_raw_weight=float(raw.reshape(-1)[0]),reference_bgr_nchw_u8_sha256=hashlib.sha256(r.cpu().numpy().astype('uint8').tobytes()).hexdigest(),distorted_bgr_nchw_u8_sha256=hashlib.sha256(d.cpu().numpy().astype('uint8').tobytes()).hexdigest())
    sys.path.insert(0,a.upstream)
    from model.PieAPPv0pt1_PT import PieAPP
    oracle=PieAPP(1,1).cuda().eval();oracle.load_state_dict(state)
    first=oracle.compute_score(d,r);repeat=oracle.compute_score(d,r)
    assert torch.equal(first[2],repeat[2]),'public weight changed on repeat'
    receipt.update(public_weight=float(first[2].reshape(-1)[0]),public_weight_repeat=float(repeat[2].reshape(-1)[0]),signed_epsilon=float((raw+1e-6).reshape(-1)[0]),abs_epsilon=float((raw.abs()+1e-6).reshape(-1)[0]))
    receipt['conclusion']='Compare retained public output to signed/absolute hypotheses; input search uses only the independently written network.'
with open(a.out,'x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
