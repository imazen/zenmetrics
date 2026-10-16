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
p.add_argument('--joint',action='store_true')
p.add_argument('--steps',type=int,default=100)
p.add_argument('--lower',type=int,default=0)
p.add_argument('--upper',type=int,default=255)
p.add_argument('--rate',type=float,default=2.0)
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
    batch=4 if a.joint else 1
    r=torch.randint(a.lower,a.upper+1,(batch,3,64,64),device='cuda').float()
    d=(torch.randint(a.lower,a.upper+1,r.shape,device='cuda').float() if a.joint else r.clone()).requires_grad_(True)
    if a.joint:r.requires_grad_(True)
    with torch.enable_grad():
        optimizer=torch.optim.Adam([r,d] if a.joint else [d],lr=a.rate if a.joint else 5.0)
        for step in range(a.steps):
            optimizer.zero_grad()
            _,weight=model.raw(r,d)
            weight.sum().backward();optimizer.step()
            with torch.no_grad():
                d.clamp_(a.lower,a.upper)
                if a.joint:r.clamp_(a.lower,a.upper)
            if float(weight.detach().min()) < -1:break
    r=r.detach().round();d=d.detach().round()
    _,all_raw=model.raw(r,d)
    selected=int(all_raw.reshape(-1).argmin())
    r=r[selected:selected+1];d=d[selected:selected+1]
    _,raw=model.raw(r,d)
    receipt.update(input_domain=[a.lower,a.upper],learning_rate=a.rate if a.joint else 5.0,reference_bgr_nchw_f32_sha256=hashlib.sha256(r.cpu().numpy().tobytes()).hexdigest(),distorted_bgr_nchw_f32_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),search_mode="joint" if a.joint else "identity",batch=batch,selected=selected,steps=step+1,rounded_raw_weight=float(raw.reshape(-1)[0]),reference_bgr_nchw_u8_sha256=(hashlib.sha256(r.cpu().numpy().astype('uint8').tobytes()).hexdigest() if a.lower==0 and a.upper==255 else None),distorted_bgr_nchw_u8_sha256=(hashlib.sha256(d.cpu().numpy().astype('uint8').tobytes()).hexdigest() if a.lower==0 and a.upper==255 else None))
    sys.path.insert(0,a.upstream)
    from model.PieAPPv0pt1_PT import PieAPP
    oracle=PieAPP(1,1).cuda().eval();oracle.load_state_dict(state)
    first=oracle.compute_score(d,r);repeat=oracle.compute_score(d,r)
    assert torch.equal(first[2],repeat[2]),'public weight changed on repeat'
    receipt.update(public_weight=float(first[2].reshape(-1)[0]),public_weight_repeat=float(repeat[2].reshape(-1)[0]),signed_epsilon=float((raw+1e-6).reshape(-1)[0]),abs_epsilon=float((raw.abs()+1e-6).reshape(-1)[0]))
    receipt['conclusion']='Compare retained public output to signed/absolute hypotheses; input search uses only the independently written network.'
with open(a.out,'x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
