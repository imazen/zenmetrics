"""Pin the independent head to measured opaque bias/projection outputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser(description=__doc__)
for name in ('checkpoint','receipt','out'):p.add_argument('--'+name,required=True)
a=p.parse_args();torch.set_grad_enabled(False);torch.set_num_threads(16)
spec=importlib.util.spec_from_file_location('independent_pieapp',Path(__file__).with_name('network.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
model=module.PieAPP(state).eval()
image=torch.zeros((1,3,64,64))
source=json.loads(Path(a.receipt).read_text());rows=[]
for row in source['rows']:
    value=row['forced_raw_weight'];control=state.copy()
    control['fc2_weight.weight']=torch.zeros_like(state['fc2_weight.weight'])
    control['fc2_weight.bias']=torch.full_like(state['fc2_weight.bias'],value)
    if row['mode']=='projection':
        control['fc1_weight.weight']=torch.zeros_like(state['fc1_weight.weight'])
        control['fc1_weight.bias']=torch.ones_like(state['fc1_weight.bias'])
        control['fc2_weight.weight']=torch.full_like(state['fc2_weight.weight'],value/512)
        control['fc2_weight.bias']=torch.zeros_like(state['fc2_weight.bias'])
    model.load_state_dict(control)
    got=float(model(image,image)[0,1]);assert got==row['public_weight'],(row,got)
    rows.append(dict(mode=row['mode'],raw=value,official=row['public_weight'],reconstruction=got))
with open(a.out,'x') as f:json.dump(dict(cases=len(rows),rows=rows),f,indent=2);f.write('\n')
print('Independent reconstruction matches all',len(rows),'opaque operator controls exactly')
