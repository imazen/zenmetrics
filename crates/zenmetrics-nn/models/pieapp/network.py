"""Independent PieAPP network reconstruction from CVPR 2018 Fig. 4 and tensor metadata.

No official implementation or pyiqa source is imported. Execution conventions
are selected only through black-box oracle comparisons.
"""
import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_SHA256 = '0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0'


class PieAPP(nn.Module):
    def __init__(self, state, *, channel_last=False):
        super().__init__()
        self.channel_last = channel_last
        for i in range(1,12):
            shape = state[f'conv{i}.weight'].shape
            setattr(self, f'conv{i}', nn.Conv2d(shape[1],shape[0],3,padding=1))
        for name in ('fc1_score','fc2_score','fc1_weight','fc2_weight'):
            shape = state[name+'.weight'].shape
            setattr(self, name, nn.Linear(shape[1],shape[0]))
        self.ref_score_subtract = nn.Module()
        self.ref_score_subtract.weight = nn.Parameter(torch.empty_like(state['ref_score_subtract.weight']))
        self.ref_score_subtract.bias = nn.Parameter(torch.empty_like(state['ref_score_subtract.bias']))
        self.load_state_dict(state, strict=True)
        self.eval()

    def features(self, x):
        values=[]
        for i in range(1,12):
            x=F.relu(getattr(self,f'conv{i}')(x))
            if i in (3,5,7,9,11):
                value=x.permute(0,2,3,1) if self.channel_last else x
                values.append(value.flatten(1))
            if i%2==0:x=F.max_pool2d(x,2,2)
        return torch.cat(values,dim=1),values[-1]

    def raw(self, reference, distorted):
        rf,rw=self.features(reference);df,dw=self.features(distorted)
        error=self.fc2_score(F.relu(self.fc1_score(rf-df)))
        weight=self.fc2_weight(F.relu(self.fc1_weight(rw-dw)))
        return error,weight

    def forward(self, reference, distorted):
        error,weight=self.raw(reference,distorted)
        # Scalar normalization and signed weights match controlled public outputs.
        error=(error/100)*self.ref_score_subtract.weight+self.ref_score_subtract.bias
        weight=weight+1e-6
        return torch.cat((error,weight),dim=1)
