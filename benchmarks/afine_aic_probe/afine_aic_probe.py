# Probe which preprocessing AIC-4 used for its A-FINE column.
# Scores a handful of PTC pairs under candidate transforms with the
# upstream torch pipeline and compares to the published values.
import csv, math, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

UP = '/home/lilith/tmp/pyiqa-research/up_src/ChrisDud0257_AFINE'
sys.path.insert(0, os.path.join(UP, 'QuickInference'))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'clip_model', os.path.join(UP, 'QuickInference', 'CLIP_ReturnFea', 'clip', 'model.py'))
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
from utils.network_archs import (AFINEQhead, AFINEDhead, AFINENLM_NR_Fit,
                                 AFINENLM_FR_Fit_with_limit, AFINELearnLambda)

torch.set_grad_enabled(False)
clip_model = cm.build_model(torch.jit.load('/home/lilith/tmp/nn-port-scratch/afine-ckpt/ViT-B-32.pt', map_location='cpu').state_dict()).float()
ck = torch.load('/home/lilith/tmp/nn-port-scratch/afine-ckpt/afine.pth', map_location='cpu', weights_only=False)
clip_model.load_state_dict(ck['finetuned_clip']); clip_model.eval()
qh = AFINEQhead(); qh.load_state_dict(ck['natural']); qh.eval()
dh = AFINEDhead(); dh.load_state_dict(ck['fidelity']); dh.eval()
sfr = AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01)
sfr.load_state_dict(ck['fidelity_scale']); sfr.eval()
snr = AFINENLM_NR_Fit(yita1=2,yita2=-2,yita3=4.9592,yita4=21.5968)
snr.load_state_dict(ck['natural_scale']); snr.eval()
ad = AFINELearnLambda(k=5); ad.load_state_dict(ck['adapter']); ad.eval()

MEAN=(0.48145466,0.4578275,0.40821073); STD=(0.26862954,0.26130258,0.27577711)

def norm01(img):  # PIL RGB -> normalized [1,3,H,W]
    t = torch.from_numpy(np.asarray(img, np.float32)/255.).permute(2,0,1)
    return torch.stack([(t[i]-m)/s for i,(m,s) in enumerate(zip(MEAN,STD))]).unsqueeze(0)

def pad32(t):  # upstream: bicubic-upsample normalized tensor to mult of 32
    _,_,h,w = t.shape
    nh,nw = h+(32-h%32)%32, w+(32-w%32)%32
    if nh!=h or nw!=w:
        t = F.interpolate(t,size=(nh,nw),mode='bicubic',align_corners=False)
    return t

def score_pair(rt, dt):  # normalized tensors in, afine_all out
    _,fd = clip_model.encode_image(dt); _,fr = clip_model.encode_image(rt)
    nd = qh(dt,fd); nr = qh(rt,fr); fid = dh(dt,rt,fd,fr)
    return ad(snr(nd),snr(nr),sfr(fid)).item()

def t_resize329_cc288(img, resample):  # torchvision semantics via PIL? use torch functional
    import torchvision.transforms.functional as tvf
    t = torch.from_numpy(np.asarray(img,np.float32)/255.).permute(2,0,1)
    t = tvf.resize(t, 329, interpolation=resample, antialias=True)
    t = tvf.center_crop(t, 288)
    return t.permute(1,2,0).numpy()*255.

import torchvision
IM = torchvision.transforms.InterpolationMode
variants = {
    'native_pad32':      lambda img: pad32(norm01(img)),
    'aa329cc288_both':   lambda img: pad32(norm01(Image.fromarray(t_resize329_cc288(img,IM.BICUBIC).astype(np.uint8)))),
    'aa329cc288_nnref':  None,  # special-cased below
    'cc288_only':        lambda img: pad32(norm01(img.crop(((img.width-288)//2,(img.height-288)//2,(img.width+288)//2,(img.height+288)//2)))),
    'resize288_anam':    lambda img: pad32(norm01(img.resize((288,288),Image.BICUBIC))),
    'rs512_bicubic':     lambda img: pad32(norm01(img.resize((int(img.width*512/min(img.size)), int(img.height*512/min(img.size))),Image.BICUBIC))),
}

pairs=[]
pub={}
for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv')):
    pub[r['distorted']]=float(r['A-FINE'])
rows=list(csv.DictReader(open('/home/lilith/tmp/nn-phase0/aic-crop/pairs.tsv'),delimiter='\t'))
sel = sys.argv[1].split(',') if len(sys.argv)>1 else [rows[0]['distorted'],rows[1]['distorted'],rows[10]['distorted'],rows[30]['distorted'],rows[60]['distorted']]
for r in rows:
    if r['distorted'] in sel: pairs.append((r['ref_path'],r['dist_path'],r['distorted']))

for rp,dp,name in pairs:
    rim,dim = Image.open(rp).convert('RGB'), Image.open(dp).convert('RGB')
    line=f'{name} pub={pub[name]:+.4f} | '
    for vn,fn in variants.items():
        if vn=='aa329cc288_nnref':
            continue
        s = score_pair(fn(rim), fn(dim))
        line += f'{vn}={s:+.4f} '
    # CKDN convention: bicubic-AA on dist, nearest on ref
    rd = Image.fromarray(t_resize329_cc288(dim,IM.BICUBIC).astype(np.uint8))
    rr = Image.fromarray(t_resize329_cc288(rim,IM.NEAREST).astype(np.uint8))
    s = score_pair(pad32(norm01(rr)), pad32(norm01(rd)))
    line += f'aaD_nnR={s:+.4f} |'
    print(line, flush=True)
