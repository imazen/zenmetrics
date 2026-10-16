import csv, os, sys
import numpy as np, torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image
UP = '/home/lilith/tmp/pyiqa-research/up_src/ChrisDud0257_AFINE'
sys.path.insert(0, os.path.join(UP, 'QuickInference'))
import importlib.util
spec = importlib.util.spec_from_file_location('clip_model', os.path.join(UP,'QuickInference','CLIP_ReturnFea','clip','model.py'))
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
from utils.network_archs import (AFINEQhead, AFINEDhead, AFINENLM_NR_Fit, AFINENLM_FR_Fit_with_limit, AFINELearnLambda)
torch.set_grad_enabled(False)
clip = cm.build_model(torch.jit.load('/home/lilith/tmp/nn-port-scratch/afine-ckpt/ViT-B-32.pt',map_location='cpu').state_dict()).float()
ck = torch.load('/home/lilith/tmp/nn-port-scratch/afine-ckpt/afine.pth', map_location='cpu', weights_only=False)
clip.load_state_dict(ck['finetuned_clip']); clip.eval()
qh=AFINEQhead(); qh.load_state_dict(ck['natural']); qh.eval()
dh=AFINEDhead(); dh.load_state_dict(ck['fidelity']); dh.eval()
sfr_t=AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01); sfr_t.load_state_dict(ck['fidelity_scale']); sfr_t.eval()
snr_t=AFINENLM_NR_Fit(yita1=2,yita2=-2,yita3=4.9592,yita4=21.5968); snr_t.load_state_dict(ck['natural_scale']); snr_t.eval()
ad_t=AFINELearnLambda(k=5); ad_t.load_state_dict(ck['adapter']); ad_t.eval()
MEAN=(0.48145466,0.4578275,0.40821073); STD=(0.26862954,0.26130258,0.27577711)
IM=torchvision_IM=__import__('torchvision').transforms.InterpolationMode
def lt(img):
    t=torch.from_numpy(np.asarray(img,np.float32)/255.).permute(2,0,1)
    return torch.stack([(t[i]-m)/s for i,(m,s) in enumerate(zip(MEAN,STD))]).unsqueeze(0)
def rs_cc(img, size, resample, aa=True):  # torchvision resize shorter->size, center_crop size
    t=torch.from_numpy(np.asarray(img,np.float32)/255.).permute(2,0,1)
    t=tvf.resize(t,size,interpolation=resample,antialias=aa)
    t=tvf.center_crop(t,size if isinstance(size,int) else size[0])
    t=torch.stack([(t[i]-m)/s for i,(m,s) in enumerate(zip(MEAN,STD))])
    return t.unsqueeze(0)
def score(rt,dt):
    _,fd=clip.encode_image(dt); _,fr=clip.encode_image(rt)
    nd=qh(dt,fd); nr=qh(rt,fr); fid=dh(dt,rt,fd,fr)
    return ad_t(snr_t(nd),snr_t(nr),sfr_t(fid)).item()
pub={r['distorted']:float(r['A-FINE']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub/metrics_subset.csv'))}
rows=list(csv.DictReader(open('/home/lilith/tmp/aic-sub/pairs.tsv'),delimiter='\t'))
sel=[rows[0]['distorted'],rows[1]['distorted'],rows[10]['distorted'],rows[30]['distorted'],rows[60]['distorted']]
for r in rows:
    if r['distorted'] not in sel: continue
    rim,dim=Image.open(r['ref_path']).convert('RGB'),Image.open(r['dist_path']).convert('RGB')
    line=f"{r['distorted']} pub={pub[r['distorted']]:+.4f} |"
    for tag,fn in [
        ('clip224both', lambda i: rs_cc(i,224,IM.BICUBIC)),
        ('clip224bic_noAA', lambda i: rs_cc(i,224,IM.BICUBIC,aa=False)),
        ('ckdnconv', None),
        ('cc832x928', lambda i: lt(i.crop(((i.width-832)//2,(i.height-928)//2,(i.width+832)//2,(i.height+928)//2)))),
    ]:
        if tag=='ckdnconv':
            s=score(rs_cc(dim,329,IM.BICUBIC),rs_cc(rim,329,IM.NEAREST))
        else:
            s=score(fn(rim),fn(dim))
        line+=f' {tag}={s:+.4f}'
    print(line,flush=True)
