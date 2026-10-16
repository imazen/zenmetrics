# Systematic probe: which pipeline variant produces the published A-FINE
# column? Grid over: image scale, BGR swap, ImageNet vs CLIP norm,
# finetuned vs vanilla CLIP trunk.
import csv, os, sys
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

CLIP_PTH='/home/lilith/tmp/nn-port-scratch/afine-ckpt/ViT-B-32.pt'
AF_PTH='/home/lilith/tmp/nn-port-scratch/afine-ckpt/afine.pth'
base_sd = torch.jit.load(CLIP_PTH, map_location='cpu').state_dict()
ck = torch.load(AF_PTH, map_location='cpu', weights_only=False)
clip_ft = cm.build_model(base_sd).float(); clip_ft.load_state_dict(ck['finetuned_clip']); clip_ft.eval()
clip_vanilla = cm.build_model(base_sd).float(); clip_vanilla.eval()
qh = AFINEQhead(); qh.load_state_dict(ck['natural']); qh.eval()
dh = AFINEDhead(); dh.load_state_dict(ck['fidelity']); dh.eval()
sfr = AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01)
sfr.load_state_dict(ck['fidelity_scale']); sfr.eval()
snr = AFINENLM_NR_Fit(yita1=2,yita2=-2,yita3=4.9592,yita4=21.5968)
snr.load_state_dict(ck['natural_scale']); snr.eval()
ad = AFINELearnLambda(k=5); ad.load_state_dict(ck['adapter']); ad.eval()

CLIP_MEAN=(0.48145466,0.4578275,0.40821073); CLIP_STD=(0.26862954,0.26130258,0.27577711)
IN_MEAN=(0.485,0.456,0.406); IN_STD=(0.229,0.224,0.225)

def load_t(img, mean, std, bgr=False):
    a = np.asarray(img, np.float32)/255.
    if bgr: a = a[:,:,::-1].copy()
    t = torch.from_numpy(a).permute(2,0,1)
    return torch.stack([(t[i]-m)/s for i,(m,s) in enumerate(zip(mean,std))]).unsqueeze(0)

def pad32(t):
    _,_,h,w = t.shape
    nh,nw = h+(32-h%32)%32, w+(32-w%32)%32
    return F.interpolate(t,size=(nh,nw),mode='bicubic',align_corners=False) if (nh,nw)!=(h,w) else t

def score(clip, rt, dt):
    _,fd = clip.encode_image(dt); _,fr = clip.encode_image(rt)
    nd=qh(dt,fd); nr=qh(rt,fr); fid=dh(dt,rt,fd,fr)
    return ad(snr(nd),snr(nr),sfr(fid)).item()

pub={r['distorted']:float(r['A-FINE']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub/metrics_subset.csv'))}
rows=list(csv.DictReader(open('/home/lilith/tmp/aic-sub/pairs.tsv'),delimiter='\t'))
sel = sys.argv[1].split(',') if len(sys.argv)>1 else [rows[0]['distorted'],rows[10]['distorted'],rows[30]['distorted']]

for r in rows:
    if r['distorted'] not in sel: continue
    rim_full,dim_full = Image.open(r['ref_path']).convert('RGB'),Image.open(r['dist_path']).convert('RGB')
    line = f"{r['distorted']} pub={pub[r['distorted']]:+.4f} | "
    for scale in (1.0, 0.75, 0.5, 0.25):
        rim,dim = rim_full,dim_full
        if scale != 1.0:
            rim = rim.resize((max(32,int(rim.width*scale)),max(32,int(rim.height*scale))),Image.BICUBIC)
            dim = dim.resize((max(32,int(dim.width*scale)),max(32,int(dim.height*scale))),Image.BICUBIC)
        s = score(clip_ft, pad32(load_t(rim,CLIP_MEAN,CLIP_STD)), pad32(load_t(dim,CLIP_MEAN,CLIP_STD)))
        line += f's{scale}={s:+.4f} '
    rim,dim = rim_full,dim_full
    s = score(clip_ft, pad32(load_t(rim,CLIP_MEAN,CLIP_STD,True)), pad32(load_t(dim,CLIP_MEAN,CLIP_STD,True)))
    line += f'| bgr={s:+.4f} '
    s = score(clip_ft, pad32(load_t(rim,IN_MEAN,IN_STD)), pad32(load_t(dim,IN_MEAN,IN_STD)))
    line += f'| inet={s:+.4f} '
    s = score(clip_vanilla, pad32(load_t(rim,CLIP_MEAN,CLIP_STD)), pad32(load_t(dim,CLIP_MEAN,CLIP_STD)))
    line += f'| vanilla={s:+.4f} |'
    print(line, flush=True)
