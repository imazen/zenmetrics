# Dump every A-FINE output quantity on AIC pairs at native pad32,
# to identify which term the published column is.
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
def norm01(img):
    t = torch.from_numpy(np.asarray(img, np.float32)/255.).permute(2,0,1)
    return torch.stack([(t[i]-m)/s for i,(m,s) in enumerate(zip(MEAN,STD))]).unsqueeze(0)
def pad32(t):
    _,_,h,w = t.shape
    nh,nw = h+(32-h%32)%32, w+(32-w%32)%32
    return F.interpolate(t,size=(nh,nw),mode='bicubic',align_corners=False) if (nh,nw)!=(h,w) else t

def dump(rt, dt):
    _,fd = clip_model.encode_image(dt); _,fr = clip_model.encode_image(rt)
    nd = qh(dt,fd); nr = qh(rt,fr); fid = dh(dt,rt,fd,fr)
    nds,nrs,fids = snr(nd),snr(nr),sfr(fid)
    allv = ad(nds,nrs,fids)
    return nd.item(), nr.item(), fid.item(), nds.item(), nrs.item(), fids.item(), allv.item()

pub={r['distorted']:float(r['A-FINE']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
rows=list(csv.DictReader(open('/home/lilith/tmp/nn-phase0/aic-crop/pairs.tsv'),delimiter='\t'))
sel = sys.argv[1].split(',') if len(sys.argv)>1 else [rows[0]['distorted'],rows[1]['distorted'],rows[10]['distorted'],rows[30]['distorted'],rows[60]['distorted']]
for r in rows:
    if r['distorted'] not in sel: continue
    rim = pad32(norm01(Image.open(r['ref_path']).convert('RGB')))
    dim = pad32(norm01(Image.open(r['dist_path']).convert('RGB')))
    nd,nr,fid,nds,nrs,fids,allv = dump(rim,dim)
    nd2,nr2,fid2,nds2,nrs2,fids2,allv2 = dump(dim,rim)  # swapped
    print(f"{r['distorted']} pub={pub[r['distorted']]:+.4f}")
    print(f"  nd={nd:+.3f} nr={nr:+.3f} fid={fid:+.3f} | nr_scale(d)={nds:+.4f} nr_scale(r)={nrs:+.4f} fr_scale={fids:+.4f} all={allv:+.4f}")
    print(f"  swapped: all={allv2:+.4f}  (nds={nds2:+.4f} nrs={nrs2:+.4f} fids={fids2:+.4f})", flush=True)
