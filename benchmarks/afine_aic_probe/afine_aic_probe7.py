# afine_all under: trained params vs DEFAULT constructor params (no scale
# checkpoint), and afine_all at several resize targets. Pub col is ~0.5-1.4
# higher than native afine_all.
import csv, os, sys
import numpy as np, torch
import torch.nn.functional as F
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

# two NLM configurations: trained vs default-constructor
sfr_t=AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01); sfr_t.load_state_dict(ck['fidelity_scale']); sfr_t.eval()
snr_t=AFINENLM_NR_Fit(yita1=2,yita2=-2,yita3=4.9592,yita4=21.5968); snr_t.load_state_dict(ck['natural_scale']); snr_t.eval()
ad_t=AFINELearnLambda(k=5); ad_t.load_state_dict(ck['adapter']); ad_t.eval()
sfr_d=AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01); sfr_d.eval()
snr_d=AFINENLM_NR_Fit(yita1=2,yita2=-2,yita3=4.9592,yita4=21.5968); snr_d.eval()
ad_d=AFINELearnLambda(k=5); ad_d.eval()
MEAN=(0.48145466,0.4578275,0.40821073); STD=(0.26862954,0.26130258,0.27577711)
def lt(img):
    t=torch.from_numpy(np.asarray(img,np.float32)/255.).permute(2,0,1)
    return torch.stack([(t[i]-m)/s for i,(m,s) in enumerate(zip(MEAN,STD))]).unsqueeze(0)
def pad32(t):
    _,_,h,w=t.shape; nh,nw=h+(32-h%32)%32,w+(32-w%32)%32
    return F.interpolate(t,size=(nh,nw),mode='bicubic',align_corners=False) if (nh,nw)!=(h,w) else t
def terms(rim,dim):
    _,fd=clip.encode_image(dim); _,fr=clip.encode_image(rim)
    return qh(dim,fd).item(),qh(rim,fr).item(),dh(dim,rim,fd,fr).item()
def allv(nd,nr,fid,sn,sf,a):
    nds=sn(torch.tensor(nd)); nrs=sn(torch.tensor(nr)); fs=sf(torch.tensor(fid))
    return a(nds,nrs,fs).item(), fs.item()

pub={r['distorted']:float(r['A-FINE']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub/metrics_subset.csv'))}
rows=list(csv.DictReader(open('/home/lilith/tmp/aic-sub/pairs.tsv'),delimiter='\t'))
sel=[rows[0]['distorted'],rows[1]['distorted'],rows[10]['distorted'],rows[30]['distorted'],rows[60]['distorted']]
for r in rows:
    if r['distorted'] not in sel: continue
    rf,df=Image.open(r['ref_path']).convert('RGB'),Image.open(r['dist_path']).convert('RGB')
    nd,nr,fid = terms(pad32(lt(rf)),pad32(lt(df)))
    a_t,f_t = allv(nd,nr,fid,snr_t,sfr_t,ad_t)
    a_d,f_d = allv(nd,nr,fid,snr_d,sfr_d,ad_d)
    line=f"{r['distorted']} pub={pub[r['distorted']]:+.4f} | all_trained={a_t:+.4f} all_default={a_d:+.4f} fid_raw={fid:+.4f} fids_d={f_d:+.4f} | resize-shorter:"
    for tgt in (224,288,384,512):
        s=tgt/min(rf.size)
        ri=rf.resize((max(32,int(rf.width*s)),max(32,int(rf.height*s))),Image.BICUBIC)
        di=df.resize((max(32,int(df.width*s)),max(32,int(df.height*s))),Image.BICUBIC)
        nd2,nr2,fid2=terms(pad32(lt(ri)),pad32(lt(di)))
        av,_=allv(nd2,nr2,fid2,snr_t,sfr_t,ad_t)
        line+=f' {tgt}={av:+.4f}'
    print(line,flush=True)
