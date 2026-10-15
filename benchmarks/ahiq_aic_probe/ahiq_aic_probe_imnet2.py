exec(open('/home/lilith/tmp/nn-port-scratch/ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np, os, csv
from PIL import Image
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
M=np.array([.485,.456,.406],np.float32); S=np.array([.229,.224,.225],np.float32)
def prepn(pil):
    a=np.asarray(pil.convert('RGB'),dtype=np.float32)/255.0
    a=(a-M)/S
    return torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0)
def grid(d,r,ny,nx):
    h,w=d.shape[2],d.shape[3]
    ts=[round(i*(h-224)/(ny-1)) for i in range(ny)]; ls=[round(i*(w-224)/(nx-1)) for i in range(nx)]
    return sum(score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224]) for t in ts for l in ls)/(ny*nx)
def s5(d,r):
    h,w=d.shape[2],d.shape[3]
    return sum(score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w)[:5])/5
def s10(d,r):
    df,rf=torch.flip(d,[3]),torch.flip(r,[3])
    return (s5(d,r)+s5(df,rf))/2
def s9(d,r):
    h,w=d.shape[2],d.shape[3]
    return sum(score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w))/9
def s18(d,r):
    df,rf=torch.flip(d,[3]),torch.flip(r,[3])
    return (s9(d,r)+s9(df,rf))/2
for dn in ['PTC_S01_AVIF_01.png','PTC_S02_CCM_01.png','PTC_S04_JAI_01.png']:
    rn=dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r=prepn(Image.open(os.path.join(IMG,rn))); d=prepn(Image.open(os.path.join(IMG,dn)))
    out={'pub':pub[dn],'g2':grid(d,r,2,2),'g3':grid(d,r,3,3),'g4':grid(d,r,4,4),'s5':s5(d,r),'s10':s10(d,r),'s9':s9(d,r),'s18':s18(d,r)}
    print(dn, {k:round(v,4) for k,v in out.items()}, flush=True)
