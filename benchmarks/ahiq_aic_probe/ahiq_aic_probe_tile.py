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
def tile224(d,r):
    # center-crop to largest 224-multiple, tile non-overlapping
    h,w=d.shape[2],d.shape[3]
    H=(h//224)*224; W=(w//224)*224
    t0=(h-H)//2; l0=(w-W)//2
    s=0.0;n=0
    for t in range(t0,t0+H,224):
        for l in range(l0,l0+W,224):
            s+=score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224]);n+=1
    return s/n
def tile224_off0(d,r):
    # top-left origin tiling (no centering)
    h,w=d.shape[2],d.shape[3]
    s=0.0;n=0
    for t in range(0,h-223,224):
        for l in range(0,w-223,224):
            s+=score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224]);n+=1
    return s/n
for dn in sorted(f for f in os.listdir(IMG) if '_Ref_' not in f)[:4]:
    rn=dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r=prepn(Image.open(os.path.join(IMG,rn))); d=prepn(Image.open(os.path.join(IMG,dn)))
    print(dn, 'pub=%.4f'%pub[dn], 'tile_c=%.4f'%tile224(d,r), 'tile_0=%.4f'%tile224_off0(d,r), flush=True)
