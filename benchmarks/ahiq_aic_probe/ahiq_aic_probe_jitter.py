exec(open('/home/lilith/tmp/nn-port-scratch/ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np, os, csv
from PIL import Image
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
M=np.array([.485,.456,.406],np.float32); S=np.array([.229,.224,.225],np.float32)
def prepn(pil):
    a=np.asarray(pil.convert('RGB'),dtype=np.float32)/255.0
    return torch.from_numpy(((a-M)/S).transpose(2,0,1)).unsqueeze(0)
for dn in ['PTC_S04_JAI_01.png','PTC_S01_AVIF_01.png']:
    rn=dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r=prepn(Image.open(os.path.join(IMG,rn))); d=prepn(Image.open(os.path.join(IMG,dn)))
    h,w=d.shape[2],d.shape[3]
    def sN(n,seed):
        np.random.seed(seed); s=0.0
        for i in range(n):
            t=np.random.randint(0,h-224); l=np.random.randint(0,w-224)
            s+=score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224])
        return s/n
    outs={}
    for n in [10,12,16,20]:
        outs[f's{n}'] = sN(n,1919)
    jit=[sN(20,sd) for sd in [1919,1,2,3,42,123]]
    outs['s20_jit_mean']=float(np.mean(jit)); outs['s20_jit_std']=float(np.std(jit))
    # median of 20 seeded crops
    np.random.seed(1919); vals=[]
    for i in range(20):
        t=np.random.randint(0,h-224); l=np.random.randint(0,w-224)
        vals.append(score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224]))
    outs['med20']=float(np.median(vals))
    print(dn, 'pub=%.4f'%pub[dn], {k:round(v,4) for k,v in outs.items()}, flush=True)
