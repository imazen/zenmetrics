exec(open('/home/lilith/tmp/nn-port-scratch/ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np, os, csv
from PIL import Image
IMG = '/home/lilith/tmp/aic-sub-crop/img'
rows = list(csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv')))
pub = {r['distorted']: float(r['AHIQ']) for r in rows}
order = [r['distorted'] for r in rows]
M=np.array([.485,.456,.406],np.float32); S=np.array([.229,.224,.225],np.float32)
def prepn(pil):
    a=np.asarray(pil.convert('RGB'),dtype=np.float32)/255.0
    return torch.from_numpy(((a-M)/S).transpose(2,0,1)).unsqueeze(0)
# upstream default: seed once, stream of 20 randint draws per image continues across the list
np.random.seed(1919)
done=0
for dn in order:
    rn=dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r=prepn(Image.open(os.path.join(IMG,rn))); d=prepn(Image.open(os.path.join(IMG,dn)))
    h,w=d.shape[2],d.shape[3]
    s=0.0
    for i in range(20):
        t=np.random.randint(0,h-224); l=np.random.randint(0,w-224)
        s+=score(d[:,:,t:t+224,l:l+224],r[:,:,t:t+224,l:l+224])
    s/=20
    done+=1
    print(f'{dn}: pub={pub[dn]:.4f} stream20={s:.4f} d={s-pub[dn]:+.4f}', flush=True)
    if done>=5: break
