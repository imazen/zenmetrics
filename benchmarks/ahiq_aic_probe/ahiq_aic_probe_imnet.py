exec(open('/home/lilith/tmp/nn-port-scratch/ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np, os, csv
from PIL import Image
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
files = sorted(f for f in os.listdir(IMG) if '_Ref_' not in f)

def prepn(pil):
    a = np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.0
    a = (a - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0)

rows = []
for dn in files[:6]:
    rn = dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r = prepn(Image.open(os.path.join(IMG, rn))); d = prepn(Image.open(os.path.join(IMG, dn)))
    h, w = d.shape[2], d.shape[3]
    s9 = sum(score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w))/9
    np.random.seed(1919); s20=0.0
    for i in range(20):
        t=np.random.randint(0,h-224); l=np.random.randint(0,w-224)
        s20 += score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224])
    s20/=20
    rows.append((dn, pub[dn], s9, s20))
    print(f'{dn}: pub={pub[dn]:.6f} imnet_s9={s9:.6f} imnet_s20={s20:.6f}', flush=True)
diffs9=[abs(p-a) for _,p,a,_ in rows]; diffs20=[abs(p-a) for _,p,_,a in rows]
import statistics
print('med|d| s9', statistics.median(diffs9), 's20', statistics.median(diffs20))
