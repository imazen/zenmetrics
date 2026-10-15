exec(open('/home/lilith/tmp/nn-port-scratch/ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np, os, csv
from PIL import Image
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
dn = 'PTC_S02_CCM_01.png'; rn = 'PTC_S02_Ref_00.png'

def prepv(pil, mode):
    a = np.asarray(pil.convert('RGB'), dtype=np.float32) / 255.0
    if mode == 'pm1': a = (a - 0.5) / 0.5
    elif mode == 'imagenet':
        a = (a - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)
    elif mode == 'bgr_pm1': a = ((a[...,::-1].copy()) - 0.5) / 0.5
    elif mode == 'u8': a = np.asarray(pil.convert('RGB'), dtype=np.float32)  # raw 0-255
    return torch.from_numpy(a.transpose(2,0,1)).unsqueeze(0)

for mode in ['pm1','u8','imagenet','bgr_pm1']:
    r = prepv(Image.open(os.path.join(IMG, rn)), mode)
    d = prepv(Image.open(os.path.join(IMG, dn)), mode)
    h, w = d.shape[2], d.shape[3]
    try:
        s9 = sum(score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w))/9
        print(f'{mode}: {s9:.4f}  (pub {pub[dn]:.4f})', flush=True)
    except Exception as e:
        print(mode, 'ERR', e, flush=True)
