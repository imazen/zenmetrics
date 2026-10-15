exec(open('ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np
from PIL import Image
import os, csv
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
files = set(os.listdir(IMG))
sel = [f for f in sorted(files) if '_Ref_' not in f][:2]
for dn in sel:
    rn = dn.split('_')[0]+'_'+dn.split('_')[1]+'_Ref_00.png'
    r = prep(Image.open(os.path.join(IMG, rn)))
    d = prep(Image.open(os.path.join(IMG, dn)))
    h, w = d.shape[2], d.shape[3]
    def grid(ny, nx):
        ts = [round(i*(h-224)/(ny-1)) for i in range(ny)]
        ls = [round(i*(w-224)/(nx-1)) for i in range(nx)]
        s = 0.0; n=0
        for t in ts:
            for l in ls:
                s += score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224]); n+=1
        return s/n
    def flips9():
        s=0.0
        for (dd,rr) in [(d,r),(torch.flip(d,[3]),torch.flip(r,[3]))]:
            for t,l in crops9(h,w):
                s += score(dd[:,:,t:t+224,l:l+224], rr[:,:,t:t+224,l:l+224])
        return s/18
    out = {'pub':pub[dn]}
    for g in [(2,2),(3,3),(4,4),(5,5),(6,6)]:
        out['g%d%d'%g] = grid(*g)
    out['flip18'] = flips9()
    # swapped args
    out['swap9'] = sum(score(r[:,:,t:t+224,l:l+224], d[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w))/9
    print(dn, {k: round(v,4) for k,v in out.items()})
