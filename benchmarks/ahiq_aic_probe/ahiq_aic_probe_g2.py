exec(open('/home/lilith/tmp/nn-port-scratch/ahiq_aic_probe.py').read().split("import random\nIMG")[0])
import numpy as np, os, csv
from PIL import Image
IMG = '/home/lilith/tmp/aic-sub-crop/img'
pub = {r['distorted']: float(r['AHIQ']) for r in csv.DictReader(open('/home/lilith/tmp/aic-sub-crop/metrics_subset.csv'))}
dn = 'PTC_S02_CCM_01.png'; rn = 'PTC_S02_Ref_00.png'
r = prep(Image.open(os.path.join(IMG, rn))); d = prep(Image.open(os.path.join(IMG, dn)))
h, w = d.shape[2], d.shape[3]
cache = {}
def s(t, l):
    key=(t,l)
    if key not in cache:
        cache[key] = score(d[:,:,t:t+224,l:l+224], r[:,:,t:t+224,l:l+224])
        print(f'  pos {key} -> {cache[key]:.4f}', flush=True)
    return cache[key]
def grid(ny,nx):
    ts=[round(i*(h-224)/(ny-1)) for i in range(ny)]; ls=[round(i*(w-224)/(nx-1)) for i in range(nx)]
    return sum(s(t,l) for t in ts for l in ls)/(ny*nx)
print('pub', pub[dn], flush=True)
for g in [(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]:
    print(f'g{g[0]}x{g[1]} = {grid(*g):.4f}', flush=True)
fl=sum(s(t,l) for t,l in crops9(h,w))
df,rf = torch.flip(d,[3]), torch.flip(r,[3])
sfl=sum(score(df[:,:,t:t+224,l:l+224], rf[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w))
print(f's9={fl/9:.4f} s9flip={sfl/9:.4f} flipavg={(fl+sfl)/18:.4f}', flush=True)
sw=sum(score(r[:,:,t:t+224,l:l+224], d[:,:,t:t+224,l:l+224]) for t,l in crops9(h,w))/9
print(f'swap9={sw:.4f}', flush=True)
