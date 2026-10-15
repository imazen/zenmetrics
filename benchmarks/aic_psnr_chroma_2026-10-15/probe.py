import csv, sys, numpy as np
from PIL import Image
pub = {r['distorted']: r for r in csv.DictReader(open('/home/lilith/tmp/aic-sub/metrics_subset.csv'))}
pairs = list(csv.DictReader(open('/home/lilith/tmp/aic-sub/pairs.tsv'), delimiter='\t'))
def load(p): return np.asarray(Image.open(p).convert('RGB')).astype(np.float64)
def yuv(img):
    r,g,b = img[...,0],img[...,1],img[...,2]
    y = 16 + (65.481*r + 128.553*g + 24.966*b)/255
    cb = 128 + (-37.797*r - 74.203*g + 112*b)/255
    cr = 128 + (112*r - 93.786*g - 18.214*b)/255
    return y,cb,cr
def rnd(x): return np.floor(x+0.5)
def sws(img):
    # ffmpeg swscale rgb24 -> yuv444p, BT.601 limited, RGB2YUV_SHIFT=15 integer path
    S=15; r,g,b=[img[...,i].astype(np.int64) for i in range(3)]
    def c(v): return int(round(v*(1<<S)))
    RY,GY,BY=c(0.299*219/255),c(0.587*219/255),c(0.114*219/255)
    RU,GU,BU=c(-0.169*224/255),c(-0.331*224/255),c(0.500*224/255)
    RV,GV,BV=c(0.500*224/255),c(-0.419*224/255),c(-0.081*224/255)
    y=(RY*r+GY*g+BY*b+(16<<S)+(1<<(S-1)))>>S
    u=(RU*r+GU*g+BU*b+(128<<S)+(1<<(S-1)))>>S
    v=(RV*r+GV*g+BV*b+(128<<S)+(1<<(S-1)))>>S
    return y.astype(float),u.astype(float),v.astype(float)
def sub420(p):
    h,w=p.shape; h2,w2=h//2*2,w//2*2
    q=p[:h2,:w2]; return (q[0::2,0::2]+q[1::2,0::2]+q[0::2,1::2]+q[1::2,1::2])/4
def psnr(a,b,peak=255.0):
    m=np.mean((a-b)**2); return 10*np.log10(peak*peak/m) if m>0 else float('inf')
variants = {
 'f444_round': lambda im: [rnd(c) for c in yuv(im)],
 'f444_float': lambda im: list(yuv(im)),
 'sws444': sws,
 'f420_avg_then_round': lambda im: [rnd(c) if i==0 else rnd(sub420(c)) for i,c in enumerate(yuv(im))],
 'f420_round_then_avg_round': lambda im: [rnd(c) if i==0 else rnd(sub420(rnd(c))) for i,c in enumerate(yuv(im))],
 'f420_avg_float': lambda im: [rnd(c) if i==0 else sub420(c) for i,c in enumerate(yuv(im))],
}
res = {k: {'Y':[], 'Cb':[], 'Cr':[], '611':[]} for k in variants}
for p in pairs:
    R,D = load(p['ref_path']), load(p['dist_path']); row = pub[p['distorted']]
    for k,f in variants.items():
        yr,cbr,crr = f(R); yd,cbd,crd = f(D)
        py,pcb,pcr = psnr(yr,yd),psnr(cbr,cbd),psnr(crr,crd)
        p611 = (6*py+pcb+pcr)/8
        for key,val,col in (('Y',py,'PSNR-Y'),('Cb',pcb,'PSNR-Cb'),('Cr',pcr,'PSNR-Cr'),('611',p611,'PSNR-YCbCr611')):
            res[k][key].append(abs(val-float(row[col])))
print(f"n={len(pairs)}")
for k,v in res.items():
    print(k, '  '.join(f"{c}: med {np.median(a):.2e} max {np.max(a):.2e}" for c,a in v.items()))
