import csv, numpy as np
from PIL import Image
exec(open('probe.py').read().split('variants = {')[0])
Kr,Kb=0.299,0.114; Kg=1-Kr-Kb
def mk(cb,cr,dtype=np.float64,rounder=rnd,scale=255.0):
    def f(im):
        im=im.astype(dtype); r,g,b=im[...,0],im[...,1],im[...,2]
        return [rounder(dtype(128)+(cb[0]*r+cb[1]*g+cb[2]*b)/dtype(scale)), rounder(dtype(128)+(cr[0]*r+cr[1]*g+cr[2]*b)/dtype(scale))]
    return f
exact_cb=[-Kr/(1-Kb)/2*224, -Kg/(1-Kb)/2*224, 0.5*224]
exact_cr=[0.5*224, -Kg/(1-Kr)/2*224, -Kb/(1-Kr)/2*224]
base_cb=[-37.797,-74.203,112.0]; base_cr=[112.0,-93.786,-18.214]
V={
 'base': mk(base_cb,base_cr),
 'exact': mk(exact_cb,exact_cr),
 '3dec': mk([-0.148*255,-0.291*255,0.439*255],[0.439*255,-0.368*255,-0.071*255]),
 'base_f32': mk(base_cb,base_cr,np.float32,lambda x: np.floor(x+np.float32(0.5))),
 'base_rint_even': mk(base_cb,base_cr,rounder=np.rint),
 'base_trunc': mk(base_cb,base_cr,rounder=np.floor),
 'exact_f32': mk(exact_cb,exact_cr,np.float32,lambda x: np.floor(x+np.float32(0.5))),
}
# integer libjpeg-style 16-bit fixed point (ffmpeg rgb->yuv "BT.601 limited" as in many tools)
def fixed(bits):
    def f(im):
        r,g,b=[im[...,i].astype(np.int64) for i in range(3)]
        q=lambda c: int(np.floor(c/255*(1<<bits)+0.5))
        cb=(q(base_cb[0])*r+q(base_cb[1])*g+q(base_cb[2])*b+(128<<bits)+(1<<(bits-1)))>>bits
        cr=(q(base_cr[0])*r+q(base_cr[1])*g+q(base_cr[2])*b+(128<<bits)+(1<<(bits-1)))>>bits
        return [cb.astype(float),cr.astype(float)]
    return f
for b in (8,14,15,16): V[f'fixed{b}']=fixed(b)
res={k:{'Cb':[],'Cr':[]} for k in V}
for p in pairs:
    R,D=load(p['ref_path']),load(p['dist_path']); row=pub[p['distorted']]
    for k,f in V.items():
        a=f(R); b2=f(D)
        res[k]['Cb'].append(abs(psnr(a[0],b2[0])-float(row['PSNR-Cb'])))
        res[k]['Cr'].append(abs(psnr(a[1],b2[1])-float(row['PSNR-Cr'])))
for k,v in res.items(): print(f"{k:16s}", '  '.join(f"{c}: med {np.median(a):.2e} max {np.max(a):.2e}" for c,a in v.items()))
